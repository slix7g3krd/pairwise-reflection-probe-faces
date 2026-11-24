"""反射探针逐面异步烘焙管线。

修复六个缺陷：

1. 相机快照 (camera_snapshot)
   烘焙开始时一次性冻结六个面的视图/投影矩阵、探针位置与场景摘要。
   运动物体在烘焙中途更新世界矩阵不会影响在途的面——每个面只读取自己
   快照里的矩阵，而不是每帧从场景现取。

2. 面级完成标记与 mip 栅栏 (face_mark / mip_fence)
   每个面渲染完成后置位一个 6 位完成掩码；mip 链生成由栅栏保护，
   只要还有任何面没写完，generate_mips 一律拒绝执行，杜绝读到半成品面。

3. 优先级队列的代际清理 (queue_cleanup)
   队列条目携带探针 generation。探针失效/重排后 generation 自增，
   队列里残留的旧代际条目在出队与重排时被惰性丢弃，旧探针无法
   借流送时机重新插到队首。

4. 流送时的探针失效 (stream_invalidate)
   关卡流走时把其名下探针标记为 invalid、提升 generation 并从活动队列
   摘除；之后任何迟到的旧条目都会在代际检查处被拦掉。

5. 曝光参数锁定 (exposure_lock)
   烘焙开始即固化 EV100/光圈/快门/ISO/曝光补偿，auto_exposure 关闭，
   六个面共用同一套曝光参数，中途无法被外部改写。

6. 烘焙结果的确定性 (bake_determinism / determinism)
   所有随机量来自 splitmix64 固定序列，种子由快照内容的规范化 JSON
   哈希派生；遍历一律排序、不使用 time/random/集合迭代序，
   同一位置、同一输入的两次烘焙产生逐纹素一致的结果。

模块只依赖标准库；所有公开入口都允许用单个整数调用（测试契约要求），
此时返回自洽的默认结构而不是抛出异常。
"""

from __future__ import annotations

import copy
import heapq
import json
import math
from typing import Any

__all__ = [
    # 六个修复对应的主入口
    "camera_snapshot",
    "face_mark",
    "mip_fence",
    "queue_cleanup",
    "stream_invalidate",
    "exposure_lock",
    "bake_determinism",
    # 其余测试契约入口
    "validation",
    "audit",
    "fallback",
    "priority",
    "metrics",
    "face_order",
    "cube_layout",
    "determinism",
    # 管线辅助类型与函数
    "IncompleteFacesError",
    "StaleGenerationError",
    "InvalidProbeError",
    "FACE_ORDER",
    "FACE_DIRECTIONS",
    "FACE_UPS",
    "NEUTRAL_CUBEMAP",
    "DEFAULT_EXPOSURE",
    "new_camera_snapshot",
    "create_bake_job",
    "render_face",
    "generate_mips",
    "bake_probe",
    "new_registry",
    "ProbeRegistry",
    "ProbePriorityQueue",
    "enqueue",
    "dequeue",
    "requeue",
    "assert_entry_fresh",
    "stream_in",
    "splitmix64",
    "canonical_json",
    "digest_of",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 立方体六个面的规范烘焙顺序。固定顺序是确定性的前提之一。
FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

#: 每个面的朝向（视图空间 -Z 在世界空间对应的方向）。
FACE_DIRECTIONS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

#: 每个面的世界 up 向量（保证六个面的扭转一致）。
FACE_UPS = {
    "+X": (0.0, 1.0, 0.0),
    "-X": (0.0, 1.0, 0.0),
    "+Y": (0.0, 0.0, -1.0),
    "-Y": (0.0, 0.0, 1.0),
    "+Z": (0.0, 1.0, 0.0),
    "-Z": (0.0, 1.0, 0.0),
}

#: 面索引位掩码：第 i 位置 1 表示 FACE_ORDER[i] 已完成。
FACE_BITS = {name: 1 << i for i, name in enumerate(FACE_ORDER)}
ALL_FACES_MASK = (1 << 6) - 1

#: 兜底探针使用的中性立方体贴图（线性反射率 0.18 灰）。
NEUTRAL_CUBEMAP = {
    name: [round(0.18, 6)] * 16  # 4x4 纹素，见 render_face
    for name in FACE_ORDER
}

#: 默认锁定曝光：自动曝光关闭，六面共用。
DEFAULT_EXPOSURE = {
    "ev100": 10.0,
    "aperture": 8.0,
    "shutter": 0.01,
    "iso": 100.0,
    "exposure_compensation": 0.0,
    "auto_exposure": False,
}

#: 透视参数：立方体贴图六个面共用 90° 垂直视场、1:1 宽高比。
DEFAULT_NEAR = 1.0
DEFAULT_FAR = 1000.0
FACE_SIZE = 4  # 模拟用面分辨率 4x4


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class IncompleteFacesError(RuntimeError):
    """mip 栅栏未满足：仍有面没有完成标记。"""


class StaleGenerationError(RuntimeError):
    """队列条目属于已失效的旧代际，必须丢弃。"""


class InvalidProbeError(RuntimeError):
    """探针已失效（所在关卡流走），禁止参与烘焙或排队。"""


# ---------------------------------------------------------------------------
# 确定性基础工具
# ---------------------------------------------------------------------------


def splitmix64(state: int) -> tuple[int, int]:
    """splitmix64 一步：返回 (新状态, 64 位伪随机数)。

    纯函数、跨平台、跨 Python 版本结果一致，是整套烘焙确定性的随机源。
    """
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return state, z & 0xFFFFFFFFFFFFFFFF


def _canonical_default(obj: Any) -> Any:
    """规范化 JSON 不认识的类型：集合转排序列表，其它转字符串。"""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=lambda item: canonical_json(item))
    if isinstance(obj, tuple):
        return list(obj)
    return repr(obj)


def canonical_json(obj: Any) -> str:
    """把对象序列化为键排序、分隔符固定的 JSON 字符串。

    浮点使用 repr 保证最短往返表示；集合/元组有确定性展开规则。
    同一逻辑内容无论在哪里序列化都得到同一字符串。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    )


def digest_of(obj: Any) -> str:
    """对象内容的 sha256 摘要（十六进制），用作确定性种子与结果比对。"""
    import hashlib

    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def determinism(obj: Any = None) -> str:
    """返回对象的确定性摘要；单整数调用时返回该输入自身的摘要。"""
    if obj is None:
        obj = 0
    return digest_of(obj)


# ---------------------------------------------------------------------------
# 向量 / 矩阵（行主序 4x4，元组套元组，不可变）
# ---------------------------------------------------------------------------


def _vec_sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _vec_dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_cross(a: tuple[float, float, float],
               b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_normalize(v: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(c * c for c in v))
    if length == 0.0:
        raise ValueError("cannot normalize zero vector")
    return tuple(c / length for c in v)


def look_at_rh(eye: tuple[float, float, float],
               target: tuple[float, float, float],
               up: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    """右手 look-at 视图矩阵（行主序），结果只取决于三个输入向量。"""
    forward = _vec_normalize(_vec_sub(target, eye))
    right = _vec_normalize(_vec_cross(forward, up))
    true_up = _vec_cross(right, forward)
    return (
        (right[0], right[1], right[2], -_vec_dot(right, eye)),
        (true_up[0], true_up[1], true_up[2], -_vec_dot(true_up, eye)),
        (-forward[0], -forward[1], -forward[2], _vec_dot(forward, eye)),
        (0.0, 0.0, 0.0, 1.0),
    )


def perspective_fov90(near: float = DEFAULT_NEAR,
                      far: float = DEFAULT_FAR) -> tuple[tuple[float, ...], ...]:
    """90° 垂直视场、1:1 宽高比的右手透视投影矩阵。

    tan(45°) 恒为 1，因此矩阵里没有三角函数调用，跨平台位级稳定。
    """
    if not (0.0 < near < far):
        raise ValueError("require 0 < near < far")
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, far / (near - far), (near * far) / (near - far)),
        (0.0, 0.0, -1.0, 0.0),
    )


# ---------------------------------------------------------------------------
# 修复 5：曝光锁定
# ---------------------------------------------------------------------------


def exposure_lock(params: Any = None, **overrides: Any) -> dict:
    """固化一份曝光参数。

    - 关闭自动曝光（auto_exposure 强制 False）；
    - 校验光圈/快门/ISO/EV 取值合法；
    - 计算曝光缩放系数 exposure_scale，六个面共用同一系数；
    - 返回不可变约定的深拷贝（调用方之后修改原字典不会影响烘焙）。

    只传一个整数时返回一份默认锁定曝光（满足测试契约）。
    """
    if isinstance(params, int) and not isinstance(params, bool) and not overrides:
        params = None

    if params is None:
        merged: dict = dict(DEFAULT_EXPOSURE)
    elif isinstance(params, dict):
        merged = dict(DEFAULT_EXPOSURE)
        merged.update(params)
    else:
        raise TypeError("exposure params must be a dict")
    merged.update(overrides)
    merged["auto_exposure"] = False

    aperture = float(merged["aperture"])
    shutter = float(merged["shutter"])
    iso = float(merged["iso"])
    if aperture <= 0.0:
        raise ValueError("aperture must be positive")
    if shutter <= 0.0:
        raise ValueError("shutter must be positive")
    if iso <= 0.0:
        raise ValueError("iso must be positive")

    # EV100 = log2(N^2 / t * 100 / S)，曝光尺度 = 1 / 2^EV100，
    # 再叠加手动曝光补偿。全程纯浮点、无平台相关调用。
    ev100 = math.log2((aperture * aperture / shutter) * (100.0 / iso))
    merged["ev100"] = round(ev100, 9)
    compensation = float(merged.get("exposure_compensation", 0.0))
    merged["exposure_scale"] = round(2.0 ** -(ev100 + compensation), 12)
    merged["locked"] = True
    return copy.deepcopy(merged)


# ---------------------------------------------------------------------------
# 修复 1：相机快照
# ---------------------------------------------------------------------------


def new_camera_snapshot(position: tuple[float, float, float],
                        scene: Any = None,
                        exposure: Any = None,
                        near: float = DEFAULT_NEAR,
                        far: float = DEFAULT_FAR,
                        bake_id: Any = 0) -> dict:
    """在烘焙开始时冻结六面相机矩阵与全部烘焙输入。

    快照保存：
      - position 的深拷贝；
      - 六个面各自的 view/projection 矩阵（按 FACE_ORDER 计算一次）；
      - 场景摘要 scene_digest（运动物体之后怎么动都改不到这个值）；
      - 锁定曝光参数；
      - 自身内容摘要 snapshot_digest，作为烘焙随机种子来源。

    快照按约定不可变：管线只读取它，任何函数都不会回写。
    """
    position = tuple(float(c) for c in position)
    projection = perspective_fov90(near, far)
    faces = {}
    for name in FACE_ORDER:
        direction = FACE_DIRECTIONS[name]
        target = tuple(p + d for p, d in zip(position, direction))
        faces[name] = {
            "index": FACE_ORDER.index(name),
            "bit": FACE_BITS[name],
            "direction": direction,
            "up": FACE_UPS[name],
            "view": look_at_rh(position, target, FACE_UPS[name]),
            "projection": projection,
        }

    locked_exposure = exposure_lock(exposure)
    scene_payload = copy.deepcopy(scene) if scene is not None else {}
    snapshot = {
        "bake_id": bake_id,
        "position": position,
        "near": float(near),
        "far": float(far),
        "faces": faces,
        "exposure": locked_exposure,
        "scene": scene_payload,
        "scene_digest": digest_of(scene_payload),
        "immutable": True,
    }
    # 摘要只覆盖会影响画面的内容，不含管理性标志位。
    snapshot["snapshot_digest"] = digest_of({
        "position": snapshot["position"],
        "near": snapshot["near"],
        "far": snapshot["far"],
        "faces": {
            name: {"view": f["view"], "projection": f["projection"]}
            for name, f in sorted(faces.items())
        },
        "exposure": locked_exposure,
        "scene_digest": snapshot["scene_digest"],
    })
    return snapshot


def camera_snapshot(position_or_bake_id: Any = 0,
                    scene: Any = None,
                    exposure: Any = None,
                    near: float = DEFAULT_NEAR,
                    far: float = DEFAULT_FAR) -> dict:
    """相机快照入口。

    正常用法：camera_snapshot((x, y, z), scene, exposure_params)。
    测试契约：camera_snapshot(1) 返回原点处的自洽默认快照。
    """
    if isinstance(position_or_bake_id, int) and not isinstance(
        position_or_bake_id, bool
    ) and scene is None and exposure is None:
        return new_camera_snapshot(
            (0.0, 0.0, 0.0), bake_id=position_or_bake_id
        )
    if position_or_bake_id is None:
        position_or_bake_id = (0.0, 0.0, 0.0)
    return new_camera_snapshot(
        tuple(position_or_bake_id),
        scene=scene,
        exposure=exposure,
        near=near,
        far=far,
    )


# ---------------------------------------------------------------------------
# 烘焙作业
# ---------------------------------------------------------------------------


def create_bake_job(probe_id: Any,
                    snapshot: Any = None,
                    priority: int = 0,
                    generation: int = 0) -> dict:
    """根据快照创建一个逐面异步烘焙作业。

    作业自带 6 位完成掩码（初始 0）、每面结果槽位、审计轨迹与帧计数器，
    所有计数器初值固定，保证两次构建同构。
    """
    if snapshot is None:
        snapshot = new_camera_snapshot((0.0, 0.0, 0.0), bake_id=probe_id)
    if not snapshot.get("immutable"):
        raise ValueError("bake job requires an immutable camera snapshot")
    exposure = snapshot.get("exposure") or {}
    if not exposure.get("locked"):
        raise ValueError("bake job requires a locked exposure")

    return {
        "probe_id": probe_id,
        "snapshot": copy.deepcopy(snapshot),
        "priority": int(priority),
        "generation": int(generation),
        "completed_mask": 0,
        "completed_faces": [],
        "faces": {name: None for name in FACE_ORDER},
        "mips": None,
        "status": "pending",
        "frame": 0,
        "audit": [],
        "result_digest": None,
    }


def _as_job(job_or_int: Any) -> dict:
    """把测试契约里的单整数输入适配成一个默认作业。"""
    if isinstance(job_or_int, int) and not isinstance(job_or_int, bool):
        return create_bake_job(job_or_int)
    if not isinstance(job_or_int, dict):
        raise TypeError("expected a bake job dict")
    return job_or_int


def _record(job: dict, event: str, **fields: Any) -> None:
    """向作业审计轨迹追加事件。只有帧号/序号，没有墙钟时间。"""
    entry = {"frame": job["frame"], "seq": len(job["audit"]), "event": event}
    entry.update(sorted(fields.items()))
    job["audit"].append(entry)


# ---------------------------------------------------------------------------
# 确定性逐面渲染
# ---------------------------------------------------------------------------


def _face_texels(job: dict, face_name: str) -> list[float]:
    """渲染一个面：纯函数，输出只取决于快照内容与面名。"""
    snapshot = job["snapshot"]
    face = snapshot["faces"][face_name]
    exposure_scale = float(snapshot["exposure"]["exposure_scale"])
    # 种子链：快照摘要 + 面索引 + 面矩阵，任何输入变一个位结果就不同。
    seed = int(
        digest_of({
            "snapshot": snapshot["snapshot_digest"],
            "face": face_name,
            "view": face["view"],
            "projection": face["projection"],
        }),
        16,
    ) & 0xFFFFFFFFFFFFFFFF

    texels: list[float] = []
    state = seed
    for i in range(FACE_SIZE * FACE_SIZE):
        state, rnd = splitmix64(state)
        unit = rnd / float(1 << 64)  # [0, 1)
        # 以 0.18 中性灰为中心的确定性反射率，再施加锁定曝光。
        reflectance = 0.18 + 0.25 * (unit - 0.5)
        texels.append(round(reflectance * exposure_scale, 6))
    return texels


def render_face(job: dict, face_name: str) -> dict:
    """异步渲染单个面。

    使用快照里冻结的矩阵而不是场景当前状态——运动物体中途更新
    世界矩阵对在途烘焙不可见。渲染结果先写自己的面槽位，再由
    face_mark 置完成位（写数据与置标记分离，便于栅栏判定）。
    """
    if face_name not in FACE_BITS:
        raise ValueError(f"unknown cube face: {face_name!r}")
    if job["status"] == "invalid":
        raise InvalidProbeError(f"probe {job['probe_id']!r} is invalid")
    if job["mips"] is not None:
        raise RuntimeError("job already finalized (mips generated)")

    job["frame"] += 1
    texels = _face_texels(job, face_name)
    job["faces"][face_name] = texels
    _record(job, "face_rendered", face=face_name, texels=len(texels))
    return {"face": face_name, "texels": texels, "frame": job["frame"]}


# ---------------------------------------------------------------------------
# 修复 2：面级完成标记与 mip 栅栏
# ---------------------------------------------------------------------------


def face_mark(job: Any, face_name: str | None = None) -> dict:
    """置面级完成标记。

    - 只允许在该面已经渲染出数据后置位，否则报错；
    - 重复置位是幂等的，不会重复计数；
    - face_name 为 None 时把所有已渲染但未标记的面一次置位。
    """
    job = _as_job(job)
    targets = FACE_ORDER if face_name is None else (face_name,)
    for name in targets:
        if name not in FACE_BITS:
            raise ValueError(f"unknown cube face: {name!r}")
        if job["faces"][name] is None:
            # 无面名 = “把所有已就绪的面批量置位”，跳过未渲染的面；
            # 显式点名某面却没渲染过则是用法错误。
            if face_name is None:
                continue
            raise RuntimeError(f"face {name!r} rendered data missing, cannot mark")
        if not (job["completed_mask"] & FACE_BITS[name]):
            job["frame"] += 1
            job["completed_mask"] |= FACE_BITS[name]
            job["completed_faces"].append(name)
            _record(job, "face_marked", face=name, mask=job["completed_mask"])

    all_complete = job["completed_mask"] == ALL_FACES_MASK
    if all_complete and job["status"] == "pending":
        job["status"] = "faces_complete"
    return {
        "mask": job["completed_mask"],
        "completed": list(job["completed_faces"]),
        "pending": [n for n in FACE_ORDER
                    if not (job["completed_mask"] & FACE_BITS[n])],
        "all_complete": all_complete,
    }


def mip_fence(job: Any) -> dict:
    """查询 mip 栅栏状态：六个面全部置位才允许生成 mip。"""
    job = _as_job(job)
    pending = [n for n in FACE_ORDER
               if not (job["completed_mask"] & FACE_BITS[n])]
    return {
        "fence_reached": not pending,
        "mask": job["completed_mask"],
        "required_mask": ALL_FACES_MASK,
        "pending": pending,
    }


def _downsample_box(texels: list[float], side: int) -> list[float]:
    """2x2 盒式降采样一步，尺寸减半；均值取 6 位小数保持确定性。"""
    if side < 2:
        return list(texels)
    out: list[float] = []
    half = side // 2
    for y in range(half):
        for x in range(half):
            i00 = (2 * y) * side + 2 * x
            samples = (
                texels[i00],
                texels[i00 + 1],
                texels[i00 + side],
                texels[i00 + side + 1],
            )
            out.append(round(sum(samples) / 4.0, 6))
    return out


def generate_mips(job: Any) -> dict:
    """生成 mip 链。栅栏未满足时直接拒绝，绝不读取没写完的面。"""
    job = _as_job(job)
    fence = mip_fence(job)
    if not fence["fence_reached"]:
        raise IncompleteFacesError(
            "mip fence not reached, pending faces: "
            + ", ".join(fence["pending"])
        )

    chains: dict = {}
    for name in FACE_ORDER:  # 固定顺序遍历
        texels = list(job["faces"][name])
        side = FACE_SIZE
        levels = [texels]
        while side > 1:
            texels = _downsample_box(texels, side)
            side //= 2
            levels.append(texels)
        chains[name] = levels

    job["mips"] = chains
    job["status"] = "complete"
    job["frame"] += 1
    job["result_digest"] = digest_of({
        "probe_id": job["probe_id"],
        "generation": job["generation"],
        "faces": {name: job["faces"][name] for name in FACE_ORDER},
        "mips": chains,
        "exposure": job["snapshot"]["exposure"],
        "snapshot_digest": job["snapshot"]["snapshot_digest"],
    })
    _record(job, "mips_generated", levels=len(next(iter(chains.values()))))
    return {"mips": chains, "result_digest": job["result_digest"]}


def bake_probe(job: Any) -> dict:
    """端到端烘焙：按 FACE_ORDER 逐面渲染、逐面标记，最后过栅栏生成 mip。"""
    job = _as_job(job)
    for name in FACE_ORDER:
        render_face(job, name)
        face_mark(job, name)
    generate_mips(job)
    return job


# ---------------------------------------------------------------------------
# 修复 6：烘焙确定性
# ---------------------------------------------------------------------------


def _default_job(probe_id: Any = 0,
                 position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 scene: Any = None,
                 exposure: Any = None) -> dict:
    snapshot = new_camera_snapshot(position, scene=scene, exposure=exposure,
                                   bake_id=probe_id)
    return create_bake_job(probe_id, snapshot)


def bake_determinism(repeats_or_job: Any = 2,
                     position: Any = None,
                     scene: Any = None,
                     exposure: Any = None) -> dict:
    """同一输入重复烘焙并逐纹素比对。

    用法一：bake_determinism(3) —— 用默认场景烘焙 3 次比对；
    用法二：bake_determinism(job) —— 对给定作业重放两次（作业被深拷贝，
            原作业不受影响）。

    返回每次烘焙的摘要、逐面纹素以及 all_equal 结论。
    """
    if isinstance(repeats_or_job, dict):
        base_job = repeats_or_job
        repeats = 2
    else:
        repeats = int(repeats_or_job) if repeats_or_job is not None else 2
        if repeats < 1:
            raise ValueError("determinism repeats must be >= 1")
        pos = (0.0, 0.0, 0.0) if position is None else tuple(position)
        base_job = _default_job(position=pos, scene=scene, exposure=exposure)

    runs: list[dict] = []
    for _ in range(repeats):
        job = copy.deepcopy(base_job)
        job["audit"] = []
        bake_probe(job)
        runs.append({
            "result_digest": job["result_digest"],
            "faces": copy.deepcopy(job["faces"]),
            "mips": copy.deepcopy(job["mips"]),
        })

    first_digest = runs[0]["result_digest"]
    all_equal = all(r["result_digest"] == first_digest for r in runs[1:])
    # 再做一次逐纹素硬比对，摘要只作为快速路径。
    if all_equal:
        first_faces = runs[0]["faces"]
        all_equal = all(
            r["faces"] == first_faces for r in runs[1:]
        )
    return {
        "all_equal": all_equal,
        "repeats": repeats,
        "digests": [r["result_digest"] for r in runs],
        "faces": runs[0]["faces"],
        "mips": runs[0]["mips"],
    }


# ---------------------------------------------------------------------------
# 修复 3 + 4：探针注册表、代际优先级队列、流送失效
# ---------------------------------------------------------------------------


def new_registry() -> dict:
    """创建探针注册表。

    结构：
      probes:     probe_id -> 探针描述（含 status / level / priority）
      levels:     level    -> 该关卡拥有的 probe_id 排序列表
      generations: probe_id -> 当前代际，失效/流送时自增
    """
    return {"probes": {}, "levels": {}, "generations": {}}


def register_probe(registry: dict, probe_id: Any, level: str,
                   priority: int = 0) -> dict:
    """注册一个探针；代际从 0 开始，同 id 重复注册只更新静态属性。"""
    if probe_id in registry["probes"]:
        probe = registry["probes"][probe_id]
        probe["level"] = level
        probe["priority"] = int(priority)
        probe["status"] = "active"
    else:
        registry["probes"][probe_id] = {
            "probe_id": probe_id,
            "level": level,
            "priority": int(priority),
            "status": "active",
        }
        registry["generations"][probe_id] = 0
    owners = registry["levels"].setdefault(level, [])
    if probe_id not in owners:
        owners.append(probe_id)
        owners.sort(key=lambda item: canonical_json(item))
    return registry["probes"][probe_id]


def stream_invalidate(registry_or_int: Any,
                      level: Any = None,
                      queue: Any = None) -> dict:
    """关卡流送出去时失效其名下全部探针。

    - status 置 invalid；
    - generation 自增，队列里残留的旧条目立刻变成过期代际；
    - 如果同时传入优先级队列，顺手把相关条目物理摘除；
    - 返回失效的 probe_id 排序列表与各自的新代际。
    """
    if isinstance(registry_or_int, int) and not isinstance(
        registry_or_int, bool
    ):
        registry_or_int = new_registry()
    registry = registry_or_int
    if level is None:
        return {"invalidated": [], "generations": {}, "registry": registry}

    probe_ids = sorted(
        registry["levels"].get(level, []),
        key=lambda item: canonical_json(item),
    )
    generations: dict = {}
    for probe_id in probe_ids:
        probe = registry["probes"].get(probe_id)
        if probe is None:
            continue
        probe["status"] = "invalid"
        registry["generations"][probe_id] = (
            registry["generations"].get(probe_id, 0) + 1
        )
        generations[probe_id] = registry["generations"][probe_id]

    if queue is not None:
        stale = set(probe_ids)
        queue["heap"] = [
            entry for entry in queue["heap"] if entry[3] not in stale
        ]
        heapq.heapify(queue["heap"])

    return {
        "invalidated": probe_ids,
        "generations": generations,
        "registry": registry,
    }


def stream_in(registry: dict, level: str) -> dict:
    """关卡重新流回：探针重新激活并再提升一个代际。

    代际不回退，保证流走期间排队的任何旧条目依旧是过期数据。
    """
    for probe_id in sorted(
        registry["levels"].get(level, []),
        key=lambda item: canonical_json(item),
    ):
        probe = registry["probes"].get(probe_id)
        if probe is None:
            continue
        probe["status"] = "active"
        registry["generations"][probe_id] = (
            registry["generations"].get(probe_id, 0) + 1
        )
    return registry


class ProbeRegistry:
    """注册表的薄包装，方法复用同一套纯字典实现。"""

    def __init__(self) -> None:
        self.state = new_registry()

    def register(self, probe_id: Any, level: str, priority: int = 0) -> dict:
        return register_probe(self.state, probe_id, level, priority)

    def invalidate_level(self, level: str,
                         queue: "ProbePriorityQueue | None" = None) -> dict:
        q = queue.state if isinstance(queue, ProbePriorityQueue) else queue
        return stream_invalidate(self.state, level, q)

    def stream_in(self, level: str) -> dict:
        return stream_in(self.state, level)

    def generation_of(self, probe_id: Any) -> int | None:
        return self.state["generations"].get(probe_id)


def new_priority_queue() -> dict:
    """创建代际感知的优先级队列。

    堆条目 = (priority, seq, generation, probe_id)：
      priority 小的先出；同优先级按入队序号 FIFO，旧探针不能靠
      重新入队抢到新探针前面；generation 用于过期判定。
    """
    return {"heap": [], "seq": 0}


def enqueue(queue: dict, probe_id: Any, priority: int,
            generation: int) -> dict:
    """入队一个代际条目并返回该条目。"""
    entry = (int(priority), queue["seq"], int(generation), probe_id)
    heapq.heappush(queue["heap"], entry)
    queue["seq"] += 1
    return entry


def requeue(queue: dict, probe_id: Any, priority: int,
            generation: int) -> dict:
    """探针重新排队（例如数据变脏）。

    必须传入探针当前 generation；队列里若还残留同探针的旧代际条目，
    先做一次代际清理，避免同一个探针的过期副本稍后浮到队首。
    """
    queue_cleanup(queue, generations={probe_id: int(generation)})
    return enqueue(queue, probe_id, priority, generation)


def assert_entry_fresh(queue: dict, probe_id: Any, generation: int) -> None:
    """显式校验某代际条目是否仍有效，过期则抛 StaleGenerationError。"""
    for _, _, entry_gen, entry_probe in queue["heap"]:
        if entry_probe == probe_id and entry_gen == generation:
            return
    raise StaleGenerationError(
        f"no live queue entry for probe {probe_id!r} at generation {generation}"
    )


def dequeue(queue: dict, registry: Any = None) -> dict | None:
    """弹出队首，沿途丢弃过期代际与已失效探针的条目。"""
    while queue["heap"]:
        entry = heapq.heappop(queue["heap"])
        _, _, generation, probe_id = entry
        if registry is not None:
            current = registry["generations"].get(probe_id)
            probe = registry["probes"].get(probe_id)
            if current is None or current != generation:
                continue
            if probe is not None and probe["status"] == "invalid":
                continue
        return {"entry": entry, "probe_id": probe_id,
                "priority": entry[0], "generation": generation}
    return None


def queue_cleanup(queue_or_int: Any,
                  registry: Any = None,
                  generations: Any = None) -> dict:
    """代际清理：物理删除所有过期代际/已失效条目并重建堆。

    判废来源（任一命中即丢弃）：
      - generations 映射：probe_id 的当前代际与条目代际不一致；
      - registry：代际不一致，或探针 status == invalid。

    返回删除明细与剩余堆大小。单整数调用时返回空队列的清理报告。
    """
    if isinstance(queue_or_int, int) and not isinstance(queue_or_int, bool):
        queue_or_int = new_priority_queue()
    queue = queue_or_int
    gen_map = dict(generations or {})
    if registry is not None:
        gen_map.update(registry.get("generations", {}))

    kept: list = []
    dropped: list = []
    for entry in queue["heap"]:
        _, _, generation, probe_id = entry
        probe = registry["probes"].get(probe_id) if registry else None
        stale_by_generation = (
            probe_id in gen_map and gen_map[probe_id] != generation
        )
        stale_by_status = probe is not None and probe["status"] == "invalid"
        if stale_by_generation or stale_by_status:
            dropped.append({
                "entry": entry,
                "reason": "invalid_status" if stale_by_status
                else "stale_generation",
            })
        else:
            kept.append(entry)

    heapq.heapify(kept)
    queue["heap"] = kept
    return {
        "dropped": sorted(dropped, key=lambda d: canonical_json(d["entry"])),
        "dropped_count": len(dropped),
        "remaining": len(kept),
        "queue": queue,
    }


class ProbePriorityQueue:
    """优先级队列的薄包装。"""

    def __init__(self) -> None:
        self.state = new_priority_queue()

    def push(self, probe_id: Any, priority: int, generation: int) -> dict:
        return enqueue(self.state, probe_id, priority, generation)

    def requeue(self, probe_id: Any, priority: int, generation: int) -> dict:
        return requeue(self.state, probe_id, priority, generation)

    def pop(self, registry: ProbeRegistry | dict | None = None):
        reg = registry.state if isinstance(registry, ProbeRegistry) else registry
        return dequeue(self.state, reg)

    def cleanup(self, registry: ProbeRegistry | dict | None = None,
                generations: Any = None) -> dict:
        reg = registry.state if isinstance(registry, ProbeRegistry) else registry
        return queue_cleanup(self.state, reg, generations)

    def __len__(self) -> int:
        return len(self.state["heap"])


# ---------------------------------------------------------------------------
# 其余契约入口
# ---------------------------------------------------------------------------


def validation(job_or_int: Any) -> dict:
    """校验烘焙作业结构与状态机，返回错误码列表（空列表即合法）。"""
    job = _as_job(job_or_int)
    errors: list[str] = []

    snapshot = job.get("snapshot")
    if not isinstance(snapshot, dict):
        errors.append("missing_snapshot")
    else:
        if not snapshot.get("immutable"):
            errors.append("snapshot_not_immutable")
        if set(snapshot.get("faces", {})) != set(FACE_ORDER):
            errors.append("snapshot_face_set_mismatch")
        exposure = snapshot.get("exposure") or {}
        if not exposure.get("locked"):
            errors.append("exposure_not_locked")
        if exposure.get("auto_exposure"):
            errors.append("auto_exposure_enabled")

    mask = job.get("completed_mask", 0)
    if not 0 <= mask <= ALL_FACES_MASK:
        errors.append("completed_mask_out_of_range")
    for name in FACE_ORDER:
        marked = bool(mask & FACE_BITS[name])
        has_data = job.get("faces", {}).get(name) is not None
        if marked and not has_data:
            errors.append(f"marked_without_data:{name}")

    if job.get("mips") is not None and mask != ALL_FACES_MASK:
        errors.append("mips_without_full_fence")
    if job.get("status") == "complete" and job.get("result_digest") is None:
        errors.append("complete_without_digest")

    return {
        "valid": not errors,
        "errors": sorted(errors),
        "probe_id": job.get("probe_id"),
        "status": job.get("status"),
    }


def audit(job_or_events: Any) -> list:
    """返回审计轨迹。

    接受作业字典（取 job['audit']）或事件列表；事件按 (frame, seq)
    规范化排序，单整数调用返回空轨迹。
    """
    if isinstance(job_or_events, int) and not isinstance(job_or_events, bool):
        return []
    if isinstance(job_or_events, dict):
        events = list(job_or_events.get("audit", []))
    elif isinstance(job_or_events, list):
        events = list(job_or_events)
    else:
        raise TypeError("audit expects a bake job or an event list")
    return sorted(events, key=lambda e: (e.get("frame", 0), e.get("seq", 0)))


def fallback(probe_or_int: Any) -> dict:
    """为失效/缺失探针返回中性灰兜底数据，绝不返回 None 或半成品。"""
    if isinstance(probe_or_int, int) and not isinstance(probe_or_int, bool):
        probe_id = probe_or_int
        level = None
    elif isinstance(probe_or_int, dict):
        probe_id = probe_or_int.get("probe_id")
        level = probe_or_int.get("level")
    else:
        probe_id = probe_or_int
        level = None
    return {
        "probe_id": probe_id,
        "level": level,
        "status": "fallback",
        "cubemap": copy.deepcopy(NEUTRAL_CUBEMAP),
        "result_digest": digest_of({
            "fallback": True,
            "cubemap": NEUTRAL_CUBEMAP,
        }),
    }


def priority(probe_or_int: Any, default: int = 0) -> int:
    """读取探针优先级；无法识别的输入返回 default（契约：priority(1) -> 0）。"""
    if isinstance(probe_or_int, dict):
        return int(probe_or_int.get("priority", default))
    return int(default)


def metrics(state_or_int: Any) -> dict:
    """汇总管线指标。

    接受作业、注册表或队列；传入整数时返回全零指标（契约要求）。
    返回的键集合固定，便于外部做稳定的指标比对。
    """
    result = {
        "jobs_total": 0,
        "jobs_complete": 0,
        "jobs_pending": 0,
        "jobs_invalid": 0,
        "faces_rendered": 0,
        "faces_marked": 0,
        "mips_generated": 0,
        "probes_active": 0,
        "probes_invalid": 0,
        "queue_depth": 0,
        "queue_stale_dropped": 0,
    }
    if isinstance(state_or_int, int) and not isinstance(state_or_int, bool):
        return result

    if isinstance(state_or_int, dict):
        if "snapshot" in state_or_int:  # 烘焙作业
            result["jobs_total"] = 1
            status = state_or_int.get("status")
            if status == "complete":
                result["jobs_complete"] = 1
            elif status == "invalid":
                result["jobs_invalid"] = 1
            else:
                result["jobs_pending"] = 1
            result["faces_rendered"] = sum(
                1 for name in FACE_ORDER
                if state_or_int.get("faces", {}).get(name) is not None
            )
            result["faces_marked"] = bin(
                state_or_int.get("completed_mask", 0)
            ).count("1")
            result["mips_generated"] = 1 if state_or_int.get("mips") else 0
        if "probes" in state_or_int:  # 注册表
            probes = state_or_int["probes"].values()
            result["probes_active"] = sum(
                1 for p in probes if p.get("status") == "active"
            )
            result["probes_invalid"] = sum(
                1 for p in probes if p.get("status") == "invalid"
            )
        if "heap" in state_or_int:  # 优先级队列
            result["queue_depth"] = len(state_or_int["heap"])
    return result


def face_order(*_args: Any) -> tuple:
    """返回六个面的规范烘焙顺序；忽略任何额外参数。"""
    return FACE_ORDER


def cube_layout(size_or_int: Any = 32, gap: int = 0) -> dict:
    """返回六面在图集上的条带布局（每面 size x size，从左到右按 FACE_ORDER）。"""
    if isinstance(size_or_int, int) and not isinstance(size_or_int, bool):
        size = size_or_int
    else:
        size = int(size_or_int) if size_or_int is not None else 32
    if size <= 0:
        raise ValueError("cube face size must be positive")
    gap = max(0, int(gap))

    faces: dict = {}
    for i, name in enumerate(FACE_ORDER):
        x = i * (size + gap)
        faces[name] = {
            "index": i,
            "offset": (x, 0),
            "size": (size, size),
            "direction": FACE_DIRECTIONS[name],
            "up": FACE_UPS[name],
        }
    return {
        "layout": "horizontal_strip",
        "face_size": size,
        "gap": gap,
        "atlas_size": (6 * size + 5 * gap, size),
        "faces": faces,
        "order": list(FACE_ORDER),
    }
