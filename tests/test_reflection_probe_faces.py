import pytest

import graphics.reflection_probe_faces as mod
from graphics.reflection_probe_faces import (
    FACE_ORDER,
    IncompleteFacesError,
    StaleGenerationError,
    bake_probe,
    camera_snapshot,
    create_bake_job,
    cube_layout,
    dequeue,
    enqueue,
    exposure_lock,
    face_mark,
    face_order,
    fallback,
    generate_mips,
    metrics,
    mip_fence,
    new_priority_queue,
    new_registry,
    queue_cleanup,
    register_probe,
    render_face,
    stream_in,
    stream_invalidate,
    bake_determinism,
    determinism,
    audit,
    validation,
)

# 注意：原用例写成 (1)，在 Python 中是整数而非单元素元组，
# fn(*1) 会在调用函数前由字节码抛出 TypeError，任何实现都无法满足。
# 修正为 (1,) 后，契约语义（每个入口接受单整数并返回非 None）保持不变。
CASES = [
    ("camera_snapshot", (1,)),
    ("face_mark", (1,)),
    ("mip_fence", (1,)),
    ("queue_cleanup", (1,)),
    ("stream_invalidate", (1,)),
    ("exposure_lock", (1,)),
    ("bake_determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("priority", (1,)),
    ("metrics", (1,)),
    ("face_order", (1,)),
    ("cube_layout", (1,)),
    ("determinism", (1,)),
]


@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None


# ---------------------------------------------------------------------------
# 修复 1：六面渲染的相机快照
# ---------------------------------------------------------------------------


def _moving_scene():
    return {
        "objects": [
            {"id": "cube", "position": [1.0, 0.0, 0.0]},
            {"id": "orb", "position": [0.0, 2.0, -3.0]},
        ]
    }


def test_snapshot_freezes_six_face_matrices():
    scene = _moving_scene()
    snap = camera_snapshot((0.0, 0.0, 0.0), scene=scene)
    assert snap["immutable"] is True
    assert list(snap["faces"]) == list(FACE_ORDER)
    # 六个面各有独立视图矩阵，共享同一投影矩阵
    views = {name: snap["faces"][name]["view"] for name in FACE_ORDER}
    assert len({repr(v) for v in views.values()}) == 6
    projections = {repr(snap["faces"][name]["projection"]) for name in FACE_ORDER}
    assert len(projections) == 1


def test_snapshot_immune_to_scene_mutation_mid_bake():
    scene = _moving_scene()
    snap1 = camera_snapshot((0.0, 0.0, 0.0), scene=scene)

    # 烘焙中途运动物体更新世界矩阵
    scene["objects"][0]["position"] = [99.0, 99.0, 99.0]
    snap2 = camera_snapshot((0.0, 0.0, 0.0), scene=_moving_scene())

    # 旧快照内容不受影响；同输入重建的快照与旧快照逐位一致
    assert snap1["scene_digest"] == snap2["scene_digest"]
    assert snap1["snapshot_digest"] == snap2["snapshot_digest"]
    for name in FACE_ORDER:
        assert snap1["faces"][name]["view"] == snap2["faces"][name]["view"]


def test_face_render_reads_snapshot_not_live_scene():
    scene = _moving_scene()
    snap = camera_snapshot((0.0, 0.0, 0.0), scene=scene)
    job = create_bake_job("p1", snap)

    render_face(job, "+X")
    first = list(job["faces"]["+X"])

    # 场景随后被运动物体改写，再渲染其它面，结果仍只由快照决定
    scene["objects"][0]["position"] = [99.0, 99.0, 99.0]
    render_face(job, "-X")
    job2 = create_bake_job("p2", camera_snapshot((0.0, 0.0, 0.0),
                                                 scene=_moving_scene()))
    render_face(job2, "+X")
    assert job["faces"]["+X"] == first == job2["faces"]["+X"]


# ---------------------------------------------------------------------------
# 修复 2：面级完成标记与 mip 栅栏
# ---------------------------------------------------------------------------


def test_face_mark_sets_bits_individually():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    render_face(job, "+X")
    mark = face_mark(job, "+X")
    assert mark["mask"] == 1
    assert mark["all_complete"] is False
    # 幂等：重复标记不改变掩码
    assert face_mark(job, "+X")["mask"] == 1


def test_mark_requires_rendered_data():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    with pytest.raises(RuntimeError):
        face_mark(job, "+X")  # 还没渲染就想置完成位


def test_mip_fence_blocks_until_all_six_faces_done():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    for i, name in enumerate(FACE_ORDER):
        render_face(job, name)
        fence = mip_fence(job)
        if i < 5:
            assert fence["fence_reached"] is False
            with pytest.raises(IncompleteFacesError):
                generate_mips(job)  # 未完成的面绝不允许被 mip 读取
        face_mark(job, name)

    assert mip_fence(job)["fence_reached"] is True
    result = generate_mips(job)
    assert len(result["mips"]["+X"]) == 3  # 4 -> 2 -> 1 三级
    assert job["status"] == "complete"


def test_fence_reports_pending_faces():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    render_face(job, "+Z")
    face_mark(job, "+Z")
    fence = mip_fence(job)
    assert "+Z" not in fence["pending"]
    assert set(fence["pending"]) == set(FACE_ORDER) - {"+Z"}
    assert fence["mask"] == 1 << FACE_ORDER.index("+Z")


# ---------------------------------------------------------------------------
# 修复 3：优先级队列的代际清理
# ---------------------------------------------------------------------------


def test_queue_drops_stale_generation_entries():
    queue = new_priority_queue()
    enqueue(queue, "probeA", priority=5, generation=0)
    enqueue(queue, "probeB", priority=9, generation=0)

    # probeA 失效后以新代际重新排队
    enqueue(queue, "probeA", priority=5, generation=1)
    report = queue_cleanup(queue, generations={"probeA": 1, "probeB": 0})
    assert report["dropped_count"] == 1
    assert report["dropped"][0]["reason"] == "stale_generation"

    out = dequeue(queue)
    assert out["probe_id"] == "probeA" and out["generation"] == 1


def test_stale_generation_raises_typed_error():
    queue = new_priority_queue()
    enqueue(queue, "probeA", priority=5, generation=0)
    queue_cleanup(queue, generations={"probeA": 1})
    with pytest.raises(StaleGenerationError):
        mod.assert_entry_fresh(queue, "probeA", 0)


def test_stale_probe_cannot_reach_queue_head_on_streaming():
    """复现缺陷：流送时旧探针被重新排到队首。"""
    registry = new_registry()
    register_probe(registry, "old", level="L1", priority=1)
    register_probe(registry, "new", level="L2", priority=5)
    queue = new_priority_queue()
    enqueue(queue, "old", 1, registry["generations"]["old"])
    enqueue(queue, "new", 5, registry["generations"]["new"])

    # L1 流走：old 失效、代际自增
    stream_invalidate(registry, "L1", queue)
    assert registry["probes"]["old"]["status"] == "invalid"

    # 即使堆里还有 old 的物理条目，出队也必须跳过它
    popped = dequeue(queue, registry)
    assert popped["probe_id"] == "new"
    assert dequeue(queue, registry) is None


def test_same_priority_is_fifo_no_requeue_jumping():
    queue = new_priority_queue()
    enqueue(queue, "a", priority=3, generation=0)
    enqueue(queue, "b", priority=3, generation=0)
    # b 以同优先级重新排队，不能插到 a 前面
    queue_cleanup(queue, generations={"b": 1})
    enqueue(queue, "b", priority=3, generation=1)
    order = [dequeue(queue)["probe_id"], dequeue(queue)["probe_id"]]
    assert order == ["a", "b"]


# ---------------------------------------------------------------------------
# 修复 4：流送时的探针失效
# ---------------------------------------------------------------------------


def test_stream_invalidate_marks_all_level_probes():
    registry = new_registry()
    register_probe(registry, "p1", "Town", priority=0)
    register_probe(registry, "p2", "Town", priority=1)
    register_probe(registry, "p3", "Forest", priority=2)

    result = stream_invalidate(registry, "Town")
    assert result["invalidated"] == ["p1", "p2"]
    assert registry["probes"]["p1"]["status"] == "invalid"
    assert registry["probes"]["p2"]["status"] == "invalid"
    assert registry["probes"]["p3"]["status"] == "active"
    # 代际单调递增
    assert registry["generations"]["p1"] == 1
    assert registry["generations"]["p2"] == 1


def test_stream_in_keeps_generation_forward_only():
    registry = new_registry()
    register_probe(registry, "p1", "Town")
    stream_invalidate(registry, "Town")
    stream_in(registry, "Town")
    assert registry["probes"]["p1"]["status"] == "active"
    assert registry["generations"]["p1"] == 2  # 不回退到 0


def test_invalid_probe_rejects_baking():
    registry = new_registry()
    register_probe(registry, "p1", "Town")
    job = create_bake_job("p1", camera_snapshot((0.0, 0.0, 0.0)))
    stream_invalidate(registry, "Town")
    job["status"] = "invalid"
    with pytest.raises(Exception):
        render_face(job, "+X")


# ---------------------------------------------------------------------------
# 修复 5：曝光参数锁定
# ---------------------------------------------------------------------------


def test_exposure_lock_is_immutable_and_shared():
    params = {"ev100": 9.0, "aperture": 5.6, "shutter": 0.02, "iso": 200.0}
    locked = exposure_lock(params)
    assert locked["locked"] is True
    assert locked["auto_exposure"] is False

    snap = camera_snapshot((0.0, 0.0, 0.0), exposure=params)
    # 六面共用同一份锁定曝光
    assert snap["exposure"]["exposure_scale"] == locked["exposure_scale"]

    # 外部事后改输入字典，快照不受影响
    params["iso"] = 3200.0
    assert snap["exposure"]["iso"] == 200.0


def test_exposure_lock_rejects_bad_values():
    with pytest.raises(ValueError):
        exposure_lock({"aperture": 0.0})
    with pytest.raises(ValueError):
        exposure_lock({"shutter": -1.0})
    with pytest.raises(ValueError):
        exposure_lock({"iso": 0.0})


def test_auto_exposure_forced_off():
    locked = exposure_lock({"auto_exposure": True})
    assert locked["auto_exposure"] is False


# ---------------------------------------------------------------------------
# 修复 6：烘焙结果的确定性
# ---------------------------------------------------------------------------


def test_same_position_bakes_identically():
    scene = _moving_scene()
    job_a = create_bake_job("p", camera_snapshot((1.0, 2.0, 3.0), scene=scene))
    job_b = create_bake_job("p", camera_snapshot((1.0, 2.0, 3.0), scene=scene))
    bake_probe(job_a)
    bake_probe(job_b)
    assert job_a["result_digest"] == job_b["result_digest"]
    assert job_a["faces"] == job_b["faces"]
    assert job_a["mips"] == job_b["mips"]


def test_bake_determinism_report():
    report = bake_determinism(3, position=(1.0, 2.0, 3.0), scene=_moving_scene())
    assert report["all_equal"] is True
    assert report["repeats"] == 3
    assert len(set(report["digests"])) == 1


def test_different_inputs_produce_different_outputs():
    job1 = create_bake_job("a", camera_snapshot((0.0, 0.0, 0.0),
                                                scene=_moving_scene()))
    scene2 = _moving_scene()
    scene2["objects"][0]["position"] = [5.0, 5.0, 5.0]
    job2 = create_bake_job("b", camera_snapshot((0.0, 0.0, 0.0), scene=scene2))
    bake_probe(job1)
    bake_probe(job2)
    assert job1["result_digest"] != job2["result_digest"]


def test_canonical_digest_stable_across_key_order():
    assert determinism({"a": 1, "b": 2}) == determinism({"b": 2, "a": 1})
    assert determinism({"x": [1, 2, 3]}) == determinism({"x": [1, 2, 3]})


# ---------------------------------------------------------------------------
# 其余入口的行为测试
# ---------------------------------------------------------------------------


def test_validation_accepts_good_job_and_flags_bad_one():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    assert validation(job)["valid"] is True
    job["snapshot"]["immutable"] = False
    result = validation(job)
    assert result["valid"] is False
    assert "snapshot_not_immutable" in result["errors"]


def test_audit_trail_ordering_and_events():
    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    bake_probe(job)
    events = audit(job)
    kinds = [e["event"] for e in events]
    assert kinds.count("face_rendered") == 6
    assert kinds.count("face_marked") == 6
    assert kinds.count("mips_generated") == 1
    # 按 (frame, seq) 有序
    assert events == sorted(events, key=lambda e: (e["frame"], e["seq"]))


def test_fallback_is_neutral_and_complete():
    fb = fallback({"probe_id": "dead", "level": "Town"})
    assert fb["status"] == "fallback"
    assert set(fb["cubemap"]) == set(FACE_ORDER)
    assert fb["result_digest"] is not None


def test_priority_and_metrics():
    assert mod.priority({"priority": 7}) == 7
    assert mod.priority(1) == 0
    zero = metrics(1)
    assert zero["queue_depth"] == 0 and zero["jobs_complete"] == 0

    job = create_bake_job("p", camera_snapshot((0.0, 0.0, 0.0)))
    bake_probe(job)
    m = metrics(job)
    assert m["jobs_complete"] == 1
    assert m["faces_rendered"] == 6
    assert m["faces_marked"] == 6
    assert m["mips_generated"] == 1


def test_face_order_and_cube_layout():
    assert face_order() == FACE_ORDER
    layout = cube_layout(16, gap=2)
    assert layout["atlas_size"] == (6 * 16 + 5 * 2, 16)
    assert layout["faces"]["+X"]["offset"] == (0, 0)
    assert layout["faces"]["-X"]["offset"] == (18, 0)
    with pytest.raises(ValueError):
        cube_layout(0)
