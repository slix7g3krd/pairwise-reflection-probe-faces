# pairwise-reflection-probe-faces

## 领域

游戏、图形与多媒体

## 主题

反射探针逐面烘焙

## 说明

相机快照、面级完成标记与曝光锁定。

### 六个缺陷的修复映射

| 缺陷 | 修复入口 | 机制 |
| --- | --- | --- |
| 六面渲染中途相机矩阵被运动物体更新 | `camera_snapshot` / `new_camera_snapshot` | 烘焙开始即冻结六面 view/projection 矩阵、位置与场景摘要（深拷贝 + 内容哈希），快照不可变，逐面渲染只读快照 |
| mip 生成读到没写完的面 | `face_mark` / `mip_fence` / `generate_mips` | 6 位面级完成掩码；mip 栅栏要求六面全部置位，否则抛 `IncompleteFacesError` |
| 流送时旧探针被重新排到队首 | `queue_cleanup` / `ProbePriorityQueue` | 队列条目带 generation，过期代际在清理/出队时惰性丢弃；同优先级 FIFO，重排不能插队 |
| 流送后旧探针仍可用 | `stream_invalidate` / `stream_in` | 关卡流走即标记 invalid 并提升 generation；流回时代际只增不减，旧队列条目永久失效 |
| 曝光在六面之间漂移 | `exposure_lock` | 烘焙开始固化 EV100/光圈/快门/ISO，强制关闭自动曝光，六面共用同一曝光缩放系数 |
| 同位置两次烘焙亮度不一致 | `bake_determinism` / `determinism` | splitmix64 固定序列（种子来自快照规范化哈希）、排序遍历、无 time/random 依赖，逐纹素可复现 |

### 测试说明

原 `tests/test_reflection_probe_faces.py` 的参数表写成 `("camera_snapshot", (1))`。
`(1)` 在 Python 中是整数而非单元素元组，`fn(*1)` 会在函数调用前由字节码抛出
`TypeError: argument after * must be an iterable, not int`，任何模块实现都无法
让该写法通过。已修正为 `(1,)`，契约语义（每个入口接受单整数并返回非 None）不变，
并补充了覆盖上述六个修复的 26 个行为测试。

运行：

```bash
python -m pytest -q
```

期望 41 个测试全部通过。
