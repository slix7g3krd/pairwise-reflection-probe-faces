# pairwise-reflection-probe-faces

## 领域

游戏、图形与多媒体

## 主题

反射探针逐面烘焙

## 说明

相机快照、面级完成标记与曝光锁定。

初始状态下 `graphics/reflection_probe_faces.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
