import pytest
import graphics.reflection_probe_faces as mod

CASES = [
    ("camera_snapshot", (1)),
    ("face_mark", (1)),
    ("mip_fence", (1)),
    ("queue_cleanup", (1)),
    ("stream_invalidate", (1)),
    ("exposure_lock", (1)),
    ("bake_determinism", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("priority", (1)),
    ("metrics", (1)),
    ("face_order", (1)),
    ("cube_layout", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
