import numpy as np
import pytest

from mocap.smoothing import EMASmoother


def test_first_update_returns_raw_unchanged():
    sm = EMASmoother(alpha=0.5)
    raw = np.array([1.0, 2.0, 3.0])
    out = sm.update(raw)
    assert np.allclose(out, raw)


def test_ema_converges_toward_constant_target():
    sm = EMASmoother(alpha=0.8)
    target = np.array([10.0, 0.0, 0.0])
    sm.update(np.array([0.0, 0.0, 0.0]))
    for _ in range(200):
        out = sm.update(target)
    assert np.allclose(out, target, atol=1e-3)


def test_ema_smooths_noisy_step():
    sm = EMASmoother(alpha=0.9)
    sm.update(np.array([0.0]))
    out = sm.update(np.array([100.0]))
    # 강한 스무딩(alpha=0.9)이므로 급격한 변화가 크게 억제되어야 함
    assert 0.0 < out[0] < 30.0


def test_missing_frame_holds_last_value():
    sm = EMASmoother(alpha=0.5)
    sm.update(np.array([5.0, 5.0]))
    held = sm.update(None)
    assert np.allclose(held, np.array([5.0, 5.0]))


def test_missing_frame_before_any_data_returns_none():
    sm = EMASmoother(alpha=0.5)
    assert sm.update(None) is None


def test_reappearance_resets_without_blending():
    sm = EMASmoother(alpha=0.9)
    sm.update(np.array([0.0, 0.0]))
    sm.update(None)  # 추적 끊김
    sm.update(None)
    new_raw = np.array([50.0, 50.0])
    out = sm.update(new_raw)
    # 복귀 시 직전 값과 섞이지 않고 raw 그대로 즉시 반영되어야 함
    assert np.allclose(out, new_raw)


def test_reset_clears_state():
    sm = EMASmoother(alpha=0.5)
    sm.update(np.array([1.0]))
    sm.reset()
    out = sm.update(np.array([9.0]))
    assert np.allclose(out, np.array([9.0]))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        EMASmoother(alpha=1.0)
    with pytest.raises(ValueError):
        EMASmoother(alpha=-0.1)
