import numpy as np
import pytest

mp_available = True
try:
    import mediapipe  # noqa: F401
except ImportError:
    mp_available = False

pytestmark = pytest.mark.skipif(not mp_available, reason="mediapipe가 설치되어 있지 않음")


def test_blank_frame_returns_all_none():
    from mocap.tracker import HolisticTracker

    with HolisticTracker(model_complexity=0) as tracker:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        raw, results = tracker.process(blank)

    assert raw.pose_world is None
    assert raw.pose_pixel is None
    assert raw.left_hand_pixel is None
    assert raw.right_hand_pixel is None
    assert raw.face_pixel is None
    assert raw.image_size == (320, 240)


def test_draw_landmarks_overlay_does_not_crash_without_detections():
    from mocap.tracker import HolisticTracker, draw_landmarks_overlay

    with HolisticTracker(model_complexity=0) as tracker:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        raw, results = tracker.process(blank)
        overlay = draw_landmarks_overlay(blank, results)

    assert overlay.shape == blank.shape


def test_process_multiple_frames_does_not_crash():
    from mocap.tracker import HolisticTracker

    rng = np.random.default_rng(0)
    with HolisticTracker(model_complexity=0) as tracker:
        for _ in range(3):
            noisy = rng.integers(0, 255, size=(180, 240, 3), dtype=np.uint8)
            raw, _ = tracker.process(noisy)
            assert raw.image_size == (240, 180)
