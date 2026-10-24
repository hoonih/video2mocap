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


def test_create_face_tracker_falls_back_without_model(tmp_path, capsys):
    from mocap.tracker import create_face_tracker

    assert create_face_tracker(None) is None
    assert create_face_tracker(str(tmp_path / "missing.task")) is None
    assert "Holistic" in capsys.readouterr().err


def test_face_blendshape_tracker_on_blank_frames():
    import os

    from mocap.tracker import DEFAULT_FACE_MODEL_PATH, create_face_tracker

    if not os.path.exists(DEFAULT_FACE_MODEL_PATH):
        pytest.skip("face_landmarker.task 없음 (assets/models/README.md 참고)")
    with create_face_tracker(DEFAULT_FACE_MODEL_PATH) as tracker:
        for i in range(3):
            points, blendshapes = tracker.process(np.zeros((240, 320, 3), np.uint8), i * 33, (160, 120))
            assert points is None and blendshapes is None
