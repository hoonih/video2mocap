import os

import numpy as np

from mocap import landmarks
from mocap.renderer import CapsuleRenderer
from mocap.retarget import FrameData

SCRATCH_DIR = os.environ.get(
    "MOCAP_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-hong-yehun-PycharmProjects-research/80c1bbc3-1ebf-463c-b01e-c8c9a7e08bbc/scratchpad",
)


def _full_body_pose() -> np.ndarray:
    pose = landmarks.build_neutral_pose()
    # 오른팔을 굽혀서 시각적으로 자세 구분이 되도록 변형
    pose[landmarks.RIGHT_ELBOW] = (0.35, -0.75, 0.05)
    pose[landmarks.RIGHT_WRIST] = (0.15, -0.95, 0.15)
    return pose


def _fake_hand(wrist_world: np.ndarray, spread: float = 0.03) -> np.ndarray:
    hand = np.zeros((landmarks.NUM_HAND_LANDMARKS, 3))
    hand[landmarks.HAND_WRIST] = wrist_world
    for finger_i, finger in enumerate(landmarks.HAND_FINGERS):
        for joint_i, joint_idx in enumerate(finger):
            offset = np.array(
                [
                    (finger_i - 2) * spread * 0.6,
                    -spread * (joint_i + 1),
                    0.0,
                ]
            )
            hand[joint_idx] = wrist_world + offset
    return hand


def _fake_face(nose_world: np.ndarray) -> np.ndarray:
    # landmarks.FACE_KEY_ORDER 순서: nose_tip, left_eye, right_eye, mouth_left, mouth_right, mouth_top, mouth_bottom
    offsets = {
        "nose_tip": (0.0, 0.0, -0.03),
        "left_eye": (-0.03, -0.02, -0.03),
        "right_eye": (0.03, -0.02, -0.03),
        "mouth_left": (-0.02, 0.03, -0.03),
        "mouth_right": (0.02, 0.03, -0.03),
        "mouth_top": (0.0, 0.025, -0.03),
        "mouth_bottom": (0.0, 0.035, -0.03),
    }
    pts = [nose_world + np.array(offsets[name]) for name in landmarks.FACE_KEY_ORDER]
    return np.stack(pts, axis=0)


def test_full_body_render_produces_nonblank_image():
    pose = _full_body_pose()
    frame = FrameData(pose=pose)
    renderer = CapsuleRenderer(width=640, height=720, upper_body=False, yaw_deg=15.0)

    img = renderer.render(frame)

    assert img.shape == (720, 640, 3)
    assert img.dtype == np.uint8
    assert img.std() > 1.0  # 완전히 단색이 아니어야 함(뭔가 그려짐)

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    from cv2 import imwrite

    imwrite(os.path.join(SCRATCH_DIR, "render_full_body.png"), img)


def test_upper_body_with_hands_and_face_render():
    pose = _full_body_pose()
    left_hand = _fake_hand(pose[landmarks.LEFT_WRIST])
    right_hand = _fake_hand(pose[landmarks.RIGHT_WRIST])
    face = _fake_face(pose[landmarks.NOSE])

    frame = FrameData(pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
    renderer = CapsuleRenderer(width=480, height=640, upper_body=True, yaw_deg=0.0)

    img = renderer.render(frame)

    assert img.shape == (640, 480, 3)
    assert img.std() > 1.0

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    from cv2 import imwrite

    imwrite(os.path.join(SCRATCH_DIR, "render_upper_body_hands_face.png"), img)


def test_orbit_changes_projection_over_frames():
    pose = _full_body_pose()
    frame = FrameData(pose=pose)
    renderer = CapsuleRenderer(width=320, height=360, upper_body=False, orbit_deg_per_frame=30.0)

    img1 = renderer.render(frame)
    img2 = renderer.render(frame)

    assert not np.array_equal(img1, img2)  # 회전이 누적되어 다른 프레임이 나와야 함


def test_render_without_hands_or_face_does_not_crash():
    pose = _full_body_pose()
    frame = FrameData(pose=pose, left_hand=None, right_hand=None, face=None)
    renderer = CapsuleRenderer(width=200, height=200, upper_body=False)

    img = renderer.render(frame)
    assert img.shape == (200, 200, 3)
