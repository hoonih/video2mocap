import math

import numpy as np
import pytest

from mocap import landmarks
from mocap.retarget import (
    FrameData,
    assign_hands,
    carry_face_offsets,
    compute_scale,
    face_offsets,
    face_to_world,
    hand_to_world,
    head_angles,
    head_center,
    head_rotation_from_face,
    head_rotation_from_pose,
    hold_missing,
    synthesize_lower_body,
)

from .face_fixture import canonical_face_pixel


def _pose_pixel_with_wrists(left_xy, right_xy):
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose[landmarks.LEFT_WRIST][:2] = left_xy
    pose[landmarks.RIGHT_WRIST][:2] = right_xy
    return pose


def _hand_near(xy):
    hand = np.zeros((landmarks.NUM_HAND_LANDMARKS, 3))
    hand[landmarks.HAND_WRIST][:2] = xy
    return hand


# --- assign_hands -----------------------------------------------------------

def test_assign_hands_fixes_mislabeled_hands():
    pose_pixel = _pose_pixel_with_wrists((100, 100), (200, 100))
    hand_near_left_wrist = _hand_near((100, 100))
    hand_near_right_wrist = _hand_near((200, 100))

    # mediapipe가 라벨을 뒤바꿔 줬다고 가정 (left_hand_pixel 인자에 오른쪽 손목 근처 손을 전달)
    assigned_left, assigned_right = assign_hands(
        pose_pixel, left_hand_pixel=hand_near_right_wrist, right_hand_pixel=hand_near_left_wrist
    )

    assert np.allclose(assigned_left, hand_near_left_wrist)
    assert np.allclose(assigned_right, hand_near_right_wrist)


def test_assign_hands_single_hand_goes_to_nearest_side():
    pose_pixel = _pose_pixel_with_wrists((100, 100), (200, 100))
    hand = _hand_near((190, 100))

    assigned_left, assigned_right = assign_hands(pose_pixel, left_hand_pixel=hand, right_hand_pixel=None)

    assert assigned_left is None
    assert np.allclose(assigned_right, hand)


def test_assign_hands_no_hands_returns_none_none():
    pose_pixel = _pose_pixel_with_wrists((100, 100), (200, 100))
    assert assign_hands(pose_pixel, None, None) == (None, None)


def test_assign_hands_without_pose_passes_through():
    hand = _hand_near((0, 0))
    left, right = assign_hands(None, hand, None)
    assert left is hand
    assert right is None


# --- compute_scale -----------------------------------------------------------

def test_compute_scale_ratio():
    pose_world = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose_world[landmarks.LEFT_SHOULDER] = (0.2, 0.0, 0.0)
    pose_world[landmarks.RIGHT_SHOULDER] = (-0.2, 0.0, 0.0)  # world shoulder width = 0.4m

    pose_pixel = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose_pixel[landmarks.LEFT_SHOULDER][:2] = (150, 50)
    pose_pixel[landmarks.RIGHT_SHOULDER][:2] = (50, 50)  # pixel shoulder width = 100px

    scale = compute_scale(pose_world, pose_pixel)
    assert scale is not None
    assert math.isclose(scale, 0.004, rel_tol=1e-6)


def test_compute_scale_none_inputs():
    assert compute_scale(None, np.zeros((33, 3))) is None
    assert compute_scale(np.zeros((33, 3)), None) is None


def test_compute_scale_zero_pixel_width_returns_none():
    pose_world = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose_pixel = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))  # 두 어깨가 같은 픽셀 위치
    assert compute_scale(pose_world, pose_pixel) is None


# --- hand_to_world / face_to_world ------------------------------------------

def test_hand_to_world_anchors_wrist_and_scales_offsets():
    hand_pixel = np.zeros((landmarks.NUM_HAND_LANDMARKS, 3))
    hand_pixel[landmarks.HAND_WRIST] = (50, 60, 70)
    hand_pixel[1] = (60, 60, 70)  # 손목보다 x가 10만큼 큼

    wrist_world = np.array([1.0, 2.0, 3.0])
    scale = 0.01

    world = hand_to_world(hand_pixel, wrist_world, scale)

    assert np.allclose(world[landmarks.HAND_WRIST], wrist_world)
    assert np.allclose(world[1], wrist_world + np.array([0.1, 0.0, 0.0]))


def test_hand_to_world_none_when_missing_input():
    assert hand_to_world(None, np.zeros(3), 0.01) is None
    assert hand_to_world(np.zeros((21, 3)), np.zeros(3), None) is None


def test_face_to_world_anchors_head_center_and_normalizes_size():
    face_pixel = canonical_face_pixel(px_per_cm=7.0)
    center = np.array([0.1, -0.6, 0.2])

    world = face_to_world(face_pixel, center)

    k = landmarks.FACE_KEY_ORDER.index
    assert world.shape == (len(landmarks.FACE_KEY_ORDER), 3)
    edge_mid = (world[k("face_left_edge")] + world[k("face_right_edge")]) / 2.0
    assert np.allclose(edge_mid, center)
    eye_span = np.linalg.norm(world[k("left_eye")] - world[k("right_eye")])
    assert eye_span == pytest.approx(landmarks.FACE_EYE_SPAN_METERS)
    # 코끝은 머리 중심보다 카메라 쪽(z 작음)에 있다
    assert world[k("nose_tip"), 2] < center[2] - 0.05


def test_face_size_does_not_depend_on_image_scale():
    small = face_offsets(canonical_face_pixel(px_per_cm=3.0))
    large = face_offsets(canonical_face_pixel(px_per_cm=12.0))
    assert np.allclose(small, large)


def test_face_to_world_none_when_missing():
    assert face_to_world(None, np.zeros(3)) is None


@pytest.mark.parametrize(
    "yaw, pitch, roll", [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, 20, 0), (0, -15, 0), (0, 0, 15), (20, -10, 5)]
)
def test_head_rotation_from_face_recovers_angles(yaw, pitch, roll):
    offsets = face_offsets(canonical_face_pixel(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll))
    est = head_angles(head_rotation_from_face(offsets))
    # 이마-턱 선이 완전히 수직이 아니라 pitch에 1도 미만의 고정 편향이 있다
    assert est == pytest.approx((yaw, pitch, roll), abs=1.5)


def test_carry_face_offsets_follows_head_rotation():
    frontal = face_offsets(canonical_face_pixel())
    turned = face_offsets(canonical_face_pixel(yaw_deg=25))
    rot_then = head_rotation_from_face(frontal)
    rot_now = head_rotation_from_face(turned)

    carried = carry_face_offsets(frontal, rot_then, rot_now)

    assert np.allclose(carried, turned, atol=2e-3)


def test_head_center_is_ear_midpoint():
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose[landmarks.LEFT_EAR] = (0.08, -0.6, 0.1)
    pose[landmarks.RIGHT_EAR] = (-0.06, -0.62, 0.12)
    assert np.allclose(head_center(pose), (0.01, -0.61, 0.11))


def test_head_rotation_from_pose_is_identity_for_neutral_pose():
    rot = head_rotation_from_pose(landmarks.build_neutral_pose())
    yaw, _, roll = head_angles(rot)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)


# --- synthesize_lower_body ---------------------------------------------------

def test_synthesize_lower_body_identity_yaw():
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose[landmarks.LEFT_SHOULDER] = (0.2, -0.5, 0.0)
    pose[landmarks.RIGHT_SHOULDER] = (-0.2, -0.5, 0.0)
    pose[landmarks.NOSE] = (0.0, -1.0, 0.0)  # 상체 값은 보존되어야 함

    out = synthesize_lower_body(pose)

    lx, ly, lz = landmarks.LOWER_BODY_TEMPLATE[landmarks.LEFT_HIP]
    assert np.allclose(out[landmarks.LEFT_HIP], (lx, ly, lz))
    assert np.allclose(out[landmarks.NOSE], (0.0, -1.0, 0.0))


def test_synthesize_lower_body_rotates_with_shoulder_yaw():
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    # 어깨가 90도 회전: left-right 벡터가 +z 방향을 향함 (vx=0, vz>0 -> yaw=90deg)
    pose[landmarks.LEFT_SHOULDER] = (0.0, -0.5, 0.2)
    pose[landmarks.RIGHT_SHOULDER] = (0.0, -0.5, -0.2)

    out = synthesize_lower_body(pose)

    lx, ly, lz = landmarks.LOWER_BODY_TEMPLATE[landmarks.LEFT_HIP]
    expected = (-lz, ly, lx)  # yaw=90도 회전 결과
    assert np.allclose(out[landmarks.LEFT_HIP], expected, atol=1e-9)


def test_synthesize_lower_body_does_not_crash_on_degenerate_shoulders():
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))  # 양쪽 어깨가 같은 위치
    out = synthesize_lower_body(pose)
    assert out.shape == (landmarks.NUM_POSE_LANDMARKS, 3)


# --- hold_missing -------------------------------------------------------------

def test_hold_missing_full_failure_returns_prev():
    prev = FrameData(pose=np.ones((33, 3)), left_hand=np.ones((21, 3)))
    result = hold_missing(prev, None, None, None, None)
    assert result is prev


def test_hold_missing_no_prev_no_pose_returns_neutral():
    result = hold_missing(None, None, None, None, None)
    assert result.pose.shape == (landmarks.NUM_POSE_LANDMARKS, 3)
    assert result.left_hand is None


def test_hold_missing_fills_only_missing_parts():
    prev = FrameData(
        pose=np.zeros((33, 3)),
        left_hand=np.full((21, 3), 9.0),
        right_hand=np.full((21, 3), 8.0),
        face=np.full((7, 3), 7.0),
    )
    new_pose = np.ones((33, 3))
    new_right_hand = np.full((21, 3), 2.0)

    result = hold_missing(prev, new_pose, None, new_right_hand, None)

    assert np.allclose(result.pose, new_pose)
    assert np.allclose(result.left_hand, prev.left_hand)  # 유지됨
    assert np.allclose(result.right_hand, new_right_hand)  # 새 값 사용
    assert np.allclose(result.face, prev.face)  # 유지됨
