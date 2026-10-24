"""좌표 리타게팅: 손 좌우 재배정, 미터 스케일 맞추기, 상체 모드 하체 합성.

MediaPipe가 주는 원본 좌표를 캐릭터에 그대로 쓸 수 있는 형태(FrameData)로 바꾸는
순수 numpy 함수들의 모음. mediapipe 자체에는 의존하지 않아 단위 테스트가 쉽다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import landmarks


@dataclass
class FrameData:
    """렌더러에 전달할 최종 프레임 데이터. pose는 항상 값이 있어야 한다."""

    pose: np.ndarray                      # (33, 3) meters
    left_hand: Optional[np.ndarray] = None   # (21, 3) meters
    right_hand: Optional[np.ndarray] = None  # (21, 3) meters
    face: Optional[np.ndarray] = None        # (N, 3) meters, landmarks.FACE_KEY_ORDER 순서
    # 출력 화면에서 어깨 중앙이 놓일 픽셀 위치. 조건 A(원본)에서 화자가 움직인 만큼 B·C도 옮긴다.
    # None이면 카메라 기본 배치를 그대로 쓴다.
    screen_anchor: Optional[np.ndarray] = None
    # ARKit 호환 블렌드셰이프 계수 (이름 → 0~1). 실측값을 증폭하지 않는다. 폴백 경로에서는 None.
    face_blendshapes: Optional[Dict[str, float]] = None


def assign_hands(
    pose_pixel: Optional[np.ndarray],
    left_hand_pixel: Optional[np.ndarray],
    right_hand_pixel: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """MediaPipe가 붙인 좌우 라벨을 무시하고, 손목이 더 가까운 팔 쪽에 손을 재배정한다.

    pose_pixel이 없으면 재배정할 기준이 없으므로 입력을 그대로 돌려준다.
    """
    if pose_pixel is None:
        return left_hand_pixel, right_hand_pixel

    candidates = [h for h in (left_hand_pixel, right_hand_pixel) if h is not None]
    if not candidates:
        return None, None

    left_wrist_px = pose_pixel[landmarks.LEFT_WRIST][:2]
    right_wrist_px = pose_pixel[landmarks.RIGHT_WRIST][:2]

    if len(candidates) == 1:
        hand = candidates[0]
        wrist_px = hand[landmarks.HAND_WRIST][:2]
        d_left = np.linalg.norm(wrist_px - left_wrist_px)
        d_right = np.linalg.norm(wrist_px - right_wrist_px)
        return (hand, None) if d_left <= d_right else (None, hand)

    # 두 손이 모두 검출된 경우: 두 배정 조합의 총 거리를 비교해 더 그럴듯한 쪽을 채택.
    hand_a, hand_b = candidates
    wrist_a = hand_a[landmarks.HAND_WRIST][:2]
    wrist_b = hand_b[landmarks.HAND_WRIST][:2]

    cost_a_left = np.linalg.norm(wrist_a - left_wrist_px) + np.linalg.norm(wrist_b - right_wrist_px)
    cost_a_right = np.linalg.norm(wrist_a - right_wrist_px) + np.linalg.norm(wrist_b - left_wrist_px)

    if cost_a_left <= cost_a_right:
        return hand_a, hand_b
    return hand_b, hand_a


def compute_scale(
    pose_world: Optional[np.ndarray], pose_pixel: Optional[np.ndarray]
) -> Optional[float]:
    """미터/픽셀 환산 비율을 '월드 어깨너비 / 픽셀 어깨너비'로 계산한다."""
    if pose_world is None or pose_pixel is None:
        return None

    world_width = np.linalg.norm(
        pose_world[landmarks.LEFT_SHOULDER] - pose_world[landmarks.RIGHT_SHOULDER]
    )
    pixel_width = np.linalg.norm(
        pose_pixel[landmarks.LEFT_SHOULDER][:2] - pose_pixel[landmarks.RIGHT_SHOULDER][:2]
    )
    if pixel_width < 1e-6:
        return None
    return float(world_width / pixel_width)


def hand_to_world(
    hand_pixel: Optional[np.ndarray], wrist_world: np.ndarray, scale: Optional[float]
) -> Optional[np.ndarray]:
    """손목(landmark 0) 기준 상대 픽셀 좌표에 scale을 곱해 wrist_world 위치에 붙인다."""
    if hand_pixel is None or scale is None:
        return None
    wrist_px = hand_pixel[landmarks.HAND_WRIST]
    relative = hand_pixel - wrist_px
    return wrist_world + relative * scale


def head_center(pose_world: np.ndarray) -> np.ndarray:
    """머리 중심(월드 좌표)을 양쪽 귀의 중점으로 잡는다. 코보다 약 10cm 뒤에 있다."""
    return (pose_world[landmarks.LEFT_EAR] + pose_world[landmarks.RIGHT_EAR]) / 2.0


def face_offsets(face_pixel: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """얼굴 핵심 포인트를 머리 중심 기준 상대 좌표(미터, landmarks.FACE_KEY_ORDER 순서)로 바꾼다.

    머리 중심은 얼굴 양쪽 가장자리의 3D 중점이다. 코끝을 기준으로 삼으면 코끝이 항상 가장
    앞에 있어서 머리가 돌아가도 얼굴 요소가 머리 구 위에서 움직이지 않는다.
    크기는 3D 눈꼬리 거리를 FACE_EYE_SPAN_METERS에 맞춘다. 고개를 돌려도 거의 변하지 않는다.
    """
    if face_pixel is None:
        return None
    eye_span_px = np.linalg.norm(
        face_pixel[landmarks.FACE_KEY_INDICES["left_eye"]]
        - face_pixel[landmarks.FACE_KEY_INDICES["right_eye"]]
    )
    if eye_span_px < 1e-6:
        return None
    scale = landmarks.FACE_EYE_SPAN_METERS / eye_span_px
    center_px = np.mean(
        [face_pixel[landmarks.FACE_KEY_INDICES[k]] for k in landmarks.FACE_CENTER_KEYS], axis=0
    )
    idxs = [landmarks.FACE_KEY_INDICES[name] for name in landmarks.FACE_KEY_ORDER]
    return (face_pixel[idxs] - center_px) * scale


def face_to_world(face_pixel: Optional[np.ndarray], head_center_world: np.ndarray) -> Optional[np.ndarray]:
    """얼굴 핵심 포인트를 머리 중심(head_center_world)에 붙인 월드 좌표로 바꾼다."""
    offsets = face_offsets(face_pixel)
    if offsets is None:
        return None
    return head_center_world + offsets


def _head_frame(x_axis: np.ndarray, forward: np.ndarray) -> Optional[np.ndarray]:
    """x축(화자 오른쪽→왼쪽)과 얼굴 정면 방향(카메라 쪽)으로 머리 회전 행렬을 만든다.

    열벡터는 (x=화자 왼쪽, y=아래, z=뒤통수 방향). 정면을 보는 자세에서 단위행렬이 된다.
    """
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        return None
    x = x_axis / x_norm
    z = -forward - np.dot(-forward, x) * x
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-9:
        return None
    z = z / z_norm
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def head_rotation_from_face(face: np.ndarray) -> Optional[np.ndarray]:
    """얼굴 포인트(face_offsets 또는 face_to_world 출력)로 머리 회전 행렬을 추정한다."""
    k = landmarks.FACE_KEY_ORDER.index
    x_axis = face[k("left_eye")] - face[k("right_eye")]
    down = face[k("chin")] - face[k("forehead")]
    return _head_frame(x_axis, np.cross(down, x_axis))


def head_rotation_from_pose(pose_world: np.ndarray) -> Optional[np.ndarray]:
    """얼굴이 검출되지 않을 때 쓰는 대체 추정: 양쪽 눈을 잇는 벡터 + 머리 중심→코 벡터.

    코가 귀보다 약간 아래에 있어서 정면 자세에서도 pitch가 0이 아니다. 절대 각도가 아니라
    프레임 간 회전 변화량(carry_face_offsets)을 구하는 데만 쓴다.
    """
    x_axis = pose_world[landmarks.LEFT_EYE] - pose_world[landmarks.RIGHT_EYE]
    forward = pose_world[landmarks.NOSE] - head_center(pose_world)
    return _head_frame(x_axis, forward)


def head_angles(rotation: np.ndarray) -> Tuple[float, float, float]:
    """회전 행렬을 (yaw, pitch, roll) 도 단위로 바꾼다.

    yaw +: 화면 오른쪽을 봄, pitch +: 아래를 봄(끄덕임), roll +: 화면 기준 시계 방향 기울임.
    """
    forward = -rotation[:, 2]
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(forward[1], math.hypot(forward[0], forward[2])))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return yaw, pitch, roll


def carry_face_offsets(
    offsets: np.ndarray, rotation_then: np.ndarray, rotation_now: np.ndarray
) -> np.ndarray:
    """마지막으로 검출된 얼굴을 그 뒤의 머리 회전만큼 돌린다 (손이 얼굴을 가린 구간용)."""
    delta = rotation_now @ rotation_then.T
    return offsets @ delta.T


def synthesize_lower_body(pose_world: np.ndarray) -> np.ndarray:
    """상체 모드용: 어깨 방향(yaw)에 맞춰 고정 하체 템플릿을 회전시켜 덮어쓴다.

    화면에 보이지 않는 골반/다리에 대한 MediaPipe의 추측값은 신뢰할 수 없으므로 버리고,
    어깨가 향한 좌우 방향만 반영한 고정 자세로 대체한다. 0~22번(상체) 인덱스는 그대로 둔다.
    """
    pose = np.array(pose_world, dtype=np.float64, copy=True)

    left_shoulder = pose[landmarks.LEFT_SHOULDER]
    right_shoulder = pose[landmarks.RIGHT_SHOULDER]
    dx = left_shoulder[0] - right_shoulder[0]
    dz = left_shoulder[2] - right_shoulder[2]
    yaw = math.atan2(dz, dx)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    for idx, (lx, ly, lz) in landmarks.LOWER_BODY_TEMPLATE.items():
        world_x = lx * cos_y - lz * sin_y
        world_z = lx * sin_y + lz * cos_y
        pose[idx] = (world_x, ly, world_z)

    return pose


def hold_missing(
    prev: Optional[FrameData],
    pose: Optional[np.ndarray],
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    face: Optional[np.ndarray],
) -> FrameData:
    """부위별 검출 실패를 직전 프레임 값으로 메운다. pose가 완전히 없으면 이전 프레임 전체를 유지한다.

    prev가 없고 pose도 없으면(영상 시작부터 검출 실패) 중립 자세로 대체한다.
    """
    if pose is None:
        if prev is not None:
            return prev
        return FrameData(pose=landmarks.build_neutral_pose(), left_hand=None, right_hand=None, face=None)

    resolved_left = left_hand if left_hand is not None else (prev.left_hand if prev else None)
    resolved_right = right_hand if right_hand is not None else (prev.right_hand if prev else None)
    resolved_face = face if face is not None else (prev.face if prev else None)

    return FrameData(pose=pose, left_hand=resolved_left, right_hand=resolved_right, face=resolved_face)
