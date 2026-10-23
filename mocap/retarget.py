"""좌표 리타게팅: 손 좌우 재배정, 미터 스케일 맞추기, 상체 모드 하체 합성.

MediaPipe가 주는 원본 좌표를 캐릭터에 그대로 쓸 수 있는 형태(FrameData)로 바꾸는
순수 numpy 함수들의 모음. mediapipe 자체에는 의존하지 않아 단위 테스트가 쉽다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import landmarks


@dataclass
class FrameData:
    """렌더러에 전달할 최종 프레임 데이터. pose는 항상 값이 있어야 한다."""

    pose: np.ndarray                      # (33, 3) meters
    left_hand: Optional[np.ndarray] = None   # (21, 3) meters
    right_hand: Optional[np.ndarray] = None  # (21, 3) meters
    face: Optional[np.ndarray] = None        # (7, 3) meters, landmarks.FACE_KEY_ORDER 순서


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


def face_to_world(
    face_pixel: Optional[np.ndarray], nose_world: np.ndarray, scale: Optional[float]
) -> Optional[np.ndarray]:
    """얼굴 핵심 포인트(눈/코/입)만 추출해 코 끝 기준으로 nose_world 위치에 붙인다."""
    if face_pixel is None or scale is None:
        return None
    nose_px = face_pixel[landmarks.FACE_KEY_INDICES["nose_tip"]]
    points = []
    for name in landmarks.FACE_KEY_ORDER:
        idx = landmarks.FACE_KEY_INDICES[name]
        relative = face_pixel[idx] - nose_px
        points.append(nose_world + relative * scale)
    return np.stack(points, axis=0)


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
