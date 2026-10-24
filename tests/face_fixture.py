"""표준 얼굴 모델을 MediaPipe FaceMesh 픽셀 출력 형식으로 바꾸는 테스트 헬퍼.

FaceMesh 픽셀 출력 규약: x = 화면 오른쪽, y = 아래, z = 깊이(작을수록 카메라에 가까움), 단위 px.
표정(눈썹 높이, 입 벌림)과 머리 회전을 조절해서 실제 face_to_world 입력으로 쓴다.
"""
from __future__ import annotations

import math
import os

import numpy as np

from mocap import landmarks

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "canonical_face_model.npy")
_CANONICAL_CM = np.load(_MODEL_PATH).astype(np.float64)  # x=화자 왼쪽, y=위, z=카메라 쪽


def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """모델 좌표(y 위, z 카메라 쪽)에서의 머리 회전. yaw +: 화면 오른쪽, pitch +: 아래, roll +: 화면 시계 방향."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def canonical_face_pixel(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    brow_raise_cm: float = 0.0,
    mouth_open_cm: float = 0.0,
    px_per_cm: float = 5.0,
    origin_px=(320.0, 180.0),
) -> np.ndarray:
    """(468, 3) FaceMesh 픽셀 좌표를 만든다.

    brow_raise_cm: 양쪽 눈썹을 위(+) 또는 아래(−)로 옮기는 양.
    mouth_open_cm: 윗입술 안쪽(13)과 아랫입술 안쪽(14) 사이를 이만큼 더 벌린다.
    """
    model = _CANONICAL_CM.copy()
    for idx in landmarks.LEFT_BROW_INDICES + landmarks.RIGHT_BROW_INDICES:
        model[idx, 1] += brow_raise_cm
    model[landmarks.FACE_KEY_INDICES["mouth_top"], 1] += mouth_open_cm / 2.0
    model[landmarks.FACE_KEY_INDICES["mouth_bottom"], 1] -= mouth_open_cm / 2.0

    # 머리 중심(양쪽 얼굴 가장자리의 중점)을 기준으로 회전
    center = (model[landmarks.FACE_KEY_INDICES["face_left_edge"]]
              + model[landmarks.FACE_KEY_INDICES["face_right_edge"]]) / 2.0
    model = (model - center) @ _rotation(yaw_deg, pitch_deg, roll_deg).T + center

    pixel = np.empty_like(model)
    pixel[:, 0] = origin_px[0] + model[:, 0] * px_per_cm
    pixel[:, 1] = origin_px[1] - model[:, 1] * px_per_cm
    pixel[:, 2] = -model[:, 2] * px_per_cm
    return pixel
