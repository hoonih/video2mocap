"""MediaPipe Holistic 래퍼: 영상 프레임 → 표준화된 원본 랜드마크(RawLandmarks).

mediapipe==0.10.14에 고정. 최신 mediapipe는 `mp.solutions` 레거시 API(Holistic,
drawing_utils)가 빠져 있을 수 있으므로 반드시 이 버전을 사용해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - 환경 안내용
    raise ImportError(
        "mediapipe가 설치되어 있지 않습니다. `pip install mediapipe==0.10.14`로 설치하세요."
    ) from exc


@dataclass
class RawLandmarks:
    """MediaPipe Holistic이 한 프레임에서 뽑아낸 원본 랜드마크."""

    pose_world: Optional[np.ndarray]        # (33, 3) meters, 골반 중심 원점
    pose_pixel: Optional[np.ndarray]        # (33, 3) x*W, y*H, z*W
    left_hand_pixel: Optional[np.ndarray]   # (21, 3) mediapipe 라벨 그대로 (좌우 미보정)
    right_hand_pixel: Optional[np.ndarray]  # (21, 3)
    face_pixel: Optional[np.ndarray]        # (468 또는 478, 3)
    image_size: Tuple[int, int]             # (width, height)


def _landmarks_to_array(landmark_list, width: int, height: int, use_pixel_z: bool = True) -> np.ndarray:
    """mediapipe의 NormalizedLandmarkList를 (N,3) 픽셀 좌표 배열로 변환한다."""
    pts = np.array(
        [(lm.x * width, lm.y * height, lm.z * width if use_pixel_z else lm.z) for lm in landmark_list.landmark],
        dtype=np.float64,
    )
    return pts


def _world_landmarks_to_array(landmark_list) -> np.ndarray:
    return np.array([(lm.x, lm.y, lm.z) for lm in landmark_list.landmark], dtype=np.float64)


class HolisticTracker:
    """mediapipe Holistic을 감싸 프레임별로 RawLandmarks를 만들어주는 클래스."""

    def __init__(self, model_complexity: int = 1) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

    def process(self, frame_bgr: np.ndarray) -> Tuple[RawLandmarks, "object"]:
        """BGR 프레임을 처리해 RawLandmarks와 mediapipe 원본 결과(오버레이 그리기용)를 반환한다."""
        height, width = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self._holistic.process(frame_rgb)

        pose_world = (
            _world_landmarks_to_array(results.pose_world_landmarks)
            if results.pose_world_landmarks
            else None
        )
        pose_pixel = (
            _landmarks_to_array(results.pose_landmarks, width, height)
            if results.pose_landmarks
            else None
        )
        left_hand_pixel = (
            _landmarks_to_array(results.left_hand_landmarks, width, height)
            if results.left_hand_landmarks
            else None
        )
        right_hand_pixel = (
            _landmarks_to_array(results.right_hand_landmarks, width, height)
            if results.right_hand_landmarks
            else None
        )
        face_pixel = (
            _landmarks_to_array(results.face_landmarks, width, height)
            if results.face_landmarks
            else None
        )

        raw = RawLandmarks(
            pose_world=pose_world,
            pose_pixel=pose_pixel,
            left_hand_pixel=left_hand_pixel,
            right_hand_pixel=right_hand_pixel,
            face_pixel=face_pixel,
            image_size=(width, height),
        )
        return raw, results

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_landmarks_overlay(frame_bgr: np.ndarray, results) -> np.ndarray:
    """원본 프레임 위에 mediapipe 랜드마크를 그려서 반환한다 (side 뷰 왼쪽 패널용)."""
    overlay = frame_bgr.copy()
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    holistic = mp.solutions.holistic

    if results.face_landmarks:
        drawing_utils.draw_landmarks(
            overlay,
            results.face_landmarks,
            holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            overlay,
            results.pose_landmarks,
            holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            overlay, results.left_hand_landmarks, holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            overlay, results.right_hand_landmarks, holistic.HAND_CONNECTIONS
        )
    return overlay
