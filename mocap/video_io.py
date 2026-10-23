"""영상 읽기/쓰기, 리사이즈, 좌우 화면 합성 유틸리티."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int


def get_video_info(path: str) -> VideoInfo:
    """영상 파일을 열어 fps/전체 프레임 수/가로/세로 정보를 읽어온다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(fps=fps, total_frames=total_frames, width=width, height=height)


def resize_to_max_height(frame: np.ndarray, max_height: int) -> np.ndarray:
    """세로가 max_height보다 크면 비율을 유지한 채 축소한다."""
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_frames(path: str, max_height: int) -> Iterator[np.ndarray]:
    """영상을 프레임 단위로 읽어 max_height 기준으로 리사이즈해서 순서대로 반환한다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield resize_to_max_height(frame, max_height)
    finally:
        cap.release()


class VideoWriter:
    """cv2.VideoWriter를 감싸는 간단한 래퍼."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int]) -> None:
        """size는 (width, height) 순서."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"출력 영상을 열 수 없습니다: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """두 이미지를 세로 높이를 맞춰서 가로로 나란히 합친다."""
    h = max(left.shape[0], right.shape[0])

    def _match_height(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / float(img.shape[0])
        new_w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    return cv2.hconcat([_match_height(left), _match_height(right)])
