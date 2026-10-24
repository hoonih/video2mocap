"""프레이밍 정규화: 영상 전체의 어깨너비 중앙값으로 세 조건(A·B·C)의 화자 크기와 위치를 맞춘다.

- 조건 A(원본): 영상마다 고정된 배율·이동 변환 1개(회전 없음)를 모든 프레임에 적용한다.
- 조건 B·C(렌더링): 같은 목표에 맞춰 카메라 거리·높이·좌우 이동을 정한다.

목표: 어깨너비 = 출력 폭 × shoulder_ratio, 어깨 중앙 = (출력 폭 × anchor_x, 출력 높이 × anchor_y).
프레임별로 따라가지 않고 영상 전체의 중앙값 하나만 쓰므로 화면이 흔들리지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import landmarks
from .renderer import CameraFraming, _rotation_y

DEFAULT_OUTPUT_SIZE = (720, 1280)
DEFAULT_SHOULDER_RATIO = 0.35
DEFAULT_ANCHOR = (0.5, 0.42)
# 몸을 튼 프레임은 어깨가 좁게 찍힌다. 크기 기준은 정면에 가까운 프레임(상위 분위수)으로 잡는다.
FRONTAL_PERCENTILE = 90
FRONTAL_MASK_PERCENTILE = 75


@dataclass
class FramingTarget:
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE  # (width, height)
    shoulder_ratio: float = DEFAULT_SHOULDER_RATIO
    anchor: Tuple[float, float] = DEFAULT_ANCHOR  # 어깨 중앙의 화면 위치 (비율)

    @property
    def shoulder_px(self) -> float:
        return self.output_size[0] * self.shoulder_ratio

    @property
    def anchor_px(self) -> np.ndarray:
        return np.array([self.output_size[0] * self.anchor[0], self.output_size[1] * self.anchor[1]])


def parse_size(text: str) -> Tuple[int, int]:
    """'720x1280' → (720, 1280)"""
    w, sep, h = text.lower().partition("x")
    if not sep or not w.isdigit() or not h.isdigit() or int(w) <= 0 or int(h) <= 0:
        raise ValueError(f"크기 형식이 잘못되었습니다 (예: 720x1280): {text}")
    return int(w), int(h)


# ---------------------------------------------------------------------------
# 조건 A: 원본 영상의 배율·이동 변환
# ---------------------------------------------------------------------------

@dataclass
class OriginalTransform:
    """원본 픽셀 p → 출력 픽셀 scale * p + offset."""

    scale: float
    offset: np.ndarray
    source_size: Tuple[int, int]
    output_size: Tuple[int, int]

    @property
    def upscaled(self) -> bool:
        """원본 해상도보다 확대해야 하는지 (화질 저하)."""
        return self.scale > 1.0 + 1e-6

    @property
    def uncovered_ratio(self) -> float:
        """원본이 덮지 못해 여백으로 채워지는 출력 면적 비율 (화자가 너무 크게 찍힌 경우)."""
        sw, sh = self.source_size
        ow, oh = self.output_size
        x0, y0 = self.offset
        x1, y1 = x0 + sw * self.scale, y0 + sh * self.scale
        covered_w = max(0.0, min(x1, ow) - max(x0, 0.0))
        covered_h = max(0.0, min(y1, oh) - max(y0, 0.0))
        return 1.0 - (covered_w * covered_h) / float(ow * oh)


@dataclass
class ShoulderTrack:
    """원본 해상도 기준 프레임별 어깨너비와 어깨 중앙 (검출 실패 프레임은 NaN)."""

    widths: np.ndarray  # (N,)
    mids: np.ndarray    # (N, 2)

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.widths)

    def frontal_width(self) -> float:
        """정면에 가까운 프레임의 어깨너비. 몸을 튼 프레임은 어깨가 좁게 찍히므로 상위 분위수를 쓴다."""
        return float(np.nanpercentile(self.widths, FRONTAL_PERCENTILE))

    def frontal_mask(self) -> np.ndarray:
        return self.detected & (np.nan_to_num(self.widths) >= np.nanpercentile(self.widths, FRONTAL_MASK_PERCENTILE))

    def median_mid(self) -> np.ndarray:
        return np.nanmedian(self.mids, axis=0)


def shoulder_track(
    pose_pixels: Sequence[Optional[np.ndarray]], pixel_size: Tuple[int, int], source_size: Tuple[int, int]
) -> Optional[ShoulderTrack]:
    """추적용 프레임의 pose 픽셀 좌표들을 원본 해상도 기준 어깨너비·중앙으로 바꾼다."""
    to_source = np.array(source_size, dtype=np.float64) / np.array(pixel_size, dtype=np.float64)
    widths = np.full(len(pose_pixels), np.nan)
    mids = np.full((len(pose_pixels), 2), np.nan)
    for i, pose in enumerate(pose_pixels):
        if pose is None:
            continue
        left = pose[landmarks.LEFT_SHOULDER, :2] * to_source
        right = pose[landmarks.RIGHT_SHOULDER, :2] * to_source
        widths[i] = np.linalg.norm(left - right)
        mids[i] = (left + right) / 2.0
    if not np.any(~np.isnan(widths)):
        return None
    return ShoulderTrack(widths=widths, mids=mids)


def original_transform(track: ShoulderTrack, source_size: Tuple[int, int], target: FramingTarget) -> OriginalTransform:
    """정면 어깨너비가 목표 폭이 되고, 어깨 중앙 중앙값이 목표 위치에 오도록 하는 고정 변환."""
    scale = target.shoulder_px / track.frontal_width()
    offset = target.anchor_px - scale * track.median_mid()
    return OriginalTransform(scale=scale, offset=offset, source_size=source_size, output_size=target.output_size)


def screen_anchors(track: ShoulderTrack, transform: OriginalTransform, smooth: float) -> np.ndarray:
    """프레임별로 조건 A 화면에서 어깨 중앙이 놓이는 위치 (N,2). 조건 B·C를 같은 자리로 옮기는 데 쓴다.

    떨림을 줄이기 위해 EMA로 다듬고, 검출이 끊긴 프레임은 직전 값을 유지한다.
    처음부터 검출이 없던 구간은 첫 검출값으로 채운다.
    """
    raw = track.mids * transform.scale + transform.offset
    out = np.empty_like(raw)
    state = None
    for i, pt in enumerate(raw):
        if not np.isnan(pt[0]):
            state = pt if state is None else smooth * state + (1.0 - smooth) * pt
        out[i] = state if state is not None else np.nan
    first = int(np.argmax(track.detected))
    out[:first] = out[first]
    return out


def apply_original_transform(
    frame: np.ndarray, transform: OriginalTransform, fill_color=(0, 0, 0)
) -> np.ndarray:
    """원본 프레임을 배율·이동만으로 출력 캔버스에 놓는다. 덮지 못한 영역은 fill_color."""
    ow, oh = transform.output_size
    canvas = np.empty((oh, ow, 3), dtype=np.uint8)
    canvas[:] = fill_color
    sh, sw = frame.shape[:2]
    new_w = max(1, int(round(sw * transform.scale)))
    new_h = max(1, int(round(sh * transform.scale)))
    interp = cv2.INTER_AREA if transform.scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    x0, y0 = (int(round(v)) for v in transform.offset)
    # 캔버스와 겹치는 영역만 복사
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + new_w, ow), min(y0 + new_h, oh)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = resized[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return canvas


# ---------------------------------------------------------------------------
# 조건 B·C: 렌더러 카메라
# ---------------------------------------------------------------------------

def _projected_shoulders(pose: np.ndarray, rot: np.ndarray) -> Tuple[float, np.ndarray]:
    """카메라 yaw로 돌린 뒤 화면 평면(x, y)에서의 어깨너비와 3D 어깨 중앙."""
    left = rot @ pose[landmarks.LEFT_SHOULDER]
    right = rot @ pose[landmarks.RIGHT_SHOULDER]
    return float(np.linalg.norm((left - right)[:2])), (left + right) / 2.0


def camera_for_target(
    poses_world: Sequence[Optional[np.ndarray]],
    focal_length: float,
    target: FramingTarget,
    yaw_deg: float = 0.0,
    track: Optional[ShoulderTrack] = None,
    transform: Optional[OriginalTransform] = None,
) -> CameraFraming:
    """렌더링(조건 B·C)의 어깨 크기가 조건 A와 같아지도록 카메라를 정한다.

    poses_world는 프레임별 월드 pose(검출 실패 프레임은 None). track·transform이 있으면
    정면 프레임에서 'A 화면 어깨너비 = 렌더링 어깨너비'가 되는 거리를 고른다(프레임별 비율의 중앙값).
    없으면 렌더링 어깨너비 중앙값을 목표 폭에 맞춘다.
    """
    rot = _rotation_y(math.radians(yaw_deg))
    measured = [(i, *_projected_shoulders(p, rot)) for i, p in enumerate(poses_world) if p is not None]
    if not measured:
        raise ValueError("어깨를 잴 수 있는 프레임이 없습니다.")
    mid = np.median(np.array([m for _, _, m in measured]), axis=0)

    if track is not None and transform is not None:
        frontal = track.frontal_mask()
        # 렌더링 폭 f*W/z 가 A 화면 폭 s*w 와 같아지는 z = f*W/(s*w)
        depths = [
            focal_length * width / (transform.scale * track.widths[i])
            for i, width, _ in measured
            if frontal[i]
        ]
        shoulder_z = float(np.median(depths))
    else:
        width = float(np.median([w for _, w, _ in measured]))
        shoulder_z = focal_length * width / target.shoulder_px

    anchor = target.anchor_px
    w, h = target.output_size
    return CameraFraming(
        distance=shoulder_z - mid[2],
        height=(anchor[1] - h / 2.0) * shoulder_z / focal_length - mid[1],
        shift_x=(anchor[0] - w / 2.0) * shoulder_z / focal_length - mid[0],
    )
