"""영상 읽기/쓰기, 리사이즈, 좌우 화면 합성 유틸리티."""
from __future__ import annotations

import json
import shutil
import subprocess
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


FFMPEG_INSTALL_HINT = (
    "ffmpeg/ffprobe를 찾을 수 없습니다. 설치 후 다시 실행하세요.\n"
    "  macOS: brew install ffmpeg\n"
    "  Ubuntu/Debian: sudo apt install ffmpeg\n"
    "  Windows: https://ffmpeg.org/download.html 에서 받아 PATH에 추가"
)


def require_ffmpeg() -> None:
    """ffmpeg와 ffprobe가 PATH에 없으면 설치 안내와 함께 RuntimeError를 낸다."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} 없음. " + FFMPEG_INSTALL_HINT)


@dataclass
class ProbeInfo:
    """ffprobe로 읽은 영상 정보. width/height는 회전 메타데이터를 반영한 '표시' 크기."""

    width: int
    height: int
    fps: float
    duration: float


def probe_video(path: str) -> ProbeInfo:
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration:stream_side_data=rotation:format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}\n{result.stderr.strip()}")
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"영상 스트림이 없습니다: {path}")
    stream = data["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) % 180 == 90:  # 스마트폰 세로 촬영: 저장은 가로, 표시는 세로
        width, height = height, width
    num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return ProbeInfo(width=width, height=height, fps=fps, duration=duration)


def read_frames_cfr(path: str, fps: float, info: ProbeInfo = None) -> Iterator[np.ndarray]:
    """ffmpeg로 영상을 고정 프레임레이트(fps)로 디코딩해 원본 해상도의 BGR 프레임을 순서대로 낸다.

    조건 A·B·C가 모두 이 프레임 시퀀스에서 만들어지므로 프레임 수가 구조적으로 같아진다.
    가변 프레임레이트(VFR)·29.97fps 스마트폰 영상도 fps에 맞춰 프레임을 복제/생략한다.
    """
    info = info or probe_video(path)
    cmd = [
        "ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
        "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3).copy()
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        returncode = proc.wait()
    if returncode != 0:
        raise IOError(f"ffmpeg 디코딩 실패: {path}\n{stderr.strip()}")


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
