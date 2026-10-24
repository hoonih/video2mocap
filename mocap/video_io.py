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


# 모바일 웹 재생용 H.264 프리셋 (iOS Safari / Android Chrome)
#  - baseline 프로파일 + yuv420p: 구형 기기 하드웨어 디코더 호환
#  - faststart: moov 아톰을 앞에 둬서 다 받기 전에 재생 시작
#  - 오디오 없음: 청각장애인 대상 연구라 불필요
WEB_H264_ARGS = (
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
    "-crf", "23", "-movflags", "+faststart", "-an",
)
WEB_H264_LEVEL = "3.1"
LEVEL_31_MAX_MACROBLOCKS = 3600  # H.264 level 3.1의 프레임당 최대 매크로블록 수 (720x1280 = 3600)


class VideoWriter:
    """BGR 프레임을 ffmpeg stdin으로 넘겨 웹 재생용 H.264 mp4로 바로 인코딩한다 (중간 파일 없음).

    ffmpeg가 없으면 설치 안내와 함께 RuntimeError를 낸다. mp4v 등 다른 코덱으로 대체하지 않는다.
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int]) -> None:
        """size는 (width, height) 순서. yuv420p 때문에 가로·세로 모두 짝수여야 한다."""
        require_ffmpeg()
        width, height = size
        if width % 2 or height % 2:
            raise ValueError(f"출력 크기는 가로·세로가 짝수여야 합니다 (yuv420p): {width}x{height}")
        self.path = path
        self.size = (width, height)
        self.frames_written = 0

        macroblocks = ((width + 15) // 16) * ((height + 15) // 16)
        # level 3.1 한도를 넘는 크기(예: side 디버그 뷰)는 level을 지정하지 않는다
        level = ["-level:v", WEB_H264_LEVEL] if macroblocks <= LEVEL_31_MAX_MACROBLOCKS else []
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{fps}",
            "-i", "-",
            *WEB_H264_ARGS, *level,
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size or frame.dtype != np.uint8 or frame.ndim != 3:
            raise ValueError(f"프레임 크기/형식이 출력과 다릅니다: {frame.shape} (기대 {self.size[1]}x{self.size[0]}x3)")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise IOError(f"ffmpeg 인코딩이 중단되었습니다: {self.path}\n{self._drain_stderr()}") from None
        self.frames_written += 1

    def _drain_stderr(self) -> str:
        return self._proc.stderr.read().decode(errors="replace").strip()

    def release(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self._drain_stderr()
        if self._proc.wait() != 0:
            raise IOError(f"ffmpeg 인코딩 실패: {self.path}\n{stderr}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # 이미 실패 중이면 ffmpeg만 정리하고 원래 예외를 살린다
            self._proc.kill()
            self._proc.wait()
            return
        self.release()


def mp4_top_level_boxes(path: str) -> list:
    """mp4 최상위 박스 이름을 파일 순서대로 돌려준다. faststart면 'moov'가 'mdat'보다 앞에 온다."""
    names = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size = int.from_bytes(header[:4], "big")
            names.append(header[4:8].decode("latin-1"))
            if size == 1:  # 64비트 크기
                size = int.from_bytes(f.read(8), "big")
                f.seek(size - 16, 1)
            elif size == 0:  # 파일 끝까지
                break
            else:
                f.seek(size - 8, 1)
    return names


@dataclass
class EncodingInfo:
    codec: str
    pix_fmt: str
    profile: str
    level: int
    width: int
    height: int
    fps: float
    frames: int
    has_audio: bool
    faststart: bool

    def web_problems(self) -> list:
        """모바일 웹 재생 프리셋과 다른 점 목록 (비어 있으면 통과)."""
        problems = []
        if self.codec != "h264":
            problems.append(f"코덱 {self.codec} (h264 필요)")
        if self.pix_fmt != "yuv420p":
            problems.append(f"픽셀 포맷 {self.pix_fmt} (yuv420p 필요)")
        if "Baseline" not in self.profile:
            problems.append(f"프로파일 {self.profile} (baseline 필요)")
        if self.level > 31:
            problems.append(f"레벨 {self.level / 10:.1f} (3.1 이하 필요)")
        if self.has_audio:
            problems.append("오디오 트랙 있음")
        if not self.faststart:
            problems.append("faststart 아님 (moov가 mdat 뒤)")
        return problems


def inspect_encoding(path: str) -> EncodingInfo:
    """ffprobe로 출력 파일의 인코딩 속성과 실제 프레임 수를 읽는다."""
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=codec_type,codec_name,pix_fmt,profile,level,width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    video = next(st for st in streams if st["codec_type"] == "video")
    num, _, den = video["avg_frame_rate"].partition("/")
    boxes = mp4_top_level_boxes(path)
    return EncodingInfo(
        codec=video["codec_name"],
        pix_fmt=video.get("pix_fmt", ""),
        profile=video.get("profile", ""),
        level=int(video.get("level", 0)),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(num) / float(den or 1),
        frames=int(video.get("nb_read_frames", 0)),
        has_audio=any(st["codec_type"] == "audio" for st in streams),
        faststart="moov" in boxes and "mdat" in boxes and boxes.index("moov") < boxes.index("mdat"),
    )


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
