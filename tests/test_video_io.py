import os

import cv2
import numpy as np
import pytest

import shutil
import subprocess

from mocap import video_io
from mocap.video_io import (
    VideoWriter,
    get_video_info,
    inspect_encoding,
    probe_video,
    read_frames,
    read_frames_cfr,
    resize_to_max_height,
    side_by_side,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")


@pytest.fixture()
def dummy_video(tmp_path) -> str:
    path = str(tmp_path / "dummy.mp4")
    width, height, fps, num_frames = 160, 120, 24.0, 10
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_get_video_info(dummy_video):
    info = get_video_info(dummy_video)
    assert info.width == 160
    assert info.height == 120
    assert info.total_frames >= 9  # 코덱에 따라 1프레임 정도 차이날 수 있음
    assert info.fps > 0


def test_get_video_info_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        get_video_info("존재하지않는파일.mp4")


def test_resize_to_max_height_shrinks_and_preserves_ratio():
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    out = resize_to_max_height(frame, max_height=100)
    assert out.shape[0] == 100
    assert out.shape[1] == 50


def test_resize_to_max_height_noop_when_smaller():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    out = resize_to_max_height(frame, max_height=100)
    assert out.shape == frame.shape


def test_read_frames_yields_expected_count_and_size(dummy_video):
    frames = list(read_frames(dummy_video, max_height=120))
    assert len(frames) >= 9
    for f in frames:
        assert f.shape[:2] == (120, 160)


def test_read_frames_resizes_when_max_height_smaller(dummy_video):
    frames = list(read_frames(dummy_video, max_height=60))
    assert frames[0].shape[0] == 60
    assert frames[0].shape[1] == 80


@needs_ffmpeg
def test_video_writer_roundtrip(tmp_path):
    out_path = str(tmp_path / "out.mp4")
    with VideoWriter(out_path, fps=15.0, size=(64, 48)) as writer:
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
    info = get_video_info(out_path)
    assert info.width == 64
    assert info.height == 48


def test_side_by_side_matches_heights_and_concatenates_width():
    left = np.zeros((100, 50, 3), dtype=np.uint8)
    right = np.zeros((80, 60, 3), dtype=np.uint8)
    combined = side_by_side(left, right)
    assert combined.shape[0] == 100
    # 오른쪽 이미지는 높이 100에 맞춰 리사이즈된 폭이어야 함
    expected_right_w = round(60 * (100 / 80))
    assert combined.shape[1] == 50 + expected_right_w


@needs_ffmpeg
def test_read_frames_cfr_converts_frame_rate(dummy_video):
    # 24fps 10프레임(0.4167초) → 30fps
    frames = list(read_frames_cfr(dummy_video, 30.0))
    assert len(frames) in (12, 13)
    assert frames[0].shape == (120, 160, 3)


@needs_ffmpeg
def test_probe_and_decode_respect_rotation_metadata(tmp_path):
    """스마트폰 세로 촬영본: 저장은 가로(64x48)지만 회전 메타데이터로 세로(48x64)로 표시된다."""
    plain, rotated = str(tmp_path / "plain.mp4"), str(tmp_path / "rotated.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-t", "0.2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", plain],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", plain, "-c", "copy", rotated],
        check=True,
    )
    info = probe_video(rotated)
    assert (info.width, info.height) == (48, 64)
    assert next(read_frames_cfr(rotated, 30.0, info)).shape == (64, 48, 3)


@needs_ffmpeg
def test_video_writer_produces_mobile_web_h264(tmp_path):
    out_path = str(tmp_path / "web.mp4")
    rng = np.random.default_rng(0)
    with VideoWriter(out_path, fps=30.0, size=(720, 1280)) as writer:
        for _ in range(17):
            writer.write(rng.integers(0, 255, size=(1280, 720, 3), dtype=np.uint8))

    enc = inspect_encoding(out_path)
    assert enc.web_problems() == []
    assert (enc.codec, enc.pix_fmt, enc.level) == ("h264", "yuv420p", 31)
    assert "Baseline" in enc.profile
    assert (enc.width, enc.height, enc.fps, enc.frames) == (720, 1280, 30.0, 17)  # 프레임 수 보존
    assert not enc.has_audio
    assert enc.faststart


@needs_ffmpeg
def test_video_writer_omits_level_when_frame_exceeds_level_31(tmp_path):
    out_path = str(tmp_path / "wide.mp4")
    with VideoWriter(out_path, fps=30.0, size=(1440, 1280)) as writer:
        writer.write(np.zeros((1280, 1440, 3), dtype=np.uint8))
    enc = inspect_encoding(out_path)
    assert enc.codec == "h264" and enc.faststart


def test_video_writer_reports_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="brew install ffmpeg"):
        VideoWriter(str(tmp_path / "x.mp4"), fps=30.0, size=(64, 48))
    assert not os.path.exists(tmp_path / "x.mp4")  # mp4v 등으로 몰래 대체하지 않음


@needs_ffmpeg
def test_video_writer_rejects_odd_size_and_wrong_frame(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(str(tmp_path / "odd.mp4"), fps=30.0, size=(65, 48))
    with VideoWriter(str(tmp_path / "ok.mp4"), fps=30.0, size=(64, 48)) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
