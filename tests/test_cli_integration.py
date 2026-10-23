import json
import os

import cv2
import numpy as np
import pytest

from mocap.cli import build_arg_parser, run
from mocap.video_io import get_video_info

mp_available = True
try:
    import mediapipe  # noqa: F401
except ImportError:
    mp_available = False

pytestmark = pytest.mark.skipif(not mp_available, reason="mediapipe가 설치되어 있지 않음")


@pytest.fixture()
def dummy_video(tmp_path) -> str:
    """사람이 없는 단색+노이즈 더미 영상을 만든다 (검출 실패 경로를 훈련시키기 위함)."""
    path = str(tmp_path / "dummy_input.mp4")
    width, height, fps, num_frames = 160, 120, 10.0, 6
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        frame = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_cli_end_to_end_side_view_with_json_export(tmp_path, dummy_video):
    out_path = str(tmp_path / "out_side.mp4")
    json_path = str(tmp_path / "pose.json")

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            dummy_video,
            "--out", out_path,
            "--view", "side",
            "--complexity", "0",
            "--export-json", json_path,
        ]
    )
    exit_code = run(args)

    assert exit_code == 0
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0

    out_info = get_video_info(out_path)
    in_info = get_video_info(dummy_video)
    assert out_info.width == in_info.width * 2  # side-by-side라 가로 2배

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == in_info.total_frames
    for rec in records:
        assert len(rec["pose"]) == 33
        assert len(rec["pose"][0]) == 3


def test_cli_end_to_end_avatar_view_upper_body(tmp_path, dummy_video):
    out_path = str(tmp_path / "out_avatar.mp4")

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            dummy_video,
            "--out", out_path,
            "--view", "avatar",
            "--upper-body",
            "--complexity", "0",
            "--smooth", "0.5",
            "--yaw", "10",
            "--orbit", "2",
        ]
    )
    exit_code = run(args)

    assert exit_code == 0
    assert os.path.exists(out_path)

    out_info = get_video_info(out_path)
    in_info = get_video_info(dummy_video)
    assert out_info.width == in_info.width  # avatar만 출력하므로 원본 폭과 동일


def test_cli_missing_input_returns_error(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args([str(tmp_path / "없는파일.mp4")])
    assert run(args) == 1


def test_cli_invalid_smooth_value_returns_error(dummy_video):
    parser = build_arg_parser()
    args = parser.parse_args([dummy_video, "--smooth", "1.5"])
    assert run(args) == 1
