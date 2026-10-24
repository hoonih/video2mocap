import json
import os

import cv2
import numpy as np
import pytest

from mocap.cli import build_arg_parser, run
from mocap.video_io import get_video_info, probe_video, read_frames_cfr


def _count_frames(path: str) -> int:
    return sum(1 for _ in read_frames_cfr(path, 30.0))


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
    assert (out_info.width, out_info.height) == (720 * 2, 1280)  # side-by-side라 가로 2배

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    # 입력(10fps, 6프레임 = 0.6초)을 30fps로 고정 변환한 프레임 수
    assert len(records) == _count_frames(dummy_video) == _count_frames(out_path)
    for rec in records:
        assert len(rec["pose"]) == 33
        assert len(rec["pose"][0]) == 3
        assert "face_blendshapes" in rec  # 사람이 없는 영상이라 값은 None


def test_retargeter_smooths_and_holds_blendshapes():
    from mocap import landmarks
    from mocap.cli import _Retargeter
    from mocap.tracker import RawLandmarks

    def raw(blendshapes):
        return RawLandmarks(
            pose_world=landmarks.build_neutral_pose(), pose_pixel=None, left_hand_pixel=None,
            right_hand_pixel=None, face_pixel=None, image_size=(100, 100), face_blendshapes=blendshapes,
        )

    rt = _Retargeter(upper_body=True, smooth=0.5)
    first = rt.step(raw({"browInnerUp": 0.8, "jawOpen": 0.2}))
    assert first.face_blendshapes["browInnerUp"] == pytest.approx(0.8)  # 첫 값은 그대로 (증폭 없음)
    assert first.face_blendshapes["browDownLeft"] == 0.0
    assert len(first.face_blendshapes) == len(landmarks.BLENDSHAPE_NAMES)
    nxt = rt.step(raw({"browInnerUp": 0.0}))
    assert nxt.face_blendshapes["browInnerUp"] == pytest.approx(0.4)  # 연속 검출 사이는 EMA
    held = rt.step(raw(None))
    assert held.face_blendshapes["browInnerUp"] == pytest.approx(0.4)  # 미검출은 유지
    back = rt.step(raw({"browInnerUp": 0.6}))
    assert back.face_blendshapes["browInnerUp"] == pytest.approx(0.6)  # 재검출 첫 프레임은 리셋 (EMASmoother 규칙)


def test_default_face_scale_is_one():
    assert build_arg_parser().parse_args(["x.mp4"]).face_scale == 1.0


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
    assert (out_info.width, out_info.height) == (720, 1280)  # 출력 크기는 입력과 무관하게 고정


def test_cli_all_conditions_share_size_fps_and_frame_count(tmp_path, dummy_video):
    """조건 A·B·C의 해상도·fps·프레임 수가 같아야 한다 (교란변인 방지)."""
    parser = build_arg_parser()
    infos = []
    for view in ("original", "skeleton", "avatar"):
        out_path = str(tmp_path / f"out_{view}.mp4")
        args = parser.parse_args(
            [dummy_video, "--out", out_path, "--view", view, "--upper-body", "--complexity", "0",
             "--output-size", "180x320"]
        )
        assert run(args) == 0
        probe = probe_video(out_path)
        infos.append((probe.width, probe.height, round(probe.fps, 3), _count_frames(out_path)))
    assert infos[0] == infos[1] == infos[2]
    assert infos[0][:3] == (180, 320, 30.0)


def test_cli_invalid_output_size_returns_error(dummy_video):
    parser = build_arg_parser()
    assert run(parser.parse_args([dummy_video, "--output-size", "720"])) == 1


def test_cli_missing_input_returns_error(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args([str(tmp_path / "없는파일.mp4")])
    assert run(args) == 1


def test_cli_invalid_smooth_value_returns_error(dummy_video):
    parser = build_arg_parser()
    args = parser.parse_args([dummy_video, "--smooth", "1.5"])
    assert run(args) == 1
