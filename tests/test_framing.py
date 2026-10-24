import numpy as np
import pytest

from mocap import framing, landmarks
from mocap.renderer import FOCAL_LENGTH_FACTOR, CapsuleRenderer, SkeletonRenderer
from mocap.retarget import FrameData

TARGET = framing.FramingTarget(output_size=(720, 1280), shoulder_ratio=0.35, anchor=(0.5, 0.42))


def _pose_pixel(mid, width):
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    pose[landmarks.LEFT_SHOULDER, :2] = (mid[0] + width / 2, mid[1])
    pose[landmarks.RIGHT_SHOULDER, :2] = (mid[0] - width / 2, mid[1])
    return pose


def test_parse_size():
    assert framing.parse_size("720x1280") == (720, 1280)
    for bad in ("720", "x1280", "0x10", "abcxdef"):
        with pytest.raises(ValueError):
            framing.parse_size(bad)


def test_original_transform_puts_frontal_shoulders_on_target():
    # 정면 프레임(폭 400) 3개 + 몸을 튼 프레임(폭 250) 1개 + 미검출 1개
    poses = [_pose_pixel((540, 700), 400)] * 3 + [_pose_pixel((600, 700), 250), None]
    track = framing.shoulder_track(poses, (1080, 1920), (1080, 1920))
    xf = framing.original_transform(track, (1080, 1920), TARGET)

    width_out = 400 * xf.scale
    assert width_out == pytest.approx(TARGET.shoulder_px)  # 몸을 튼 프레임에 끌려가지 않음
    assert xf.scale * np.array([540, 700]) + xf.offset == pytest.approx(TARGET.anchor_px)


def test_shoulder_track_rescales_tracking_pixels_to_source():
    track = framing.shoulder_track([_pose_pixel((200, 300), 100)], (405, 720), (1080, 1920))
    assert track.widths[0] == pytest.approx(100 * 1080 / 405)


def test_apply_original_transform_places_pixels():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[40:60, 90:110] = 255  # 중앙의 흰 사각형 (중심 100, 50)
    xf = framing.OriginalTransform(scale=2.0, offset=np.array([60.0, 30.0]), source_size=(200, 100), output_size=(320, 240))
    out = framing.apply_original_transform(frame, xf)
    assert out.shape == (240, 320, 3)
    assert out[130, 260].tolist() == [255, 255, 255]  # 2*(100,50)+(60,30)
    assert out[5, 5].tolist() == [0, 0, 0]
    assert xf.upscaled
    assert xf.uncovered_ratio == pytest.approx(1 - (260 * 200) / (320 * 240))


def test_camera_for_target_puts_rendered_shoulders_on_target():
    pose = landmarks.build_neutral_pose()
    focal = 1280 * FOCAL_LENGTH_FACTOR
    cam = framing.camera_for_target([pose, pose, None], focal, TARGET)
    for cls in (CapsuleRenderer, SkeletonRenderer):
        r = cls(width=720, height=1280, upper_body=True, camera=cam)
        r._begin_frame(FrameData(pose=pose))
        sh = r._project(r._camera_transform(pose[[landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER]]))
        assert np.linalg.norm(sh[0] - sh[1]) == pytest.approx(TARGET.shoulder_px, rel=1e-6)
        assert sh.mean(axis=0) == pytest.approx(TARGET.anchor_px, abs=1e-6)


def test_screen_anchor_moves_render_to_requested_position():
    pose = landmarks.build_neutral_pose()
    r = CapsuleRenderer(width=720, height=1280, upper_body=True)
    anchor = np.array([250.0, 600.0])
    r._begin_frame(FrameData(pose=pose, screen_anchor=anchor))
    sh = r._project(r._camera_transform(pose[[landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER]]))
    assert sh.mean(axis=0) == pytest.approx(anchor)


def test_screen_anchors_smooth_and_hold_missing():
    poses = [None, _pose_pixel((100, 100), 50), None, _pose_pixel((200, 100), 50)]
    track = framing.shoulder_track(poses, (400, 400), (400, 400))
    xf = framing.OriginalTransform(1.0, np.zeros(2), (400, 400), (400, 400))
    anchors = framing.screen_anchors(track, xf, smooth=0.5)
    assert anchors[0] == pytest.approx([100, 100])  # 첫 검출 전은 첫 검출값
    assert anchors[2] == pytest.approx([100, 100])  # 미검출은 유지
    assert anchors[3] == pytest.approx([150, 100])  # EMA


def test_hide_legs_removes_leg_segments():
    r = CapsuleRenderer(width=100, height=100, upper_body=True, hide_legs=True)
    assert all(not part.endswith("_leg") for _, _, part in r._limb_segments())
    assert len(CapsuleRenderer(width=100, height=100)._limb_segments()) == len(landmarks.POSE_LIMB_SEGMENTS)
