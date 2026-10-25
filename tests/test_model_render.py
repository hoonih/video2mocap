"""3D 모델 렌더 경로(Phase 4E): 카메라 계획·화면 이동 계산 (+ 모델이 있으면 실제 렌더 결정론)."""
import os
import shutil

import numpy as np
import pytest

from mocap import framing, model_render

TARGET = framing.FramingTarget(output_size=(720, 1280), shoulder_ratio=0.35, anchor=(0.5, 0.42))


def _shoulders(n=5, width=0.22, mid=(0.0, 1.27, 0.0)):
    mid = np.array(mid)
    s = np.empty((n, 2, 3))
    s[:, 0] = mid + [width / 2, 0, 0]
    s[:, 1] = mid - [width / 2, 0, 0]
    return s


def test_camera_projection_matches_threejs_convention():
    cam = model_render.ModelCamera(width=720, height=1280, fov_deg=30.0, position=np.array([0.0, 1.0, 3.0]))
    px = cam.project(np.array([[0.0, 1.0, 0.0], [0.5, 1.5, 0.0]]))
    assert px[0] == pytest.approx([360, 640])
    assert px[1][0] > 360 and px[1][1] < 640  # 모델 왼쪽(+X)은 화면 오른쪽, 위(+Y)는 화면 위
    job = cam.to_job()
    assert job["target"] == [0.0, 1.0, 0.0] and job["near"] == pytest.approx(1.5) and job["far"] == pytest.approx(6.0)


def test_plan_camera_puts_model_shoulders_at_target_width():
    shoulders = _shoulders()
    cam = model_render.plan_camera(shoulders, TARGET)
    px = cam.project(shoulders[0])
    assert np.linalg.norm(px[0] - px[1]) == pytest.approx(TARGET.shoulder_px, rel=1e-6)


def test_plan_camera_matches_condition_a_width_on_frontal_frames():
    shoulders = _shoulders(n=4)
    a_widths = np.array([300.0, 300.0, 150.0, np.nan])  # 마지막 둘은 몸을 튼 프레임·미검출
    frontal = np.array([True, True, False, False])
    cam = model_render.plan_camera(shoulders, TARGET, a_widths, frontal)
    px = cam.project(shoulders[0])
    assert np.linalg.norm(px[0] - px[1]) == pytest.approx(300.0, rel=1e-6)


def test_view_offsets_move_shoulder_mid_to_anchor():
    shoulders = _shoulders(n=2)
    cam = model_render.plan_camera(shoulders, TARGET)
    anchors = [np.array([300.0, 500.0]), None]
    offsets = model_render.view_offsets(cam, shoulders, anchors)
    moved = cam.project(shoulders[0], offsets[0]).mean(axis=0)
    assert moved == pytest.approx(anchors[0])
    assert offsets[1] == pytest.approx([0, 0])


needs_model = pytest.mark.skipif(
    not (os.path.exists(model_render.DEFAULT_MODEL_PATH) and shutil.which("node")
         and os.path.isdir(os.path.join(model_render.RENDER_DIR, "node_modules"))),
    reason="아바타 모델 또는 render/ 의존성 없음 (assets/avatar/README.md, render/package.json)",
)


@needs_model
def test_model_render_is_deterministic(tmp_path):
    from mocap import landmarks
    from mocap.retarget import FrameData

    rest = model_render.ensure_rest_pose(model_render.DEFAULT_MODEL_PATH)
    pose = landmarks.build_neutral_pose()
    frames = [FrameData(pose=pose, face_blendshapes={"browInnerUp": v}) for v in (0.0, 0.5, 1.0)]
    target = framing.FramingTarget(output_size=(180, 320))
    plan = model_render.plan_model_render(frames, rest, target)
    a = list(model_render.render_plan(plan, model_render.DEFAULT_MODEL_PATH))
    b = list(model_render.render_plan(plan, model_render.DEFAULT_MODEL_PATH))
    assert len(a) == 3 and a[0].shape == (320, 180, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[2])  # 표정이 실제로 반영됨


@needs_model
def test_bones_and_model_views_align_and_match_python_projection():
    """조건 B(bones)와 C(model)의 주요 관절 화면 좌표가 같고, Python 투영과 1px 이내로 일치해야 한다."""
    import math

    from mocap import landmarks, rig
    from mocap.retarget import FrameData

    rest = model_render.ensure_rest_pose(model_render.DEFAULT_MODEL_PATH)
    pose = landmarks.build_neutral_pose()
    raised = pose.copy()
    raised[landmarks.LEFT_ELBOW] = (0.35, -0.75, -0.1)   # 왼팔을 들어 올린 자세 (MediaPipe 좌표)
    raised[landmarks.LEFT_WRIST] = (0.2, -0.95, -0.25)
    frames = [FrameData(pose=p, face_blendshapes={"browInnerUp": 0.8}, screen_anchor=np.array(a))
              for p, a in ((pose, (90.0, 135.0)), (raised, (100.0, 130.0)))]
    plan = model_render.plan_model_render(frames, rest, framing.FramingTarget(output_size=(180, 320)))
    jb, jc = [], []
    b = list(model_render.render_plan(plan, model_render.DEFAULT_MODEL_PATH, "bones", jb))
    c = list(model_render.render_plan(plan, model_render.DEFAULT_MODEL_PATH, "model", jc))
    assert len(b) == len(c) == 2 and b[0].shape == c[0].shape == (320, 180, 3)
    for fb, fc, r, off in zip(jb, jc, plan.rigs, plan.offsets):
        assert fb.keys() == fc.keys()
        pos, _ = rig.forward_kinematics(rest, r.bones)
        for name in fb:
            assert fb[name] == pytest.approx(fc[name], abs=1e-6), name                      # B = C
            expected = plan.camera.project(np.array([pos.get(name, rest.positions[name])]), off)[0]
            assert math.dist(fc[name], expected) <= 1.0, name                                # Python 투영
    spread = max(int((f.max(axis=2).astype(int) - f.min(axis=2)).max()) for f in b)
    assert spread <= 4  # 조건 B는 회색조 선·점만
    again = list(model_render.render_plan(plan, model_render.DEFAULT_MODEL_PATH, "bones"))
    assert all(np.array_equal(x, y) for x, y in zip(b, again))  # 결정론
