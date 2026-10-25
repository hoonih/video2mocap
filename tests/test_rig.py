"""리타게팅 솔버 왕복 검증: 알려진 각도로 만든 관절 위치 → 솔버 → 순운동학 → 원래 방향과 비교."""
import math

import numpy as np
import pytest

from mocap import landmarks, rig
from mocap.retarget import FrameData

SIDES = ("left", "right")


def _synthetic_rest() -> rig.RestPose:
    """VRM 1.0 규약 T-포즈 (모델 왼쪽 +X, 위 +Y, 정면 +Z). 손바닥은 아래, 엄지는 앞쪽."""
    p = {"hips": [0, 0.9, 0], "spine": [0, 1.0, 0], "chest": [0, 1.1, 0], "upperChest": [0, 1.2, 0],
         "neck": [0, 1.32, 0], "head": [0, 1.4, 0]}
    for side, sx in (("left", 1.0), ("right", -1.0)):
        p[f"{side}Shoulder"] = [0.03 * sx, 1.28, 0]
        p[f"{side}UpperArm"] = [0.11 * sx, 1.27, 0]
        p[f"{side}LowerArm"] = [0.33 * sx, 1.265, 0]
        p[f"{side}Hand"] = [0.55 * sx, 1.26, 0]
        p[f"{side}ThumbMetacarpal"] = [0.565 * sx, 1.25, 0.025]
        p[f"{side}ThumbProximal"] = [0.59 * sx, 1.247, 0.045]
        p[f"{side}ThumbDistal"] = [0.615 * sx, 1.245, 0.06]
        for name, z, length in (("Index", 0.02, 0.025), ("Middle", 0.0, 0.028), ("Ring", -0.018, 0.026), ("Little", -0.034, 0.02)):
            x0 = 0.63 * sx
            p[f"{side}{name}Proximal"] = [x0, 1.26, z]
            p[f"{side}{name}Intermediate"] = [x0 + length * 1.4 * sx, 1.26, z]
            p[f"{side}{name}Distal"] = [x0 + length * 2.3 * sx, 1.26, z]
    return rig.RestPose({k: np.array(v, dtype=float) for k, v in p.items()})


REST = _synthetic_rest()
_FINGER_TIP_EXTRA = 0.018


def _landmarks_from_fk(local):
    """순운동학으로 MediaPipe 형식 pose(33)·손(21) 랜드마크를 만든다 (MediaPipe 좌표계로 변환)."""
    pos, rot = rig.forward_kinematics(REST, local)
    to_mp = np.array([1.0, -1.0, -1.0])
    pose = np.zeros((landmarks.NUM_POSE_LANDMARKS, 3))
    for side in SIDES:
        L = side == "left"
        pose[landmarks.LEFT_SHOULDER if L else landmarks.RIGHT_SHOULDER] = pos[f"{side}UpperArm"]
        pose[landmarks.LEFT_ELBOW if L else landmarks.RIGHT_ELBOW] = pos[f"{side}LowerArm"]
        pose[landmarks.LEFT_WRIST if L else landmarks.RIGHT_WRIST] = pos[f"{side}Hand"]
    hands = {}
    for side in SIDES:
        h = np.zeros((21, 3))
        h[0] = pos[f"{side}Hand"]
        chain = [f"{side}Thumb{s}" for s in ("Metacarpal", "Proximal", "Distal")]
        for i, bone in enumerate(chain):
            h[1 + i] = pos[bone]
        h[4] = pos[chain[2]] + rot[chain[2]] @ (REST.positions[chain[2]] - REST.positions[chain[1]])
        for base, name in ((5, "Index"), (9, "Middle"), (13, "Ring"), (17, "Little")):
            bones = [f"{side}{name}{s}" for s in ("Proximal", "Intermediate", "Distal")]
            for i, bone in enumerate(bones):
                h[base + i] = pos[bone]
            d = REST.positions[bones[2]] - REST.positions[bones[1]]
            h[base + 3] = pos[bones[2]] + rot[bones[2]] @ (d / np.linalg.norm(d) * _FINGER_TIP_EXTRA)
        hands[side] = h * to_mp
    return FrameData(pose=pose * to_mp, left_hand=hands["left"], right_hand=hands["right"]), pos, rot


def _angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))


def _bone_dirs(pos):
    return {b: pos[b] - pos[p] for b, p in rig.HUMANOID_PARENT.items() if b in pos and p in pos and p != "hips"}


def _palm(pos, side):
    return rig._palm_normal(pos[f"{side}Hand"], pos[f"{side}IndexProximal"], pos[f"{side}LittleProximal"], side)


def _roundtrip(local):
    frame, pos, _ = _landmarks_from_fk(local)
    solved = rig.RigSolver(REST).solve(frame)
    pos2, _ = rig.forward_kinematics(REST, solved.bones)
    return solved, pos, pos2


def test_rest_pose_solves_to_identity():
    solved, _, _ = _roundtrip({})
    for bone, q in solved.bones.items():
        assert q == pytest.approx([0, 0, 0, 1], abs=1e-6), bone
    assert solved.violations == []


def _ax(axis, deg):
    return rig.axis_angle(np.array(axis, float), math.radians(deg))


SIGNING_POSE = {
    # 왼팔: 앞으로 들고 팔꿈치 90도, 손바닥 위로(아래팔 비틀림), 손목 약간 굽힘
    "leftUpperArm": _ax([0, 0, 1], -50) @ _ax([0, 1, 0], -30),
    "leftLowerArm": _ax([0, 1, 0], -90) @ _ax([1, 0, 0], 150),
    "leftHand": _ax([0, 0, 1], -15),
    "leftIndexProximal": _ax([0, 1, 0], 10) @ _ax([0, 0, 1], -50),
    "leftIndexIntermediate": _ax([0, 0, 1], -70),
    "leftIndexDistal": _ax([0, 0, 1], -30),
    "leftMiddleProximal": _ax([0, 0, 1], -80),
    "leftMiddleIntermediate": _ax([0, 0, 1], -90),
    # 오른팔: 거울상에 가까운 다른 자세
    "rightUpperArm": _ax([0, 0, 1], 60) @ _ax([0, 1, 0], 20),
    "rightLowerArm": _ax([0, 1, 0], 70) @ _ax([1, 0, 0], -60),
    "rightRingProximal": _ax([0, 0, 1], 60),
    "rightLittleIntermediate": _ax([0, 0, 1], 80),
}


def test_roundtrip_recovers_bone_directions():
    solved, original, recovered = _roundtrip(SIGNING_POSE)
    for bone, d in _bone_dirs(original).items():
        if "Thumb" in bone or bone in ("neck", "head", "spine", "chest", "upperChest"):
            continue
        assert _angle_deg(d, _bone_dirs(recovered)[bone]) < 1.0, bone
    assert solved.violations == []


@pytest.mark.parametrize("side", SIDES)
def test_roundtrip_recovers_palm_direction(side):
    _, original, recovered = _roundtrip(SIGNING_POSE)
    assert _angle_deg(_palm(original, side), _palm(recovered, side)) < 1.0


@pytest.mark.parametrize("side, axis_sign", [("left", 1.0), ("right", -1.0)])
def test_palm_flip_follows_forearm_twist(side, axis_sign):
    """손바닥 아래/위만 다른 두 자세: 모델 손바닥도 따라 뒤집혀야 한다 (음운 자질)."""
    bend = {f"{side}LowerArm": _ax([0, 1, 0], -80 * axis_sign)}
    down = dict(bend)
    up = {f"{side}LowerArm": bend[f"{side}LowerArm"] @ _ax([axis_sign, 0, 0], 180)}
    _, _, pos_down = _roundtrip(down)
    _, _, pos_up = _roundtrip(up)
    assert _palm(pos_down, side)[1] < -0.9  # 손바닥 아래
    assert _palm(pos_up, side)[1] > 0.9     # 손바닥 위


def test_finger_hyperextension_is_clamped_and_logged():
    _, _, recovered = _roundtrip({})
    solved, _, recovered = _roundtrip({"leftIndexIntermediate": _ax([0, 0, 1], 60)})  # 손등 쪽으로 60도
    assert any("leftIndexIntermediateFlex" in v for v in solved.violations)
    d1 = _bone_dirs(recovered)["leftIndexIntermediate"]
    d2 = _bone_dirs(recovered)["leftIndexDistal"]
    back = np.cross(d1, d2)[2]  # 왼손: 손등 쪽 굽힘은 +Z 축 회전
    assert math.degrees(math.asin(np.clip(back / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1, 1))) <= 10.5


def test_elbow_cannot_bend_backward():
    """팔꿈치 굽힘 방향이 위팔 비틀림을 정하므로, 어떤 굽힘도 앞쪽 경첩으로 재현된다."""
    solved, original, recovered = _roundtrip({"leftLowerArm": _ax([0, 1, 0], 70)})  # 뒤쪽으로 굽힌 입력
    assert _angle_deg(_bone_dirs(original)["leftLowerArm"], _bone_dirs(recovered)["leftLowerArm"]) < 1.0
    lower_local = rig.quaternion_to_matrix(solved.bones["leftLowerArm"])
    # 로컬 아래팔 회전은 위팔 기준 앞쪽(+Z) 경첩: 휴지 아래팔 방향(+X)이 +Z 쪽으로 간다
    assert (lower_local @ np.array([1.0, 0, 0]))[2] > 0.5


def test_straight_arm_keeps_previous_twist_without_nan():
    solver = rig.RigSolver(REST)
    bent, _, _ = _landmarks_from_fk({"leftLowerArm": _ax([0, 1, 0], -60)})
    straight, _, _ = _landmarks_from_fk({})
    solver.solve(bent)
    out = solver.solve(straight)
    assert all(np.all(np.isfinite(q)) for q in out.bones.values())


def test_quaternions_are_unit_with_nonnegative_w_and_deterministic():
    frame, _, _ = _landmarks_from_fk(SIGNING_POSE)
    a = rig.RigSolver(REST).solve(frame)
    b = rig.RigSolver(REST).solve(frame)
    assert a.bones == b.bones
    for q in a.bones.values():
        assert np.linalg.norm(q) == pytest.approx(1.0) and q[3] >= 0


def test_no_leg_bones_are_driven():
    solved, _, _ = _roundtrip(SIGNING_POSE)
    assert not any(k.endswith(("UpperLeg", "LowerLeg", "Foot", "Toes")) or k == "hips" for k in solved.bones)


@pytest.mark.parametrize("side", SIDES)
def test_thumb_roundtrip(side):
    sx = 1.0 if side == "left" else -1.0
    pose = {
        f"{side}ThumbMetacarpal": _ax([0, 1, 0], -25 * sx) @ _ax([0, 0, 1], -20 * sx),  # 손바닥 쪽으로 모음
        f"{side}ThumbProximal": _ax([0, 1, 0], -40 * sx),
        f"{side}ThumbDistal": _ax([0, 1, 0], -30 * sx),
    }
    _, original, recovered = _roundtrip(pose)
    for seg in ("ThumbProximal", "ThumbDistal"):
        bone = f"{side}{seg}"
        assert _angle_deg(_bone_dirs(original)[bone], _bone_dirs(recovered)[bone]) < 3.0, bone


def test_reconcile_forearm_depth_only_fixes_implausible_wrist():
    hand = np.array([1.0, 0.0, 0.0])
    mild = np.array([1.0, 0.0, 0.4])            # 손목 약 22도: 그대로
    assert np.allclose(rig.reconcile_forearm_depth(mild, hand), mild)
    severe = np.array([1.0, 0.1, 1.5])           # 손목 약 56도: 깊이만 바꿔 35도로
    fixed = rig.reconcile_forearm_depth(severe, hand)
    assert fixed[:2] == pytest.approx(severe[:2])
    assert _angle_deg(fixed, hand) == pytest.approx(rig.RECONCILE_WRIST_DEG, abs=0.1)


# --- 얼굴 접촉 위치 보정 (IK) -----------------------------------------------------

def test_two_bone_ik_reaches_target_and_keeps_lengths():
    s, e, w = np.array([0.1, 1.3, 0.0]), np.array([0.3, 1.1, 0.1]), np.array([0.35, 1.3, 0.25])
    target = np.array([0.2, 1.45, 0.2])
    ne, nw = rig.two_bone_ik(s, e, w, target)
    assert np.linalg.norm(ne - s) == pytest.approx(np.linalg.norm(e - s))
    assert np.linalg.norm(nw - ne) == pytest.approx(np.linalg.norm(w - e))
    assert nw == pytest.approx(target, abs=1e-6)


def _rest_with_face():
    rest = _synthetic_rest()
    head = rest.positions["head"]
    # 애니메이션풍 비율: 눈꼬리 간격 넓고 눈~턱이 짧다
    rest.face_anchors = {
        "eyeOuterLeft": head + [0.068, 0.055, 0.06], "eyeOuterRight": head + [-0.068, 0.055, 0.06],
        "noseTip": head + [0, 0.02, 0.11], "mouth": head + [0, 0.005, 0.10], "chin": head + [0, -0.035, 0.08],
        "forehead": head + [0, 0.2, 0.04],
    }
    return rest


_PX_PER_M = 1000.0


def _to_image(points_model):
    """모델 좌표 → 정면 정사영 영상 픽셀 (x 오른쪽, y 아래)."""
    p = np.atleast_2d(points_model)
    return np.stack([p[:, 0] * _PX_PER_M, -p[:, 1] * _PX_PER_M], axis=1)


def _human_face_model(center):
    """사람 비율의 얼굴 점 (모델 좌표, FACE_KEY_ORDER 순서). center는 머리 중심."""
    k = landmarks.FACE_KEY_ORDER
    pts = {name: np.zeros(3) for name in k}
    pts.update({"left_eye": [0.045, 0.06, 0.07], "right_eye": [-0.045, 0.06, 0.07], "nose_tip": [0, 0.02, 0.11],
                "mouth_top": [0, -0.02, 0.095], "mouth_bottom": [0, -0.03, 0.095], "chin": [0, -0.065, 0.085],
                "forehead": [0, 0.13, 0.08], "face_left_edge": [0.07, 0.02, 0.0], "face_right_edge": [-0.07, 0.02, 0.0]})
    return np.array([np.asarray(center) + np.asarray(pts[n], float) for n in k])


def test_face_mapping_sends_anchors_to_anchors():
    rest = _rest_with_face()
    ik = rig.FaceContactIK(rest)
    face_img = _to_image(_human_face_model(np.array([0.0, 1.5, 0.0])))
    human = ik._human_anchors_2d(face_img)
    pos, rot = rig.forward_kinematics(rest, {})
    model = ik._model_anchors(pos, rot)
    for name in ("eyeLeft", "eyeRight", "mouth", "chin"):
        pt = face_img[landmarks.FACE_KEY_ORDER.index({"eyeLeft": "left_eye", "eyeRight": "right_eye", "chin": "chin"}.get(name, "mouth_top"))]
        if name == "mouth":
            k = landmarks.FACE_KEY_ORDER.index
            pt = (face_img[k("mouth_top")] + face_img[k("mouth_bottom")]) / 2
        far = model[name] + np.array([0, 0, 0.2])          # 손이 얼굴 앞 20cm에 있어도
        mapped = ik.map_point_2d(pt, human, model, current=far)
        origin, basis, _ = rig._face_frame(model["eyeLeft"], model["eyeRight"], model["chin"])
        surface = origin + basis @ (basis.T @ (model[name] - origin) + [0, 0, rig.FACE_CONTACT_SURFACE_GAP])
        assert mapped[:2] == pytest.approx(surface[:2], abs=1e-6), name   # 화면 평면 위치는 그 기준점
        assert mapped[2] == pytest.approx(far[2]), name                     # 깊이는 현재 값 유지


def test_fingertip_touching_chin_lands_on_model_chin():
    """영상에서 사람 검지 끝이 턱에 닿음 → IK 후 모델 검지 끝이 (정면에서 보아) 모델 턱 위치."""
    rest = _rest_with_face()
    pose_local = {"rightUpperArm": _ax([0, 0, 1], 70) @ _ax([0, 1, 0], 30), "rightLowerArm": _ax([0, 1, 0], 120)}
    frame, _, _ = _landmarks_from_fk(pose_local)
    tip = frame.right_hand[8] * np.array([1.0, -1.0, -1.0])
    face = _human_face_model(tip - np.array([0, -0.065, 0.085]))   # 검지 끝이 정확히 사람 턱에 오도록
    frame.face_image = _to_image(face)
    frame.right_hand_image = _to_image(frame.right_hand * np.array([1.0, -1.0, -1.0]))

    def model_tip(solver):
        out = solver.solve(frame)
        pos, rot = rig.forward_kinematics(rest, out.bones)
        seg = rest.positions["rightIndexDistal"] - rest.positions["rightIndexIntermediate"]
        return pos["rightIndexDistal"] + rot["rightIndexDistal"] @ seg, out, pos, rot

    with_ik, out, pos, rot = model_tip(rig.RigSolver(rest, face_contact=True))
    without_ik, _, _, _ = model_tip(rig.RigSolver(rest, face_contact=False))
    model_chin = rig.FaceContactIK(rest)._model_anchors(pos, rot)["chin"]
    assert out.contact["right"] == pytest.approx(1.0)
    assert np.linalg.norm((with_ik - model_chin)[:2]) < 0.003       # 정면에서 본 위치
    assert np.linalg.norm((without_ik - model_chin)[:2]) > 0.01      # FK만이면 어긋남
