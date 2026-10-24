import os

import cv2
import numpy as np
import pytest

from mocap import landmarks
from mocap.renderer import BACKGROUND_COLOR, FACE_COLOR, CapsuleRenderer, SkeletonRenderer
from mocap.retarget import FrameData, face_to_world, head_center, synthesize_lower_body

from .face_fixture import canonical_face_pixel

SCRATCH_DIR = os.environ.get(
    "MOCAP_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-hong-yehun-PycharmProjects-research/80c1bbc3-1ebf-463c-b01e-c8c9a7e08bbc/scratchpad",
)


def _full_body_pose() -> np.ndarray:
    pose = landmarks.build_neutral_pose()
    # 오른팔을 굽혀서 시각적으로 자세 구분이 되도록 변형
    pose[landmarks.RIGHT_ELBOW] = (0.35, -0.75, 0.05)
    pose[landmarks.RIGHT_WRIST] = (0.15, -0.95, 0.15)
    return pose


def _fake_hand(wrist_world: np.ndarray, spread: float = 0.03) -> np.ndarray:
    hand = np.zeros((landmarks.NUM_HAND_LANDMARKS, 3))
    hand[landmarks.HAND_WRIST] = wrist_world
    for finger_i, finger in enumerate(landmarks.HAND_FINGERS):
        for joint_i, joint_idx in enumerate(finger):
            offset = np.array(
                [
                    (finger_i - 2) * spread * 0.6,
                    -spread * (joint_i + 1),
                    0.0,
                ]
            )
            hand[joint_idx] = wrist_world + offset
    return hand


def _real_face(pose: np.ndarray, **expression) -> np.ndarray:
    """표준 얼굴 모델을 FaceMesh 픽셀 형식으로 만든 뒤 실제 face_to_world를 거친 얼굴."""
    return face_to_world(canonical_face_pixel(**expression), head_center(pose))


def test_full_body_render_produces_nonblank_image():
    pose = _full_body_pose()
    frame = FrameData(pose=pose)
    renderer = CapsuleRenderer(width=640, height=720, upper_body=False, yaw_deg=15.0)

    img = renderer.render(frame)

    assert img.shape == (720, 640, 3)
    assert img.dtype == np.uint8
    assert img.std() > 1.0  # 완전히 단색이 아니어야 함(뭔가 그려짐)

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    from cv2 import imwrite

    imwrite(os.path.join(SCRATCH_DIR, "render_full_body.png"), img)


def test_upper_body_with_hands_and_face_render():
    pose = _full_body_pose()
    left_hand = _fake_hand(pose[landmarks.LEFT_WRIST])
    right_hand = _fake_hand(pose[landmarks.RIGHT_WRIST])
    face = _real_face(pose)

    frame = FrameData(pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
    renderer = CapsuleRenderer(width=480, height=640, upper_body=True, yaw_deg=0.0)

    img = renderer.render(frame)

    assert img.shape == (640, 480, 3)
    assert img.std() > 1.0

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    from cv2 import imwrite

    imwrite(os.path.join(SCRATCH_DIR, "render_upper_body_hands_face.png"), img)


def test_orbit_changes_projection_over_frames():
    pose = _full_body_pose()
    frame = FrameData(pose=pose)
    renderer = CapsuleRenderer(width=320, height=360, upper_body=False, orbit_deg_per_frame=30.0)

    img1 = renderer.render(frame)
    img2 = renderer.render(frame)

    assert not np.array_equal(img1, img2)  # 회전이 누적되어 다른 프레임이 나와야 함


def test_render_without_hands_or_face_does_not_crash():
    pose = _full_body_pose()
    frame = FrameData(pose=pose, left_hand=None, right_hand=None, face=None)
    renderer = CapsuleRenderer(width=200, height=200, upper_body=False)

    img = renderer.render(frame)
    assert img.shape == (200, 200, 3)


# --- 얼굴 표현 (Phase 1) ---------------------------------------------------------

def _render_face(**expression):
    pose = landmarks.build_neutral_pose()
    renderer = CapsuleRenderer(width=720, height=1280, upper_body=True)
    face = _real_face(pose, **expression)
    img = renderer.render(FrameData(pose=pose, face=face))
    center = head_center(pose)
    drawn = center + (face - center) * renderer.face_scale
    proj = renderer._project(renderer._camera_transform(drawn))
    return img, {name: proj[i] for i, name in enumerate(landmarks.FACE_KEY_ORDER)}


def _is_face_color(img, pt) -> bool:
    x, y = int(round(pt[0])), int(round(pt[1]))
    return tuple(int(v) for v in img[y, x]) == FACE_COLOR


def _face_pixels(img):
    ys, xs = np.nonzero(np.all(img == np.array(FACE_COLOR, dtype=np.uint8), axis=2))
    return xs, ys


def test_frontal_face_draws_nose_eyes_brows_and_mouth():
    """회귀: 코끝 기준 컬링 때문에 얼굴이 하나도 그려지지 않던 버그 (renderer.py 구 338행)."""
    img, pts = _render_face(mouth_open_cm=1.0)

    assert _is_face_color(img, pts["nose_tip"])
    for outline in (landmarks.LEFT_EYE_OUTLINE, landmarks.RIGHT_EYE_OUTLINE, landmarks.MOUTH_OUTLINE):
        assert _is_face_color(img, np.mean([pts[n] for n in outline], axis=0)), outline
    for brow in (landmarks.LEFT_BROW_KEYS, landmarks.RIGHT_BROW_KEYS):
        assert _is_face_color(img, pts[brow[2]]), brow


def test_brow_raise_and_lower_move_brow_pixels():
    raised, pts = _render_face(brow_raise_cm=0.8)
    lowered, _ = _render_face(brow_raise_cm=-0.5)
    eye_y = pts["left_eye"][1]

    def brow_mean_y(img):
        xs, ys = _face_pixels(img)
        return ys[ys < eye_y - 8].mean()

    assert brow_mean_y(lowered) - brow_mean_y(raised) > 6.0


def test_mouth_open_enlarges_mouth_shape():
    closed, pts = _render_face()
    opened, _ = _render_face(mouth_open_cm=1.5)
    nose_y = pts["nose_tip"][1]

    def mouth_area(img):
        _, ys = _face_pixels(img)
        return int(np.sum(ys > nose_y + 10))

    assert mouth_area(opened) > mouth_area(closed) * 2


def test_head_yaw_moves_face_across_head():
    _, frontal = _render_face()
    turned_img, turned = _render_face(yaw_deg=35)
    assert turned["nose_tip"][0] - frontal["nose_tip"][0] > 20  # 코가 화면 오른쪽으로 이동
    assert _is_face_color(turned_img, turned["nose_tip"])


def test_head_shake_hides_far_side_features():
    img, pts = _render_face(yaw_deg=80)
    # 화면 오른쪽으로 고개를 돌리면 화자의 왼쪽 눈이 뒤로 넘어가 보이지 않는다
    assert not _is_face_color(img, np.mean([pts[n] for n in landmarks.LEFT_EYE_OUTLINE], axis=0))


def test_face_scale_enlarges_head_and_face_together():
    pose = landmarks.build_neutral_pose()
    face = _real_face(pose, mouth_open_cm=1.0)
    small = CapsuleRenderer(width=720, height=1280, upper_body=True, face_scale=1.0).render(FrameData(pose=pose, face=face))
    large = CapsuleRenderer(width=720, height=1280, upper_body=True, face_scale=1.4).render(FrameData(pose=pose, face=face))
    assert len(_face_pixels(large)[0]) > len(_face_pixels(small)[0]) * 1.6  # 면적은 대략 배율의 제곱


# --- 골격 렌더러 (조건 B, Phase 2) -------------------------------------------------

def _signing_frame() -> FrameData:
    pose = _full_body_pose()
    return FrameData(
        pose=pose,
        left_hand=_fake_hand(pose[landmarks.LEFT_WRIST]),
        right_hand=_fake_hand(pose[landmarks.RIGHT_WRIST]),
        face=_real_face(pose, mouth_open_cm=1.0),
    )


def _foreground(img, include_face_color=False):
    mask = np.abs(img.astype(int) - np.array(BACKGROUND_COLOR)).sum(axis=2) > 30
    if include_face_color:
        mask |= np.all(img == np.array(FACE_COLOR, dtype=np.uint8), axis=2)
    return mask


@pytest.mark.parametrize("upper_body, yaw", [(True, 0.0), (True, 25.0), (False, -15.0)])
def test_skeleton_and_avatar_share_camera(upper_body, yaw):
    kwargs = dict(width=360, height=640, upper_body=upper_body, yaw_deg=yaw)
    avatar, skeleton = CapsuleRenderer(**kwargs), SkeletonRenderer(**kwargs)
    pose = _full_body_pose()

    for attr in ("camera_distance", "camera_height", "focal_length", "yaw_rad", "face_scale", "width", "height"):
        assert getattr(avatar, attr) == getattr(skeleton, attr), attr
    assert np.array_equal(
        avatar._project(avatar._camera_transform(pose)), skeleton._project(skeleton._camera_transform(pose))
    )


def test_skeleton_lies_inside_avatar_silhouette():
    """같은 프레임에서 골격의 모든 픽셀이 아바타 실루엣(얼굴 요소 포함) 안에 있어야 한다."""
    frame = _signing_frame()
    kwargs = dict(width=720, height=1280, upper_body=True)
    avatar_img = CapsuleRenderer(**kwargs).render(frame)
    skeleton_img = SkeletonRenderer(**kwargs).render(frame)

    avatar_mask = _foreground(avatar_img, include_face_color=True).astype(np.uint8)
    avatar_mask = cv2.dilate(avatar_mask, np.ones((5, 5), np.uint8)) > 0  # 안티에일리어싱 가장자리 허용
    skeleton_mask = _foreground(skeleton_img)

    assert skeleton_mask.sum() > 1000
    assert (skeleton_mask & ~avatar_mask).sum() == 0


def test_skeleton_has_no_filled_surfaces_or_floor():
    frame = _signing_frame()
    img = SkeletonRenderer(width=720, height=1280, upper_body=False).render(frame)
    colors = img.reshape(-1, 3)
    # 회색조만 쓴다 (아바타의 파란 면, 바닥 격자 색이 없어야 함)
    spread = colors.max(axis=1).astype(int) - colors.min(axis=1).astype(int)
    assert spread.max() <= 4
    # 선과 점만이므로 전경 비율이 아바타보다 훨씬 작다
    avatar = CapsuleRenderer(width=720, height=1280, upper_body=False).render(frame)
    assert _foreground(img).mean() < _foreground(avatar).mean() * 0.5


def test_skeleton_shows_brow_and_mouth_changes():
    pose = landmarks.build_neutral_pose()
    renderer = SkeletonRenderer(width=720, height=1280, upper_body=True)

    def render(**expression):
        return renderer.render(FrameData(pose=pose, face=_real_face(pose, **expression)))

    assert not np.array_equal(render(brow_raise_cm=0.8), render(brow_raise_cm=-0.5))
    assert not np.array_equal(render(), render(mouth_open_cm=1.5))


def test_upper_body_torso_is_not_crossed():
    """회귀: 하체 템플릿 좌우가 MediaPipe와 반대여서 몸통이 나비넥타이 모양으로 꼬이던 버그."""
    pose = synthesize_lower_body(landmarks.build_neutral_pose())
    assert pose[landmarks.LEFT_HIP, 0] > pose[landmarks.RIGHT_HIP, 0]
    assert pose[landmarks.LEFT_SHOULDER, 0] > pose[landmarks.RIGHT_SHOULDER, 0]
