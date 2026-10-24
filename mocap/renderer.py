"""캡슐형(관절 인형 스타일) 3D 캐릭터 렌더러.

OpenCV만으로 원근 투영, Y축 회전 카메라, 화가의 알고리즘(depth sort)을 구현해
FrameData를 캐릭터 이미지로 그린다. 나중에 실제 3D 모델(glTF 등) 렌더러로 교체할 수
있도록 `Renderer` 인터페이스만 외부(cli.py)가 의존하게 설계했다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from . import landmarks
from .retarget import FrameData, head_center

# ---------------------------------------------------------------------------
# 카메라/투영 상수
# ---------------------------------------------------------------------------
FULL_BODY_CAMERA_DISTANCE = 3.2
FULL_BODY_CAMERA_HEIGHT = -0.6  # world y가 아래쪽 양수이므로 음수는 위쪽
UPPER_BODY_CAMERA_DISTANCE = 1.8
UPPER_BODY_CAMERA_HEIGHT = 0.25
FOCAL_LENGTH_FACTOR = 1.1  # 이미지 높이에 곱해서 초점거리로 사용

NECK_DEPTH_BIAS = 0.15  # 목이 항상 머리보다 뒤에 그려지도록 밀어주는 깊이 보정치

LIMB_THICKNESS = {
    "left_arm": 14,
    "right_arm": 14,
    "left_leg": 18,
    "right_leg": 18,
    "neck": 16,
}
HEAD_RADIUS_METERS = 0.11
# 머리와 얼굴 요소를 머리 중심 기준으로 확대하는 배율. 표정 움직임의 '비율'은 원본과 같게 두고
# 화면상 픽셀 변화량만 키운다 (docs/stimulus-spec.md 참고). 손이 닿는 얼굴 위치는 그만큼 어긋난다.
DEFAULT_FACE_SCALE = 1.4
FACE_COLOR = (30, 30, 30)
# 얼굴 요소 굵기 = 머리 반지름(px) × 비율
BROW_THICKNESS_RATIO = 0.07
FACE_OUTLINE_RATIO = 0.03
NOSE_RADIUS_RATIO = 0.04
FACE_DEPTH_EPSILON = 1e-3  # 얼굴 요소를 머리 구 바로 위에 그리기 위한 깊이 보정치
HAND_JOINT_RADIUS = 3
FINGER_THICKNESS = 3

FLOOR_HALF_SIZE = 1.5
FLOOR_GRID_STEP = 0.3
FLOOR_Y = None  # synthesize_lower_body 발 위치 기준으로 런타임에 계산

# 좌우 밝기 구분: 오른쪽 부위를 더 밝게
BASE_COLOR = np.array([170, 130, 60], dtype=np.float64)  # BGR
RIGHT_SIDE_BRIGHTEN = 1.25
LEFT_SIDE_BRIGHTEN = 0.9
OUTLINE_DARKEN = 0.45
HIGHLIGHT_BRIGHTEN = 1.6

BACKGROUND_COLOR = (24, 22, 20)  # 조건 B·C 공통 배경색

# 골격 렌더러 (조건 B): 선과 점만
SKELETON_COLOR = np.array([235, 235, 235], dtype=np.float64)
SKELETON_LEFT_DIM = 0.75  # 좌우 구분: 화자 왼쪽 부위를 약간 어둡게
SKELETON_LINE_THICKNESS = 4
SKELETON_JOINT_RADIUS = 6
SKELETON_HEAD_OUTLINE = 2
FLOOR_COLOR = (50, 48, 45)
SHADOW_COLOR = (15, 15, 15)


class Renderer(Protocol):
    """캐릭터 렌더러 인터페이스. 실제 3D 모델 렌더러로 교체할 때 이 인터페이스만 지키면 된다."""

    def render(self, frame: FrameData) -> np.ndarray:
        ...


@dataclass
class _Primitive:
    """화가의 알고리즘으로 정렬해서 그릴 하나의 그림 요소."""

    depth: float
    draw_fn: "callable"


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _clamp_color(color: np.ndarray) -> Tuple[int, int, int]:
    c = np.clip(color, 0, 255).astype(np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def _side_color(part_name: str) -> np.ndarray:
    if "right" in part_name:
        return BASE_COLOR * RIGHT_SIDE_BRIGHTEN
    if "left" in part_name:
        return BASE_COLOR * LEFT_SIDE_BRIGHTEN
    return BASE_COLOR.copy()


@dataclass
class _Head:
    """머리 구의 위치와 화면상 크기. 모든 렌더러가 같은 값을 쓴다."""

    world: np.ndarray
    cam: np.ndarray
    px: np.ndarray
    radius_px: int


@dataclass
class _FaceLayout:
    """얼굴 요소의 화면 좌표와 가시성 (landmarks.FACE_KEY_ORDER 이름 기준)."""

    pts: Dict[str, np.ndarray]
    visible: Dict[str, bool]

    def polygon(self, names) -> Optional[np.ndarray]:
        """names가 모두 보일 때만 (N,2) int32 좌표를 돌려준다."""
        if not all(self.visible[n] for n in names):
            return None
        return np.array([self.pts[n] for n in names], dtype=np.int32)


def _draw_hand_chains(img: np.ndarray, proj: np.ndarray, color: Tuple[int, int, int]) -> None:
    """손목에서 각 손가락 끝까지 선과 관절 점을 그린다. 조건 B·C가 같은 손 표현을 쓴다."""
    wrist_px = tuple(proj[landmarks.HAND_WRIST].astype(int))
    for finger in landmarks.HAND_FINGERS:
        prev_pt = wrist_px
        for joint_idx in finger:
            pt = tuple(proj[joint_idx].astype(int))
            cv2.line(img, prev_pt, pt, color, FINGER_THICKNESS, cv2.LINE_AA)
            cv2.circle(img, pt, HAND_JOINT_RADIUS, color, -1, cv2.LINE_AA)
            prev_pt = pt


class _ProjectedRenderer:
    """카메라·투영·배경·머리/얼굴 배치를 공유하는 렌더러 베이스.

    조건 B(SkeletonRenderer)와 C(CapsuleRenderer)는 화자 위치와 크기가 픽셀 단위로 같아야 하므로
    카메라 파라미터와 머리/얼굴 기하는 전부 여기서만 계산한다. 하위 클래스는 그리기만 한다.
    """

    def __init__(
        self,
        width: int,
        height: int,
        upper_body: bool = False,
        yaw_deg: float = 0.0,
        orbit_deg_per_frame: float = 0.0,
        face_scale: float = DEFAULT_FACE_SCALE,
    ) -> None:
        self.width = width
        self.height = height
        self.upper_body = upper_body
        self.yaw_rad = math.radians(yaw_deg)
        self.orbit_rad_per_frame = math.radians(orbit_deg_per_frame)
        self._orbit_accum = 0.0
        self.face_scale = face_scale

        self.camera_distance = UPPER_BODY_CAMERA_DISTANCE if upper_body else FULL_BODY_CAMERA_DISTANCE
        self.camera_height = UPPER_BODY_CAMERA_HEIGHT if upper_body else FULL_BODY_CAMERA_HEIGHT
        self.focal_length = height * FOCAL_LENGTH_FACTOR

    # -- 카메라/투영 ---------------------------------------------------------

    def _camera_transform(self, points: np.ndarray) -> np.ndarray:
        """월드 좌표(N,3)를 카메라 공간(N,3)으로 변환한다. z는 카메라 앞쪽으로 클수록 멀다."""
        rot = _rotation_y(self.yaw_rad + self._orbit_accum)
        rotated = points @ rot.T
        cam = rotated.copy()
        cam[:, 1] += self.camera_height
        cam[:, 2] += self.camera_distance
        return cam

    def _project(self, cam_points: np.ndarray) -> np.ndarray:
        """카메라 공간 좌표를 화면 픽셀 좌표(N,2)로 원근 투영한다."""
        z = np.clip(cam_points[:, 2], 0.05, None)
        x = self.focal_length * cam_points[:, 0] / z + self.width / 2.0
        y = self.focal_length * cam_points[:, 1] / z + self.height / 2.0
        return np.stack([x, y], axis=1)

    # -- 공통 기하 -------------------------------------------------------------

    def _blank_canvas(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), BACKGROUND_COLOR, dtype=np.uint8)

    def _head(self, pose: np.ndarray) -> _Head:
        world = head_center(pose)
        cam = self._camera_transform(world[np.newaxis, :])[0]
        px = self._project(cam[np.newaxis, :])[0]
        radius_px = max(4, int(self.focal_length * HEAD_RADIUS_METERS * self.face_scale / max(cam[2], 0.05)))
        return _Head(world=world, cam=cam, px=px, radius_px=radius_px)

    def _face_layout(self, face: np.ndarray, head: _Head) -> Optional[_FaceLayout]:
        """얼굴 요소를 face_scale만큼 키워 투영하고, 머리 앞쪽 반구에 있는 것만 보이게 한다."""
        face_world = head.world + (face - head.world) * self.face_scale
        cam = self._camera_transform(face_world)
        proj = self._project(cam)
        # 머리 중심보다 카메라 쪽(앞 반구)에 있는 점만 보인다. 고개를 돌리면 먼 쪽 눈썹/눈이 사라진다.
        visible = cam[:, 2] < head.cam[2]
        if not np.any(visible):
            return None
        names = landmarks.FACE_KEY_ORDER
        return _FaceLayout(
            pts={name: proj[i].astype(np.int32) for i, name in enumerate(names)},
            visible={name: bool(visible[i]) for i, name in enumerate(names)},
        )

    def _face_line_widths(self, head: _Head) -> Tuple[int, int, int]:
        """(눈썹 굵기, 윤곽선 굵기, 코 점 반지름) — 머리 크기에 비례."""
        return (
            max(2, int(round(head.radius_px * BROW_THICKNESS_RATIO))),
            max(1, int(round(head.radius_px * FACE_OUTLINE_RATIO))),
            max(2, int(round(head.radius_px * NOSE_RADIUS_RATIO))),
        )

    def _advance_orbit(self) -> None:
        self._orbit_accum += self.orbit_rad_per_frame


class CapsuleRenderer(_ProjectedRenderer):
    """캡슐형 팔다리 + 몸통 폴리곤 + 구형 머리로 이루어진 캐릭터를 그리는 렌더러 (조건 C)."""

    def render(self, frame: FrameData) -> np.ndarray:
        img = self._blank_canvas()

        cam_pose = self._camera_transform(frame.pose)
        proj_pose = self._project(cam_pose)

        if not self.upper_body:
            self._draw_floor(img)
            self._draw_shadows(img, frame.pose)

        primitives: List[_Primitive] = []
        primitives.extend(self._limb_primitives(frame.pose, cam_pose, proj_pose))
        primitives.extend(self._torso_primitives(frame.pose, cam_pose, proj_pose))
        head = self._head(frame.pose)
        primitives.extend(self._neck_primitive(cam_pose, proj_pose, head.px))
        primitives.append(self._head_primitive(head.cam, head.px, head.radius_px))

        if frame.left_hand is not None:
            primitives.extend(self._hand_primitives(frame.left_hand, "left"))
        if frame.right_hand is not None:
            primitives.extend(self._hand_primitives(frame.right_hand, "right"))

        if frame.face is not None:
            primitives.extend(self._face_primitives(frame.face, head))

        primitives.sort(key=lambda p: p.depth, reverse=True)  # 먼 것부터 그려서 가까운 게 덮도록
        for prim in primitives:
            prim.draw_fn(img)

        self._advance_orbit()
        return img

    # -- 바닥 -----------------------------------------------------------------

    def _draw_floor(self, img: np.ndarray) -> None:
        lines = []
        steps = int(FLOOR_HALF_SIZE / FLOOR_GRID_STEP)
        floor_y = self._floor_y()
        for i in range(-steps, steps + 1):
            offset = i * FLOOR_GRID_STEP
            lines.append(
                np.array([[offset, floor_y, -FLOOR_HALF_SIZE], [offset, floor_y, FLOOR_HALF_SIZE]])
            )
            lines.append(
                np.array([[-FLOOR_HALF_SIZE, floor_y, offset], [FLOOR_HALF_SIZE, floor_y, offset]])
            )
        for line in lines:
            cam = self._camera_transform(line)
            if np.any(cam[:, 2] <= 0.05):
                continue
            proj = self._project(cam).astype(int)
            cv2.line(img, tuple(proj[0]), tuple(proj[1]), FLOOR_COLOR, 1, cv2.LINE_AA)

    def _floor_y(self) -> float:
        _, heel_y, _ = landmarks.LOWER_BODY_TEMPLATE[landmarks.LEFT_HEEL]
        return float(heel_y)

    def _draw_shadows(self, img: np.ndarray, pose: np.ndarray) -> None:
        floor_y = self._floor_y()
        for heel_idx, toe_idx in (
            (landmarks.LEFT_HEEL, landmarks.LEFT_FOOT_INDEX),
            (landmarks.RIGHT_HEEL, landmarks.RIGHT_FOOT_INDEX),
        ):
            center = (pose[heel_idx] + pose[toe_idx]) / 2.0
            center = center.copy()
            center[1] = floor_y
            cam = self._camera_transform(center[np.newaxis, :])[0]
            if cam[2] <= 0.05:
                continue
            proj = self._project(cam[np.newaxis, :])[0].astype(int)
            radius_x = max(3, int(self.focal_length * 0.12 / cam[2]))
            radius_y = max(2, int(radius_x * 0.4))
            cv2.ellipse(img, tuple(proj), (radius_x, radius_y), 0, 0, 360, SHADOW_COLOR, -1, cv2.LINE_AA)

    # -- 팔다리 (캡슐) ---------------------------------------------------------

    def _limb_primitives(
        self, pose: np.ndarray, cam_pose: np.ndarray, proj_pose: np.ndarray
    ) -> List[_Primitive]:
        prims = []
        for start_idx, end_idx, part_name in landmarks.POSE_LIMB_SEGMENTS:
            depth = float((cam_pose[start_idx, 2] + cam_pose[end_idx, 2]) / 2.0)
            p1 = tuple(proj_pose[start_idx].astype(int))
            p2 = tuple(proj_pose[end_idx].astype(int))
            thickness = LIMB_THICKNESS.get(part_name, 14)
            color = _side_color(part_name)

            def draw(img, p1=p1, p2=p2, thickness=thickness, color=color):
                self._draw_capsule_segment(img, p1, p2, thickness, color)

            prims.append(_Primitive(depth=depth, draw_fn=draw))
        return prims

    def _draw_capsule_segment(
        self, img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], thickness: int, color: np.ndarray
    ) -> None:
        outline_color = _clamp_color(color * OUTLINE_DARKEN)
        base_color = _clamp_color(color)
        highlight_color = _clamp_color(color * HIGHLIGHT_BRIGHTEN)

        cv2.line(img, p1, p2, outline_color, thickness + 4, cv2.LINE_AA)
        cv2.line(img, p1, p2, base_color, thickness, cv2.LINE_AA)

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > 1e-3:
            nx, ny = -dy / length, dx / length
            offset = thickness * 0.22
            h1 = (int(p1[0] + nx * offset), int(p1[1] + ny * offset))
            h2 = (int(p2[0] + nx * offset), int(p2[1] + ny * offset))
            cv2.line(img, h1, h2, highlight_color, max(1, thickness // 4), cv2.LINE_AA)

        for p in (p1, p2):
            cv2.circle(img, p, thickness // 2, base_color, -1, cv2.LINE_AA)

    # -- 몸통(폴리곤) ----------------------------------------------------------

    def _torso_primitives(
        self, pose: np.ndarray, cam_pose: np.ndarray, proj_pose: np.ndarray
    ) -> List[_Primitive]:
        idxs = [
            landmarks.LEFT_SHOULDER,
            landmarks.RIGHT_SHOULDER,
            landmarks.RIGHT_HIP,
            landmarks.LEFT_HIP,
        ]
        depth = float(np.mean(cam_pose[idxs, 2]))
        pts = proj_pose[idxs].astype(int)
        color = _clamp_color(BASE_COLOR * 1.05)
        outline = _clamp_color(BASE_COLOR * OUTLINE_DARKEN)

        def draw(img, pts=pts, color=color, outline=outline):
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, outline, 2, cv2.LINE_AA)

        return [_Primitive(depth=depth, draw_fn=draw)]

    # -- 목 (항상 머리 뒤) ------------------------------------------------------

    def _neck_primitive(
        self, cam_pose: np.ndarray, proj_pose: np.ndarray, head_px: np.ndarray
    ) -> List[_Primitive]:
        shoulder_mid_idx = (landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER)
        p1 = tuple(
            (
                (proj_pose[shoulder_mid_idx[0]] + proj_pose[shoulder_mid_idx[1]]) / 2.0
            ).astype(int)
        )
        p2 = tuple(head_px.astype(int))
        depth = float((cam_pose[shoulder_mid_idx[0], 2] + cam_pose[shoulder_mid_idx[1], 2]) / 2.0)
        depth += NECK_DEPTH_BIAS  # 항상 머리보다 뒤에 그리기 위한 강제 바이어스
        color = BASE_COLOR.copy()

        def draw(img, p1=p1, p2=p2, color=color):
            self._draw_capsule_segment(img, p1, p2, LIMB_THICKNESS["neck"], color)

        return [_Primitive(depth=depth, draw_fn=draw)]

    # -- 머리 (구) -------------------------------------------------------------

    def _head_primitive(
        self, head_cam: np.ndarray, head_px: np.ndarray, radius_px: int
    ) -> _Primitive:
        center = tuple(head_px.astype(int))
        base_color = _clamp_color(BASE_COLOR * 1.1)
        outline_color = _clamp_color(BASE_COLOR * OUTLINE_DARKEN)

        # 하이라이트는 그리지 않는다. 얼굴 요소와 겹치면 눈처럼 보여 표정 판독을 방해한다.
        def draw(img, center=center, radius_px=radius_px):
            cv2.circle(img, center, radius_px, outline_color, -1, cv2.LINE_AA)
            cv2.circle(img, center, max(1, int(radius_px * 0.9)), base_color, -1, cv2.LINE_AA)

        return _Primitive(depth=float(head_cam[2]), draw_fn=draw)

    # -- 손 (손가락 체인) --------------------------------------------------------

    def _hand_primitives(self, hand_world: np.ndarray, side: str) -> List[_Primitive]:
        cam = self._camera_transform(hand_world)
        proj = self._project(cam)
        depth = float(np.mean(cam[:, 2]))
        color = _side_color(f"{side}_hand" if side == "right" else "left_hand")
        base_color = _clamp_color(color)

        def draw(img, proj=proj, base_color=base_color):
            _draw_hand_chains(img, proj, base_color)

        return [_Primitive(depth=depth, draw_fn=draw)]

    # -- 얼굴 (눈썹/눈/코/입) ----------------------------------------------------

    def _face_primitives(self, face: np.ndarray, head: _Head) -> List[_Primitive]:
        layout = self._face_layout(face, head)
        if layout is None:
            return []
        brow_thickness, outline_thickness, nose_radius = self._face_line_widths(head)

        def draw(img):
            for brow in (landmarks.LEFT_BROW_KEYS, landmarks.RIGHT_BROW_KEYS):
                line = layout.polygon(brow)
                if line is not None:
                    cv2.polylines(img, [line], False, FACE_COLOR, brow_thickness, cv2.LINE_AA)
            for outline in (landmarks.LEFT_EYE_OUTLINE, landmarks.RIGHT_EYE_OUTLINE, landmarks.MOUTH_OUTLINE):
                poly = layout.polygon(outline)
                if poly is not None:
                    cv2.fillPoly(img, [poly], FACE_COLOR, cv2.LINE_AA)
                    cv2.polylines(img, [poly], True, FACE_COLOR, outline_thickness, cv2.LINE_AA)
            if layout.visible["nose_tip"]:
                cv2.circle(img, tuple(int(v) for v in layout.pts["nose_tip"]), nose_radius, FACE_COLOR, -1, cv2.LINE_AA)

        return [_Primitive(depth=float(head.cam[2]) - FACE_DEPTH_EPSILON, draw_fn=draw)]


class SkeletonRenderer(_ProjectedRenderer):
    """배경 위에 선과 점만으로 관절·손가락·얼굴을 그리는 렌더러 (조건 B).

    CapsuleRenderer와 카메라·머리·얼굴 기하를 공유하므로 같은 프레임에서 관절이 같은 픽셀에 놓인다.
    면(캡슐, 폴리곤), 바닥, 그림자는 그리지 않는다.
    """

    def render(self, frame: FrameData) -> np.ndarray:
        img = self._blank_canvas()
        proj_pose = self._project(self._camera_transform(frame.pose))
        head = self._head(frame.pose)

        def px(idx: int) -> Tuple[int, int]:
            return tuple(int(v) for v in proj_pose[idx])

        # 몸통 윤곽
        torso = [landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER, landmarks.RIGHT_HIP, landmarks.LEFT_HIP]
        torso_color = _clamp_color(SKELETON_COLOR)
        for a, b in zip(torso, torso[1:] + torso[:1]):
            cv2.line(img, px(a), px(b), torso_color, SKELETON_LINE_THICKNESS, cv2.LINE_AA)

        # 목: 어깨 중앙 → 머리 윤곽. 머리 중심까지 그으면 선이 얼굴(입)을 가로지른다.
        shoulder_mid = (proj_pose[landmarks.LEFT_SHOULDER] + proj_pose[landmarks.RIGHT_SHOULDER]) / 2.0
        to_shoulder = shoulder_mid - head.px
        dist = float(np.linalg.norm(to_shoulder))
        if dist > head.radius_px:
            neck_top = head.px + to_shoulder / dist * head.radius_px
            cv2.line(
                img,
                tuple(int(v) for v in shoulder_mid),
                tuple(int(v) for v in neck_top),
                torso_color,
                SKELETON_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        # 팔다리
        for start_idx, end_idx, part_name in landmarks.POSE_LIMB_SEGMENTS:
            cv2.line(img, px(start_idx), px(end_idx), self._color(part_name), SKELETON_LINE_THICKNESS, cv2.LINE_AA)
        for start_idx, end_idx, part_name in landmarks.POSE_LIMB_SEGMENTS:
            for idx in (start_idx, end_idx):
                cv2.circle(img, px(idx), SKELETON_JOINT_RADIUS, self._color(part_name), -1, cv2.LINE_AA)

        # 머리 윤곽 + 얼굴
        cv2.circle(img, tuple(int(v) for v in head.px), head.radius_px, torso_color, SKELETON_HEAD_OUTLINE, cv2.LINE_AA)
        if frame.face is not None:
            self._draw_face(img, frame.face, head)

        # 손
        for hand, side in ((frame.left_hand, "left"), (frame.right_hand, "right")):
            if hand is not None:
                _draw_hand_chains(img, self._project(self._camera_transform(hand)), self._color(side))

        self._advance_orbit()
        return img

    @staticmethod
    def _color(part_name: str) -> Tuple[int, int, int]:
        dim = SKELETON_LEFT_DIM if part_name.startswith("left") else 1.0
        return _clamp_color(SKELETON_COLOR * dim)

    def _draw_face(self, img: np.ndarray, face: np.ndarray, head: _Head) -> None:
        layout = self._face_layout(face, head)
        if layout is None:
            return
        brow_thickness, outline_thickness, nose_radius = self._face_line_widths(head)
        color = _clamp_color(SKELETON_COLOR)
        for brow in (landmarks.LEFT_BROW_KEYS, landmarks.RIGHT_BROW_KEYS):
            line = layout.polygon(brow)
            if line is not None:
                cv2.polylines(img, [line], False, color, brow_thickness, cv2.LINE_AA)
        # 눈·입은 채우지 않고 윤곽선만 (입 벌림은 윤곽의 높이로 보인다)
        for outline in (landmarks.LEFT_EYE_OUTLINE, landmarks.RIGHT_EYE_OUTLINE, landmarks.MOUTH_OUTLINE):
            poly = layout.polygon(outline)
            if poly is not None:
                cv2.polylines(img, [poly], True, color, max(outline_thickness, 2), cv2.LINE_AA)
        if layout.visible["nose_tip"]:
            cv2.circle(img, tuple(int(v) for v in layout.pts["nose_tip"]), nose_radius, color, -1, cv2.LINE_AA)
