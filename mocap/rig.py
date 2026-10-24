"""FrameData(관절 3D 위치) → VRM 휴머노이드 정규화 본 로컬 회전 (FK 리타게팅, Phase 4D).

뼈 길이가 모델과 다르므로 위치는 옮기지 않고 각도만 옮긴다. 증폭 계수는 쓰지 않는다.

좌표계: VRM 1.0 정규화 규약 — 모델 정면 +Z, 모델의 왼쪽 +X, 위 +Y. 정규화 본은 휴지 자세(T-포즈)에서
회전이 모두 단위행렬이므로, 본의 전역 회전 G는 "휴지 자세 → 현재 자세" 회전이고 로컬 회전은
G_parent^T · G 이다. 휴지 자세의 뼈 방향은 모델에서 직접 읽는다 (RestPose, render.mjs dumpRest).

비틀림(twist):
- 위팔: 팔꿈치가 굽는 방향(아래팔 방향)으로 정한다. 팔꿈치는 앞쪽으로만 굽게 되어 반대로 꺾일 수 없다.
  팔이 거의 펴져 굽는 방향이 불분명하면 직전 프레임 값을 쓴다.
- 아래팔·손: 손목(0)·검지 MCP(5)·새끼 MCP(17)가 이루는 손바닥 평면의 법선으로 정한다.
  손바닥 방향은 한국수어의 음운 자질이므로 이 값이 손바닥 뒤집기를 결정한다.

손가락: MCP는 굽힘+벌림 2자유도, PIP·DIP는 굽힘 1자유도. 엄지는 중수골(CMC)을 손바닥 기준으로 방향 정렬하고,
MCP·IP는 관측된 굽힘 평면으로 최소 회전한다 (엄지는 다른 손가락과 굽힘 축이 달라 축을 가정하지 않는다).
해부학적 한계를 넘는 각도는 클램프하고 RigFrame.violations에 남긴다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import landmarks
from .retarget import FrameData, head_rotation_from_face

# MediaPipe 월드(x=화자 왼쪽, y=아래, z=카메라 반대) → 모델(x=왼쪽, y=위, z=정면)
_MP_TO_MODEL = np.array([1.0, -1.0, -1.0])
_C = np.diag(_MP_TO_MODEL)

UP = np.array([0.0, 1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])

# 해부학적 한계 (도)
ELBOW_FLEX_MAX = 160.0
WRIST_FLEX_LIMIT = (-80.0, 80.0)        # +: 손바닥 쪽으로 굽힘
WRIST_DEVIATION_LIMIT = (-35.0, 35.0)
FINGER_MCP_FLEX = (-30.0, 100.0)        # 음수 = 뒤로 젖힘
FINGER_MCP_ABDUCTION = 30.0             # 모델 휴지 자세 기준 좌우 벌림 한계
FINGER_PIP_FLEX = (-10.0, 120.0)
FINGER_DIP_FLEX = (-10.0, 90.0)
THUMB_JOINT_FLEX = (-30.0, 90.0)
VIOLATION_TOLERANCE_DEG = 2.0           # 이보다 작게 넘은 것은 추정 잡음으로 보고 기록하지 않는다

# 손목 각이 이보다 크면 몸·손 모델의 깊이 불일치로 보고 아래팔 깊이를 보정한다 (reconcile_forearm_depth)
RECONCILE_WRIST_DEG = 35.0

# 손가락이 손바닥 평면에 남은 성분(0~1)이 이 범위보다 작으면 벌림각을 믿지 않는다 (주먹)
ABDUCTION_FADE = (0.25, 0.6)

# 팔이 이만큼 펴지면(아래팔이 위팔과 이루는 각의 sin) 위팔 비틀림을 직전 값으로 유지하기 시작
STRAIGHT_ARM_SIN = (0.10, 0.30)

_FINGERS = (  # (VRM 이름, MediaPipe 손 랜드마크 [MCP, PIP, DIP, TIP])
    ("Index", (5, 6, 7, 8)),
    ("Middle", (9, 10, 11, 12)),
    ("Ring", (13, 14, 15, 16)),
    ("Little", (17, 18, 19, 20)),
)
_THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP → Metacarpal, Proximal, Distal


# ---------------------------------------------------------------------------
# 회전 유틸
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _frame(direction: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """direction을 첫 축, ref의 수직 성분을 둘째 축으로 하는 정규 직교 기저 (열벡터)."""
    x = _normalize(direction)
    y = ref - np.dot(ref, x) * x
    if np.linalg.norm(x) < 1e-9 or np.linalg.norm(y) < 1e-9:
        return None
    y = _normalize(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def align(rest_dir, rest_ref, cur_dir, cur_ref) -> Optional[np.ndarray]:
    """휴지 자세의 (방향, 기준 방향)을 현재의 (방향, 기준 방향)으로 보내는 회전 행렬."""
    f0, f1 = _frame(rest_dir, rest_ref), _frame(cur_dir, cur_ref)
    if f0 is None or f1 is None:
        return None
    return f1 @ f0.T


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = _normalize(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * kx + (1 - math.cos(angle_rad)) * (kx @ kx)


def half_rotation(r: np.ndarray) -> np.ndarray:
    """회전 행렬의 절반 회전 (같은 축, 절반 각)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return np.eye(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # 180도: 대칭 성분에서 축을 구한다
        axis = np.sqrt(np.maximum(np.diag(r) + 1, 0) / 2)
    return axis_angle(axis, angle / 2)


def to_quaternion(r: np.ndarray) -> List[float]:
    """회전 행렬 → [x, y, z, w] (three.js 순서). w ≥ 0으로 부호를 고정한다 (결정론)."""
    m = r
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """axis 둘레로 a를 b로 돌리는 부호 있는 각 (라디안)."""
    return math.atan2(float(np.dot(np.cross(a, b), _normalize(axis))), float(np.dot(a, b)))


def _swing(rest_dir: np.ndarray, target_dir: np.ndarray, max_deg: float, label: str, violations: List[str]) -> np.ndarray:
    """rest_dir를 target_dir로 보내는 최소 회전. 회전각이 max_deg를 넘으면 클램프하고 기록한다."""
    axis = np.cross(rest_dir, target_dir)
    angle = math.degrees(math.atan2(np.linalg.norm(axis), float(np.dot(rest_dir, target_dir))))
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3)
    angle = _clamp_deg(angle, 0.0, max_deg, label, violations)
    return axis_angle(axis, math.radians(angle))


def _flex_abduction(v, a, n, s) -> Tuple[float, float, float]:
    """손 좌표계(a: 손 방향, n: 손바닥 쪽, s: 옆)에서 (굽힘°, 벌림°, 손바닥 평면 성분 크기)."""
    v = _normalize(v)
    va, vn, vs = float(np.dot(v, a)), float(np.dot(v, n)), float(np.dot(v, s))
    in_plane = math.hypot(va, vs)
    return math.degrees(math.atan2(vn, in_plane)), math.degrees(math.atan2(vs, va)), in_plane


def reconcile_forearm_depth(fore_dir: np.ndarray, hand_dir: np.ndarray) -> np.ndarray:
    """손목 각이 지나치게 크면 아래팔의 깊이(z) 성분만 손 방향 쪽으로 옮겨 RECONCILE_WRIST_DEG까지 줄인다.

    몸(Holistic pose)과 손(hand) 모델이 깊이를 따로 추정해 서로 어긋나면 손목이 30~40도 더 꺾여 보인다
    (input2.mp4: 화면 평면 손목 각 중앙값 ±20도 vs 3D 30~40도). 카메라가 정면이라 깊이 성분을 바꿔도
    화면상의 아래팔은 거의 같다. 손목 각이 그 이하이면 실제 손목 굽힘으로 보고 건드리지 않는다.
    """
    def angle(f):
        return math.degrees(math.acos(np.clip(np.dot(_normalize(f), _normalize(hand_dir)), -1, 1)))

    if angle(fore_dir) <= RECONCILE_WRIST_DEG:
        return fore_dir
    fx, fy = fore_dir[0], fore_dir[1]
    hx, hy, hz = hand_dir
    denom = fx * hx + fy * hy
    if denom <= 1e-9:
        return fore_dir
    best = np.array([fx, fy, hz * (fx * fx + fy * fy) / denom])  # (fx, fy, z)와 손 방향의 각이 최소인 z
    if angle(best) >= RECONCILE_WRIST_DEG:
        return best
    lo, hi = 0.0, 1.0  # 원래 z에서 best z로 가는 비율 중 손목 각이 정확히 기준이 되는 지점
    for _ in range(30):
        mid = (lo + hi) / 2
        if angle(fore_dir + mid * (best - fore_dir)) > RECONCILE_WRIST_DEG:
            lo = mid
        else:
            hi = mid
    return fore_dir + hi * (best - fore_dir)


def _clamp_deg(value: float, lo: float, hi: float, label: str, violations: List[str]) -> float:
    if value < lo - VIOLATION_TOLERANCE_DEG or value > hi + VIOLATION_TOLERANCE_DEG:
        violations.append(f"{label}={value:.0f}°")
    return min(max(value, lo), hi)


# ---------------------------------------------------------------------------
# 모델 휴지 자세
# ---------------------------------------------------------------------------

@dataclass
class RestPose:
    """모델의 정규화 휴지 자세에서 본의 월드 위치 (VRM 1.0 규약 좌표)."""

    positions: Dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str) -> "RestPose":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls({k: np.asarray(v, dtype=np.float64) for k, v in data["positions"].items()})

    def direction(self, bone: str, child: str) -> np.ndarray:
        return _normalize(self.positions[child] - self.positions[bone])

    def palm_normal(self, side: str) -> np.ndarray:
        """손바닥이 향하는 방향. 손 랜드마크와 같은 공식으로 계산해 부호를 맞춘다."""
        p = self.positions
        return _palm_normal(p[f"{side}Hand"], p[f"{side}IndexProximal"], p[f"{side}LittleProximal"], side)


def _palm_normal(wrist: np.ndarray, index_mcp: np.ndarray, little_mcp: np.ndarray, side: str) -> np.ndarray:
    """손목·검지 MCP·새끼 MCP로 손바닥이 향하는 방향(손가락이 굽어 들어가는 쪽).

    왼손·오른손은 세 점의 회전 방향이 반대다. 예: 왼손 T-포즈(손가락 +X, 엄지 쪽 +Z, 손바닥 아래)에서
    cross(검지-손목, 새끼-손목) = +Y(손등) 이므로 부호를 뒤집어 -Y(손바닥)를 돌려준다.
    """
    n = np.cross(index_mcp - wrist, little_mcp - wrist)
    return _normalize(-n if side == "left" else n)


# ---------------------------------------------------------------------------
# 솔버
# ---------------------------------------------------------------------------

@dataclass
class RigFrame:
    bones: Dict[str, List[float]]                       # 본 이름 → [x, y, z, w]
    violations: List[str] = field(default_factory=list)  # 클램프한 해부학적 한계 위반


class RigSolver:
    """프레임 순서대로 호출한다 (팔이 펴졌을 때 위팔 비틀림을 직전 값으로 유지하기 위한 상태)."""

    def __init__(self, rest: RestPose) -> None:
        self.rest = rest
        self._prev_elbow_ref: Dict[str, np.ndarray] = {}
        self._prev_palm: Dict[str, np.ndarray] = {}

    def solve(self, frame: FrameData) -> RigFrame:
        violations: List[str] = []
        bones: Dict[str, np.ndarray] = {}
        pose = frame.pose * _MP_TO_MODEL

        chest = self._torso(pose, bones)
        self._head(frame, chest, bones)
        for side in ("left", "right"):
            hand = frame.left_hand if side == "left" else frame.right_hand
            self._arm(side, pose, None if hand is None else hand * _MP_TO_MODEL, chest, bones, violations)
        return RigFrame(bones={k: to_quaternion(v) for k, v in bones.items()}, violations=violations)

    # -- 몸통·머리 -------------------------------------------------------------

    def _torso(self, pose: np.ndarray, bones: Dict[str, np.ndarray]) -> np.ndarray:
        """어깨선(모델 왼쪽 +X)과 세계 위쪽으로 몸통 회전. spine과 chest가 절반씩 나눠 갖는다."""
        shoulders = pose[landmarks.LEFT_SHOULDER] - pose[landmarks.RIGHT_SHOULDER]
        g = align(np.array([1.0, 0, 0]), UP, shoulders, UP)
        if g is None:
            g = np.eye(3)
        half = half_rotation(g)
        bones["spine"] = half
        bones["chest"] = half.T @ g
        bones["upperChest"] = np.eye(3)
        return g

    def _head(self, frame: FrameData, chest: np.ndarray, bones: Dict[str, np.ndarray]) -> None:
        """얼굴 랜드마크로 추정한 머리 회전. 목과 머리가 절반씩 나눠 갖는다. 얼굴이 없으면 몸통을 따른다."""
        head = chest
        if frame.face is not None:
            r_mp = head_rotation_from_face(frame.face)
            if r_mp is not None:
                head = _C @ r_mp @ _C
        rel = chest.T @ head
        half = half_rotation(rel)
        bones["neck"] = half
        bones["head"] = half.T @ rel

    # -- 팔·손 -----------------------------------------------------------------

    def _arm(
        self,
        side: str,
        pose: np.ndarray,
        hand: Optional[np.ndarray],
        chest: np.ndarray,
        bones: Dict[str, np.ndarray],
        violations: List[str],
    ) -> None:
        rest = self.rest
        cap = side.capitalize()
        idx = {
            "shoulder": landmarks.LEFT_SHOULDER if side == "left" else landmarks.RIGHT_SHOULDER,
            "elbow": landmarks.LEFT_ELBOW if side == "left" else landmarks.RIGHT_ELBOW,
            "wrist": landmarks.LEFT_WRIST if side == "left" else landmarks.RIGHT_WRIST,
        }
        s, e, w = pose[idx["shoulder"]], pose[idx["elbow"]], pose[idx["wrist"]]
        upper_dir, fore_dir = e - s, w - e
        if hand is not None:
            fore_dir = reconcile_forearm_depth(fore_dir, hand[9] - hand[0])

        rest_upper = rest.direction(f"{side}UpperArm", f"{side}LowerArm")
        rest_fore = rest.direction(f"{side}LowerArm", f"{side}Hand")
        rest_hand = rest.direction(f"{side}Hand", f"{side}MiddleProximal")
        rest_palm = rest.palm_normal(side)

        # 위팔: 팔꿈치가 굽는 방향 = 아래팔의 위팔 수직 성분. 휴지 자세에서 팔꿈치는 앞(+Z)으로 굽는다.
        elbow_ref = fore_dir - np.dot(fore_dir, _normalize(upper_dir)) * _normalize(upper_dir)
        bend = np.linalg.norm(elbow_ref) / max(np.linalg.norm(fore_dir), 1e-9)
        weight = float(np.clip((bend - STRAIGHT_ARM_SIN[0]) / (STRAIGHT_ARM_SIN[1] - STRAIGHT_ARM_SIN[0]), 0, 1))
        fallback = self._prev_elbow_ref.get(side, chest @ FORWARD)
        elbow_ref = weight * _normalize(elbow_ref) + (1 - weight) * fallback
        self._prev_elbow_ref[side] = _normalize(elbow_ref)
        g_upper = align(rest_upper, FORWARD, upper_dir, elbow_ref)

        elbow_flex = math.degrees(math.acos(np.clip(np.dot(_normalize(upper_dir), _normalize(fore_dir)), -1, 1)))
        if elbow_flex > ELBOW_FLEX_MAX + VIOLATION_TOLERANCE_DEG:
            violations.append(f"{side}Elbow={elbow_flex:.0f}°")

        # 아래팔: 손바닥 법선으로 비틀림을 정한다. 손이 없으면 직전 손바닥 방향(없으면 위팔을 따름).
        if hand is not None:
            palm = _palm_normal(hand[0], hand[5], hand[17], side)
            self._prev_palm[side] = palm
        else:
            palm = self._prev_palm.get(side, (g_upper if g_upper is not None else chest) @ rest_palm)
        g_fore = align(rest_fore, rest_palm, fore_dir, palm)

        if g_upper is None or g_fore is None:
            return  # 퇴화한 자세(뼈 길이 0 등): 이 팔은 휴지 자세로 둔다
        bones[f"{side}Shoulder"] = np.eye(3)
        bones[f"{side}UpperArm"] = chest.T @ g_upper
        bones[f"{side}LowerArm"] = g_upper.T @ g_fore

        if hand is None:
            return
        g_hand = self._wrist(side, hand, palm, g_fore, rest_fore, rest_hand, rest_palm, violations)
        bones[f"{side}Hand"] = g_fore.T @ g_hand
        self._fingers(side, cap, hand, palm, g_hand, bones, violations)

    def _wrist(self, side, hand, palm, g_fore, rest_fore, rest_hand, rest_palm, violations) -> np.ndarray:
        """손 방향(손목→중지 MCP)을 아래팔 기준 굽힘·편위 각으로 바꿔 한계를 건 뒤 다시 조립한다."""
        # 아래팔 좌표계에서의 관측 손 방향
        hand_dir_local = g_fore.T @ _normalize(hand[9] - hand[0])
        a0 = rest_fore                                       # 아래팔 축
        n0 = _normalize(rest_palm - np.dot(rest_palm, a0) * a0)  # 손바닥 쪽
        s0 = np.cross(a0, n0)
        flex = math.degrees(math.atan2(np.dot(hand_dir_local, n0), np.dot(hand_dir_local, a0)))
        dev = math.degrees(math.atan2(np.dot(hand_dir_local, s0), np.dot(hand_dir_local, a0)))
        # 모델 휴지 자세의 손 방향이 아래팔과 정확히 일직선이 아닐 수 있으므로 그 각을 기준으로 잰다
        rest_flex = math.degrees(math.atan2(np.dot(rest_hand, n0), np.dot(rest_hand, a0)))
        rest_dev = math.degrees(math.atan2(np.dot(rest_hand, s0), np.dot(rest_hand, a0)))
        flex = _clamp_deg(flex - rest_flex, *WRIST_FLEX_LIMIT, f"{side}WristFlex", violations) + rest_flex
        dev = _clamp_deg(dev - rest_dev, *WRIST_DEVIATION_LIMIT, f"{side}WristDev", violations) + rest_dev
        f, d = math.radians(flex), math.radians(dev)
        target_local = _normalize(a0 * math.cos(f) * math.cos(d) + n0 * math.sin(f) + s0 * math.cos(f) * math.sin(d))
        g = align(rest_hand, rest_palm, g_fore @ target_local, palm)
        return g if g is not None else g_fore

    def _fingers(self, side, cap, hand, palm, g_hand, bones, violations) -> None:
        rest = self.rest
        rest_palm = rest.palm_normal(side)
        rest_hand_dir = rest.direction(f"{side}Hand", f"{side}MiddleProximal")
        # 관측 손 좌표계 (a: 손목→중지 MCP, n: 손바닥 쪽, s: 옆)
        a = _normalize(hand[9] - hand[0])
        n = _normalize(palm - np.dot(palm, a) * a)
        s_axis = np.cross(a, n)
        # 모델 휴지 손 좌표계
        a0 = rest_hand_dir
        n0 = _normalize(rest_palm - np.dot(rest_palm, a0) * a0)
        s0 = np.cross(a0, n0)

        for name, (mcp, pip, dip, tip) in _FINGERS:
            prox, inter, dist = (f"{side}{name}{seg}" for seg in ("Proximal", "Intermediate", "Distal"))
            rest_prox_dir = rest.direction(prox, inter)
            rest_inter_dir = rest.direction(inter, dist)
            rest_dist_dir = self._distal_direction(dist, inter)
            v1, v2, v3 = hand[pip] - hand[mcp], hand[dip] - hand[pip], hand[tip] - hand[dip]

            # MCP: 손 좌표계에서 굽힘(손바닥 쪽 +)과 벌림
            flex, abd, in_plane = _flex_abduction(v1, a, n, s_axis)
            _, rest_abd, _ = _flex_abduction(rest_prox_dir, a0, n0, s0)
            flex = _clamp_deg(flex, *FINGER_MCP_FLEX, f"{prox}Flex", violations)
            # 많이 굽히면(주먹) 손바닥 평면 성분이 작아져 벌림각이 정의되지 않는다 → 휴지 벌림으로 서서히 되돌린다
            trust = float(np.clip((in_plane - ABDUCTION_FADE[0]) / (ABDUCTION_FADE[1] - ABDUCTION_FADE[0]), 0, 1))
            abd = rest_abd + trust * (abd - rest_abd)
            abd = _clamp_deg(abd - rest_abd, -FINGER_MCP_ABDUCTION, FINGER_MCP_ABDUCTION, f"{prox}Abd", violations) + rest_abd
            f, b = math.radians(flex), math.radians(abd)
            target = a0 * math.cos(f) * math.cos(b) + n0 * math.sin(f) + s0 * math.cos(f) * math.sin(b)
            g_prox = align(rest_prox_dir, n0, g_hand @ target, g_hand @ n0)
            if g_prox is None:
                continue
            bones[prox] = g_hand.T @ g_prox

            # PIP·DIP: 손가락 자체의 굽힘 축 둘레 1자유도. 모델 휴지 자세의 굽은 각을 빼고 적용한다.
            k0 = np.cross(rest_prox_dir, n0)  # 휴지 자세에서 손바닥 쪽으로 굽히는 축
            # 관측 굽힘 축: 손가락의 '손바닥 평면 안 방향'과 손바닥 법선의 외적. 손가락이 굽어도 퇴화하지 않는다.
            b = math.radians(abd)
            k_obs = np.cross(a * math.cos(b) + s_axis * math.sin(b), n)
            pip_flex = _clamp_deg(math.degrees(_signed_angle(v1, v2, k_obs)), *FINGER_PIP_FLEX, f"{inter}Flex", violations)
            dip_flex = _clamp_deg(math.degrees(_signed_angle(v2, v3, k_obs)), *FINGER_DIP_FLEX, f"{dist}Flex", violations)
            pip_rest = math.degrees(_signed_angle(rest_prox_dir, rest_inter_dir, k0))
            dip_rest = math.degrees(_signed_angle(rest_inter_dir, rest_dist_dir, k0))
            k1 = np.cross(rest_inter_dir, n0)
            bones[inter] = axis_angle(k0, math.radians(pip_flex - pip_rest))
            bones[dist] = axis_angle(k1, math.radians(dip_flex - dip_rest))

        # 엄지: 손가락과 달리 약 90도 돌아가 있어 손끝이 손바닥을 가로질러 굽는다. 굽힘 축을 가정하지 않고
        # 관측된 굽힘 평면으로 최소 회전시키며, 각의 크기에만 해부학적 한계를 건다.
        meta, prox, dist = (f"{side}Thumb{seg}" for seg in ("Metacarpal", "Proximal", "Distal"))
        cmc, mcp, ip, tip = _THUMB
        v1, v2, v3 = hand[mcp] - hand[cmc], hand[ip] - hand[mcp], hand[tip] - hand[ip]
        rest_meta = rest.direction(meta, prox)
        rest_prox = rest.direction(prox, dist)
        rest_dist = self._distal_direction(dist, prox)
        g_meta = align(rest_meta, n0, v1, n)
        if g_meta is None:
            return
        bones[meta] = g_hand.T @ g_meta
        g_parent = g_meta
        for bone, rest_dir, observed, label in ((prox, rest_prox, v2, f"{prox}Flex"), (dist, rest_dist, v3, f"{dist}Flex")):
            local = _swing(rest_dir, g_parent.T @ _normalize(observed), THUMB_JOINT_FLEX[1], label, violations)
            bones[bone] = local
            g_parent = g_parent @ local

    def _distal_direction(self, distal: str, parent: str) -> np.ndarray:
        """끝마디는 자식 본이 없으므로 휴지 자세 방향을 부모 마디 방향과 같다고 본다."""
        return self.rest.direction(parent, distal)


# ---------------------------------------------------------------------------
# 순운동학 (검증용: 솔버 결과를 모델 뼈대에 적용했을 때의 관절 위치)
# ---------------------------------------------------------------------------

def _humanoid_parents() -> Dict[str, str]:
    parents = {"spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck"}
    for side in ("left", "right"):
        parents.update({
            f"{side}Shoulder": "upperChest", f"{side}UpperArm": f"{side}Shoulder",
            f"{side}LowerArm": f"{side}UpperArm", f"{side}Hand": f"{side}LowerArm",
            f"{side}ThumbMetacarpal": f"{side}Hand", f"{side}ThumbProximal": f"{side}ThumbMetacarpal",
            f"{side}ThumbDistal": f"{side}ThumbProximal",
        })
        for name, _ in _FINGERS:
            parents[f"{side}{name}Proximal"] = f"{side}Hand"
            parents[f"{side}{name}Intermediate"] = f"{side}{name}Proximal"
            parents[f"{side}{name}Distal"] = f"{side}{name}Intermediate"
    return parents


HUMANOID_PARENT = _humanoid_parents()


def quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def forward_kinematics(rest: RestPose, local: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """로컬 회전(행렬 또는 [x,y,z,w])을 휴지 자세에 적용한 (본 월드 위치, 본 전역 회전). hips는 고정."""
    positions: Dict[str, np.ndarray] = {"hips": rest.positions["hips"]}
    rotations: Dict[str, np.ndarray] = {"hips": np.eye(3)}

    def resolve(bone: str) -> None:
        if bone in positions:
            return
        parent = HUMANOID_PARENT[bone]
        resolve(parent)
        r = local.get(bone, np.eye(3))
        r = quaternion_to_matrix(r) if np.shape(r) == (4,) else np.asarray(r)
        positions[bone] = positions[parent] + rotations[parent] @ (rest.positions[bone] - rest.positions[parent])
        rotations[bone] = rotations[parent] @ r

    for bone in HUMANOID_PARENT:
        if bone in rest.positions:
            resolve(bone)
    return positions, rotations
