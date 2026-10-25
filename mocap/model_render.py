"""조건 B·C의 3D 모델 렌더 경로 (Phase 4E): 추적 결과 → 리그 → three.js 오프라인 렌더 → 프레임.

카메라는 정면(−Z 방향)을 보는 고정 원근 카메라 하나다. 크기·위치 정규화는 Phase 3와 같은 기준을 쓴다.
- 거리: 정면 프레임에서 모델 어깨너비(화면상)가 조건 A의 어깨너비와 같아지도록 정한다.
- 위치: 프레임마다 조건 A 화면의 어깨 중앙(FrameData.screen_anchor)으로 그림 전체를 평행 이동한다
  (three.js setViewOffset, 원근 불변). 이동량은 여기서 three.js와 같은 투영식으로 계산한다.

렌더는 render/render.mjs가 헤드리스 Chromium에서 프레임을 하나씩 수동 전진시켜 PNG로 쓴다.
같은 입력이면 같은 픽셀이 나와야 한다 (결정론).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import framing, landmarks, rig
from .retarget import FrameData

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_DIR = os.path.join(PROJECT_ROOT, "render")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "avatar", "VRoid_V110_Female_v1.1.3.vrm")

# 조건 B·C 공통 고정값 (docs/stimulus-spec.md에 기록)
CAMERA_FOV_DEG = 30.0
BACKGROUND_RGB = (20, 22, 24)  # 기존 렌더러 BACKGROUND_COLOR(BGR 24,22,20)와 같은 색
LIGHTS = (
    {"type": "ambient", "intensity": 0.6},
    {"type": "directional", "intensity": 1.2, "position": [0.5, 1.5, 2.0]},
)
MODEL_STYLE = {"browColor": [45, 35, 30], "hideBangs": {"margin": 0.02, "browLow": 0.012, "browHigh": 0.065, "frontOffset": 0.03}}
FRAME_DT = 1.0 / 30.0
NEAR_FAR_RATIO = (0.5, 2.0)  # 카메라 near/far = 피사체 거리 × 이 비율 (깊이 정밀도 → 결정론)  # 스프링본(머리카락) 갱신 간격. 출력 fps와 맞춘다.


def rest_pose_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".rest.json"


def _run_node(job: dict, workdir: str) -> None:
    job_path = os.path.join(workdir, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f)
    if not os.path.exists(os.path.join(RENDER_DIR, "dist", "scene.js")):
        subprocess.run(["npm", "run", "build"], cwd=RENDER_DIR, check=True, capture_output=True)
    result = subprocess.run(["node", os.path.join(RENDER_DIR, "render.mjs"), job_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"3D 렌더 실패:\n{result.stderr.strip()[-2000:]}")


def ensure_rest_pose(model_path: str) -> rig.RestPose:
    """모델의 휴지 자세 본 위치를 읽는다. 없으면 렌더러로 덤프해 모델 옆에 저장한다."""
    path = rest_pose_path(model_path)
    if not os.path.exists(path):
        with tempfile.TemporaryDirectory() as tmp:
            _run_node(_base_job(model_path, 64, 64, {"fov": CAMERA_FOV_DEG, "position": [0, 1, 3], "target": [0, 1, 0]},
                                frames=[], out_dir=tmp) | {"dumpRest": path}, tmp)
    return rig.RestPose.load(path)


# ---------------------------------------------------------------------------
# 카메라 (three.js PerspectiveCamera와 같은 투영)
# ---------------------------------------------------------------------------

@dataclass
class ModelCamera:
    width: int
    height: int
    fov_deg: float
    position: np.ndarray  # 모델 좌표 (정면 +Z, 위 +Y). 카메라는 −Z 방향을 본다.

    @property
    def focal(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    def project(self, points: np.ndarray, offset: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
        """모델 좌표 (N,3) → 화면 픽셀 (N,2). offset은 setViewOffset으로 준 화면 이동량."""
        rel = np.asarray(points, dtype=np.float64) - self.position
        depth = -rel[:, 2]
        x = self.width / 2.0 + self.focal * rel[:, 0] / depth + offset[0]
        y = self.height / 2.0 - self.focal * rel[:, 1] / depth + offset[1]
        return np.stack([x, y], axis=1)

    def to_job(self) -> dict:
        """three.js 카메라 설정. near/far는 피사체 거리(카메라 z, 모델은 z≈0)의 0.5~2배로 좁힌다.

        범위가 넓으면 깊이 정밀도가 떨어져 머리카락·얼굴 경계의 거의 같은 깊이의 두 면이 실행마다 다르게
        그려졌다 (60프레임×8회 중 최대 12프레임에서 1픽셀 차이 → x264가 이후 프레임을 다르게 부호화).
        이 범위로 좁힌 뒤 8회 모두 픽셀 단위로 같았다.
        """
        p = [float(v) for v in self.position]
        distance = abs(p[2])
        return {"fov": self.fov_deg, "position": p, "target": [p[0], p[1], 0.0],
                "near": NEAR_FAR_RATIO[0] * distance, "far": NEAR_FAR_RATIO[1] * distance}


def model_shoulders(rest: rig.RestPose, rigs: Sequence[rig.RigFrame]) -> np.ndarray:
    """프레임별 모델 어깨 관절(UpperArm 본) 위치 (N, 2, 3): [왼쪽, 오른쪽]."""
    out = np.empty((len(rigs), 2, 3))
    for i, r in enumerate(rigs):
        pos, _ = rig.forward_kinematics(rest, r.bones)
        out[i, 0], out[i, 1] = pos["leftUpperArm"], pos["rightUpperArm"]
    return out


def plan_camera(
    shoulders: np.ndarray,
    target: framing.FramingTarget,
    a_widths: Optional[np.ndarray] = None,
    frontal: Optional[np.ndarray] = None,
) -> ModelCamera:
    """정면 프레임에서 모델의 화면 어깨너비가 조건 A와 같아지도록 카메라 거리를 정한다.

    a_widths: 프레임별 조건 A 화면의 어깨너비(px, 미검출 NaN). frontal: 정면 프레임 마스크.
    없으면 모델 어깨너비 중앙값을 목표 폭(target.shoulder_px)에 맞춘다.
    카메라 x·y는 어깨 중앙 중앙값 정면에 둔다. 프레임별 위치는 view offset이 맞춘다.
    """
    w, h = target.output_size
    focal = (h / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG) / 2.0)
    widths = np.linalg.norm((shoulders[:, 0] - shoulders[:, 1])[:, :2], axis=1)  # 화면 평면 어깨너비 (m)
    mids = shoulders.mean(axis=1)
    if a_widths is not None and frontal is not None and np.any(frontal & ~np.isnan(a_widths)):
        sel = frontal & ~np.isnan(a_widths)
        depth = float(np.median(focal * widths[sel] / a_widths[sel]))  # 모델 폭 f·W/d = A 폭
    else:
        depth = focal * float(np.median(widths)) / target.shoulder_px
    mid = np.median(mids, axis=0)
    return ModelCamera(width=w, height=h, fov_deg=CAMERA_FOV_DEG, position=np.array([mid[0], mid[1], mid[2] + depth]))


def view_offsets(camera: ModelCamera, shoulders: np.ndarray, anchors: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """프레임별 화면 이동량: 모델 어깨 중앙이 조건 A의 어깨 중앙(anchor)에 오도록."""
    mids = camera.project(shoulders.mean(axis=1))
    out = np.zeros((len(mids), 2))
    for i, anchor in enumerate(anchors):
        if anchor is not None and not np.any(np.isnan(anchor)):
            out[i] = np.asarray(anchor) - mids[i]
    return out


# ---------------------------------------------------------------------------
# 렌더 작업
# ---------------------------------------------------------------------------

def _base_job(model_path: str, width: int, height: int, camera: dict, frames: list, out_dir: str) -> dict:
    return {
        "model": model_path, "width": width, "height": height,
        "background": list(BACKGROUND_RGB), "lights": [dict(l) for l in LIGHTS], "style": MODEL_STYLE,
        "camera": camera, "frames": frames, "outDir": out_dir,
    }


@dataclass
class ModelRenderPlan:
    camera: ModelCamera
    frames: List[dict]          # render.mjs 프레임 (bones, blendshapes, dt, viewOffset)
    rigs: List[rig.RigFrame]
    shoulders: np.ndarray       # (N, 2, 3) 모델 좌표
    offsets: np.ndarray         # (N, 2)


def plan_model_render(
    frames: Sequence[FrameData],
    rest: rig.RestPose,
    target: framing.FramingTarget,
    a_widths: Optional[np.ndarray] = None,
    frontal: Optional[np.ndarray] = None,
) -> ModelRenderPlan:
    """추적 결과(FrameData, screen_anchor 포함)로 리그·카메라·프레임별 화면 이동을 계산한다."""
    solver = rig.RigSolver(rest)
    rigs = [solver.solve(f) for f in frames]
    shoulders = model_shoulders(rest, rigs)
    camera = plan_camera(shoulders, target, a_widths, frontal)
    offsets = view_offsets(camera, shoulders, [f.screen_anchor for f in frames])
    job_frames = [
        {
            "bones": r.bones,
            "blendshapes": f.face_blendshapes or {},
            "dt": 0.0 if i == 0 else FRAME_DT,
            "viewOffset": [float(o[0]), float(o[1])],
        }
        for i, (f, r, o) in enumerate(zip(frames, rigs, offsets))
    ]
    return ModelRenderPlan(camera=camera, frames=job_frames, rigs=rigs, shoulders=shoulders, offsets=offsets)


def render_plan(plan: ModelRenderPlan, model_path: str, view: str = "model") -> Iterator[np.ndarray]:
    """계획을 헤드리스 렌더러로 돌려 BGR 프레임을 순서대로 낸다. 임시 PNG는 끝나면 지운다."""
    if view != "model":
        raise ValueError(f"지원하지 않는 3D 뷰: {view}")
    workdir = tempfile.mkdtemp(prefix="model_render_")
    try:
        out_dir = os.path.join(workdir, "frames")
        job = _base_job(model_path, plan.camera.width, plan.camera.height, plan.camera.to_job(), plan.frames, out_dir)
        keep = os.environ.get("MOCAP_KEEP_RENDER_JOB")  # 디버그: 렌더 작업 JSON을 이 경로에 남긴다
        if keep:
            with open(keep, "w", encoding="utf-8") as f:
                json.dump(dict(job, outDir=""), f)
        _run_node(job, workdir)
        for i in range(len(plan.frames)):
            img = cv2.imread(os.path.join(out_dir, f"{i:06d}.png"), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"렌더 프레임이 없습니다: {i}")
            yield img
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
