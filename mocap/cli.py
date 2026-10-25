"""명령줄 인터페이스: 전체 파이프라인(추적 → 리타게팅 → 스무딩 → 렌더링)을 실행한다."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import framing, landmarks, retarget, video_io
from .renderer import (
    DEFAULT_FACE_SCALE,
    FOCAL_LENGTH_FACTOR,
    CameraFraming,
    CapsuleRenderer,
    SkeletonRenderer,
)
from .smoothing import EMASmoother
from .tracker import (
    DEFAULT_FACE_MODEL_PATH,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    HolisticTracker,
    create_face_tracker,
)

DEFAULT_SMOOTH_ALPHA = 0.3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mocap",
        description=(
            "영상 속 사람의 동작을 추적해서 3D 캐릭터가 똑같이 따라 움직이는 영상을 만듭니다."
        ),
    )
    parser.add_argument("input", help="입력 영상 파일 경로 (mp4)")
    parser.add_argument("--out", default=None, help="출력 영상 파일 경로 (기본값: <입력파일명>_mocap.mp4)")
    parser.add_argument(
        "--upper-body",
        action="store_true",
        help="가슴 위만 나오는 영상 모드. 하체는 어깨 방향에 맞춘 고정 자세로 대체합니다.",
    )
    parser.add_argument(
        "--view",
        choices=["side", "avatar", "skeleton", "original", "model"],
        default="side",
        help=(
            "side: 정규화한 원본(랜드마크 표시)과 캐릭터를 나란히, avatar: 캐릭터만(조건 C), "
            "skeleton: 선과 점으로 된 골격만(조건 B), original: 원본을 같은 크기·위치로 정규화(조건 A), "
            "model: 3D 모델(VRM) 렌더(조건 C, 새 경로) "
            "(기본값: side)"
        ),
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="카메라 좌우 회전 각도(도), 기본값 0")
    parser.add_argument(
        "--orbit", type=float, default=0.0, help="프레임마다 추가로 회전할 각도(도/프레임), 기본값 0"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=DEFAULT_SMOOTH_ALPHA,
        help="떨림 보정 강도 0.0~0.9 (클수록 부드럽지만 반응이 느려짐), 기본값 0.3",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe 모델 복잡도 0/1/2 (높을수록 정확하지만 느림), 기본값 1",
    )
    parser.add_argument(
        "--max-height", type=int, default=720, help="추적용 프레임 최대 세로 픽셀 (출력 크기와 무관, 기본값 720)"
    )
    parser.add_argument(
        "--output-size",
        default="{}x{}".format(*framing.DEFAULT_OUTPUT_SIZE),
        help="출력 해상도 WxH (기본값 720x1280). 모든 조건이 이 크기로 나온다",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="출력 고정 프레임레이트 (기본값 30)")
    parser.add_argument(
        "--shoulder-ratio",
        type=float,
        default=framing.DEFAULT_SHOULDER_RATIO,
        help=f"어깨너비 중앙값이 화면 폭에서 차지할 비율 (기본값 {framing.DEFAULT_SHOULDER_RATIO})",
    )
    parser.add_argument(
        "--anchor-y",
        type=float,
        default=framing.DEFAULT_ANCHOR[1],
        help=f"어깨 중앙의 세로 위치 비율 (0=위, 1=아래, 기본값 {framing.DEFAULT_ANCHOR[1]})",
    )
    parser.add_argument(
        "--face-model",
        default=DEFAULT_FACE_MODEL_PATH,
        help="MediaPipe Face Landmarker 모델(.task) 경로. 없으면 Holistic 얼굴 좌표로 대신 (블렌드셰이프 없음)",
    )
    parser.add_argument(
        "--no-blendshapes",
        action="store_true",
        help="Face Landmarker를 쓰지 않고 Holistic 얼굴 좌표만 쓴다 (폴백 경로 강제)",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "avatar", "VRoid_V110_Female_v1.1.3.vrm"),
        help="--view model에서 쓸 VRM 모델 경로 (assets/avatar/README.md 참고)",
    )
    parser.add_argument(
        "--show-legs",
        action="store_true",
        help="상체 모드에서도 합성 다리를 그린다 (기본값: 상체 모드에서는 다리를 숨김)",
    )
    parser.add_argument(
        "--face-scale",
        type=float,
        default=DEFAULT_FACE_SCALE,
        help=f"머리·얼굴 확대 배율 (표정 판독용), 기본값 {DEFAULT_FACE_SCALE}",
    )
    parser.add_argument(
        "--export-json", default=None, help="프레임별 3D 관절 좌표를 저장할 JSON 파일 경로"
    )
    return parser


def _default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_mocap.mp4"


def _frame_data_to_json_record(frame_index: int, frame: retarget.FrameData) -> dict:
    return {
        "frame": frame_index,
        "pose": frame.pose.tolist(),
        "left_hand": frame.left_hand.tolist() if frame.left_hand is not None else None,
        "right_hand": frame.right_hand.tolist() if frame.right_hand is not None else None,
        "face": frame.face.tolist() if frame.face is not None else None,
        "face_blendshapes": frame.face_blendshapes,
    }


def _print_progress(stage: str, current: int, total: int) -> None:
    if total > 0:
        pct = min(current / total, 1.0) * 100.0
        sys.stdout.write(f"\r{stage} 중: {current}/{total} 프레임 ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r{stage} 중: {current} 프레임")
    sys.stdout.flush()


class _Retargeter:
    """프레임별 RawLandmarks → FrameData 변환 (스케일 유지, 손 배정, 얼굴 유지, 스무딩)."""

    def __init__(self, upper_body: bool, smooth: float) -> None:
        self.upper_body = upper_body
        self.pose_smoother = EMASmoother(alpha=smooth)
        self.left_hand_smoother = EMASmoother(alpha=smooth)
        self.right_hand_smoother = EMASmoother(alpha=smooth)
        self.face_smoother = EMASmoother(alpha=smooth)
        self.blendshape_smoother = EMASmoother(alpha=smooth)
        self.prev_frame_data: Optional[retarget.FrameData] = None
        self.last_scale: Optional[float] = None
        # 마지막으로 검출된 얼굴(머리 중심 기준 오프셋)과 그때의 머리 회전. 손이 얼굴을 가리는
        # 구간에서는 이 얼굴을 머리 회전 변화량만큼 돌려서 계속 그린다.
        self.face_ref: Optional[tuple] = None

    def step(self, raw) -> retarget.FrameData:
        scale = retarget.compute_scale(raw.pose_world, raw.pose_pixel)
        if scale is not None:
            self.last_scale = scale
        elif self.last_scale is not None:
            scale = self.last_scale

        assigned_left_px, assigned_right_px = retarget.assign_hands(
            raw.pose_pixel, raw.left_hand_pixel, raw.right_hand_pixel
        )

        pose_world = raw.pose_world
        left_hand_world = None
        right_hand_world = None
        face_rel = None

        if pose_world is not None:
            if self.upper_body:
                pose_world = retarget.synthesize_lower_body(pose_world)
            if assigned_left_px is not None and scale is not None:
                left_hand_world = retarget.hand_to_world(
                    assigned_left_px, pose_world[landmarks.LEFT_WRIST], scale
                )
            if assigned_right_px is not None and scale is not None:
                right_hand_world = retarget.hand_to_world(
                    assigned_right_px, pose_world[landmarks.RIGHT_WRIST], scale
                )
            head_rot = retarget.head_rotation_from_pose(pose_world)
            face_rel = retarget.face_offsets(raw.face_pixel)
            if face_rel is not None:
                self.face_ref = (face_rel, head_rot)
            elif self.face_ref is not None and self.face_ref[1] is not None and head_rot is not None:
                face_rel = retarget.carry_face_offsets(self.face_ref[0], self.face_ref[1], head_rot)

        smoothed_pose = self.pose_smoother.update(pose_world)
        smoothed_left = self.left_hand_smoother.update(left_hand_world)
        smoothed_right = self.right_hand_smoother.update(right_hand_world)
        smoothed_face_rel = self.face_smoother.update(face_rel)
        smoothed_face = None
        if smoothed_pose is not None and smoothed_face_rel is not None:
            smoothed_face = retarget.head_center(smoothed_pose) + smoothed_face_rel

        frame_data = retarget.hold_missing(
            self.prev_frame_data, smoothed_pose, smoothed_left, smoothed_right, smoothed_face
        )
        # 블렌드셰이프도 다른 부위와 같은 EMA로 떨림만 줄인다 (증폭 없음). 미검출 프레임은 직전 값 유지.
        coeffs = None
        if raw.face_blendshapes is not None:
            coeffs = np.array([raw.face_blendshapes.get(n, 0.0) for n in landmarks.BLENDSHAPE_NAMES])
        smoothed_coeffs = self.blendshape_smoother.update(coeffs)
        if smoothed_coeffs is not None:
            frame_data = replace(
                frame_data,
                face_blendshapes={n: float(v) for n, v in zip(landmarks.BLENDSHAPE_NAMES, smoothed_coeffs)},
            )
        self.prev_frame_data = frame_data
        return frame_data


@dataclass
class TrackResult:
    """1패스 추적 결과. 2패스(출력)는 이 값만으로 A·B·C를 만든다."""

    probe: video_io.ProbeInfo
    fps: float
    frames: List[retarget.FrameData]
    raws: list                      # tracker.RawLandmarks, 프레임별
    pixel_size: Tuple[int, int]     # 추적에 쓴 프레임 크기 (width, height)


def track_video(
    path: str,
    fps: float,
    max_height: int,
    complexity: int,
    upper_body: bool,
    smooth: float,
    face_model: Optional[str] = DEFAULT_FACE_MODEL_PATH,
) -> TrackResult:
    """1패스: 고정 fps로 디코딩해 전체 프레임을 추적하고 FrameData 시퀀스를 만든다.

    몸·손은 Holistic(추적용 축소 프레임), 얼굴은 Face Landmarker(원본 해상도)로 추적한다.
    face_model이 None이거나 쓸 수 없으면 Holistic 얼굴 좌표만 쓴다 (블렌드셰이프 없음).
    """
    probe = video_io.probe_video(path)
    face_tracker = create_face_tracker(face_model)
    retargeter = _Retargeter(upper_body=upper_body, smooth=smooth)
    frames: List[retarget.FrameData] = []
    raws = []
    pixel_size = (probe.width, probe.height)
    expected = int(round(probe.duration * fps)) if probe.duration > 0 else 0
    try:
        with HolisticTracker(model_complexity=complexity) as tracker:
            for frame_bgr in video_io.read_frames_cfr(path, fps, probe):
                small = video_io.resize_to_max_height(frame_bgr, max_height)
                pixel_size = (small.shape[1], small.shape[0])
                raw, _ = tracker.process(small)
                if face_tracker is not None:
                    timestamp_ms = int(round(len(frames) * 1000.0 / fps))
                    face_points, blendshapes = face_tracker.process(frame_bgr, timestamp_ms, pixel_size)
                    if face_points is not None:
                        raw.face_pixel = face_points
                        raw.face_blendshapes = blendshapes
                raws.append(raw)
                frames.append(retargeter.step(raw))
                _print_progress("추적", len(frames), expected)
    finally:
        if face_tracker is not None:
            face_tracker.close()
    sys.stdout.write("\n")
    return TrackResult(probe=probe, fps=fps, frames=frames, raws=raws, pixel_size=pixel_size)


def compute_framing(
    track: TrackResult, target: framing.FramingTarget, focal_length: float, yaw_deg: float, smooth: float
) -> Tuple[framing.OriginalTransform, Optional[CameraFraming]]:
    """영상 전체의 어깨 통계로 조건 A 변환과 조건 B·C 카메라를 정하고, 프레임별 화면 위치를 채운다."""
    source_size = (track.probe.width, track.probe.height)
    shoulders = framing.shoulder_track([r.pose_pixel for r in track.raws], track.pixel_size, source_size)
    if shoulders is None:
        # 사람이 한 번도 검출되지 않음: 원본은 화면에 맞춰 넣고, 렌더러는 기본 카메라를 쓴다
        print("경고: 영상 전체에서 사람이 검출되지 않아 프레이밍 정규화를 건너뜁니다.", file=sys.stderr)
        fit = min(target.output_size[0] / source_size[0], target.output_size[1] / source_size[1])
        offset = (np.array(target.output_size) - np.array(source_size) * fit) / 2.0
        return framing.OriginalTransform(fit, offset, source_size, target.output_size), None
    transform = framing.original_transform(shoulders, source_size, target)
    poses = [f.pose if r.pose_world is not None else None for f, r in zip(track.frames, track.raws)]
    camera = framing.camera_for_target(poses, focal_length, target, yaw_deg, shoulders, transform)
    # 조건 A에서 화자가 움직이면 B·C도 같은 화면 위치로 옮긴다
    anchors = framing.screen_anchors(shoulders, transform, smooth)
    track.frames = [replace(f, screen_anchor=a) for f, a in zip(track.frames, anchors)]
    return transform, camera


def _overlay_landmarks(canvas: np.ndarray, raw, track: TrackResult, transform: framing.OriginalTransform) -> None:
    """디버그용(side 뷰): 추적 랜드마크를 조건 A 화면 좌표로 옮겨 그린다."""
    to_source = np.array([track.probe.width, track.probe.height], dtype=np.float64) / np.array(track.pixel_size)

    def to_out(pts: np.ndarray) -> np.ndarray:
        return pts[:, :2] * to_source * transform.scale + transform.offset

    if raw.pose_pixel is not None:
        pts = to_out(raw.pose_pixel)
        for a, b in POSE_CONNECTIONS:
            cv2.line(canvas, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (80, 220, 80), 2, cv2.LINE_AA)
    for hand in (raw.left_hand_pixel, raw.right_hand_pixel):
        if hand is not None:
            pts = to_out(hand)
            for a, b in HAND_CONNECTIONS:
                cv2.line(canvas, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (60, 160, 255), 1, cv2.LINE_AA)
    if raw.face_pixel is not None:
        for pt in to_out(raw.face_pixel)[::4]:
            cv2.circle(canvas, tuple(pt.astype(int)), 1, (200, 200, 200), -1)


def _plan_model_frames(track: TrackResult, target: framing.FramingTarget, transform, model_path: str):
    """3D 모델 경로: 리그·카메라를 계산하고 렌더 프레임 이터레이터를 돌려준다 (Phase 4E)."""
    from . import model_render

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"아바타 모델이 없습니다: {model_path} (assets/avatar/README.md 참고)")
    rest = model_render.ensure_rest_pose(model_path)
    source_size = (track.probe.width, track.probe.height)
    shoulders = framing.shoulder_track([r.pose_pixel for r in track.raws], track.pixel_size, source_size)
    a_widths = frontal = None
    if shoulders is not None:
        a_widths = shoulders.widths * transform.scale  # 조건 A 화면에서의 어깨너비 (px)
        frontal = shoulders.frontal_mask()
    plan = model_render.plan_model_render(track.frames, rest, target, a_widths, frontal)
    over = sum(1 for r in plan.rigs if r.violations)
    print(f"리그: 해부학적 한계를 넘어 클램프한 프레임 {over}/{len(plan.rigs)} ({over / max(len(plan.rigs), 1):.1%})")
    print("3D 렌더 중 (헤드리스 Chromium)...")
    return model_render.render_plan(plan, model_path)


def run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input}", file=sys.stderr)
        return 1

    if not 0.0 <= args.smooth < 1.0:
        print("오류: --smooth 값은 0.0 이상 1.0 미만이어야 합니다.", file=sys.stderr)
        return 1

    try:
        output_size = framing.parse_size(args.output_size)
        video_io.require_ffmpeg()
    except (ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    out_path = args.out or _default_output_path(args.input)
    target = framing.FramingTarget(
        output_size=output_size,
        shoulder_ratio=args.shoulder_ratio,
        anchor=(0.5, args.anchor_y),
    )

    # -- 1패스: 추적 ------------------------------------------------------------
    face_model = None if args.no_blendshapes else args.face_model
    track = track_video(
        args.input, args.fps, args.max_height, args.complexity, args.upper_body, args.smooth, face_model
    )
    n_frames = len(track.frames)
    print(
        f"입력: {args.input} ({track.probe.width}x{track.probe.height}, {track.probe.fps:.2f}fps "
        f"→ {args.fps:g}fps 고정, {n_frames}프레임)"
    )

    # -- 프레이밍 ---------------------------------------------------------------
    width, height = output_size
    focal_length = height * FOCAL_LENGTH_FACTOR
    transform, camera = compute_framing(track, target, focal_length, args.yaw, args.smooth)
    if args.view in ("original", "side"):
        if transform.upscaled:
            print(f"경고: 원본을 {transform.scale:.2f}배 확대합니다 (화자가 작게 찍힘, 화질 저하).", file=sys.stderr)
        if transform.uncovered_ratio > 0.001:
            print(
                f"경고: 원본이 출력 화면의 {transform.uncovered_ratio:.1%}를 덮지 못해 여백이 생깁니다 "
                "(화자가 크게 찍혔거나 원본 비율이 다름).",
                file=sys.stderr,
            )

    hide_legs = args.upper_body and not args.show_legs
    renderer = None
    model_frames = None
    if args.view == "model":
        model_frames = _plan_model_frames(track, target, transform, args.model)
    elif args.view != "original":
        renderer_cls = SkeletonRenderer if args.view == "skeleton" else CapsuleRenderer
        renderer = renderer_cls(
            width=width,
            height=height,
            upper_body=args.upper_body,
            yaw_deg=args.yaw,
            orbit_deg_per_frame=args.orbit,
            face_scale=args.face_scale,
            camera=camera,
            hide_legs=hide_legs,
        )

    # -- 2패스: 출력 ------------------------------------------------------------
    view_label = {
        "side": "나란히 보기", "avatar": "캐릭터만 보기(조건 C)",
        "skeleton": "골격만 보기(조건 B)", "original": "원본 정규화(조건 A)",
        "model": "3D 모델(조건 C)",
    }[args.view]
    print(f"출력: {out_path} ({width}x{height}, {view_label})")
    output_width = width * 2 if args.view == "side" else width
    needs_source = args.view in ("original", "side")
    sources = video_io.read_frames_cfr(args.input, args.fps, track.probe) if needs_source else None

    written = 0
    with video_io.VideoWriter(out_path, fps=args.fps, size=(output_width, height)) as writer:
        for i, frame_data in enumerate(track.frames):
            if needs_source:
                source = next(sources, None)
                if source is None:
                    print("오류: 2패스 디코딩 프레임 수가 1패스와 다릅니다.", file=sys.stderr)
                    return 1
                original = framing.apply_original_transform(source, transform)
            if args.view == "original":
                out = original
            elif model_frames is not None:
                out = next(model_frames)
            else:
                rendered = renderer.render(frame_data)
                if args.view == "side":
                    _overlay_landmarks(original, track.raws[i], track, transform)
                    out = video_io.side_by_side(original, rendered)
                else:
                    out = rendered
            writer.write(out)
            written += 1
            _print_progress("출력", written, n_frames)
    if sources is not None:
        sources.close()
    sys.stdout.write("\n")

    # -- 출력 검증: 프레임 수와 웹 재생 프리셋 ------------------------------------------
    enc = video_io.inspect_encoding(out_path)
    if not (written == enc.frames == n_frames):
        print(
            f"오류: 프레임 수 불일치 (입력 {n_frames}, 인코더 입력 {written}, 출력 파일 {enc.frames})",
            file=sys.stderr,
        )
        return 1
    problems = enc.web_problems()
    if problems and args.view != "side":
        print(f"오류: 모바일 웹 재생 조건 위반: {', '.join(problems)}", file=sys.stderr)
        return 1
    print(
        f"처리 완료. 프레임 수 검증: 입력 {n_frames} = 출력 {enc.frames} / "
        f"{enc.codec} {enc.profile} L{enc.level / 10:.1f} {enc.pix_fmt}, faststart={enc.faststart}"
        + (" (side 뷰는 level 3.1 한도 초과라 모바일 프리셋 검사 제외)" if args.view == "side" else "")
    )

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(
                [_frame_data_to_json_record(i, fd) for i, fd in enumerate(track.frames)], f, ensure_ascii=False
            )
        print(f"포즈 데이터 저장: {args.export_json}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
