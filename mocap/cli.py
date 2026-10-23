"""명령줄 인터페이스: 전체 파이프라인(추적 → 리타게팅 → 스무딩 → 렌더링)을 실행한다."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

from . import landmarks, retarget, video_io
from .renderer import CapsuleRenderer
from .smoothing import EMASmoother
from .tracker import HolisticTracker, draw_landmarks_overlay

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
        choices=["side", "avatar"],
        default="side",
        help="side: 원본과 캐릭터를 나란히 출력, avatar: 캐릭터만 출력 (기본값: side)",
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
        "--max-height", type=int, default=720, help="처리용 프레임 최대 세로 픽셀 (기본값 720)"
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
    }


def _print_progress(current: int, total: int) -> None:
    if total > 0:
        pct = current / total * 100.0
        sys.stdout.write(f"\r처리 중: {current}/{total} 프레임 ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r처리 중: {current} 프레임")
    sys.stdout.flush()


def run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input}", file=sys.stderr)
        return 1

    if not 0.0 <= args.smooth < 1.0:
        print("오류: --smooth 값은 0.0 이상 1.0 미만이어야 합니다.", file=sys.stderr)
        return 1

    out_path = args.out or _default_output_path(args.input)
    info = video_io.get_video_info(args.input)

    renderer = CapsuleRenderer(
        width=info.width,
        height=info.height,
        upper_body=args.upper_body,
        yaw_deg=args.yaw,
        orbit_deg_per_frame=args.orbit,
    )

    pose_smoother = EMASmoother(alpha=args.smooth)
    left_hand_smoother = EMASmoother(alpha=args.smooth)
    right_hand_smoother = EMASmoother(alpha=args.smooth)
    face_smoother = EMASmoother(alpha=args.smooth)

    output_width = info.width * 2 if args.view == "side" else info.width
    json_records: List[dict] = []
    prev_frame_data: Optional[retarget.FrameData] = None
    last_scale: Optional[float] = None

    print(f"입력: {args.input} ({info.width}x{info.height}, {info.fps:.1f}fps, {info.total_frames}프레임)")
    print(f"출력: {out_path} ({'나란히 보기' if args.view == 'side' else '캐릭터만 보기'})")

    with HolisticTracker(model_complexity=args.complexity) as tracker, video_io.VideoWriter(
        out_path, fps=info.fps, size=(output_width, info.height)
    ) as writer:
        frame_index = 0
        for frame_bgr in video_io.read_frames(args.input, max_height=args.max_height):
            raw, mp_results = tracker.process(frame_bgr)

            scale = retarget.compute_scale(raw.pose_world, raw.pose_pixel)
            if scale is not None:
                last_scale = scale
            elif last_scale is not None:
                scale = last_scale

            assigned_left_px, assigned_right_px = retarget.assign_hands(
                raw.pose_pixel, raw.left_hand_pixel, raw.right_hand_pixel
            )

            pose_world = raw.pose_world
            left_hand_world = None
            right_hand_world = None
            face_world = None

            if pose_world is not None:
                if args.upper_body:
                    pose_world = retarget.synthesize_lower_body(pose_world)
                if assigned_left_px is not None and scale is not None:
                    left_hand_world = retarget.hand_to_world(
                        assigned_left_px, pose_world[landmarks.LEFT_WRIST], scale
                    )
                if assigned_right_px is not None and scale is not None:
                    right_hand_world = retarget.hand_to_world(
                        assigned_right_px, pose_world[landmarks.RIGHT_WRIST], scale
                    )
                if raw.face_pixel is not None and scale is not None:
                    face_world = retarget.face_to_world(
                        raw.face_pixel, pose_world[landmarks.NOSE], scale
                    )

            smoothed_pose = pose_smoother.update(pose_world)
            smoothed_left = left_hand_smoother.update(left_hand_world)
            smoothed_right = right_hand_smoother.update(right_hand_world)
            smoothed_face = face_smoother.update(face_world)

            frame_data = retarget.hold_missing(
                prev_frame_data, smoothed_pose, smoothed_left, smoothed_right, smoothed_face
            )
            prev_frame_data = frame_data

            avatar_img = renderer.render(frame_data)

            if args.view == "side":
                overlay = draw_landmarks_overlay(frame_bgr, mp_results)
                combined = video_io.side_by_side(overlay, avatar_img)
            else:
                combined = avatar_img

            writer.write(combined)

            if args.export_json:
                json_records.append(_frame_data_to_json_record(frame_index, frame_data))

            frame_index += 1
            _print_progress(frame_index, info.total_frames)

    sys.stdout.write("\n")
    print("처리 완료.")

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(json_records, f, ensure_ascii=False)
        print(f"포즈 데이터 저장: {args.export_json}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
