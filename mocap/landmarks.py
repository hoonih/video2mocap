"""관절 인덱스, 손가락 체인, 얼굴 키포인트, 하체 템플릿 등 프로젝트 전역 상수 모음.

MediaPipe Pose/Hand/Face 랜드마크 인덱스는 MediaPipe 공식 토폴로지를 그대로 따른다.
여기서 정의하는 상수는 다른 모듈(tracker/retarget/renderer)이 공유해서 사용한다.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pose 랜드마크 인덱스 (mediapipe pose_landmark 33점 중 이 프로젝트에서 쓰는 것)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_POSE_LANDMARKS = 33

# 상체 모드에서 실제 검출값을 버리고 고정 하체 템플릿으로 덮어쓸 인덱스들
LOWER_BODY_INDICES = (
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

# 렌더러가 캡슐(팔다리) 세그먼트를 그릴 때 사용하는 (시작, 끝, 부위명) 목록.
# 부위명은 좌우 밝기 구분과 굵기 결정에 쓰인다.
POSE_LIMB_SEGMENTS = (
    (LEFT_SHOULDER, LEFT_ELBOW, "left_arm"),
    (LEFT_ELBOW, LEFT_WRIST, "left_arm"),
    (RIGHT_SHOULDER, RIGHT_ELBOW, "right_arm"),
    (RIGHT_ELBOW, RIGHT_WRIST, "right_arm"),
    (LEFT_HIP, LEFT_KNEE, "left_leg"),
    (LEFT_KNEE, LEFT_ANKLE, "left_leg"),
    (RIGHT_HIP, RIGHT_KNEE, "right_leg"),
    (RIGHT_KNEE, RIGHT_ANKLE, "right_leg"),
    (LEFT_ANKLE, LEFT_FOOT_INDEX, "left_leg"),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX, "right_leg"),
)

# 목: 어깨 중앙 <-> 머리. 렌더러에서 항상 머리보다 뒤에 그리도록 depth bias를 준다.
NECK_TOP = NOSE

# ---------------------------------------------------------------------------
# Hand 랜드마크 (21점, 손목=0)
# ---------------------------------------------------------------------------
HAND_WRIST = 0
NUM_HAND_LANDMARKS = 21

# 손가락별 [관절1, 관절2, 관절3, 끝] — 손목(0)에서 이어진다.
HAND_FINGERS = (
    (1, 2, 3, 4),      # 엄지
    (5, 6, 7, 8),       # 검지
    (9, 10, 11, 12),    # 중지
    (13, 14, 15, 16),   # 약지
    (17, 18, 19, 20),   # 새끼
)

# ---------------------------------------------------------------------------
# Face 랜드마크 (mediapipe face mesh 468점 중 필요한 핵심 포인트만 추출)
# ---------------------------------------------------------------------------
FACE_KEY_INDICES = {
    "nose_tip": 1,
    "left_eye": 263,
    "right_eye": 33,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
}
# FrameData.face 배열에서의 순서 (face_to_world의 출력 순서와 일치해야 함)
FACE_KEY_ORDER = (
    "nose_tip", "left_eye", "right_eye",
    "mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
)
NUM_FACE_KEY_POINTS = len(FACE_KEY_ORDER)

# ---------------------------------------------------------------------------
# 상체 모드용 고정 하체 템플릿 (미터 단위, 골반 원점 기준 로컬 좌표)
# mediapipe world 좌표계: x=오른쪽, y=아래, z=카메라 반대 방향
# ---------------------------------------------------------------------------
STANDING_HIP_HALF_WIDTH = 0.09
THIGH_LENGTH = 0.42
SHIN_LENGTH = 0.42
ANKLE_TO_HEEL = 0.03
FOOT_LENGTH = 0.16
HEEL_HEIGHT = 0.03

# LOWER_BODY_INDICES 순서와 동일한 순서로 (x, y, z) 로컬 좌표 정의.
# y가 클수록(양수) 아래쪽(발쪽)이다.
LOWER_BODY_TEMPLATE = {
    LEFT_HIP: (-STANDING_HIP_HALF_WIDTH, 0.0, 0.0),
    RIGHT_HIP: (STANDING_HIP_HALF_WIDTH, 0.0, 0.0),
    LEFT_KNEE: (-STANDING_HIP_HALF_WIDTH, THIGH_LENGTH, 0.0),
    RIGHT_KNEE: (STANDING_HIP_HALF_WIDTH, THIGH_LENGTH, 0.0),
    LEFT_ANKLE: (-STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH, 0.0),
    RIGHT_ANKLE: (STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH, 0.0),
    LEFT_HEEL: (-STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH + HEEL_HEIGHT, -ANKLE_TO_HEEL),
    RIGHT_HEEL: (STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH + HEEL_HEIGHT, -ANKLE_TO_HEEL),
    LEFT_FOOT_INDEX: (-STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH + HEEL_HEIGHT, FOOT_LENGTH),
    RIGHT_FOOT_INDEX: (STANDING_HIP_HALF_WIDTH, THIGH_LENGTH + SHIN_LENGTH + HEEL_HEIGHT, FOOT_LENGTH),
}

# ---------------------------------------------------------------------------
# 상반신 기본값 (영상 시작부터 검출이 안 될 때 쓰는 최후의 fallback용 중립 자세)
# ---------------------------------------------------------------------------
TORSO_LENGTH = 0.5
SHOULDER_HALF_WIDTH = 0.19
UPPER_ARM_LENGTH = 0.28
FOREARM_LENGTH = 0.25
NECK_LENGTH = 0.1
HEAD_OFFSET = 0.12
ARM_OUTWARD_OFFSET = 0.02


def build_neutral_pose() -> np.ndarray:
    """검출 이력이 전혀 없을 때 사용할 '팔 내리고 서 있는' 중립 자세를 만든다."""
    pose = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float64)
    for idx, (lx, ly, lz) in LOWER_BODY_TEMPLATE.items():
        pose[idx] = (lx, ly, lz)

    pose[LEFT_SHOULDER] = (-SHOULDER_HALF_WIDTH, -TORSO_LENGTH, 0.0)
    pose[RIGHT_SHOULDER] = (SHOULDER_HALF_WIDTH, -TORSO_LENGTH, 0.0)
    pose[LEFT_ELBOW] = (
        -SHOULDER_HALF_WIDTH - ARM_OUTWARD_OFFSET,
        -TORSO_LENGTH + UPPER_ARM_LENGTH,
        0.0,
    )
    pose[RIGHT_ELBOW] = (
        SHOULDER_HALF_WIDTH + ARM_OUTWARD_OFFSET,
        -TORSO_LENGTH + UPPER_ARM_LENGTH,
        0.0,
    )
    pose[LEFT_WRIST] = (
        -SHOULDER_HALF_WIDTH - ARM_OUTWARD_OFFSET,
        -TORSO_LENGTH + UPPER_ARM_LENGTH + FOREARM_LENGTH,
        0.0,
    )
    pose[RIGHT_WRIST] = (
        SHOULDER_HALF_WIDTH + ARM_OUTWARD_OFFSET,
        -TORSO_LENGTH + UPPER_ARM_LENGTH + FOREARM_LENGTH,
        0.0,
    )
    pose[NOSE] = (0.0, -TORSO_LENGTH - NECK_LENGTH - HEAD_OFFSET, 0.0)
    return pose
