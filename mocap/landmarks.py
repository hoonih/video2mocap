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
LEFT_EAR = 7
RIGHT_EAR = 8
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
# 좌우는 MediaPipe 명명 규칙(화자 기준)을 따른다. 화자의 왼쪽 = 화면 오른쪽.
# 눈썹은 안쪽 끝이 아니라 바깥 끝(관자놀이 쪽)부터 미간 쪽 순서로 나열한다.
LEFT_BROW_INDICES = (300, 293, 334, 296, 336)
RIGHT_BROW_INDICES = (70, 63, 105, 66, 107)
LEFT_BROW_KEYS = tuple(f"left_brow_{i}" for i in range(len(LEFT_BROW_INDICES)))
RIGHT_BROW_KEYS = tuple(f"right_brow_{i}" for i in range(len(RIGHT_BROW_INDICES)))

FACE_KEY_INDICES = {
    "nose_tip": 1,
    "left_eye": 263,          # 바깥 눈꼬리
    "left_eye_inner": 362,
    "left_eye_top": 386,
    "left_eye_bottom": 374,
    "right_eye": 33,          # 바깥 눈꼬리
    "right_eye_inner": 133,
    "right_eye_top": 159,
    "right_eye_bottom": 145,
    "mouth_left": 291,
    "mouth_right": 61,
    "mouth_top": 13,          # 윗입술 안쪽
    "mouth_bottom": 14,       # 아랫입술 안쪽
    "forehead": 10,
    "chin": 152,
    "face_left_edge": 454,
    "face_right_edge": 234,
    **dict(zip(LEFT_BROW_KEYS, LEFT_BROW_INDICES)),
    **dict(zip(RIGHT_BROW_KEYS, RIGHT_BROW_INDICES)),
}
# FrameData.face 배열에서의 순서 (face_to_world의 출력 순서와 일치해야 함)
FACE_KEY_ORDER = tuple(FACE_KEY_INDICES)

# 얼굴 메시에서 머리 중심으로 쓰는 점: 양쪽 얼굴 가장자리(귀 앞)의 3D 중점.
# FaceMesh의 z 원점은 얼굴 앞면 근처라서 코끝 z는 이 중점보다 약 10cm 앞에 있다.
FACE_CENTER_KEYS = ("face_left_edge", "face_right_edge")

# 양쪽 바깥 눈꼬리(263-33) 사이 3D 거리(미터). MediaPipe canonical face model 기준값.
# 얼굴 크기를 어깨너비 대신 이 값으로 맞춘다. 어깨 기반 환산은 몸을 틀면 얼굴이 최대 50% 커진다.
FACE_EYE_SPAN_METERS = 0.0889

# 눈 윤곽 폴리곤(바깥 꼬리 → 위 → 안쪽 꼬리 → 아래), 입 폴리곤(오른쪽 → 위 → 왼쪽 → 아래)
LEFT_EYE_OUTLINE = ("left_eye", "left_eye_top", "left_eye_inner", "left_eye_bottom")
RIGHT_EYE_OUTLINE = ("right_eye", "right_eye_top", "right_eye_inner", "right_eye_bottom")
MOUTH_OUTLINE = ("mouth_right", "mouth_top", "mouth_left", "mouth_bottom")
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
EYE_HALF_WIDTH = 0.03
EYE_RAISE = 0.03
EAR_HALF_WIDTH = 0.07
EAR_DEPTH = 0.1  # 귀(머리 중심)는 코보다 이만큼 뒤에 있다 (z 양수 = 카메라 반대쪽)


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
    nose_y = -TORSO_LENGTH - NECK_LENGTH - HEAD_OFFSET
    pose[NOSE] = (0.0, nose_y, 0.0)
    pose[LEFT_EYE] = (EYE_HALF_WIDTH, nose_y - EYE_RAISE, EAR_DEPTH * 0.2)
    pose[RIGHT_EYE] = (-EYE_HALF_WIDTH, nose_y - EYE_RAISE, EAR_DEPTH * 0.2)
    pose[LEFT_EAR] = (EAR_HALF_WIDTH, nose_y, EAR_DEPTH)
    pose[RIGHT_EAR] = (-EAR_HALF_WIDTH, nose_y, EAR_DEPTH)
    return pose
