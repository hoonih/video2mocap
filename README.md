# mocap — 영상 모션캡처 → 3D 캐릭터 애니메이션

사람이 나오는 mp4 영상을 입력하면, 영상 속 사람의 동작을 추적해서 3D 캐릭터가 똑같이
따라 움직이는 영상을 만들어주는 순수 Python CLI 도구입니다. Blender 같은 외부 3D 프로그램
없이 `mediapipe` / `opencv-python` / `numpy`만으로 동작합니다.

## 설치

**ffmpeg가 필요합니다** (`brew install ffmpeg` / `sudo apt install ffmpeg`). 입력 디코딩에 씁니다.

얼굴 블렌드셰이프용 모델은 따로 받아야 합니다: `assets/models/README.md` 참고.


Python 3.9~3.12 환경을 권장합니다.

```bash
pip install -r requirements.txt
# 또는
pip install -e .
```

`mediapipe==0.10.14`로 버전이 고정되어 있습니다. 최신 mediapipe는 이 프로젝트가 쓰는
`mp.solutions.holistic` / `mp.solutions.drawing_utils` 같은 레거시 API가 빠져 있을 수
있으니 이 버전을 꼭 사용하세요.

## 사용법

```bash
python -m mocap input.mp4
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--out 경로.mp4` | 출력 영상 경로 | `<입력파일명>_mocap.mp4` |
| `--upper-body` | 상체 모드 (가슴 위만 나오는 영상용) | 꺼짐 |
| `--view side\|avatar\|skeleton\|original` | `side`: 정규화한 원본(랜드마크 표시)+캐릭터 나란히, `avatar`: 캐릭터만(조건 C), `skeleton`: 선과 점으로 된 골격만(조건 B), `original`: 원본을 같은 크기·위치로 정규화(조건 A) | `side` |
| `--yaw 각도` | 카메라 좌우 회전 각도(도) | `0` |
| `--orbit 각도` | 프레임마다 추가로 회전할 각도(도/프레임) | `0` |
| `--smooth 0.0~0.9` | 떨림 보정 강도 (클수록 부드럽지만 반응 느려짐) | `0.3` |
| `--complexity 0\|1\|2` | MediaPipe 모델 복잡도 (높을수록 정확하지만 느림) | `1` |
| `--max-height 픽셀` | 추적용 프레임 최대 세로 크기 (출력 크기와 무관) | `720` |
| `--output-size WxH` | 출력 해상도. 모든 조건이 이 크기로 나옴 | `720x1280` |
| `--fps 값` | 출력 고정 프레임레이트 (입력이 29.97·가변이어도 맞춤) | `30` |
| `--shoulder-ratio 비율` | 정면 어깨너비가 화면 폭에서 차지할 비율 | `0.35` |
| `--anchor-y 비율` | 어깨 중앙의 세로 위치 (0=위, 1=아래) | `0.42` |
| `--show-legs` | 상체 모드에서도 합성 다리를 그림 | 꺼짐(다리 숨김) |
| `--face-scale 배율` | 머리·얼굴 확대 배율 | `1.0` |
| `--face-model 경로` | Face Landmarker 모델(블렌드셰이프). 없으면 Holistic 얼굴 좌표로 대신 | `assets/models/face_landmarker.task` |
| `--no-blendshapes` | Face Landmarker를 끄고 Holistic 얼굴 좌표만 사용 | 꺼짐 |
| `--export-json 경로.json` | 프레임별 3D 관절 좌표를 JSON으로 저장 | 없음 |

### 예시

전신이 나오는 영상, 원본과 캐릭터를 나란히 출력:

```bash
python -m mocap dance.mp4 --out dance_mocap.mp4
```

상체만 나오는 웹캠 영상(가슴 위), 캐릭터만 출력하고 카메라를 15도 돌려서 보기:

```bash
python -m mocap webcam_upper.mp4 --upper-body --view avatar --yaw 15 --out result.mp4
```

계속 회전하는 카메라로 캐릭터를 보면서, 포즈 데이터도 JSON으로 함께 저장:

```bash
python -m mocap input.mp4 --view avatar --orbit 1.5 --export-json pose.json
```

떨림이 심할 때 스무딩을 더 강하게:

```bash
python -m mocap input.mp4 --smooth 0.6
```

## 동작 방식 요약

1. **추적** (`mocap/tracker.py`): MediaPipe Holistic으로 프레임마다 몸 전체(33점) +
   양손(21점×2) + 얼굴(468점)을 검출합니다. 몸은 미터 단위 3D 좌표(`pose_world_landmarks`)를
   그대로 쓰고, 손/얼굴은 정규화 픽셀 좌표만 나오므로 별도로 미터로 환산합니다.
2. **리타게팅** (`mocap/retarget.py`):
   - 손목이 더 가까운 팔 쪽에 손을 재배정합니다(MediaPipe의 좌우 라벨은 신뢰하지 않음).
   - "월드 어깨너비 / 픽셀 어깨너비" 비율로 손·얼굴의 픽셀 좌표를 미터로 환산해 손목/코
     위치에 붙입니다.
   - `--upper-body`일 때는 화면에 보이지 않는 골반/다리의 MediaPipe 추측값을 버리고,
     어깨가 향한 좌우 방향(yaw)에 맞춰 고정된 서 있는 하체 자세로 대체합니다.
   - 인식이 실패한 프레임은 직전 프레임 값을 유지합니다.
3. **스무딩** (`mocap/smoothing.py`): 부위별(몸/왼손/오른손/얼굴)로 독립적인 지수이동평균
   (EMA)을 적용해 떨림을 줄입니다. 추적이 끊겼다가 다시 잡힌 부위는 스무딩 없이 즉시
   새 값으로 리셋해서, 오래된 값과 섞여 미끄러지는 현상을 막습니다.
4. **렌더링** (`mocap/renderer.py`): OpenCV로 직접 그리는 캡슐형 캐릭터에 원근 투영과
   화가의 알고리즘(depth sort)을 적용해 3D처럼 보이게 그립니다. `Renderer` 인터페이스만
   지키면 나중에 실제 3D 모델(glTF 등) 렌더러로 교체할 수 있습니다.

## 코드 구조

```
mocap/
  cli.py          # argparse, 전체 파이프라인 실행
  tracker.py      # MediaPipe Holistic → 표준 프레임 데이터 (RawLandmarks)
  retarget.py     # 하체 합성, 좌표 정규화, 손/얼굴 스케일 맞추기, FrameData
  smoothing.py    # 부위별 EMA 스무더
  renderer.py     # Renderer 인터페이스 + 캡슐 캐릭터 렌더러
  landmarks.py    # 관절 인덱스, 손가락 체인, 얼굴 키포인트, 하체 템플릿 상수
  video_io.py     # 영상 읽기/쓰기, 리사이즈, 좌우 합성
tests/            # pytest 단위/통합 테스트
```

## 테스트

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

`smoothing`/`retarget`/`video_io` 테스트는 mediapipe 없이도 동작합니다. `tracker`와
`cli` 통합 테스트는 mediapipe가 설치되어 있어야 하며, 사람이 없는 더미 영상으로 전체
파이프라인이 에러 없이 끝까지 도는지 확인합니다. 렌더러 테스트는 가짜 포즈 데이터로
PNG를 실제로 생성해 눈으로 확인할 수 있습니다.

## 한계

- **한 명만 추적**: MediaPipe Holistic은 화면에 여러 사람이 있어도 한 명만 인식합니다.
- **제자리 동작만 반영**: 사람의 실제 이동(걷기, 점프해서 위치가 바뀌는 것 등)은 반영되지
  않습니다. `pose_world_landmarks`가 항상 골반을 원점으로 삼기 때문에, 캐릭터는 제자리에서
  팔다리/몸통 움직임만 따라 하고 화면 속 위치 이동은 하지 않습니다.
- **팔/다리 비틀림(twist) 미반영**: 어깨-팔꿈치-손목의 위치만으로 자세를 잡기 때문에,
  팔뚝이나 다리를 축 방향으로 비트는 회전(예: 손바닥을 뒤집는 동작)은 캐릭터에 나타나지
  않습니다.
- **상체 모드의 하체는 항상 고정 자세**: 어깨 방향(좌우 회전)만 반영하고, 실제 다리
  동작(걷기, 앉기 등)은 표현하지 않습니다.
- **깊이 추정 오차**: MediaPipe의 3D 추정은 카메라 각도가 극단적이거나 몸의 일부가 잘려
  나가면 부정확해질 수 있습니다.
- **저조도/가려짐에서 검출 불안정**: 조명이 어둡거나 신체 일부가 다른 물체에 가려지면
  검출이 끊기며, 이 경우 직전 프레임 값으로 유지되다가 스무딩이 리셋되면서 약간 튀는
  느낌이 날 수 있습니다.
- **렌더러는 스타일라이즈드 캡슐 캐릭터**: 실제 사람 형태나 텍스처가 있는 3D 모델이
  아니라, 관절 인형 스타일의 단순한 캡슐 렌더링입니다. `Renderer` 인터페이스를 구현하는
  다른 렌더러(예: glTF 모델 렌더러)로 교체할 수 있도록 설계했습니다.
