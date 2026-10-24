# 모델 파일

저장소에는 넣지 않는다 (`.gitignore`). 아래 명령으로 받는다.

## `face_landmarker.task`

MediaPipe Tasks API Face Landmarker. 얼굴 478점 + ARKit 호환 블렌드셰이프 52개(`_neutral` 포함).

```bash
mkdir -p assets/models
curl -L -o assets/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
shasum -a 256 assets/models/face_landmarker.task
# 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff (2026-10-25 받은 파일)
```

- 출처: Google MediaPipe (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- 라이선스: Apache License 2.0
- 버전: float16 / 1
- 이 파일이 없으면 `python -m mocap`은 경고를 내고 Holistic 얼굴 좌표만 쓴다 (블렌드셰이프 없음).
