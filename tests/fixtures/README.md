# 테스트 픽스처

## `canonical_face_model.npy`

MediaPipe FaceMesh의 표준 얼굴 모델(canonical face model) 정점 468개, `(468, 3)` float32, 단위 cm.
특정 인물이 아닌 평균 얼굴이다. 실제 인물의 얼굴 기하 데이터를 공개 저장소에 넣지 않기 위해
얼굴 관련 테스트는 이 모델을 FaceMesh 출력 형식으로 변환해서 쓴다 (`tests/face_fixture.py`).

- 출처: https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/modules/face_geometry/data/canonical_face_model.obj
- 라이선스: Apache License 2.0 (MediaPipe)
- 좌표계: x = 화자 왼쪽(화면 오른쪽), y = 위, z = 카메라 쪽
