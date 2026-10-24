# 리타게팅 라이브러리 선정 (Phase 4A)

조사일: 2026-10-25. 상태: **선정 완료 (2026-10-25 승인)** — 직접 FK 솔버 + three-vrm.

MediaPipe Holistic 출력(몸 33점 월드 좌표, 손 21점, 얼굴)을 휴머노이드 리그의 뼈 회전으로 바꾸는
기존 라이브러리를 비교했다. 얼굴은 Phase 4B에서 Face Landmarker 블렌드셰이프 계수를 직접 쓰므로,
얼굴 솔버는 평가 기준에서 참고로만 본다.

## 비교표

| 후보 | 라이선스 | 최근 활동 | 형태 | Holistic 입력 | 출력 리그 | 눈썹 독립 제어 | 손가락 | 아래팔 비틀림 |
|---|---|---|---|---|---|---|---|---|
| **Kalidokit** (yeemachine) | MIT | npm 마지막 릴리스 1.1.5 (2022-02-05). README에 "공식 지원 중단(deprecated)" 명시. 2025-08 예제 이관 커밋만 있음. 이슈 31개 | JS 라이브러리 | ✅ (pose 33 월드+2D, hand 21, face 468/478) | VRM·Live2D용으로 설계. 오일러 각만 내므로 다른 리그에도 쓸 수는 있으나 축 규약이 VRM 정규화 본 기준 | ❌ `brow` 단일 값 | ⚠️ 엄지 외 손가락은 굽힘 1축만(벌림 없음). 엄지 3축이 굽힘 값 하나에 상수를 곱해 함께 움직임 | ❌ ±0.3 rad(±17°)로 잘림 |
| **XR Animator** (ButzYung) | **CC BY-NC-SA 4.0** (비상업·동일조건) | 활발 (2026-10) | 앱 (라이브러리 아님) | ✅ | VRM·MMD | ✅ (ARKit 52) | ✅ | 미기재 |
| **SysMocap** (xianfei) | MPL-2.0 | 2026-06 | Electron 데스크톱 앱 | ✅ | VRM·glTF·FBX | 미기재 | 미기재 | Kalidokit 기반으로 보여 동일 한계 추정 |
| **BlendArMocap** (cgtinker) | GPL-3.0 | 저자가 "더 이상 유지보수할 수 없음" 명시 | Blender 애드온 | ✅ | Blender Rigify 전용 | 드라이버 방식 | ✅ | 미기재 |
| **vid2anim** (mehulcode12) | MIT | 2026-10 (별 2개, 포크 0) | Python + Blender | ✅ | Mixamo·VRM·Rigify·UE5 | 미기재 | 굽힘만 기재 | 미기재 |
| **three-vrm** (pixiv) | MIT | 활발 (2026-10) | three.js 로더·런타임 | (솔버 아님) | VRM 정규화 휴머노이드 본 | 모델에 사용자 정의 표정이 있으면 ✅ | — | — |

### Kalidokit 소스 확인 결과 (v1.1.5 `dist/`)

README만으로는 판단하기 어려워 소스를 직접 읽었다.

- `HandSolver`: `hand[side+"Wrist"].x = clamp(... * 2 * invert, -0.3, 0.3); // twist` → **손바닥 뒤집기를
  표현할 수 없다.** 손바닥 방향은 한국수어의 음운 자질이라 치명적이다.
- `HandSolver`: 엄지 3관절의 x·y·z가 모두 굽힘 각도 하나(`trackedFinger.z`)에 서로 다른 상수를 곱한
  값이다. 엄지 위치가 중요한 지문자에서 손 모양이 틀어진다.
- `calcArms.rigArm`: 위팔 ×2.3, 아래팔 ×2.14, 손목 ×2.3 등 "자연스러운 움직임"을 위한 증폭 계수와
  오프셋(−0.3, −0.4)이 곳곳에 있다. **아바타가 원본보다 과장되게 움직이므로 조건 A·C 사이의 교란
  요인이 된다.** 표정을 증폭하지 않기로 한 원칙과도 충돌한다.
- `PoseSolver.solve`: 손목이 골반 근처(월드 y > 0.1)에 있으면 팔을 휴지 자세로 되돌린다. 수어 문장의
  시작·끝 자세에서 팔이 튈 수 있다.

## 결론

조건을 모두 만족하는 라이브러리가 없다.

- 라이브러리 형태이고 라이선스가 맞는 것은 Kalidokit뿐인데, 지원이 중단되었고 손바닥 방향·손가락
  벌림·움직임 충실도에서 연구 요구를 만족하지 못한다.
- 손가락·표정이 갖춰진 XR Animator는 비상업 라이선스 앱이다.
- 나머지는 앱이거나(SysMocap), 유지보수가 끊겼거나(BlendArMocap), 검증이 부족하다(vid2anim).

## 선정 결과

계획서의 '리타게팅은 직접 구현하지 않는다' 원칙의 **예외로 승인**되었다. 조건을 만족하는 라이브러리가
없기 때문이다. 직접 구현 범위는 '관절 위치 → 뼈 회전' 계산으로 한정하고, 리그 규약·렌더링은 기존
라이브러리를 쓴다.

### 구성

**뼈 회전 계산은 직접 구현하고(기하학적 FK 솔버), 휴머노이드 리그 추상화와 렌더링은 검증된
라이브러리를 쓴다.**

| 구성 요소 | 선택 | 라이선스 |
|---|---|---|
| 관절 위치 → 뼈 회전 | `mocap/rig.py` 직접 구현 (Python, pytest로 검증) | 이 저장소 |
| 휴머노이드 본 규약 | three-vrm 정규화 본 (VRM 모델일 때) 또는 본 이름 매핑 표 (일반 glTF) | MIT |
| 렌더링 | three.js | MIT |
| 얼굴 | MediaPipe Face Landmarker 블렌드셰이프 (Phase 4B) | Apache-2.0 |

근거:
- 직접 구현할 부분(손바닥 법선으로 아래팔 비틀림 풀기, MCP 벌림, 해부학적 한계)은 Phase 4D에서
  이미 직접 구현하라고 정한 범위다. Kalidokit를 써도 이 부분은 새로 짜야 하고, 남는 것은 위팔·
  아래팔·손가락 굽힘인데 이 역시 증폭 계수를 걷어내면 "휴지 자세 방향 → 관측 방향" 회전 한 줄이다.
- 증폭 계수가 없으므로 원본 움직임을 그대로 옮긴다 (조건 A와의 교란 제거).
- 솔버가 Python에 있으면 기존 추적 결과(`FrameData`)를 그대로 쓰고, 브라우저는 계산된 쿼터니언만
  적용한다. 렌더 결정론성 확보와 단위 테스트가 쉬워진다.
- 리타게팅에서 가장 까다로운 "모델마다 다른 본 축 규약"은 three-vrm 정규화 본이 흡수한다.
  그래서 Phase 4C 모델은 **VRM 1.0 + ARKit 52 사용자 정의 표정(Perfect Sync)** 을 우선 찾는다.
  VRM 표준 프리셋에는 눈썹이 없지만, 사용자 정의 표정에 `browInnerUp` 등이 따로 있으면 독립 제어가
  가능하다. 실제로 들어 있는지는 Phase 4C에서 모프 이름을 덤프해 확인한다.

## 출처

- Kalidokit: https://github.com/yeemachine/kalidokit (MIT), npm `kalidokit@1.1.5`
- XR Animator: https://github.com/ButzYung/SystemAnimatorOnline (CC BY-NC-SA 4.0)
- SysMocap: https://github.com/xianfei/SysMocap (MPL-2.0)
- BlendArMocap: https://github.com/cgtinker/BlendArMocap (GPL-3.0)
- vid2anim: https://github.com/mehulcode12/vid2anim (MIT)
- three-vrm: https://github.com/pixiv/three-vrm (MIT)
