# 아바타 모델

모델 파일은 저장소에 넣지 않는다 (`.gitignore`). 아래에서 받아 이 폴더에 둔다.

## PerfectSync 샘플 (VRoid 공식 모델 + ARKit 52 블렌드셰이프)

```bash
for f in VRoid_V110_Female_v1.1.3.vrm VRoid_V110_Male_v1.1.3.vrm; do
  curl -L -o "assets/avatar/$f" "https://github.com/hinzka/52blendshapes-for-VRoid-face/raw/main/$f"
done
shasum -a 256 assets/avatar/*.vrm
```

| 파일 | SHA-256 (2026-10-25 받은 파일) |
|---|---|
| `VRoid_V110_Female_v1.1.3.vrm` | `36d6d242999d580bd0d3bcd8656bb2d4d5d4839f187d8198226bda905e1d9114` |
| `VRoid_V110_Male_v1.1.3.vrm` | `7526838f4a45086a1d0abb23e9f38ad3f0103889f1efa70d2605ef25e7e6ef99` |

- 출처: https://github.com/hinzka/52blendshapes-for-VRoid-face (작성자 @hinzka)
- 원본: VRoid Studio 공식 모델 (pixiv)
- 형식: VRM 0.x (UniVRM 0.79.0 내보내기), 휴머노이드 본 54개 (손가락 손가락당 3개, 30개)
- 표정: ARKit 52개 모프 타깃 (이름이 ARKit와 정확히 같음: `browInnerUp`, `browDownLeft`, ...) +
  같은 이름의 BlendShapeGroup (`BrowInnerUp` 등, PascalCase)

### 라이선스

- 저장소 README의 이용 조건 (2026-10-25 확인): 개인·상업 이용 자유, VRM 파일·수정 데이터의
  재배포·판매 가능, 크레딧 표기 불필요. 개별 이용 문의에는 응하지 않음.
- VRM 파일 내부 메타: `licenseName: Redistribution_Prohibited`, `commercialUssageName: Allow`,
  `allowedUserName: Everyone`. **README와 메타가 재배포에 대해 엇갈리므로 모델 파일은 재배포하지
  않는다** (이 저장소에 넣지 않음). 렌더링 결과(자극 영상·논문 그림)는 재배포가 아니다.
- 원본 VRoid Studio 모델: pixiv 공식 도움말상 VRoid Studio 정식판으로 만든 모델은 상업적 이용 가능.
  프리셋 아이템의 권리는 pixiv에 있으나 이용은 허락된다.
