"""부위별 지수이동평균(EMA) 스무더.

떨림 보정을 위해 매 프레임의 원본 좌표를 직전 스무딩 값과 섞는다. 단, 추적이 끊겼다가
다시 잡힌 부위는 오래된 값과 섞으면 순간적으로 미끄러지듯 이동하는 느낌이 나므로,
복귀한 첫 프레임에는 스무딩 없이 원본 값으로 즉시 초기화한다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EMASmoother:
    """단일 부위(pose 또는 한쪽 손, 얼굴)의 좌표를 지수이동평균으로 부드럽게 만든다."""

    def __init__(self, alpha: float) -> None:
        """alpha: 0.0~0.9 권장. 클수록 더 부드럽지만 반응이 느려진다."""
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha는 0.0 이상 1.0 미만이어야 합니다.")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None
        self._was_missing = True  # 아직 값이 없으므로 최초 입력은 리셋으로 취급

    def update(self, raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """새 원본 좌표를 반영하고 스무딩된 좌표를 반환한다.

        raw가 None이면(이번 프레임에 검출 실패) 마지막 값을 그대로 유지해서 반환한다.
        raw가 들어왔는데 직전 프레임에서 이 부위가 없었다면, 스무딩 없이 raw로 즉시
        리셋한다(스무딩 상태 초기화). 그 외에는 일반적인 EMA로 갱신한다.
        """
        if raw is None:
            self._was_missing = True
            return self._state

        raw = np.asarray(raw, dtype=np.float64)

        if self._was_missing or self._state is None:
            self._state = raw.copy()
        else:
            self._state = self.alpha * self._state + (1.0 - self.alpha) * raw

        self._was_missing = False
        return self._state

    def reset(self) -> None:
        """스무딩 상태를 완전히 비운다."""
        self._state = None
        self._was_missing = True
