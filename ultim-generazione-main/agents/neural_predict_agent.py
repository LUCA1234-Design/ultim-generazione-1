"""
Predictive Neural Engine (lightweight sequence model, Phase 14).
Implements a Markov-style probability matrix + trend features.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentResult
from config.settings import (
    NEURAL_PREDICT_ENABLED,
    NEURAL_PREDICT_LOOKBACK,
    NEURAL_PREDICT_HORIZON,
)

logger = logging.getLogger("NeuralPredictAgent")


class NeuralPredictAgent(BaseAgent):
    """Estimates bullish/bearish probability for next N candles."""

    def __init__(self):
        super().__init__("neural_predict", initial_weight=0.15)

    @staticmethod
    def _transition_matrix(signs: pd.Series) -> np.ndarray:
        # states: 0=bear, 1=bull
        mat = np.ones((2, 2), dtype=float)  # Laplace smoothing
        values = signs.values.astype(int)
        for i in range(1, len(values)):
            prev = 1 if values[i - 1] > 0 else 0
            nxt = 1 if values[i] > 0 else 0
            mat[prev, nxt] += 1.0
        mat = mat / mat.sum(axis=1, keepdims=True)
        return mat

    def analyse(self, symbol: str, interval: str, df, *args, **kwargs) -> Optional[AgentResult]:
        if not NEURAL_PREDICT_ENABLED:
            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=0.5,
                direction="neutral",
                confidence=0.2,
                details=["neural_predict:disabled"],
                metadata={},
            )

        if df is None or len(df) < 40 or "close" not in df.columns:
            return None

        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(closes) < 40:
            return None

        lookback = int(max(40, NEURAL_PREDICT_LOOKBACK))
        closes = closes.iloc[-lookback:]
        returns = closes.pct_change().dropna()
        if len(returns) < 20:
            return None

        signs = returns.apply(lambda x: 1 if x > 0 else -1)
        mat = self._transition_matrix(signs)
        horizon = int(max(1, NEURAL_PREDICT_HORIZON))
        mat_n = np.linalg.matrix_power(mat, horizon)

        last_state = 1 if float(returns.iloc[-1]) > 0 else 0
        prob_vec = mat_n[last_state]
        p_bear = float(np.clip(prob_vec[0], 0.0, 1.0))
        p_bull = float(np.clip(prob_vec[1], 0.0, 1.0))

        trend = float((closes.iloc[-1] / closes.iloc[0]) - 1.0)
        trend_bias = float(np.clip(trend * 3.0, -0.12, 0.12))
        p_bull = float(np.clip(p_bull + trend_bias, 0.0, 1.0))
        p_bear = float(np.clip(1.0 - p_bull, 0.0, 1.0))

        if p_bull > p_bear + 0.03:
            direction = "long"
        elif p_bear > p_bull + 0.03:
            direction = "short"
        else:
            direction = "neutral"

        confidence = float(np.clip(abs(p_bull - p_bear), 0.2, 0.98))
        score = float(max(p_bull, p_bear))
        details = [f"p_bull@{horizon}={p_bull:.2%}", f"p_bear@{horizon}={p_bear:.2%}"]

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=score,
            direction=direction,
            confidence=confidence,
            details=details,
            metadata={
                "prob_bull": p_bull,
                "prob_bear": p_bear,
                "horizon": horizon,
                "trend_bias": trend_bias,
            },
        )
