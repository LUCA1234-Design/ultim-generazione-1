"""
Liquidity Agent for V17.
Phase 2 proxy implementation using OHLCV data (REST-style approximation).
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent


class LiquidityAgent(BaseAgent):
    """Estimates liquidity pressure using volume spike + flow metrics."""

    def __init__(self):
        super().__init__("liquidity", initial_weight=1.0)

    def analyze(self, symbol: str, interval: str, df: pd.DataFrame) -> Dict[str, Any]:
        neutral = {
            "signal": 0,
            "confidence": 0.0,
            "details": {"liquidity_score": 0.0, "reason": "insufficient_data"},
        }

        if df is None or len(df) < 30:
            return neutral
        required = {"high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return neutral

        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(df["close"], errors="coerce").ffill()
        high = pd.to_numeric(df["high"], errors="coerce").ffill()
        low = pd.to_numeric(df["low"], errors="coerce").ffill()

        if close.isna().all() or volume.sum() <= 0:
            return neutral

        recent_volume = float(volume.iloc[-5:].mean())
        baseline_series = volume.iloc[-25:-5]
        baseline_volume = float(baseline_series.mean()) if len(baseline_series) > 0 else 0.0
        if baseline_volume <= 0:
            baseline_volume = float(volume.iloc[:-1].mean()) if len(volume) > 1 else recent_volume

        volume_ratio = recent_volume / max(baseline_volume, 1e-9)
        volume_spike = float(np.clip((volume_ratio - 1.0) / 1.5, 0.0, 1.0))

        returns = close.pct_change().fillna(0.0)
        flow_window = min(20, len(close))
        flow_numerator = float((returns.iloc[-flow_window:] * volume.iloc[-flow_window:]).sum())
        flow_denominator = float(volume.iloc[-flow_window:].sum())
        flow_bias = float(np.clip(flow_numerator / max(flow_denominator, 1e-9), -1.0, 1.0))

        price_range = (high - low).replace(0, np.nan)
        mfm = (((close - low) - (high - close)) / price_range).fillna(0.0)
        cmf_raw = (mfm * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
        cmf = float(np.clip(cmf_raw.fillna(0.0).iloc[-1], -1.0, 1.0))

        directional_bias = float(np.clip((0.55 * flow_bias) + (0.45 * cmf), -1.0, 1.0))
        liquidity_score = float(np.clip(directional_bias * (0.5 + 0.5 * volume_spike), -1.0, 1.0))

        signal = 1 if liquidity_score > 0.15 else -1 if liquidity_score < -0.15 else 0
        confidence = float(np.clip(abs(liquidity_score), 0.0, 1.0))

        return {
            "signal": signal,
            "confidence": confidence,
            "details": {
                "liquidity_score": liquidity_score,
                "volume_ratio": float(volume_ratio),
                "volume_spike": volume_spike,
                "flow_bias": flow_bias,
                "cmf": cmf,
            },
        }

    def analyse(self, symbol: str, interval: str, df, *args, **kwargs) -> AgentResult:
        payload = self.analyze(symbol, interval, df)
        signal = int(payload.get("signal", 0))
        confidence = float(np.clip(payload.get("confidence", 0.0), 0.0, 1.0))
        direction = "long" if signal > 0 else "short" if signal < 0 else "neutral"
        details_dict = payload.get("details", {}) or {}
        details = [f"{k}={v}" for k, v in details_dict.items()]
        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=confidence,
            direction=direction,
            confidence=confidence,
            details=details,
            metadata={"signal": signal, "details": details_dict},
        )
