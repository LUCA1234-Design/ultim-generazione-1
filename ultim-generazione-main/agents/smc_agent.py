"""
Smart Money Concepts agent (Phase 12).
Detects Fair Value Gaps (FVG) and simple Order Blocks (OB) and proposes a
mitigation limit-entry price.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentResult


class SMCAgent(BaseAgent):
    """Institutional-style setup detector for FVG/OB zones."""

    def __init__(self):
        super().__init__("smc", initial_weight=0.12)

    @staticmethod
    def _latest_fvg(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 3:
            return None
        high_2 = float(df["high"].iloc[-3])
        low_2 = float(df["low"].iloc[-3])
        high_0 = float(df["high"].iloc[-1])
        low_0 = float(df["low"].iloc[-1])

        if high_2 < low_0:
            zone_low, zone_high = high_2, low_0
            return {
                "type": "fvg_bullish",
                "direction": "long",
                "zone_low": zone_low,
                "zone_high": zone_high,
                "entry": (zone_low + zone_high) / 2.0,
                "strength": (zone_high - zone_low) / max(abs(zone_high), 1e-9),
            }
        if low_2 > high_0:
            zone_low, zone_high = high_0, low_2
            return {
                "type": "fvg_bearish",
                "direction": "short",
                "zone_low": zone_low,
                "zone_high": zone_high,
                "entry": (zone_low + zone_high) / 2.0,
                "strength": (zone_high - zone_low) / max(abs(zone_high), 1e-9),
            }
        return None

    @staticmethod
    def _latest_order_block(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 6:
            return None
        prev = df.iloc[-2]
        cur = df.iloc[-1]
        avg_range = float((df["high"] - df["low"]).iloc[-6:-1].mean())
        impulse = float(cur["high"] - cur["low"])
        if avg_range <= 0:
            return None
        impulse_mult = impulse / avg_range
        is_down_candle = float(prev["close"]) < float(prev["open"])
        is_up_candle = float(prev["close"]) > float(prev["open"])
        has_bullish_impulse = float(cur["close"]) > float(prev["high"])
        has_bearish_impulse = float(cur["close"]) < float(prev["low"])
        is_strong_impulse = impulse_mult > 1.1

        # Bullish OB: last down candle before strong up impulse.
        if is_down_candle and has_bullish_impulse and is_strong_impulse:
            zone_low = min(float(prev["open"]), float(prev["close"]))
            zone_high = max(float(prev["open"]), float(prev["close"]))
            return {
                "type": "ob_bullish",
                "direction": "long",
                "zone_low": zone_low,
                "zone_high": zone_high,
                "entry": (zone_low + zone_high) / 2.0,
                "strength": min(1.0, impulse_mult / 3.0),
            }

        # Bearish OB: last up candle before strong down impulse.
        if is_up_candle and has_bearish_impulse and is_strong_impulse:
            zone_low = min(float(prev["open"]), float(prev["close"]))
            zone_high = max(float(prev["open"]), float(prev["close"]))
            return {
                "type": "ob_bearish",
                "direction": "short",
                "zone_low": zone_low,
                "zone_high": zone_high,
                "entry": (zone_low + zone_high) / 2.0,
                "strength": min(1.0, impulse_mult / 3.0),
            }
        return None

    def analyse(self, symbol: str, interval: str, df, direction: str = "neutral") -> Optional[AgentResult]:
        if df is None or len(df) < 10:
            return None
        requested_direction = str(direction or "neutral").lower()
        fvg = self._latest_fvg(df)
        ob = self._latest_order_block(df)

        candidates = [c for c in (fvg, ob) if c is not None]
        if not candidates:
            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details=["no_smc_setup"],
                metadata={"setup": "none"},
            )

        best = max(candidates, key=lambda x: float(x.get("strength", 0.0)))
        setup_dir = str(best.get("direction", "neutral"))
        directional_alignment = 1.0 if requested_direction in {"neutral", setup_dir} else 0.55
        setup_score = float(np.clip(best.get("strength", 0.0) * 3.0, 0.0, 1.0))
        score = float(np.clip(0.45 + 0.55 * setup_score, 0.0, 1.0)) * directional_alignment

        details = [best["type"], f"zone=({best['zone_low']:.4f}-{best['zone_high']:.4f})"]
        if requested_direction != "neutral" and requested_direction != setup_dir:
            details.append("direction_mismatch")

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(score, 0.0, 1.0)),
            direction=setup_dir,
            confidence=float(np.clip(score, 0.0, 1.0)),
            details=details,
            metadata={
                "setup": best["type"],
                "zone_low": float(best["zone_low"]),
                "zone_high": float(best["zone_high"]),
                "limit_entry": float(best["entry"]),
                "direction_match": bool(requested_direction in {"neutral", setup_dir}),
            },
        )
