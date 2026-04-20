"""
Market Gravity Agent (Phase 11).
Uses BTC/ETH trend state to adjust or veto altcoin directional signals:
- vetoes shorts during strong BTC bullish momentum
- boosts longs when BTC (and optionally ETH) are bullish
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from agents.base_agent import BaseAgent, AgentResult
from data import data_store

# Phase-11 tuning constants:
# - penalty/boost magnitudes control how much BTC/ETH macro context shifts confidence.
# - veto threshold blocks shorts when BTC bullish momentum is exceptionally strong.
# - trend/momentum thresholds define when BTC/ETH are treated as bullish.
_SHORT_PENALTY_BTC_BULLISH = -0.35
_SHORT_PENALTY_BTC_ETH_BULLISH = -0.45
_LONG_BOOST_BTC_BULLISH = 0.08
_LONG_BOOST_BTC_ETH_BULLISH = 0.12
_BTC_STRONG_BULLISH_VETO_THRESHOLD = 0.95
_BULLISH_TREND_THRESHOLD = 0.75
_MOMENTUM_POSITIVE_THRESHOLD = 0.0025
_MOMENTUM_LOOKBACK_PERIODS = 6
_FAST_EMA_SPAN = 9
_SLOW_EMA_SPAN = 21


class MarketGravityAgent(BaseAgent):
    """Macro filter based on BTC/ETH trend alignment."""

    def __init__(self) -> None:
        super().__init__(name="market_gravity", initial_weight=1.0)

    def analyse(self, symbol: str, interval: str, _df=None, direction: str = "neutral") -> AgentResult:
        direction = (direction or "neutral").lower()
        btc_strength, btc_bullish = self._trend_strength("BTCUSDT", interval)
        eth_strength, eth_bullish = self._trend_strength("ETHUSDT", interval)

        adjustment = 0.0
        veto = False
        details = [f"btc_strength={btc_strength:.2f}", f"eth_strength={eth_strength:.2f}"]

        if direction == "short" and btc_bullish:
            adjustment = _SHORT_PENALTY_BTC_BULLISH if not eth_bullish else _SHORT_PENALTY_BTC_ETH_BULLISH
            veto = btc_strength >= _BTC_STRONG_BULLISH_VETO_THRESHOLD
            details.append("short_vs_bullish_btc")
        elif direction == "long" and btc_bullish:
            adjustment = _LONG_BOOST_BTC_BULLISH if not eth_bullish else _LONG_BOOST_BTC_ETH_BULLISH
            details.append("long_with_bullish_btc")
        else:
            details.append("neutral_gravity")

        base_score = (btc_strength + eth_strength) / 2.0
        score = float(np.clip(base_score + adjustment, 0.0, 1.0))
        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=score,
            direction=direction if direction in ("long", "short") else "neutral",
            confidence=float(max(btc_strength, eth_strength)),
            details=details,
            metadata={
                "score_adjustment": adjustment,
                "veto": veto,
                "btc_bullish": btc_bullish,
                "eth_bullish": eth_bullish,
                "btc_strength": btc_strength,
                "eth_strength": eth_strength,
            },
        )

    def _trend_strength(self, symbol: str, interval: str) -> Tuple[float, bool]:
        df = data_store.get_df(symbol, interval)
        if df is None or len(df) < 30 or "close" not in df.columns:
            return 0.0, False

        closes = df["close"].dropna()
        if len(closes) < 30:
            return 0.0, False

        closes = closes.astype(float)
        fast_ema = closes.ewm(span=_FAST_EMA_SPAN, adjust=False).mean().iloc[-1]
        slow_ema = closes.ewm(span=_SLOW_EMA_SPAN, adjust=False).mean().iloc[-1]
        last = closes.iloc[-1]
        ref = (
            closes.iloc[-_MOMENTUM_LOOKBACK_PERIODS]
            if len(closes) > _MOMENTUM_LOOKBACK_PERIODS
            else closes.iloc[0]
        )
        momentum = (last - ref) / max(abs(ref), 1e-9)

        strength = 0.0
        if fast_ema > slow_ema:
            strength += 0.50
        if last > fast_ema:
            strength += 0.25
        if momentum > _MOMENTUM_POSITIVE_THRESHOLD:
            strength += 0.25

        strength = float(np.clip(strength, 0.0, 1.0))
        return strength, strength >= _BULLISH_TREND_THRESHOLD
