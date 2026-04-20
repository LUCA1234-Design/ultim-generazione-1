"""
Statistical arbitrage / pairs trading agent (Phase 13).
Generates delta-neutral LONG/SHORT pair signals using spread Z-score.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from agents.base_agent import AgentResult, BaseAgent
from config.settings import (
    PAIRS_TRADING_CANDIDATE_PAIRS,
    PAIRS_TRADING_COOLDOWN_SECONDS,
    PAIRS_TRADING_ENABLED,
    PAIRS_TRADING_INTERVALS,
    PAIRS_TRADING_LOOKBACK,
    PAIRS_TRADING_MIN_CORRELATION,
    PAIRS_TRADING_ZSCORE_ENTRY,
)
from data import data_store


class PairsTradingAgent(BaseAgent):
    """Detects mean-reversion opportunities across configured crypto pairs."""

    def __init__(self) -> None:
        super().__init__("pairs_trading", initial_weight=0.0)
        self._last_signal_ts: Dict[str, float] = {}
        self._pair_map: Dict[str, list[Tuple[str, str]]] = {}
        for left, right in PAIRS_TRADING_CANDIDATE_PAIRS:
            l = str(left or "").upper()
            r = str(right or "").upper()
            if not l or not r or l == r:
                continue
            self._pair_map.setdefault(l, []).append((l, r))
            self._pair_map.setdefault(r, []).append((l, r))

    @staticmethod
    def _as_close(symbol: str, interval: str, lookback: int):
        df = data_store.get_df(symbol, interval)
        if df is None or df.empty or "close" not in df.columns:
            return None
        close = df["close"].dropna().astype(float)
        if len(close) < lookback:
            return None
        return close.iloc[-lookback:]

    def analyse(self, symbol: str, interval: str, _df=None) -> AgentResult | None:
        if not PAIRS_TRADING_ENABLED:
            return None
        interval = str(interval or "")
        if interval not in set(PAIRS_TRADING_INTERVALS):
            return None

        symbol = str(symbol or "").upper()
        for left, right in self._pair_map.get(symbol, []):
            key = f"{left}:{right}:{interval}"
            if time.time() - self._last_signal_ts.get(key, 0.0) < PAIRS_TRADING_COOLDOWN_SECONDS:
                continue

            close_l = self._as_close(left, interval, PAIRS_TRADING_LOOKBACK)
            close_r = self._as_close(right, interval, PAIRS_TRADING_LOOKBACK)
            if close_l is None or close_r is None:
                continue

            n = min(len(close_l), len(close_r))
            x = np.log(close_l.iloc[-n:].to_numpy(dtype=float))
            y = np.log(close_r.iloc[-n:].to_numpy(dtype=float))
            if n < 30:
                continue

            ret_x = np.diff(x)
            ret_y = np.diff(y)
            if ret_x.size < 10 or ret_y.size < 10:
                continue
            corr = float(np.corrcoef(ret_x, ret_y)[0, 1])
            if not np.isfinite(corr) or corr < PAIRS_TRADING_MIN_CORRELATION:
                continue

            x_var = float(np.var(x))
            if x_var <= 1e-12:
                continue
            hedge_ratio = float(np.cov(y, x)[0, 1] / x_var)
            spread = y - hedge_ratio * x
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread))
            if spread_std <= 1e-12:
                continue
            zscore = float((spread[-1] - spread_mean) / spread_std)

            if abs(zscore) < PAIRS_TRADING_ZSCORE_ENTRY:
                continue

            if zscore > 0:
                long_symbol, short_symbol = left, right
            else:
                long_symbol, short_symbol = right, left

            self._last_signal_ts[key] = time.time()
            confidence = float(np.clip(abs(zscore) / (PAIRS_TRADING_ZSCORE_ENTRY * 2.0), 0.0, 1.0))
            return AgentResult(
                agent_name=self.name,
                symbol=f"{left}/{right}",
                interval=interval,
                score=confidence,
                direction="neutral",
                confidence=confidence,
                details=[
                    f"pair={left}/{right}",
                    f"z={zscore:+.2f}",
                    f"corr={corr:.2f}",
                    f"long={long_symbol}",
                    f"short={short_symbol}",
                ],
                metadata={
                    "signal_type": "delta_neutral_pairs",
                    "pair": [left, right],
                    "long_symbol": long_symbol,
                    "short_symbol": short_symbol,
                    "zscore": zscore,
                    "correlation": corr,
                    "hedge_ratio": hedge_ratio,
                    "spread": float(spread[-1]),
                },
            )
        return None
