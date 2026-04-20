"""
Sector rotation / money-flow heatmap (Phase 12).
Computes sector-level momentum and returns a confluence score adjustment.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from agents.base_agent import BaseAgent, AgentResult
from data import data_store

SECTOR_MAP: Dict[str, str] = {
    "BTCUSDT": "majors",
    "ETHUSDT": "majors",
    "BNBUSDT": "majors",
    "SOLUSDT": "layer1",
    "AVAXUSDT": "layer1",
    "NEARUSDT": "layer1",
    "SUIUSDT": "layer1",
    "APTUSDT": "layer1",
    "DOGEUSDT": "meme",
    "SHIBUSDT": "meme",
    "PEPEUSDT": "meme",
    "WIFUSDT": "meme",
    "FLOKIUSDT": "meme",
    "FETUSDT": "ai",
    "TAOUSDT": "ai",
    "RNDRUSDT": "ai",
    "AGIXUSDT": "ai",
    "INJUSDT": "defi",
    "AAVEUSDT": "defi",
    "UNIUSDT": "defi",
    "LINKUSDT": "oracle",
}


class SectorRotationAgent(BaseAgent):
    """Sector momentum detector returning confluence boost/penalty."""

    def __init__(self):
        super().__init__("sector_rotation", initial_weight=0.08)

    @staticmethod
    def _symbol_sector(symbol: str) -> str:
        return SECTOR_MAP.get(str(symbol or "").upper(), "other")

    @staticmethod
    def _sector_symbols(sector: str) -> List[str]:
        if sector == "other":
            return []
        return [sym for sym, sec in SECTOR_MAP.items() if sec == sector]

    def _sector_flow_score(self, sector: str, interval: str) -> float:
        peers = self._sector_symbols(sector)
        if not peers:
            return 0.0
        values: List[float] = []
        for sym in peers:
            df = data_store.get_df(sym, interval)
            if df is None or len(df) < 25:
                continue
            close = df["close"]
            volume = df["volume"]
            ret = float((close.iloc[-1] / close.iloc[-5]) - 1.0)
            vol_now = float(volume.iloc[-1])
            vol_avg = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            if not np.isfinite(ret) or not np.isfinite(vol_now) or not np.isfinite(vol_avg):
                continue
            vol_ratio = (vol_now / vol_avg) if vol_avg > 0 else 1.0
            candidate = ret * np.clip(vol_ratio, 0.5, 3.0)
            if np.isfinite(candidate):
                values.append(float(candidate))
        if not values:
            return 0.0
        return float(np.clip(np.mean(values), -0.20, 0.20))

    def analyse(self, symbol: str, interval: str, _df=None, direction: str = "neutral") -> AgentResult:
        sector = self._symbol_sector(symbol)
        flow = self._sector_flow_score(sector, interval)
        adjustment = float(np.clip(flow * 1.5, -0.12, 0.12))
        hot = adjustment >= 0.03
        cold = adjustment <= -0.03
        direction = str(direction or "neutral").lower()
        if direction == "short":
            adjustment *= -1.0

        details = [f"sector={sector}", f"flow={flow:+.4f}", f"adj={adjustment:+.3f}"]
        if hot:
            details.append("hot_sector")
        elif cold:
            details.append("dead_sector")

        confidence = min(1.0, abs(adjustment) / 0.12) if adjustment != 0 else 0.0
        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(0.5 + adjustment, 0.0, 1.0)),
            direction=direction if direction in {"long", "short"} else "neutral",
            confidence=float(confidence),
            details=details,
            metadata={
                "sector": sector,
                "flow_score": float(flow),
                "confluence_adjustment": float(adjustment),
            },
        )
