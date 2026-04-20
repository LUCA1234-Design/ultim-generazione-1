"""
On-Chain Whale Tracker agent (Phase 14).
Tracks large transfers to/from exchanges and derives directional pressure.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from agents.base_agent import BaseAgent, AgentResult
from config.settings import (
    ONCHAIN_TRACKING_ENABLED,
    ONCHAIN_WHALE_ALERT_API_KEY,
    ONCHAIN_ALERT_MIN_USD,
    ONCHAIN_CACHE_TTL_SECONDS,
    ONCHAIN_EXCHANGE_LABELS,
    ONCHAIN_STABLECOINS,
)

logger = logging.getLogger("OnChainAgent")


class OnChainAgent(BaseAgent):
    """Detects whale transfers and estimates pre-book directional pressure."""

    def __init__(self):
        super().__init__("onchain", initial_weight=0.12)
        self._cache: Dict[str, Tuple[float, AgentResult]] = {}

    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        upper = (symbol or "").upper()
        for quote in ("USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH"):
            if upper.endswith(quote):
                return upper[: -len(quote)] or upper
        return upper

    @staticmethod
    def _normalize_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        token = str(raw.get("symbol") or raw.get("token") or "").upper()
        amount_usd = raw.get("amount_usd", raw.get("usd", raw.get("amount", 0.0)))
        from_owner = str(raw.get("from_owner") or raw.get("from", "")).lower()
        to_owner = str(raw.get("to_owner") or raw.get("to", "")).lower()
        exchange_like = any(lbl in from_owner or lbl in to_owner for lbl in ONCHAIN_EXCHANGE_LABELS)
        if not exchange_like:
            return None
        try:
            amount_usd = float(amount_usd)
        except (TypeError, ValueError):
            return None
        if amount_usd < ONCHAIN_ALERT_MIN_USD:
            return None
        direction = "to_exchange" if any(lbl in to_owner for lbl in ONCHAIN_EXCHANGE_LABELS) else "from_exchange"
        return {
            "token": token,
            "amount_usd": amount_usd,
            "direction": direction,
            "from": from_owner,
            "to": to_owner,
        }

    def _fetch_whale_events(self, symbol: str) -> List[Dict[str, Any]]:
        if not ONCHAIN_WHALE_ALERT_API_KEY:
            return []
        try:
            resp = requests.get(
                "https://api.whale-alert.io/v1/transactions",
                params={
                    "api_key": ONCHAIN_WHALE_ALERT_API_KEY,
                    "min_value": int(ONCHAIN_ALERT_MIN_USD),
                    "currency": self._extract_base_asset(symbol).lower(),
                },
                timeout=2.5,
            )
            payload = resp.json() if resp.ok else {}
            txs = payload.get("transactions", []) if isinstance(payload, dict) else []
            events: List[Dict[str, Any]] = []
            for tx in txs:
                norm = self._normalize_event(
                    {
                        "symbol": tx.get("symbol"),
                        "amount_usd": tx.get("amount_usd"),
                        "from_owner": (tx.get("from") or {}).get("owner"),
                        "to_owner": (tx.get("to") or {}).get("owner"),
                    }
                )
                if norm is not None:
                    events.append(norm)
            return events
        except Exception as exc:
            logger.debug(f"whale fetch failed: {exc}")
            return []

    def _score_events(self, symbol: str, events: List[Dict[str, Any]]) -> Tuple[float, str, float]:
        if not events:
            return 0.0, "neutral", 0.2

        base = self._extract_base_asset(symbol)
        stable = set(ONCHAIN_STABLECOINS)
        bullish_pressure = 0.0
        bearish_pressure = 0.0

        for ev in events:
            token = str(ev.get("token", "")).upper()
            amount_usd = float(ev.get("amount_usd", 0.0))
            scale = float(np.clip(amount_usd / (ONCHAIN_ALERT_MIN_USD * 4.0), 0.0, 3.0))
            direction = ev.get("direction")

            if token in stable:
                if direction == "to_exchange":
                    bullish_pressure += scale
                else:
                    bearish_pressure += scale
            elif token == base:
                if direction == "to_exchange":
                    bearish_pressure += scale
                else:
                    bullish_pressure += scale

        delta = bullish_pressure - bearish_pressure
        confidence = float(np.clip(abs(delta) / 3.0, 0.2, 0.95))
        if delta > 0.15:
            return float(np.clip(0.50 + delta * 0.25, 0.0, 1.0)), "long", confidence
        if delta < -0.15:
            return float(np.clip(0.50 + abs(delta) * 0.25, 0.0, 1.0)), "short", confidence
        return 0.5, "neutral", confidence

    def analyse(self, symbol: str, interval: str, df, *args, **kwargs) -> Optional[AgentResult]:
        if not ONCHAIN_TRACKING_ENABLED:
            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=0.5,
                direction="neutral",
                confidence=0.2,
                details=["onchain:disabled"],
                metadata={"alerts": []},
            )

        now = time.time()
        cache_key = f"{symbol}:{interval}"
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]) < ONCHAIN_CACHE_TTL_SECONDS:
            return cached[1]

        events = self._fetch_whale_events(symbol)
        score, direction, confidence = self._score_events(symbol, events)
        if events:
            top = sorted(events, key=lambda x: x.get("amount_usd", 0.0), reverse=True)[0]
            alert = (
                f"whale:{top.get('direction')} {top.get('token')} "
                f"${float(top.get('amount_usd', 0.0)):,.0f}"
            )
            details = [alert]
        elif ONCHAIN_WHALE_ALERT_API_KEY:
            details = ["onchain:no_significant_transfers"]
        else:
            details = ["onchain:placeholder_no_api_key"]

        result = AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(score, 0.0, 1.0)),
            direction=direction,
            confidence=confidence,
            details=details,
            metadata={
                "alerts": events[:5],
                "alert_count": len(events),
                "has_api_key": bool(ONCHAIN_WHALE_ALERT_API_KEY),
            },
        )
        self._cache[cache_key] = (now, result)
        return result
