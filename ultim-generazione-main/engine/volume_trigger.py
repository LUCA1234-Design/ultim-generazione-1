"""
Micro-momentum trigger for final execution confirmation.
"""
import logging
from typing import Tuple, Dict, Any

from data.binance_client import fetch_futures_depth

logger = logging.getLogger("VolumeTrigger")


class VolumeTrigger:
    """Confirm directional micro-pressure from immediate order book depth."""

    def __init__(self, depth_limit: int = 10, min_imbalance: float = 0.05, timeout_sec: float = 0.35):
        self.depth_limit = max(5, int(depth_limit))
        self.min_imbalance = max(0.0, float(min_imbalance))
        self.timeout_sec = max(0.05, float(timeout_sec))

    def confirm(self, symbol: str, direction: str) -> Tuple[bool, Dict[str, Any]]:
        direction = (direction or "").lower()
        if direction not in ("long", "short"):
            return False, {"reason": "invalid_direction"}

        depth = fetch_futures_depth(symbol, limit=self.depth_limit, timeout=self.timeout_sec)
        bids = depth.get("bids", []) if isinstance(depth, dict) else []
        asks = depth.get("asks", []) if isinstance(depth, dict) else []

        bid_qty = self._sum_levels_qty(bids)
        ask_qty = self._sum_levels_qty(asks)
        total = bid_qty + ask_qty
        if total <= 0:
            return False, {"reason": "empty_depth", "bid_qty": bid_qty, "ask_qty": ask_qty}

        imbalance = (bid_qty - ask_qty) / total
        is_confirmed = (
            imbalance >= self.min_imbalance
            if direction == "long"
            else imbalance <= -self.min_imbalance
        )
        return is_confirmed, {
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "imbalance": imbalance,
            "min_imbalance": self.min_imbalance,
        }

    @staticmethod
    def _sum_levels_qty(levels) -> float:
        total = 0.0
        for lvl in levels:
            try:
                total += float(lvl[1])
            except Exception:
                continue
        return total
