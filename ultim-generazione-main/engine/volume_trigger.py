"""
Micro-momentum trigger for final execution confirmation.
"""
import logging
from typing import Tuple, Dict, Any

from data.binance_client import fetch_futures_depth

logger = logging.getLogger("VolumeTrigger")
_MIN_DEPTH_LIMIT = 5
_MIN_TIMEOUT_SEC = 0.05


class VolumeTrigger:
    """Confirm directional micro-pressure from immediate order book depth."""

    def __init__(self, depth_limit: int = 10, min_imbalance: float = 0.05, timeout_sec: float = 0.35):
        """Configure micro-momentum checks.

        Args:
            depth_limit: Number of depth levels per side to aggregate.
            min_imbalance: Minimum normalized imbalance to confirm momentum,
                computed as (bid_qty - ask_qty) / (bid_qty + ask_qty).
            timeout_sec: Max REST wait per check to keep execution responsive.
        """
        self.depth_limit = max(_MIN_DEPTH_LIMIT, int(depth_limit))
        self.min_imbalance = max(0.0, float(min_imbalance))
        self.timeout_sec = max(_MIN_TIMEOUT_SEC, float(timeout_sec))

    def confirm(self, symbol: str, direction: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate immediate book pressure for a directional signal.

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True when micro-momentum confirms the direction.
                - dict: Diagnostics (imbalance/quantities on success, reason on failure).
        """
        direction = (direction or "").lower()
        if direction not in ("long", "short"):
            return False, {"reason": "invalid_direction"}

        depth = fetch_futures_depth(symbol, limit=self.depth_limit, timeout=self.timeout_sec)
        if not isinstance(depth, dict) or not depth:
            logger.debug(f"{symbol} volume trigger depth unavailable: {depth}")
            return False, {"reason": "depth_unavailable"}
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])

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
