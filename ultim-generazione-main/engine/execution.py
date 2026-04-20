"""
Execution Engine for V17.
Paper trading (default, PAPER_TRADING=True) simulates orders and tracks P&L.
Real mode uses Binance Futures futures_create_order() via binance_client.
"""
import logging
import time
import datetime
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from config.settings import (
    PAPER_TRADING,
    ACCOUNT_BALANCE,
    LEVERAGE,
    MAX_DAILY_LOSS_USDT,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
    TRAINING_MODE,
    MAX_CANDLES_IN_TRADE,
    DEAD_TRADE_TIMEOUT_PNL_BAND_PCT,
    DYNAMIC_TRAILING_BREAKEVEN_PCT,
    DYNAMIC_TRAILING_LOCK_PCT,
    DYNAMIC_TRAILING_LOCK_SL_PCT,
)
from data.binance_client import place_futures_order
from memory import experience_db

logger = logging.getLogger("Execution")
_POSITION_SIZE_EPSILON = 1e-8

# Maximum age per interval before a position is force-closed (in seconds)
if TRAINING_MODE:
    _MAX_POSITION_AGE = {"15m": 3600, "1h": 7200, "4h": 14400}  # 1h, 2h, 4h
else:
    _MAX_POSITION_AGE = {"15m": 86400, "1h": 172800, "4h": 259200}  # 1d, 2d, 3d
# Dynamic trailing-stop percentages (distance from current price)
_TRAIL_PCT_AT_TP1 = 0.006
_TRAIL_PCT_AT_TP2 = 0.002
_MIN_TRAIL_DISTANCE = 1e-9
_ENTRY_PRICE_EPSILON = 1e-9
# ATR-based trailing scales from 1.2x ATR near TP1 to 0.6x ATR near TP2.
_ATR_TRAIL_MULT_AT_TP1 = 1.2
_ATR_TRAIL_MULT_AT_TP2 = 0.6
if _TRAIL_PCT_AT_TP2 >= _TRAIL_PCT_AT_TP1:
    raise ValueError("Dynamic trailing requires _TRAIL_PCT_AT_TP2 < _TRAIL_PCT_AT_TP1")
_INTERVAL_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}
_HARD_TIMEOUT_CANDLE_BUFFER = 1
_TRAILING_STAGE_LABELS = {1: "BREAKEVEN", 2: "TRAIL +1%"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents an open (or closed) position."""
    position_id: str
    symbol: str
    interval: str
    direction: str          # "long" | "short"
    entry_price: float
    size: float             # base currency units
    sl: float
    tp1: float
    tp2: float
    strategy: str = ""
    open_time: float = field(default_factory=time.time)
    close_time: Optional[float] = None
    close_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"   # "open" | "closed" | "sl_hit" | "tp1_hit" | "tp2_hit"
    tp1_hit: bool = False
    tp2_hit: bool = False
    realized_pnl: float = 0.0
    initial_size: Optional[float] = None
    initial_sl: Optional[float] = None
    initial_atr: Optional[float] = None
    decision_id: str = ""
    paper: bool = True
    trailing_stage: int = 0

    def __post_init__(self) -> None:
        if self.initial_size is None:
            self.initial_size = self.size
        if self.initial_sl is None:
            self.initial_sl = self.sl

    @property
    def is_tp1_hit(self) -> bool:
        return self.tp1_hit

    @is_tp1_hit.setter
    def is_tp1_hit(self, value: bool) -> None:
        self.tp1_hit = bool(value)

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "size": self.size,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "close_price": self.close_price,
            "pnl": self.pnl,
            "status": self.status,
            "tp1_hit": self.tp1_hit,
            "is_tp1_hit": self.tp1_hit,
            "tp2_hit": self.tp2_hit,
            "initial_size": self.initial_size,
            "initial_sl": self.initial_sl,
            "realized_pnl": self.realized_pnl,
            "paper": self.paper,
            "trailing_stage": self.trailing_stage,
        }


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Handles order placement in paper or live mode."""

    def __init__(self, paper_trading: bool = PAPER_TRADING,
                 initial_balance: float = ACCOUNT_BALANCE):
        self._lock = threading.RLock()
        self.paper_trading = paper_trading
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._open_positions: Dict[str, Position] = {}   # position_id → Position
        self._closed_positions: List[Position] = []
        self._total_pnl = 0.0
        self._trade_count = self._restore_trade_count()
        self._win_count = 0
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._current_day = datetime.datetime.now(datetime.timezone.utc).date()
        logger.info(
            f"ExecutionEngine: {'PAPER' if paper_trading else 'LIVE'} trading | "
            f"balance={initial_balance} | restored_trades={self._trade_count}"
        )

    def _restore_trade_count(self) -> int:
        try:
            return int(experience_db.get_completed_trade_count(paper_only=self.paper_trading))
        except Exception as e:
            logger.warning(f"ExecutionEngine trade counter restore error: {e}")
            return 0
    def _roll_day_if_needed(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self._current_day:
            self._current_day = today
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            logger.info("🔄 Daily risk counters reset")

    def is_risk_blocked(self) -> tuple[bool, str]:
        self._roll_day_if_needed()

        daily_loss_usdt = max(0.0, -self._daily_pnl)
        daily_loss_pct = (
            (daily_loss_usdt / self._initial_balance) * 100
            if self._initial_balance > 0 else 0.0
        )

        if daily_loss_usdt >= MAX_DAILY_LOSS_USDT:
            return True, "max_daily_loss_usdt"

        if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            return True, "max_daily_loss_pct"

        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return True, "max_consecutive_losses"

        return False, ""

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, symbol: str, interval: str, direction: str,
                      entry_price: float, size: float, sl: float,
                      tp1: float, tp2: float, strategy: str = "",
                      decision_id: str = "",
                      initial_atr: Optional[float] = None,
                      force_paper: bool = False) -> Optional[Position]:
        """Open a new position (paper or live)."""
        with self._lock:
            direction = (direction or "").lower()
            if direction not in ("long", "short"):
                logger.error(f"Invalid direction for open_position: {direction}")
                return None
            effective_paper = bool(self.paper_trading or force_paper)

            risk = abs(entry_price - sl)
            if tp1 is None:
                tp1 = entry_price + risk if direction == "long" else entry_price - risk
            if tp2 is None:
                tp2 = entry_price + 2.0 * risk if direction == "long" else entry_price - 2.0 * risk

            pos_id = str(uuid.uuid4())[:8]
            pos = Position(
                position_id=pos_id,
                symbol=symbol,
                interval=interval,
                direction=direction,
                entry_price=entry_price,
                size=size,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                strategy=strategy,
                decision_id=decision_id,
                initial_atr=initial_atr,
                paper=effective_paper,
            )

            if effective_paper:
                self._open_positions[pos_id] = pos
                logger.info(
                    f"📄 PAPER OPEN [{pos_id}] {symbol} {direction.upper()} "
                    f"@ {entry_price:.4f} size={size} sl={sl:.4f} tp1={tp1:.4f}"
                )
            else:
                # Real execution
                side = "BUY" if direction == "long" else "SELL"
                order = place_futures_order(symbol, side, "MARKET", size)
                if order is None:
                    logger.error(f"Failed to open live position for {symbol}")
                    return None
                self._open_positions[pos_id] = pos
                logger.info(f"✅ LIVE OPEN [{pos_id}] {symbol} {direction.upper()} {order}")

            return pos

    def close_position(
        self,
        position_id: str,
        close_price: float,
        reason: str = "manual",
        from_exchange: bool = False,
    ) -> Optional[Position]:
        """Close an open position."""
        with self._lock:
            pos = self._open_positions.pop(position_id, None)
            if pos is None:
                return None

            pos.close_price = close_price
            pos.close_time = time.time()
            pos.pnl = pos.unrealised_pnl(close_price)
            pos.status = reason
            self._roll_day_if_needed()
            self._balance += pos.pnl
            self._total_pnl += pos.pnl
            self._daily_pnl += pos.pnl
            self._trade_count += 1

            if pos.pnl > 0:
                self._win_count += 1
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

            self._closed_positions.append(pos)
            if len(self._closed_positions) > 1000:
                del self._closed_positions[:100]
            emoji = "✅" if pos.pnl > 0 else "❌"
            logger.info(
                f"{emoji} {'PAPER ' if pos.paper else ''}CLOSE [{position_id}] "
                f"{pos.symbol} {pos.direction.upper()} @ {close_price:.4f} "
                f"PnL={pos.pnl:+.4f} ({reason})"
            )

            if not self.paper_trading and not pos.paper and not from_exchange:
                side = "SELL" if pos.direction == "long" else "BUY"
                close_order = place_futures_order(pos.symbol, side, "MARKET", pos.size, reduce_only=True)
                if close_order is None:
                    logger.warning(f"Failed to place reduce-only close order for {pos.symbol}")

            return pos

    def process_user_stream_event(self, event: Dict[str, Any]) -> List[Position]:
        """Apply Binance User Data Stream events to local live-position state."""
        if self.paper_trading or not isinstance(event, dict):
            return []

        event_type = str(event.get("e", ""))
        if event_type == "ORDER_TRADE_UPDATE":
            return self._apply_order_trade_update(event.get("o") or {})
        if event_type == "ACCOUNT_UPDATE":
            return self._apply_account_update(event.get("a") or {})
        return []

    def _apply_order_trade_update(self, order: Dict[str, Any]) -> List[Position]:
        status = str(order.get("X", ""))
        execution_type = str(order.get("x", ""))
        if status != "FILLED" or execution_type != "TRADE":
            return []

        symbol = str(order.get("s", "")).upper()
        side = str(order.get("S", "")).upper()
        if not symbol or side not in {"BUY", "SELL"}:
            return []

        try:
            filled_qty = float(order.get("l") or order.get("z") or 0.0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        try:
            fill_price = float(order.get("ap") or order.get("L") or 0.0)
        except (TypeError, ValueError):
            fill_price = 0.0

        order_type = str(order.get("o", ""))
        reduce_only = bool(order.get("R"))
        if not reduce_only:
            return []

        reason = "exchange_fill"
        if "STOP" in order_type:
            reason = "sl_hit"
        elif "TAKE_PROFIT" in order_type:
            reason = "tp2_hit"

        return self._apply_external_fill(
            symbol=symbol,
            side=side,
            quantity=filled_qty,
            fill_price=fill_price,
            reason=reason,
            order_type=order_type,
        )

    def _apply_account_update(self, account_update: Dict[str, Any]) -> List[Position]:
        closed_positions: List[Position] = []
        positions = account_update.get("P") or []
        if not isinstance(positions, list):
            return closed_positions

        for item in positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("s", "")).upper()
            if not symbol:
                continue
            try:
                amount = float(item.get("pa", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(amount) > _POSITION_SIZE_EPSILON:
                continue
            try:
                event_price = float(item.get("ep") or 0.0)
            except (TypeError, ValueError):
                event_price = 0.0

            with self._lock:
                symbol_positions = [
                    (pos_id, pos)
                    for pos_id, pos in self._open_positions.items()
                    if pos.symbol.upper() == symbol
                ]
                for pos_id, pos in symbol_positions:
                    close_price = event_price if event_price > 0 else pos.entry_price
                    closed = self.close_position(
                        pos_id,
                        close_price=close_price,
                        reason="account_update_flat",
                        from_exchange=True,
                    )
                    if closed is not None:
                        closed_positions.append(closed)
        return closed_positions

    def _apply_external_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        reason: str,
        order_type: str,
    ) -> List[Position]:
        closed_positions: List[Position] = []
        with self._lock:
            candidates = [
                (pos_id, pos)
                for pos_id, pos in self._open_positions.items()
                if pos.symbol.upper() == symbol
                and ((pos.direction == "long" and side == "SELL") or (pos.direction == "short" and side == "BUY"))
            ]

        if not candidates:
            return closed_positions

        remaining = max(float(quantity or 0.0), 0.0)
        close_all_matching = remaining <= 0.0
        for pos_id, pos in candidates:
            close_qty = pos.size if close_all_matching else min(pos.size, remaining)
            if close_qty <= 0:
                continue

            close_px = fill_price if fill_price > 0 else pos.entry_price
            is_full_close = close_qty >= (pos.size - _POSITION_SIZE_EPSILON)
            if is_full_close:
                closed = self.close_position(
                    pos_id,
                    close_price=close_px,
                    reason=reason,
                    from_exchange=True,
                )
                if closed is not None:
                    closed_positions.append(closed)
            else:
                with self._lock:
                    if pos.direction == "long":
                        partial_pnl = (close_px - pos.entry_price) * close_qty
                    else:
                        partial_pnl = (pos.entry_price - close_px) * close_qty
                    pos.realized_pnl += partial_pnl
                    self._roll_day_if_needed()
                    self._balance += partial_pnl
                    self._total_pnl += partial_pnl
                    self._daily_pnl += partial_pnl
                    pos.size = max(pos.size - close_qty, 0.0)
                    if "TAKE_PROFIT" in order_type and not pos.tp1_hit:
                        pos.tp1_hit = True
                        pos.sl = pos.entry_price
                logger.info(
                    f"🎯 LIVE PARTIAL CLOSE [{pos.position_id}] {pos.symbol} "
                    f"qty={close_qty:.6f} @ {close_px:.4f} "
                    f"PnL={partial_pnl:+.4f} ({order_type})"
                )

            if not close_all_matching:
                remaining -= close_qty
                if remaining <= _POSITION_SIZE_EPSILON:
                    break
        return closed_positions

    def check_position_levels(self, symbol: str, current_price: float) -> List[Position]:
        """Check all open positions for SL/TP hits and return closed positions."""
        with self._lock:
            self._roll_day_if_needed()
            to_close: List[Tuple[str, float, str]] = []
            closed_positions: List[Position] = []

            for pos_id, pos in list(self._open_positions.items()):
                if pos.symbol != symbol:
                    continue

                now_ts = time.time()
                self._apply_phase10_dynamic_trailing(pos, current_price)

                # Dead-trade timeout by candles: close only if the trade is stuck in a narrow range.
                candles_elapsed = (now_ts - pos.open_time) / self._interval_seconds(pos.interval)
                pnl_pct = self._position_profit_pct(pos, current_price)
                if candles_elapsed > MAX_CANDLES_IN_TRADE and abs(pnl_pct) <= DEAD_TRADE_TIMEOUT_PNL_BAND_PCT:
                    logger.info(
                        f"⏳ DEAD-TRADE TIMEOUT [{pos_id}] {pos.symbol}/{pos.interval} — "
                        f"candles={candles_elapsed:.1f} pnl={pnl_pct:+.2f}%"
                    )
                    to_close.append((pos_id, current_price, "timeout_dead_trade"))
                    continue

                # Hard timeout fallback for very old positions.
                min_timeout = self._interval_seconds(pos.interval) * (
                    MAX_CANDLES_IN_TRADE + _HARD_TIMEOUT_CANDLE_BUFFER
                )
                max_age = max(_MAX_POSITION_AGE.get(pos.interval, 172800), min_timeout)
                if now_ts - pos.open_time > max_age:
                    logger.info(
                        f"⏰ TIMEOUT [{pos_id}] {pos.symbol}/{pos.interval} — "
                        f"open for >{max_age}s, closing at {current_price:.4f}"
                    )
                    to_close.append((pos_id, current_price, "timeout"))
                    continue

                if pos.direction == "long":
                    if current_price <= pos.sl:
                        to_close.append((pos_id, current_price, "sl_hit"))
                    elif not pos.tp1_hit and current_price >= pos.tp1:
                        close_size = pos.size * 0.5
                        close_pnl = (current_price - pos.entry_price) * close_size
                        self._execute_tp1_scale_out(pos, close_size, close_pnl)
                        pos.tp1_hit = True
                        pos.is_tp1_hit = True
                        pos.sl = pos.entry_price
                        logger.info(
                            f"🎯 TP1 hit [{pos_id}] {pos.symbol} — 50% closed for profit "
                            f"(PnL={close_pnl:+.4f}), SL to breakeven"
                        )
                        if not self.paper_trading and not pos.paper:
                            partial_order = place_futures_order(
                                pos.symbol, "SELL", "MARKET", close_size, reduce_only=True
                            )
                            if partial_order is None:
                                logger.warning(f"Failed to place TP1 partial reduce-only order for {pos.symbol}")
                    elif pos.tp1_hit and not pos.tp2_hit and current_price >= pos.tp2:
                        pos.tp2_hit = True
                        to_close.append((pos_id, current_price, "tp2_hit"))
                    elif pos.tp1_hit and not pos.tp2_hit:
                        trail_distance = self._dynamic_trail_distance(pos, current_price)
                        new_sl = current_price - trail_distance
                        if new_sl > pos.sl:
                            pos.sl = new_sl
                            logger.debug(f"📈 Trail SL [{pos_id}] {pos.symbol} → {new_sl:.4f}")
                else:
                    if current_price >= pos.sl:
                        to_close.append((pos_id, current_price, "sl_hit"))
                    elif not pos.tp1_hit and current_price <= pos.tp1:
                        close_size = pos.size * 0.5
                        close_pnl = (pos.entry_price - current_price) * close_size
                        self._execute_tp1_scale_out(pos, close_size, close_pnl)
                        pos.tp1_hit = True
                        pos.is_tp1_hit = True
                        pos.sl = pos.entry_price
                        logger.info(
                            f"🎯 TP1 hit [{pos_id}] {pos.symbol} — 50% closed for profit "
                            f"(PnL={close_pnl:+.4f}), SL to breakeven"
                        )
                        if not self.paper_trading and not pos.paper:
                            partial_order = place_futures_order(
                                pos.symbol, "BUY", "MARKET", close_size, reduce_only=True
                            )
                            if partial_order is None:
                                logger.warning(f"Failed to place TP1 partial reduce-only order for {pos.symbol}")
                    elif pos.tp1_hit and not pos.tp2_hit and current_price <= pos.tp2:
                        pos.tp2_hit = True
                        to_close.append((pos_id, current_price, "tp2_hit"))
                    elif pos.tp1_hit and not pos.tp2_hit:
                        trail_distance = self._dynamic_trail_distance(pos, current_price)
                        new_sl = current_price + trail_distance
                        if new_sl < pos.sl:
                            pos.sl = new_sl
                            logger.debug(f"📉 Trail SL [{pos_id}] {pos.symbol} → {new_sl:.4f}")

            for pos_id, price, reason in to_close:
                closed = self.close_position(pos_id, price, reason)
                if closed is not None:
                    closed_positions.append(closed)

            return closed_positions

    @staticmethod
    def _interval_seconds(interval: str) -> int:
        raw = str(interval or "").strip().lower()
        if raw in _INTERVAL_SECONDS:
            return int(_INTERVAL_SECONDS[raw])
        if len(raw) >= 2 and raw[:-1].isdigit():
            value = int(raw[:-1])
            unit = raw[-1]
            if unit == "m":
                return value * 60
            if unit == "h":
                return value * 3600
        logger.warning(
            f"Unknown interval '{interval}' in ExecutionEngine; falling back to 3600s (1h). "
            "Please verify interval format (e.g., 15m, 1h, 4h) or add mapping to _INTERVAL_SECONDS."
        )
        return 3600

    @staticmethod
    def _position_profit_pct(pos: Position, current_price: float) -> float:
        if abs(pos.entry_price) < _ENTRY_PRICE_EPSILON:
            logger.warning(
                f"Invalid entry_price for position [{pos.position_id}] {pos.symbol}: "
                f"{pos.entry_price}. Returning 0.0%% PnL."
            )
            return 0.0
        if pos.direction == "long":
            return ((current_price - pos.entry_price) / pos.entry_price) * 100.0
        return ((pos.entry_price - current_price) / pos.entry_price) * 100.0

    def _apply_phase10_dynamic_trailing(self, pos: Position, current_price: float) -> None:
        """Phase 10 dynamic SL progression: breakeven at +1%, lock +1% at +2%."""
        pnl_pct = self._position_profit_pct(pos, current_price)
        target_sl: Optional[float] = None
        stage = pos.trailing_stage

        if pnl_pct >= DYNAMIC_TRAILING_LOCK_PCT:
            lock_mult = DYNAMIC_TRAILING_LOCK_SL_PCT / 100.0
            target_sl = (
                pos.entry_price * (1.0 + lock_mult)
                if pos.direction == "long"
                else pos.entry_price * (1.0 - lock_mult)
            )
            stage = 2
        elif pnl_pct >= DYNAMIC_TRAILING_BREAKEVEN_PCT:
            target_sl = pos.entry_price
            stage = 1

        if target_sl is None:
            return

        moved = False
        if pos.direction == "long" and target_sl > pos.sl:
            pos.sl = target_sl
            moved = True
        elif pos.direction == "short" and target_sl < pos.sl:
            pos.sl = target_sl
            moved = True

        if moved and stage > pos.trailing_stage:
            pos.trailing_stage = stage
            label = _TRAILING_STAGE_LABELS.get(stage, f"STAGE {stage}")
            logger.info(
                f"🛡️ DYNAMIC SL [{pos.position_id}] {pos.symbol} "
                f"stage={label} pnl={pnl_pct:+.2f}% -> SL={pos.sl:.4f}"
            )

    def _dynamic_trail_distance(self, pos: Position, current_price: float) -> float:
        """Return trailing distance for post-TP1 management.

        Args:
            pos: Active position with TP1/TP2 and direction metadata.
            current_price: Latest market price for the position symbol.

        Returns:
            Positive trailing distance (price units). The distance tightens
            linearly from `_TRAIL_PCT_AT_TP1` to `_TRAIL_PCT_AT_TP2` as price
            progresses from TP1 toward TP2.
        """
        move_total = abs(pos.tp2 - pos.tp1)
        if move_total == 0:
            logger.warning(
                f"Invalid TP configuration for dynamic trailing [{pos.position_id}] "
                f"{pos.symbol}: tp1={pos.tp1:.4f}, tp2={pos.tp2:.4f}"
            )
            # Fallback to "TP1 stage" behavior (widest trail) when TP geometry is invalid.
            progress = 0.0
            trail_pct = _TRAIL_PCT_AT_TP1
        else:
            if pos.direction == "long":
                progress = max(0.0, min(1.0, (current_price - pos.tp1) / move_total))
            else:
                progress = max(0.0, min(1.0, (pos.tp1 - current_price) / move_total))
            trail_pct = _TRAIL_PCT_AT_TP1 + (
                (_TRAIL_PCT_AT_TP2 - _TRAIL_PCT_AT_TP1) * progress
            )

        # ATR-adaptive component:
        # widen stop when initial ATR is high; tighten as price progresses to TP2.
        atr_ref = float(abs(pos.initial_atr or 0.0))
        if atr_ref <= 0.0:
            atr_ref = abs(pos.entry_price - pos.initial_sl)
            logger.debug(
                f"ATR fallback for trailing [{pos.position_id}] {pos.symbol}: "
                f"using entry/SL distance {atr_ref:.6f}"
            )
        atr_mult = _ATR_TRAIL_MULT_AT_TP1 + (
            (_ATR_TRAIL_MULT_AT_TP2 - _ATR_TRAIL_MULT_AT_TP1) * progress
        )
        atr_distance = atr_ref * atr_mult
        pct_distance = current_price * trail_pct
        return max((atr_distance + pct_distance) * 0.5, _MIN_TRAIL_DISTANCE)

    def _execute_tp1_scale_out(self, pos: Position, close_size: float, close_pnl: float) -> None:
        """Apply TP1 partial close accounting and reduce open size.

        Args:
            pos: Position being partially reduced at TP1.
            close_size: Quantity closed at TP1 (50% of current position size).
            close_pnl: Realized PnL for the closed quantity based on trigger price.
        """
        pos.realized_pnl += close_pnl
        self._balance += close_pnl
        self._total_pnl += close_pnl
        self._daily_pnl += close_pnl
        pos.size -= close_size

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            self._roll_day_if_needed()
            risk_blocked, risk_reason = self.is_risk_blocked()

            return {
                "paper_trading": self.paper_trading,
                "balance": self._balance,
                "initial_balance": self._initial_balance,
                "total_pnl": self._total_pnl,
                "pnl_pct": self._total_pnl / self._initial_balance * 100,
                "trade_count": self._trade_count,
                "win_count": self._win_count,
                "win_rate": self._win_count / max(self._trade_count, 1),
                "open_positions": len(self._open_positions),
                "daily_pnl": self._daily_pnl,
                "consecutive_losses": self._consecutive_losses,
                "risk_blocked": risk_blocked,
                "risk_block_reason": risk_reason,
            }

    def get_open_positions(self) -> List[Position]:
        with self._lock:
            return list(self._open_positions.values())

    def get_closed_positions(self, limit: int = 50) -> List[Position]:
        with self._lock:
            return self._closed_positions[-limit:]
