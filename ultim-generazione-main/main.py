"""
V18 Agentic AI Trading System — Main Orchestrator
Transforms V16 "Cecchino Istituzionale" into a multi-agent adaptive system.
"""
import gc
import logging
from queue import Empty, Queue
import sys
import threading
import time
from typing import Dict, Any, Optional, List

# ---- Config ----
from config.settings import (
    PAPER_TRADING, SIGNAL_ONLY, ACCOUNT_BALANCE, HG_ENABLED, HG_MONITOR_ALL,
    HG_MIN_QUOTE_VOL, SYMBOLS_LIMIT, TELEGRAM_TEST_ON_START,
    STARTUP_TIMEOUT, POLL_CLOSED_ENABLE, DB_PATH,
    HEARTBEAT_INTERVAL, HEARTBEAT_ENABLED,
    TRAINING_MODE, TRAINING_TARGET_TRADES,
    SNIPER_FUSION_THRESHOLD, SNIPER_MIN_FUSION_SCORE,
    SNIPER_MIN_AGENT_CONFIRMATIONS, SNIPER_MIN_RR,
    SNIPER_NON_OPTIMAL_HOUR_PENALTY, SNIPER_SIGNAL_COOLDOWN_BY_TF,
    SNIPER_MAX_OPEN_POSITIONS,
    SENTIMENT_ENABLED, SENTIMENT_UPDATE_INTERVAL_SECONDS, SENTIMENT_TTL_SECONDS,
    CRYPTO_PANIC_API_KEY, LM_STUDIO_URL, LM_STUDIO_MODEL,
)
import config.settings as _cfg  # Used for runtime threshold updates on Training → Sniper switch

# ---- Data layer ----
from data import data_store
from data.binance_client import get_client, fetch_futures_klines, fetch_exchange_info, fetch_futures_ticker
from data.websocket_manager import (
    start_websockets, startup_health_check, start_rest_fallback,
    register_callbacks,
)
from data.user_data_stream import UserDataStreamManager

# ---- Indicators (imported so available globally) ----
import indicators.technical  # noqa
import indicators.smart_money  # noqa

# ---- Agents ----
from agents.pattern_agent import PatternAgent
from agents.regime_agent import RegimeAgent
from agents.confluence_agent import ConfluenceAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.meta_agent import MetaAgent
from agents.sentiment_agent import SentimentAgent
from agents.smc_agent import SMCAgent
from agents.sector_rotation_agent import SectorRotationAgent
from agents.pairs_trading_agent import PairsTradingAgent
from agents.onchain_agent import OnChainAgent
from agents.neural_predict_agent import NeuralPredictAgent

# ---- Engine ----
from engine.decision_fusion import DecisionFusion
from engine.execution import ExecutionEngine
from engine.event_processor import EventProcessor

# ---- Memory ----
from memory import experience_db
from memory.performance_tracker import PerformanceTracker

# ---- Evolution engine ----
from evolution.evolution_engine import EvolutionEngine

# ---- Services ----
from services.notification_worker import (
    start_notification_worker,
    enqueue_signal_notification,
    enqueue_pairs_signal_notification,
)

# ---- Notifications ----
from notifications.telegram_service import (
    send_message,
    test_connection,
    build_startup_message,
    build_stats_message,
    build_heartbeat_message,
    notify_position_closed,
    send_early_exit_alert,
)
from dashboard.app import DashboardState, start_dashboard_server


# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v17.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Main")

# ---------------------------------------------------------------------------
# Global symbol universe
# ---------------------------------------------------------------------------

symbols_whitelist = []    # Top N by volume (for divergence/pattern scanning)
symbols_hg_all = []       # All USDT-M perpetual futures (for HG scan)

# Global flag: True once Training Mode has completed and Sniper Mode is active
_sniper_mode_active: bool = False

def load_top_symbols(limit: int = SYMBOLS_LIMIT) -> None:
    global symbols_whitelist
    try:
        info = fetch_exchange_info()
        tickers = fetch_futures_ticker()
        qvol_map = {t.get("symbol"): float(t.get("quoteVolume", 0)) for t in tickers if t.get("symbol")}
        valid = [
            s["symbol"] for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and qvol_map.get(s["symbol"], 0) > 0
        ]
        valid.sort(key=lambda sym: qvol_map.get(sym, 0), reverse=True)
        symbols_whitelist = valid[:limit]
        logger.info(f"✅ Loaded {len(symbols_whitelist)} top USDT-M symbols")
    except Exception as e:
        logger.error(f"load_top_symbols: {e}")
        symbols_whitelist = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_all_futures_symbols() -> None:
    global symbols_hg_all
    try:
        info = fetch_exchange_info()
        symbols_hg_all = sorted([
            s["symbol"] for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ])
        logger.info(f"💎 Loaded {len(symbols_hg_all)} HG symbols")
    except Exception as e:
        logger.error(f"load_all_futures_symbols: {e}")
        symbols_hg_all = list(symbols_whitelist)


def filter_hg_symbols_by_liquidity(min_quote_usdt: float = HG_MIN_QUOTE_VOL) -> None:
    global symbols_hg_all
    try:
        tickers = fetch_futures_ticker()
        qvol = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in tickers if "symbol" in t}
        symbols_hg_all = [s for s in symbols_hg_all if qvol.get(s, 0) >= min_quote_usdt]
        logger.info(f"💎 HG after liquidity filter (>{min_quote_usdt}): {len(symbols_hg_all)}")
    except Exception as e:
        logger.warning(f"filter_hg_symbols_by_liquidity: {e}")


def load_universes() -> None:
    load_top_symbols(SYMBOLS_LIMIT)
    load_all_futures_symbols()
    filter_hg_symbols_by_liquidity(HG_MIN_QUOTE_VOL)


# ---------------------------------------------------------------------------
# Historical data preloading
# ---------------------------------------------------------------------------

def preload_historical(symbols, label: str = "") -> None:
    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        for interval in ["15m", "1h", "4h"]:
            try:
                klines = fetch_futures_klines(sym, interval, limit=500)
                if klines:
                    data_store.store_historical(sym, interval, klines)
                time.sleep(0.05)
            except Exception as e:
                logger.debug(f"preload {sym} {interval}: {e}")
        if idx % 20 == 0 or idx == total:
            logger.info(f"📊 {label} preload: {idx}/{total}")


# ---------------------------------------------------------------------------
# Agent & engine wiring
# ---------------------------------------------------------------------------

def build_system(dashboard_state: Optional[DashboardState] = None):
    """Instantiate and wire all V18 components.

    Returns
    -------
    processor        : EventProcessor
    meta             : MetaAgent
    tracker          : PerformanceTracker
    execution        : ExecutionEngine
    risk             : RiskAgent
    strategy         : StrategyAgent
    confluence       : ConfluenceAgent
    pattern          : PatternAgent
    decision_context : Dict[str, Dict[str, Any]] — runtime signal context cache
    """
    logger.info("🔧 Building V18 agent system...")

    pattern = PatternAgent()
    regime = RegimeAgent()
    confluence = ConfluenceAgent()
    risk = RiskAgent()
    strategy = StrategyAgent()
    smc = SMCAgent()
    sector_rotation = SectorRotationAgent()
    pairs_trading = PairsTradingAgent()
    onchain = OnChainAgent()
    neural_predict = NeuralPredictAgent()
    meta = MetaAgent(agents=[pattern, regime, confluence, risk, strategy, smc, onchain, neural_predict])

    fusion = DecisionFusion()
    execution = ExecutionEngine(paper_trading=PAPER_TRADING, initial_balance=ACCOUNT_BALANCE)
    tracker = PerformanceTracker()

    # decision_id -> context used later on position close
    decision_context: Dict[str, Dict[str, Any]] = {}

    # Load historical win rates from DB into RiskAgent
    try:
        from memory.experience_db import get_agent_win_rates
        db_win_rates = get_agent_win_rates()
        for key, wr in db_win_rates.items():
            risk.set_win_rate(key, wr)
    except Exception as e:
        logger.debug(f"Could not load win rates from DB: {e}")

    def on_signal(fusion_result, agent_results, position):
        """Signal callback: keep fast path light and offload heavy work."""
        try:
            queued = enqueue_signal_notification(
                fusion_result=fusion_result,
                agent_results=agent_results,
                position=position,
            )
            if not queued:
                logger.error(
                    f"Failed to queue signal notification for "
                    f"{fusion_result.symbol} [{fusion_result.interval}]"
                )
        except Exception as e:
            logger.error(f"Signal enqueue error: {e}")

        if dashboard_state is not None:
            try:
                dashboard_state.add_log(
                    f"SIGNAL {fusion_result.symbol}/{fusion_result.interval} "
                    f"{fusion_result.decision} score={fusion_result.final_score:.3f}"
                )
            except Exception as _dashboard_log_err:
                logger.debug(f"dashboard signal log error: {_dashboard_log_err}")

        # Save runtime context for later close handling
        try:
            # Extract regime from agent_results
            _regime = "unknown"
            _regime_result = agent_results.get("regime")
            if _regime_result and hasattr(_regime_result, "metadata") and _regime_result.metadata:
                _regime = _regime_result.metadata.get("regime", "unknown")
            decision_context[fusion_result.decision_id] = {
                "symbol": fusion_result.symbol,
                "interval": fusion_result.interval,
                "decision": fusion_result.decision,
                "agent_scores": dict(fusion_result.agent_scores or {}),
                "agent_directions": {
                    name: getattr(result, "direction", "")
                    for name, result in agent_results.items()
                },
                "agent_results": dict(agent_results),
                "regime": _regime,
            }
        except Exception as e:
            logger.error(f"Decision context cache error: {e}")

    def on_pairs_signal(pair_result):
        try:
            queued = enqueue_pairs_signal_notification(pair_result)
            if not queued:
                logger.error("Failed to queue pairs trading notification")
        except Exception as e:
            logger.error(f"Pairs signal enqueue error: {e}")

        if dashboard_state is not None:
            try:
                meta = pair_result.metadata or {}
                dashboard_state.add_log(
                    f"PAIRS {pair_result.symbol}/{pair_result.interval} "
                    f"LONG={meta.get('long_symbol')} SHORT={meta.get('short_symbol')} "
                    f"z={float(meta.get('zscore', 0.0)):+.2f}"
                )
            except Exception as _dashboard_pairs_log_err:
                logger.debug(f"dashboard pairs log error: {_dashboard_pairs_log_err}")

    processor = EventProcessor(
        pattern_agent=pattern,
        regime_agent=regime,
        confluence_agent=confluence,
        risk_agent=risk,
        strategy_agent=strategy,
        meta_agent=meta,
        fusion=fusion,
        execution=execution,
        on_signal=on_signal,
        smc_agent=smc,
        sector_rotation_agent=sector_rotation,
        pairs_trading_agent=pairs_trading,
        onchain_agent=onchain,
        neural_predict_agent=neural_predict,
        on_pairs_signal=on_pairs_signal,
    )

    logger.info("✅ V18 agent system ready")
    return processor, meta, tracker, execution, risk, strategy, confluence, pattern, decision_context


# ---------------------------------------------------------------------------
# Position monitoring thread
# ---------------------------------------------------------------------------

def _position_monitor(
    processor: EventProcessor,
    tracker: PerformanceTracker,
    decision_context: Dict[str, Dict[str, Any]],
    interval_sec: int = 10,
    evolution_engine: Optional["EvolutionEngine"] = None,
    dashboard_state: Optional[DashboardState] = None,
) -> None:
    """Periodically update SL/TP levels for open positions using latest prices."""
    while True:
        try:
            open_pos = processor.execution.get_open_positions()
            for pos in open_pos:
                df = data_store.get_df(pos.symbol, pos.interval)
                if df is None or df.empty:
                    continue

                current_price = float(df["close"].iloc[-1])
                closed_positions = processor.execution.check_position_levels(pos.symbol, current_price)

                for closed in closed_positions:
                    _handle_closed_position(
                        closed=closed,
                        processor=processor,
                        tracker=tracker,
                        decision_context=decision_context,
                        evolution_engine=evolution_engine,
                        dashboard_state=dashboard_state,
                    )

        except Exception as e:
            logger.debug(f"position_monitor error: {e}")

        time.sleep(interval_sec)


def _handle_closed_position(
    closed,
    processor: EventProcessor,
    tracker: PerformanceTracker,
    decision_context: Dict[str, Dict[str, Any]],
    evolution_engine: Optional["EvolutionEngine"] = None,
    dashboard_state: Optional[DashboardState] = None,
) -> None:
    """Apply side-effects after a position closes (stats, DB, notifications, feedback loops)."""
    # Track performance
    try:
        tracker.record_position(closed)
    except Exception as e:
        logger.error(f"tracker.record_position error: {e}")

    # Send manual early-exit alert for signal-only mode
    try:
        manual_reason = _manual_exit_reason_for_alert(closed)
        if SIGNAL_ONLY and getattr(closed, "paper", False) and manual_reason:
            send_early_exit_alert(closed, reason=manual_reason)
    except Exception as e:
        logger.error(f"send_early_exit_alert error: {e}")

    # Notify close
    try:
        notify_position_closed(closed)
    except Exception as e:
        logger.error(f"notify_position_closed error: {e}")

    if dashboard_state is not None:
        try:
            dashboard_state.add_log(
                f"CLOSED {closed.symbol}/{closed.interval} {closed.direction} "
                f"status={closed.status} pnl={(closed.pnl or 0.0):+.4f}"
            )
        except Exception as _dashboard_close_log_err:
            logger.debug(f"dashboard close log error: {_dashboard_close_log_err}")

    # Update decision outcome in DB
    try:
        if closed.decision_id:
            experience_db.update_decision_outcome(
                decision_id=closed.decision_id,
                outcome=closed.status,
                pnl=closed.pnl or 0.0,
            )
    except Exception as e:
        logger.error(f"update_decision_outcome error: {e}")

    # After updating decision outcome, adapt the fusion threshold
    try:
        correct = (closed.pnl or 0.0) > 0
        processor.fusion.adapt_threshold(correct, 0.0)
    except Exception as e:
        logger.error(f"adapt_threshold error: {e}")

    # Save agent outcomes in DB
    try:
        ctx = decision_context.get(closed.decision_id, {})
        agent_scores = ctx.get("agent_scores", {})
        agent_directions = ctx.get("agent_directions", {})
        correct = (closed.pnl or 0.0) > 0

        # Extract pattern tags for the pattern agent row
        pattern_tags = ""
        pattern_ctx = ctx.get("agent_results", {}).get("pattern")
        if pattern_ctx and hasattr(pattern_ctx, "details"):
            pattern_tags = ",".join(str(d) for d in list(pattern_ctx.details)[:10])

        for agent_name, score in agent_scores.items():
            experience_db.save_agent_outcome(
                decision_id=closed.decision_id,
                agent_name=agent_name,
                score=float(score),
                direction=str(agent_directions.get(agent_name, "")),
                correct=correct,
                pattern_tags=pattern_tags if agent_name == "pattern" else "",
            )
    except Exception as e:
        logger.error(f"save_agent_outcome error: {e}")

    # Update StrategyAgent with trade outcome
    try:
        strategy_name = closed.strategy
        if strategy_name:
            processor.strategy.update_strategy_outcome(
                strategy_name,
                was_profitable=(closed.pnl or 0) > 0,
            )
    except Exception as e:
        logger.error(f"update_strategy_outcome error: {e}")

    # Record outcome in MetaAgent for weight adjustment
    try:
        ctx = decision_context.get(closed.decision_id, {})
        stored_agent_results = ctx.get("agent_results", {})
        regime = ctx.get("regime", "unknown")
        if stored_agent_results and hasattr(processor.meta, "record_outcome"):
            was_correct = (closed.pnl or 0.0) > 0
            processor.meta.record_outcome(
                closed.decision_id,
                stored_agent_results,
                was_correct,
                regime=regime,
            )
    except Exception as e:
        logger.error(f"meta.record_outcome error: {e}")

    # Clean runtime context
    try:
        if closed.decision_id in decision_context:
            decision_context.pop(closed.decision_id, None)
    except Exception as e:
        logger.debug(f"decision_context cleanup error: {e}")

    # Notify evolution engine of closed trade
    try:
        if evolution_engine is not None:
            ctx = decision_context.get(getattr(closed, "decision_id", None), {})
            if not ctx and hasattr(processor, "get_decision_context"):
                ctx = processor.get_decision_context(getattr(closed, "decision_id", None)) or {}
            evolution_engine.on_trade_close(closed, ctx)
    except Exception as e:
        logger.error(f"evolution_engine.on_trade_close error: {e}")


def _manual_exit_reason_for_alert(closed: Any) -> Optional[str]:
    """Return manual early-exit reason label for signal-only mode, or None."""
    status = str(getattr(closed, "status", "") or "").lower()
    if status in {"timeout", "timeout_dead_trade"}:
        return "Timeout (Trade Morto)"
    # After TP1, SL is moved/protected in engine.execution.check_position_levels;
    # an SL hit here indicates an early trailing/protected exit.
    if status == "sl_hit" and (
        getattr(closed, "tp1_hit", False)
        or int(getattr(closed, "trailing_stage", 0) or 0) > 0
    ):
        return "Trailing stop"
    if "momentum" in status or "reversal" in status:
        return "Momentum esaurito"
    return None


def _user_data_event_consumer(
    processor: EventProcessor,
    tracker: PerformanceTracker,
    decision_context: Dict[str, Dict[str, Any]],
    event_queue: "Queue[Dict[str, Any]]",
    evolution_engine: Optional["EvolutionEngine"] = None,
    dashboard_state: Optional[DashboardState] = None,
) -> None:
    """Consume private User Data Stream events and apply immediate execution updates."""
    while True:
        try:
            event = event_queue.get(timeout=1.0)
        except Empty:
            continue
        except Exception as e:
            logger.debug(f"user_data_event_consumer queue error: {e}")
            continue

        try:
            closed_positions = processor.execution.process_user_stream_event(event)
            for closed in closed_positions:
                _handle_closed_position(
                    closed=closed,
                    processor=processor,
                    tracker=tracker,
                    decision_context=decision_context,
                    evolution_engine=evolution_engine,
                    dashboard_state=dashboard_state,
                )
        except Exception as e:
            logger.error(f"user_data_event_consumer processing error: {e}")


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

def _heartbeat_loop(processor: EventProcessor, interval_sec: int) -> None:
    """Send periodic heartbeat messages via Telegram so the user knows the bot is alive."""
    import traceback
    logger.info("🫀 Heartbeat thread started, first beat in 120s")
    start_time = time.time()
    time.sleep(120)  # Wait 2 min before first heartbeat
    consecutive_errors = 0
    while True:
        try:
            uptime_sec = time.time() - start_time
            hours = int(uptime_sec // 3600)
            minutes = int((uptime_sec % 3600) // 60)

            stats = processor.get_stats()
            processed = stats.get("processed", 0)
            signals = stats.get("signals", 0)
            skip_reasons = stats.get("skip_reasons", {})
            exec_stats = stats.get("execution", {})
            open_pos = exec_stats.get("open_positions", 0)
            balance = exec_stats.get("balance") or 0
            risk_blocked = exec_stats.get("risk_blocked", False)
            fusion_threshold = stats.get("fusion_threshold", 0.0)
            last_signal_info = stats.get("last_signal", "")

            # Build training-mode status string for heartbeat
            try:
                completed_trades = experience_db.get_completed_trade_count()
            except Exception:
                completed_trades = 0
            if _sniper_mode_active:
                training_status = "🎯 Sniper Mode attivo"
            else:
                training_status = f"📚 Trade: {completed_trades}/{TRAINING_TARGET_TRADES} → Training Mode"

            msg = build_heartbeat_message(
                uptime_hours=hours,
                uptime_minutes=minutes,
                processed=processed,
                signals=signals,
                open_positions=open_pos,
                balance=balance,
                risk_blocked=risk_blocked,
                skip_reasons=skip_reasons,
                fusion_threshold=fusion_threshold,
                last_signal_info=last_signal_info,
                training_status=training_status,
            )
            send_message(msg)
            logger.info("🫀 Heartbeat sent")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Heartbeat error: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 3:
                try:
                    send_message("🔴 V18 HEARTBEAT — system alive but stats unavailable")
                except Exception:
                    pass
        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Periodic reporting thread
# ---------------------------------------------------------------------------

def _report_loop(processor: EventProcessor, tracker: PerformanceTracker,
                  meta: MetaAgent, interval_sec: int = 3600) -> None:
    """Send periodic performance reports via Telegram."""
    import traceback
    logger.info("📊 Report thread started, first report in 60s")
    time.sleep(60)  # Give system time to start
    consecutive_errors = 0
    while True:
        try:
            exec_stats = processor.execution.get_stats()
            perf_summary = tracker.get_summary()
            agent_report = meta.get_report()
            msg = build_stats_message(exec_stats, perf_summary, agent_report)
            send_message(msg)
            logger.info("📊 Periodic report sent")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"_report_loop error: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 3:
                try:
                    send_message("🔴 V18 REPORT — system alive but report unavailable")
                except Exception:
                    pass
        time.sleep(interval_sec)


def _dashboard_state_snapshot(
    processor: EventProcessor,
    meta: MetaAgent,
    monitored_symbols: List[str],
) -> Dict[str, Any]:
    stats = processor.get_stats()
    exec_stats = stats.get("execution", {})
    meta_report = meta.get_report(include_regime=False)

    mm = getattr(getattr(processor, "fusion", None), "memory_manager", None)
    sentiment_scores: Dict[str, float] = {}
    if mm is not None and hasattr(mm, "get_sentiment_score"):
        for sym in monitored_symbols:
            score = float(mm.get_sentiment_score(sym, 0.0))
            if abs(score) > 1e-12:
                sentiment_scores[sym] = score
    if len(sentiment_scores) > 20:
        sentiment_scores = dict(
            sorted(
                sentiment_scores.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:20]
        )

    return {
        "system_running": True,
        "paper_trading": bool(exec_stats.get("paper_trading", PAPER_TRADING)),
        "balance": float(exec_stats.get("balance", 0.0)),
        "global_win_rate": float(exec_stats.get("win_rate", 0.0)),
        "total_pnl": float(exec_stats.get("total_pnl", 0.0)),
        "pnl_pct": float(exec_stats.get("pnl_pct", 0.0)),
        "open_positions": int(exec_stats.get("open_positions", 0)),
        "agent_weights": {
            "pattern": meta_report.get("pattern", {}).get("weight"),
            "regime": meta_report.get("regime", {}).get("weight"),
            "confluence": meta_report.get("confluence", {}).get("weight"),
            "risk": meta_report.get("risk", {}).get("weight"),
            "sentiment": meta_report.get("sentiment", {}).get("weight"),
        },
        "last_signal": stats.get("last_signal", ""),
        "skip_reasons": stats.get("skip_reasons", {}),
        "sentiment_scores": sentiment_scores,
    }


def _dashboard_positions_snapshot(processor: EventProcessor) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    for pos in processor.execution.get_open_positions():
        current_price: Optional[float] = None
        try:
            df = data_store.get_df(pos.symbol, pos.interval)
            if df is not None and not df.empty:
                current_price = float(df["close"].iloc[-1])
        except Exception:
            current_price = None

        pnl = 0.0
        if current_price is not None:
            if pos.direction == "long":
                pnl = (current_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - current_price) * pos.size

        positions.append(
            {
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "interval": pos.interval,
                "direction": pos.direction,
                "entry_price": float(pos.entry_price),
                "current_price": current_price,
                "pnl": float(pnl),
                "strategy": pos.strategy,
                "decision_id": pos.decision_id,
            }
        )
    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sentiment_agent: Optional[SentimentAgent] = None
    user_data_stream: Optional[UserDataStreamManager] = None
    dashboard_state = DashboardState()
    logger.info("=" * 60)
    logger.info("🤖 V18 AGENTIC AI TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info("🛡️ ACTIVE MODULES:")
    logger.info("   - Regime Agent (GaussianMixture): ON")
    logger.info("   - Pattern Agent (V16 detectors + auto-calibration): ON")
    logger.info("   - Confluence Agent (Probabilistic MTF): ON")
    logger.info("   - Risk Agent (Kelly + real win rates): ON")
    logger.info("   - Strategy Agent (generation + evaluation): ON")
    logger.info("   - SMC Agent (FVG + Order Blocks): ON")
    logger.info("   - Sector Rotation (money-flow heatmap): ON")
    logger.info("   - Pairs Trading (delta-neutral stat-arb): ON")
    logger.info("   - On-Chain Whale Tracker: ON")
    logger.info("   - Neural Predictive Engine: ON")
    logger.info("   - Meta Agent (weight adjustment): ON")
    logger.info("   - Decision Fusion (weighted voting): ON")
    logger.info(f"   - Sentiment Agent (Redis narrative brain): {'ON' if SENTIMENT_ENABLED else 'OFF'}")
    logger.info(f"   - Execution: {'PAPER TRADING' if PAPER_TRADING else 'LIVE TRADING'}")
    logger.info("   - Experience DB (SQLite): ON")
    logger.info("=" * 60)

    try:
        # ---- DB init ----
        experience_db.init_db(DB_PATH)

        # ---- Async workers ----
        start_notification_worker()

        # ---- Telegram test ----
        if TELEGRAM_TEST_ON_START:
            test_connection()

        # ---- Binance client (initialise early to validate credentials) ----
        _ = get_client()

        # ---- Load symbol universes ----
        load_universes()

        if not symbols_whitelist:
            raise ValueError("❌ No symbols loaded for scanning!")

        # ---- Preload historical data ----
        logger.info(f"📥 Preloading history for {len(symbols_whitelist)} symbols (main list)...")
        preload_historical(symbols_whitelist, "MAIN")

        if HG_ENABLED and HG_MONITOR_ALL and symbols_hg_all:
            # Only preload HG symbols not already in whitelist
            hg_extra = [s for s in symbols_hg_all if s not in set(symbols_whitelist)]
            if hg_extra:
                logger.info(f"📥 Preloading history for {len(hg_extra)} HG-only symbols...")
                preload_historical(hg_extra, "HG")

        # ---- Build V18 system ----
        processor, meta, tracker, execution, risk_agent, strategy_agent, confluence_agent, pattern_agent, decision_context = build_system(
            dashboard_state=dashboard_state
        )

        # ---- Build & start Evolution Engine ----
        evolution_engine = EvolutionEngine(
            meta_agent=meta,
            fusion=processor.fusion,
            risk_agent=risk_agent,
            strategy_agent=strategy_agent,
            confluence_agent=confluence_agent,
            tracker=tracker,
            pattern_agent=pattern_agent,
        )
        evolution_engine.startup()
        logger.info("🧬 Evolution Engine started")
        logger.info("   - Loop #1 (MetaAgent → Fusion): ON")
        logger.info("   - Loop #2 (Tracker → RiskAgent): ON")
        logger.info("   - Loop #3 (Auto-tune threshold): ON")
        logger.info("   - Loop #5 (Strategy evolution): ON")
        logger.info("   - Loop #6 (MetaAgent persistence): ON")
        logger.info("   - Loop #7 (Confluence TF learning): ON")

        # ---- Wire WebSocket callbacks ----
        all_symbols = list(set(symbols_whitelist + (symbols_hg_all if HG_MONITOR_ALL else [])))
        if SENTIMENT_ENABLED:
            sentiment_agent = SentimentAgent(
                memory_manager=processor.fusion.memory_manager,
                symbols_provider=lambda: all_symbols,
                update_interval_seconds=SENTIMENT_UPDATE_INTERVAL_SECONDS,
                ttl_seconds=SENTIMENT_TTL_SECONDS,
                crypto_panic_api_key=CRYPTO_PANIC_API_KEY,
                lm_studio_url=LM_STUDIO_URL,
                lm_studio_model=LM_STUDIO_MODEL,
            )
            sentiment_agent.start()

        def ws_on_update(symbol, interval, kline):
            data_store.update_realtime(symbol, interval, kline)

        def ws_on_closed(symbol, interval, kline):
            processor.on_candle_close(symbol, interval, kline)

        register_callbacks(on_closed=ws_on_closed, on_update=ws_on_update)

        monitored_symbols = list(dict.fromkeys(symbols_whitelist + symbols_hg_all))
        threading.Thread(
            target=start_dashboard_server,
            kwargs={
                "state_provider": lambda: _dashboard_state_snapshot(processor, meta, monitored_symbols),
                "positions_provider": lambda: _dashboard_positions_snapshot(processor),
                "logs_provider": dashboard_state.get_logs,
                "host": "127.0.0.1",
                "port": 5018,
            },
            daemon=True,
            name="DashboardServer",
        ).start()
        dashboard_state.add_log("Dashboard server thread started on http://127.0.0.1:5018")

        # ---- Private User Data Stream (LIVE only) ----
        if not PAPER_TRADING:
            user_data_events: "Queue[Dict[str, Any]]" = Queue(maxsize=500)
            user_data_queue_lock = threading.Lock()

            def _on_user_event(event: Dict[str, Any]) -> None:
                with user_data_queue_lock:
                    try:
                        user_data_events.put_nowait(event)
                    except Exception:
                        logger.warning("User Data Stream event queue full - dropping oldest event")
                        try:
                            dropped_event = user_data_events.get_nowait()
                            logger.warning(
                                f"User Data Stream dropped queued event type={str(dropped_event.get('e', 'unknown'))}"
                            )
                            user_data_events.put_nowait(event)
                        except Exception as queue_recover_err:
                            logger.debug(f"User Data Stream queue recovery failed: {queue_recover_err}")

            user_data_stream = UserDataStreamManager(on_event=_on_user_event)
            uds_started = user_data_stream.start()
            if uds_started:
                threading.Thread(
                    target=_user_data_event_consumer,
                    args=(
                        processor,
                        tracker,
                        decision_context,
                        user_data_events,
                        evolution_engine,
                        dashboard_state,
                    ),
                    daemon=True,
                    name="UserDataConsumer",
                ).start()
                logger.info("⚡ User Data Stream active: execution updates now event-driven")
            else:
                logger.warning("⚠️ User Data Stream unavailable - continuing with polling/market data path")
        else:
            logger.info("ℹ️ User Data Stream skipped in PAPER_TRADING mode")

        # ---- Start WebSockets ----
        start_websockets(all_symbols, timeframes=["15m", "1h", "4h"])

        # ---- REST fallback ----
        if POLL_CLOSED_ENABLE:
            start_rest_fallback(symbols_whitelist, ws_on_closed)

        # ---- Startup health check ----
        ws_ok = startup_health_check(STARTUP_TIMEOUT)
        if ws_ok:
            logger.info("✅ All WebSockets healthy")
        else:
            logger.warning("⚠️ Some WebSockets not responding — REST fallback active")

        # ---- Background threads ----
        threading.Thread(
            target=lambda: _position_monitor(
                processor, tracker, decision_context,
                interval_sec=10,
                evolution_engine=evolution_engine,
                dashboard_state=dashboard_state,
            ),
            daemon=True,
            name="PositionMonitor",
        ).start()

        threading.Thread(
            target=_report_loop,
            args=(processor, tracker, meta),
            daemon=True,
            name="ReportLoop",
        ).start()

        if HEARTBEAT_ENABLED:
            threading.Thread(
                target=_heartbeat_loop,
                args=(processor, HEARTBEAT_INTERVAL),
                daemon=True,
                name="Heartbeat",
            ).start()

        # ---- Send startup notification ----
        send_message(build_startup_message(
            n_symbols=len(symbols_whitelist),
            n_hg=len(symbols_hg_all),
            paper=PAPER_TRADING,
        ))

        logger.info("=" * 60)
        logger.info("🚀 V18 SYSTEM OPERATIONAL — Press Ctrl+C to stop")
        logger.info("=" * 60)

        # ---- Signal handlers for graceful shutdown on SIGTERM/SIGINT ----
        import signal as _signal

        def _graceful_shutdown(signum, frame):
            logger.info(f"⚡ Signal {signum} received — saving state before exit...")
            try:
                if sentiment_agent is not None:
                    sentiment_agent.stop()
            except Exception as _sentiment_stop_err:
                logger.error(f"Sentiment agent stop error: {_sentiment_stop_err}")
            try:
                if user_data_stream is not None:
                    user_data_stream.stop()
            except Exception as _user_stream_stop_err:
                logger.error(f"User Data Stream stop error: {_user_stream_stop_err}")
            try:
                evolution_engine.shutdown()
            except Exception as _se:
                logger.error(f"Graceful shutdown save error: {_se}")
            try:
                send_message("⏹️ V18 — shutdown signal received, state saved.")
            except Exception:
                pass
            sys.exit(0)

        _signal.signal(_signal.SIGTERM, _graceful_shutdown)
        _signal.signal(_signal.SIGINT, _graceful_shutdown)

        # ---- Main loop ----
        global _sniper_mode_active
        _sniper_mode_active = not TRAINING_MODE  # True from the start if training is disabled
        _last_evolution_tick = time.time()
        while True:
            time.sleep(30)
            gc.collect()

            # Frequently push fresh win rates to RiskAgent (lightweight)
            tracker.update_risk_agent_win_rates(risk_agent, current_balance=execution.get_stats()["balance"])

            # Auto-switch: Training Mode → Sniper Mode once enough trades are completed
            if TRAINING_MODE and not _sniper_mode_active:
                try:
                    completed = experience_db.get_completed_trade_count()
                    if completed >= TRAINING_TARGET_TRADES:
                        _sniper_mode_active = True
                        logger.info(
                            f"🎓 TRAINING COMPLETATO ({completed} trade) — passaggio a Sniper Mode"
                        )
                        # Apply Sniper Mode thresholds to running components
                        processor.fusion.threshold = SNIPER_FUSION_THRESHOLD
                        _cfg.FUSION_THRESHOLD_DEFAULT = SNIPER_FUSION_THRESHOLD
                        _cfg.MIN_FUSION_SCORE = SNIPER_MIN_FUSION_SCORE
                        _cfg.MIN_AGENT_CONFIRMATIONS = SNIPER_MIN_AGENT_CONFIRMATIONS
                        _cfg.MIN_RR = SNIPER_MIN_RR
                        _cfg.NON_OPTIMAL_HOUR_PENALTY = SNIPER_NON_OPTIMAL_HOUR_PENALTY
                        _cfg.SIGNAL_COOLDOWN_BY_TF = SNIPER_SIGNAL_COOLDOWN_BY_TF
                        _cfg.MAX_OPEN_POSITIONS = SNIPER_MAX_OPEN_POSITIONS
                        try:
                            send_message(
                                f"🎓 *V18 TRAINING COMPLETATO*\n\n"
                                f"✅ {completed} trade completati\n"
                                f"🎯 Passaggio a *Sniper Mode* — soglie alzate:\n"
                                f"  • Fusion threshold: {SNIPER_FUSION_THRESHOLD}\n"
                                f"  • Min fusion score: {SNIPER_MIN_FUSION_SCORE}\n"
                                f"  • Min agents: {SNIPER_MIN_AGENT_CONFIRMATIONS}\n"
                                f"  • Min R/R: {SNIPER_MIN_RR}"
                            )
                        except Exception as _notify_err:
                            logger.error(f"Sniper Mode notification error: {_notify_err}")
                except Exception as _switch_err:
                    logger.error(f"Training→Sniper auto-switch error: {_switch_err}")

            # Evolution tick every 30 minutes (handles weight adjust, auto-tune, state save)
            if time.time() - _last_evolution_tick >= 1800:
                try:
                    evolution_engine.tick()
                except Exception as _evo_err:
                    logger.error(f"evolution_engine.tick error: {_evo_err}")
                _last_evolution_tick = time.time()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("⏹️ MANUAL SHUTDOWN (Ctrl+C)")
        logger.info("=" * 60)
        try:
            if sentiment_agent is not None:
                sentiment_agent.stop()
        except Exception as e:
            logger.error(f"sentiment_agent.stop error: {e}")
        try:
            if user_data_stream is not None:
                user_data_stream.stop()
        except Exception as e:
            logger.error(f"user_data_stream.stop error: {e}")
        try:
            evolution_engine.shutdown()
        except Exception as e:
            logger.error(f"evolution_engine.shutdown error: {e}")
        try:
            stats = processor.execution.get_stats()
            logger.info(
                f"📊 Final stats: trades={stats['trade_count']} "
                f"wr={stats['win_rate']:.1%} pnl={stats['total_pnl']:+.4f}"
            )
            send_message("⏹️ V18 Agentic AI Trading System — STOPPED")
        except Exception as e:
            logger.error(f"Shutdown cleanup error: {e}")
        logger.info("👋 V18 terminated gracefully")
        sys.exit(0)

    except Exception as e:
        logger.critical("=" * 60)
        logger.critical(f"❌ FATAL ERROR: {e}")
        logger.critical("=" * 60)
        import traceback
        traceback.print_exc()
        try:
            if user_data_stream is not None:
                user_data_stream.stop()
        except Exception as _uds_fatal_stop:
            logger.error(f"user_data_stream.stop on fatal error: {_uds_fatal_stop}")
        try:
            send_message(f"🔴 V18 FATAL ERROR\n\n{str(e)[:300]}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
