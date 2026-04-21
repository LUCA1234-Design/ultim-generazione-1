from types import SimpleNamespace

from notifications.telegram_service import build_early_exit_alert_message
from notifications.telegram_service import build_heartbeat_message
from notifications.telegram_service import build_pairs_signal_message
from notifications.telegram_service import build_signal_message
from notifications.telegram_service import build_startup_message
from notifications.telegram_service import build_stats_message
from agents.base_agent import AgentResult


def test_build_early_exit_alert_message_uses_timeout_title_for_dead_trade():
    position = SimpleNamespace(symbol="BTCUSDT", close_price=100.0, pnl=0.1)
    message = build_early_exit_alert_message(position, reason="Timeout (Trade Morto)")
    assert "⚠️ USCITA PER TIMEOUT (Trade Morto) ⚠️" in message


def test_build_pairs_signal_message_contains_delta_neutral_legs():
    pair_signal = SimpleNamespace(
        symbol="PEPEUSDT/FLOKIUSDT",
        interval="1h",
        metadata={
            "pair": ["PEPEUSDT", "FLOKIUSDT"],
            "long_symbol": "PEPEUSDT",
            "short_symbol": "FLOKIUSDT",
            "zscore": 2.35,
            "correlation": 0.91,
        },
    )
    message = build_pairs_signal_message(pair_signal)
    assert "DELTA-NEUTRAL PAIRS SIGNAL" in message
    assert "LONG `PEPEUSDT`" in message
    assert "SHORT `FLOKIUSDT`" in message


def test_build_signal_message_includes_kelly_size_and_onchain_alert():
    fusion = SimpleNamespace(decision="long", symbol="BTCUSDT", interval="1h", final_score=0.8, direction="long", decision_id="d1")
    onchain = AgentResult(
        agent_name="onchain",
        symbol="BTCUSDT",
        interval="1h",
        score=0.8,
        direction="short",
        confidence=0.8,
        details=["whale:to_exchange BTC $5,000,000"],
        metadata={"alert_count": 1},
    )
    risk = AgentResult(
        agent_name="risk",
        symbol="BTCUSDT",
        interval="1h",
        score=0.7,
        direction="long",
        confidence=0.62,
        details=["rr=2.0"],
        metadata={"rr": 2.0, "kelly": 0.08, "win_rate": 0.62},
    )
    position = SimpleNamespace(entry_price=100.0, sl=95.0, tp1=110.0, tp2=120.0, size=2.5, paper=True)
    msg = build_signal_message(
        fusion=fusion,
        agent_results={"risk": risk, "onchain": onchain},
        position=position,
    )
    assert "Kelly Size" in msg
    assert "2.5000" in msg
    assert "Leverage" in msg
    assert "On-Chain" in msg


def test_core_telegram_templates_are_rebranded_to_v18():
    heartbeat = build_heartbeat_message(
        uptime_hours=1,
        uptime_minutes=2,
        processed=10,
        signals=1,
        open_positions=0,
        balance=1000.0,
        risk_blocked=False,
        skip_reasons={},
        fusion_threshold=0.55,
    )
    startup = build_startup_message(n_symbols=5, n_hg=2, paper=True)
    report = build_stats_message(
        exec_stats={"balance": 1000.0, "total_pnl": 1.0, "pnl_pct": 0.1},
        perf_summary={"win_rate": 0.5, "wins": 1, "losses": 1, "sharpe": 0.1},
        agent_report={},
    )

    assert "V18 HEARTBEAT" in heartbeat
    assert "V18 Agentic AI Trading System — STARTED" in startup
    assert "V18 Performance Report" in report
