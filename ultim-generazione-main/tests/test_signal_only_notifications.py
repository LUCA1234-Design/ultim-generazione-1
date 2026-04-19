from types import SimpleNamespace
from unittest.mock import MagicMock

from engine.execution import Position
from main import _handle_closed_position
from services import notification_worker


def test_signal_only_worker_sends_manual_signal_format(monkeypatch):
    send_mock = MagicMock()
    save_decision_mock = MagicMock()
    def fail_on_rich_signal_call(*_args, **_kwargs):
        raise AssertionError("rich signal message must not be used")

    monkeypatch.setattr("services.notification_worker.send_message", send_mock)
    monkeypatch.setattr("services.notification_worker.send_photo", MagicMock())
    monkeypatch.setattr("services.notification_worker.data_store.get_df", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("services.notification_worker.save_decision", save_decision_mock)
    monkeypatch.setattr("services.notification_worker.build_signal_message", fail_on_rich_signal_call)

    position = Position(
        position_id="p1",
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        entry_price=100.0,
        size=1.0,
        sl=95.0,
        tp1=110.0,
        tp2=120.0,
        paper=True,
    )
    fusion = SimpleNamespace(
        decision_id="d1",
        symbol="BTCUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        threshold=0.5,
        reasoning=[],
        agent_scores={},
    )

    notification_worker._process_signal_job({
        "fusion_result": fusion,
        "agent_results": {},
        "position": position,
    })

    send_mock.assert_called_once()
    msg = send_mock.call_args[0][0]
    assert "🚨 SEGNALE OPERATIVO (MANUALE) 🚨" in msg
    assert "🪙 Moneta: BTCUSDT (LONG)" in msg
    assert "🎯 Entry: 100.0000" in msg
    assert "🛑 Stop Loss: 95.0000" in msg
    assert "💰 Take Profit 1: 110.0000" in msg
    save_decision_mock.assert_called_once()


def test_handle_closed_position_sends_manual_exit_alert_for_timeout(monkeypatch):
    tracker = SimpleNamespace(record_position=MagicMock())
    processor = SimpleNamespace(
        fusion=SimpleNamespace(adapt_threshold=MagicMock()),
        strategy=SimpleNamespace(update_strategy_outcome=MagicMock()),
        meta=SimpleNamespace(record_outcome=MagicMock()),
    )
    manual_exit_mock = MagicMock()
    notify_close_mock = MagicMock()

    monkeypatch.setattr("main.send_early_exit_alert", manual_exit_mock)
    monkeypatch.setattr("main.notify_position_closed", notify_close_mock)
    monkeypatch.setattr("main.experience_db.update_decision_outcome", MagicMock())
    monkeypatch.setattr("main.experience_db.save_agent_outcome", MagicMock())

    closed = SimpleNamespace(
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        close_price=102.0,
        pnl=2.5,
        status="timeout",
        tp1_hit=False,
        paper=True,
        strategy="",
        decision_id="",
    )

    _handle_closed_position(
        closed=closed,
        processor=processor,
        tracker=tracker,
        decision_context={},
        evolution_engine=None,
        dashboard_state=None,
    )

    manual_exit_mock.assert_called_once_with(closed, reason="Timeout")
    notify_close_mock.assert_called_once_with(closed)
