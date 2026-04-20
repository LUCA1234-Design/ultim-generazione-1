from types import SimpleNamespace

from notifications.telegram_service import build_early_exit_alert_message


def test_build_early_exit_alert_message_uses_timeout_title_for_dead_trade():
    position = SimpleNamespace(symbol="BTCUSDT", close_price=100.0, pnl=0.1)
    message = build_early_exit_alert_message(position, reason="Timeout (Trade Morto)")
    assert "⚠️ USCITA PER TIMEOUT (Trade Morto) ⚠️" in message
