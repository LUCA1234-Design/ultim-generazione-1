from types import SimpleNamespace

from notifications.telegram_service import build_early_exit_alert_message
from notifications.telegram_service import build_pairs_signal_message


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
