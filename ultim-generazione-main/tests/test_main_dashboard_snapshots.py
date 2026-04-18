from types import SimpleNamespace

import pandas as pd

from main import _dashboard_positions_snapshot, _dashboard_state_snapshot


class _FakeMemoryManager:
    def __init__(self, scores):
        self._scores = scores

    def get_sentiment_score(self, symbol, default=0.0):
        return self._scores.get(symbol, default)


class _FakeExecution:
    def __init__(self, positions):
        self._positions = positions

    def get_open_positions(self):
        return self._positions


class _FakeProcessor:
    def __init__(self, stats, positions, sentiment_scores):
        self._stats = stats
        self.execution = _FakeExecution(positions)
        self.fusion = SimpleNamespace(memory_manager=_FakeMemoryManager(sentiment_scores))

    def get_stats(self):
        return self._stats


def test_dashboard_state_snapshot_aggregates_stats_and_weights():
    processor = _FakeProcessor(
        stats={
            "execution": {
                "paper_trading": True,
                "balance": 123.0,
                "win_rate": 0.7,
                "total_pnl": 22.0,
                "pnl_pct": 1.8,
                "open_positions": 2,
            },
            "last_signal": "BTCUSDT 1h long",
            "skip_reasons": {"cooldown": 3},
        },
        positions=[],
        sentiment_scores={"BTCUSDT": 0.9, "ETHUSDT": -0.2, "XRPUSDT": 0.0},
    )
    meta = SimpleNamespace(
        get_report=lambda include_regime=False: {
            "pattern": {"weight": 1.1},
            "regime": {"weight": 0.9},
            "confluence": {"weight": 1.0},
            "risk": {"weight": 1.2},
            "sentiment": {"weight": 0.8},
        }
    )

    snapshot = _dashboard_state_snapshot(
        processor=processor,
        meta=meta,
        monitored_symbols=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    )

    assert snapshot["paper_trading"] is True
    assert snapshot["balance"] == 123.0
    assert snapshot["global_win_rate"] == 0.7
    assert snapshot["agent_weights"]["pattern"] == 1.1
    assert snapshot["last_signal"] == "BTCUSDT 1h long"
    assert snapshot["sentiment_scores"] == {"BTCUSDT": 0.9, "ETHUSDT": -0.2}


def test_dashboard_positions_snapshot_computes_long_and_short_pnl(monkeypatch):
    positions = [
        SimpleNamespace(
            position_id="p1",
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            entry_price=100.0,
            size=2.0,
            strategy="breakout",
            decision_id="d1",
        ),
        SimpleNamespace(
            position_id="p2",
            symbol="ETHUSDT",
            interval="1h",
            direction="short",
            entry_price=50.0,
            size=1.5,
            strategy="mean_revert",
            decision_id="d2",
        ),
    ]
    processor = _FakeProcessor(stats={}, positions=positions, sentiment_scores={})

    def _fake_get_df(symbol, interval):
        if symbol == "BTCUSDT":
            return pd.DataFrame({"close": [101.0]})
        if symbol == "ETHUSDT":
            return pd.DataFrame({"close": [48.0]})
        return None

    monkeypatch.setattr("main.data_store.get_df", _fake_get_df)

    out = _dashboard_positions_snapshot(processor)

    assert len(out) == 2
    btc = next(item for item in out if item["symbol"] == "BTCUSDT")
    eth = next(item for item in out if item["symbol"] == "ETHUSDT")
    assert btc["pnl"] == 2.0
    assert eth["pnl"] == 3.0

