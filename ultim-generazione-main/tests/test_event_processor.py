import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.base_agent import AgentResult
from engine.decision_fusion import FusionResult
from engine.event_processor import EventProcessor


def _agent_result(name: str, direction: str = "long", metadata: dict | None = None) -> AgentResult:
    return AgentResult(
        agent_name=name,
        symbol="SOLUSDT",
        interval="1h",
        score=0.8,
        direction=direction,
        confidence=0.8,
        details=["ok"],
        metadata=metadata or {},
    )


def _make_processor(execution: MagicMock, fusion: MagicMock) -> EventProcessor:
    pattern = MagicMock()
    pattern.safe_analyse.return_value = _agent_result("pattern")

    regime = MagicMock()
    regime.safe_analyse.return_value = _agent_result("regime", metadata={"regime": "trending"})

    confluence = MagicMock()
    confluence.safe_analyse.return_value = _agent_result("confluence", metadata={"agreeing_tfs": 3})

    risk = MagicMock()
    risk.safe_analyse.return_value = _agent_result(
        "risk",
        metadata={"entry": 100.0, "sl": 95.0, "tp1": 110.0, "tp2": 120.0, "size": 1.0},
    )

    strategy = MagicMock()
    strategy.safe_analyse.return_value = _agent_result("strategy", metadata={"strategy": "test"})

    meta = MagicMock()
    meta.safe_analyse.return_value = _agent_result("meta")

    return EventProcessor(
        pattern_agent=pattern,
        regime_agent=regime,
        confluence_agent=confluence,
        risk_agent=risk,
        strategy_agent=strategy,
        meta_agent=meta,
        fusion=fusion,
        execution=execution,
    )


def test_on_candle_close_blocks_trade_when_highly_correlated_same_direction(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = [SimpleNamespace(symbol="ETHUSDT", direction="long")]
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = object()

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d1",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)

    base_close = pd.Series(range(1, 121), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": base_close}),
        ("ETHUSDT", "1h"): pd.DataFrame({"close": base_close * 2.0}),  # corr = 1.0
        ("BTCUSDT", "1h"): pd.DataFrame({"close": base_close}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)
    monkeypatch.setattr(
        "engine.event_processor.fetch_futures_depth",
        lambda *_args, **_kwargs: {"bids": [["100.0", "1"]], "asks": [["100.2", "1"]]},
    )

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 120.0})

    assert result is None
    execution.open_position.assert_not_called()
    assert processor.get_stats()["skip_reasons"]["high_correlation"] == 1


def test_on_candle_close_allows_trade_when_correlation_data_is_missing_or_too_short(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = [SimpleNamespace(symbol="ETHUSDT", direction="long")]
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d2",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": 0.2}))

    base_close = pd.Series(range(1, 121), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": base_close}),
        ("ETHUSDT", "1h"): pd.DataFrame({"close": pd.Series([1.0])}),  # too short for corr
        ("BTCUSDT", "1h"): pd.DataFrame({"close": base_close}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 120.0})

    assert result is not None
    execution.open_position.assert_called_once()
    assert execution.open_position.call_args.kwargs["force_paper"] is True


def test_on_candle_close_skips_trade_when_micro_momentum_is_weak(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d3",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(False, {"imbalance": -0.1}))

    base_close = pd.Series(range(1, 121), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": base_close}),
        ("BTCUSDT", "1h"): pd.DataFrame({"close": base_close}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 120.0})

    assert result is None
    execution.open_position.assert_not_called()
    assert processor.get_stats()["skip_reasons"]["weak_micro_momentum"] == 1


def test_on_candle_close_skips_long_when_sentiment_is_highly_negative(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d4",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.fusion.memory_manager = MagicMock()
    processor.fusion.memory_manager.get_sentiment_score.return_value = -0.8
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": 0.3}))

    base_close = pd.Series(range(1, 121), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": base_close}),
        ("BTCUSDT", "1h"): pd.DataFrame({"close": base_close}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 120.0})

    assert result is None
    execution.open_position.assert_not_called()
    assert processor.get_stats()["skip_reasons"]["negative_news_sentiment"] == 1


def test_on_candle_close_skips_short_when_sentiment_is_highly_positive(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d5",
        symbol="SOLUSDT",
        interval="1h",
        decision="short",
        final_score=0.9,
        direction="short",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.fusion.memory_manager = MagicMock()
    processor.fusion.memory_manager.get_sentiment_score.return_value = 0.9
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": -0.3}))
    processor.market_gravity.safe_analyse = MagicMock(
        return_value=_agent_result("market_gravity", direction="short", metadata={"score_adjustment": 0.0, "veto": False})
    )

    base_close = pd.Series(range(1, 121), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": base_close}),
        ("BTCUSDT", "1h"): pd.DataFrame({"close": base_close}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 120.0})

    assert result is None
    execution.open_position.assert_not_called()
    assert processor.get_stats()["skip_reasons"]["positive_news_sentiment"] == 1


def test_correlation_check_does_not_block_at_exact_threshold(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = [SimpleNamespace(symbol="ETHUSDT", direction="long")]
    execution.is_risk_blocked.return_value = (False, "")

    fusion = MagicMock()
    fusion._threshold = 0.5
    processor = _make_processor(execution, fusion)

    close_data = pd.DataFrame({"close": pd.Series(range(1, 121), dtype=float)})
    frames = {
        ("SOLUSDT", "1h"): close_data,
        ("ETHUSDT", "1h"): close_data,
    }

    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )

    threshold = close_data["close"].corr(close_data["close"])
    blocked = processor._correlation_check("SOLUSDT", "1h", "long", threshold=threshold)
    assert blocked is None


def test_correlation_check_ignores_when_new_symbol_data_is_too_short(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = [SimpleNamespace(symbol="ETHUSDT", direction="long")]
    execution.is_risk_blocked.return_value = (False, "")

    fusion = MagicMock()
    fusion._threshold = 0.5
    processor = _make_processor(execution, fusion)

    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": pd.Series([1.0, 2.0, 3.0])}),
        ("ETHUSDT", "1h"): pd.DataFrame({"close": pd.Series(range(1, 121), dtype=float)}),
    }

    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )

    blocked = processor._correlation_check("SOLUSDT", "1h", "long")
    assert blocked is None


def test_on_candle_close_vetoes_short_when_market_gravity_detects_strong_btc_uptrend(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d6",
        symbol="SOLUSDT",
        interval="1h",
        decision="short",
        final_score=0.9,
        direction="short",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": -0.2}))

    uptrend = pd.Series(range(100, 260), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": uptrend}),
        ("BTCUSDT", "1h"): pd.DataFrame({"close": uptrend}),
        ("ETHUSDT", "1h"): pd.DataFrame({"close": uptrend * 1.2}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 259.0})

    assert result is None
    execution.open_position.assert_not_called()
    assert processor.get_stats()["skip_reasons"]["market_gravity_veto"] == 1


def test_smart_limit_entry_price_uses_bid_for_long_and_ask_for_short(monkeypatch):
    execution = MagicMock()
    fusion = MagicMock()
    fusion._threshold = 0.5
    processor = _make_processor(execution, fusion)

    order_books = [
        {"bids": [["100.0", "10"]], "asks": [["100.2", "10"]]},
        {"bids": [["99.8", "10"]], "asks": [["101.0", "10"]]},
    ]
    monkeypatch.setattr(
        "engine.event_processor.fetch_futures_depth",
        lambda *_args, **_kwargs: order_books.pop(0),
    )

    long_entry = processor._smart_limit_entry_price("SOLUSDT", "long", 100.0)
    short_entry = processor._smart_limit_entry_price("SOLUSDT", "short", 100.0)

    assert long_entry == pytest.approx(100.0 * 0.9995)
    assert short_entry == pytest.approx(101.0 * 1.0005)


def test_on_candle_close_market_gravity_boosts_long_score(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d9",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.24,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": 0.2}))

    uptrend = pd.Series(range(100, 260), dtype=float)
    frames = {
        ("SOLUSDT", "1h"): pd.DataFrame({"close": uptrend}),
        ("BTCUSDT", "1h"): pd.DataFrame({"close": uptrend}),
        ("ETHUSDT", "1h"): pd.DataFrame({"close": uptrend * 1.2}),
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)
    monkeypatch.setattr(
        "engine.event_processor.fetch_futures_depth",
        lambda *_args, **_kwargs: {"bids": [["100.0", "10"]], "asks": [["100.2", "10"]]},
    )

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 259.0})

    assert result is not None
    assert result.final_score > 0.24
    execution.open_position.assert_called_once()


def test_on_candle_close_uses_smc_limit_entry_when_available(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d10",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": 0.2}))
    processor.smc.safe_analyse = MagicMock(
        return_value=_agent_result(
            "smc",
            direction="long",
            metadata={"limit_entry": 98.75, "setup": "fvg_bullish"},
        )
    )
    processor.market_gravity.safe_analyse = MagicMock(
        return_value=_agent_result("market_gravity", direction="long", metadata={"score_adjustment": 0.0, "veto": False})
    )

    base = pd.Series(range(100, 260), dtype=float)
    frame = pd.DataFrame(
        {
            "open": base - 0.3,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": pd.Series([100.0] * len(base)),
        }
    )
    frames = {
        ("SOLUSDT", "1h"): frame,
        ("BTCUSDT", "1h"): frame,
        ("ETHUSDT", "1h"): frame,
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)
    monkeypatch.setattr(
        "engine.event_processor.fetch_futures_depth",
        lambda *_args, **_kwargs: {"bids": [["100.0", "10"]], "asks": [["100.2", "10"]]},
    )

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 259.0})

    assert result is not None
    execution.open_position.assert_called_once()
    assert execution.open_position.call_args.kwargs["entry_price"] == pytest.approx(98.75)


def test_on_candle_close_falls_back_to_smart_limit_when_smc_score_is_too_low(monkeypatch):
    execution = MagicMock()
    execution.get_open_positions.return_value = []
    execution.is_risk_blocked.return_value = (False, "")
    execution.open_position.return_value = SimpleNamespace(position_id="p1")

    fusion = MagicMock()
    fusion.fuse.return_value = FusionResult(
        decision_id="d11",
        symbol="SOLUSDT",
        interval="1h",
        decision="long",
        final_score=0.9,
        direction="long",
        agent_scores={},
        agent_results={},
        threshold=0.5,
        reasoning=[],
    )
    fusion._threshold = 0.5

    processor = _make_processor(execution, fusion)
    processor.volume_trigger.confirm = MagicMock(return_value=(True, {"imbalance": 0.2}))
    processor.smc.safe_analyse = MagicMock(
        return_value=AgentResult(
            agent_name="smc",
            symbol="SOLUSDT",
            interval="1h",
            score=0.10,
            direction="long",
            confidence=0.10,
            details=["weak"],
            metadata={"limit_entry": 98.75, "setup": "fvg_bullish"},
        )
    )
    processor.market_gravity.safe_analyse = MagicMock(
        return_value=_agent_result("market_gravity", direction="long", metadata={"score_adjustment": 0.0, "veto": False})
    )

    base = pd.Series(range(100, 260), dtype=float)
    frame = pd.DataFrame(
        {
            "open": base - 0.3,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": pd.Series([100.0] * len(base)),
        }
    )
    frames = {
        ("SOLUSDT", "1h"): frame,
        ("BTCUSDT", "1h"): frame,
        ("ETHUSDT", "1h"): frame,
    }

    monkeypatch.setattr("engine.event_processor.data_store.update_realtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "engine.event_processor.data_store.get_df",
        lambda symbol, interval: frames.get((symbol, interval)),
    )
    monkeypatch.setattr(processor, "_is_forbidden_hour", lambda: False)
    monkeypatch.setattr(processor, "_is_signal_cooled", lambda _symbol, _interval: True)
    monkeypatch.setattr(processor, "_is_optimal_hour", lambda: True)
    monkeypatch.setattr(
        "engine.event_processor.fetch_futures_depth",
        lambda *_args, **_kwargs: {"bids": [["100.0", "10"]], "asks": [["100.2", "10"]]},
    )

    result = processor.on_candle_close("SOLUSDT", "1h", {"close": 259.0})

    assert result is not None
    execution.open_position.assert_called_once()
    assert execution.open_position.call_args.kwargs["entry_price"] == pytest.approx(99.95)
