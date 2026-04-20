import numpy as np
import pandas as pd

from agents.pairs_trading_agent import PairsTradingAgent


def test_pairs_trading_agent_emits_delta_neutral_signal_on_extreme_zscore(monkeypatch):
    agent = PairsTradingAgent()
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_LOOKBACK", 60)
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_ZSCORE_ENTRY", 1.5)
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_MIN_CORRELATION", 0.5)
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_CANDIDATE_PAIRS", [("PEPEUSDT", "FLOKIUSDT")])
    agent._pair_map = {"PEPEUSDT": [("PEPEUSDT", "FLOKIUSDT")], "FLOKIUSDT": [("PEPEUSDT", "FLOKIUSDT")]}

    base = np.linspace(100.0, 130.0, 80)
    close_a = pd.Series(base)
    close_b = pd.Series(base * 1.05)
    close_b.iloc[-1] = close_b.iloc[-1] * 1.15  # force strong positive spread deviation

    frames = {
        ("PEPEUSDT", "1h"): pd.DataFrame({"close": close_a}),
        ("FLOKIUSDT", "1h"): pd.DataFrame({"close": close_b}),
    }
    monkeypatch.setattr("agents.pairs_trading_agent.data_store.get_df", lambda s, i: frames.get((s, i)))

    result = agent.analyse("PEPEUSDT", "1h")

    assert result is not None
    assert result.metadata["signal_type"] == "delta_neutral_pairs"
    assert result.metadata["long_symbol"] == "PEPEUSDT"
    assert result.metadata["short_symbol"] == "FLOKIUSDT"
    assert abs(result.metadata["zscore"]) >= 1.5


def test_pairs_trading_agent_skips_when_correlation_is_too_low(monkeypatch):
    agent = PairsTradingAgent()
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_LOOKBACK", 60)
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_MIN_CORRELATION", 0.95)
    monkeypatch.setattr("agents.pairs_trading_agent.PAIRS_TRADING_CANDIDATE_PAIRS", [("FETUSDT", "TAOUSDT")])
    agent._pair_map = {"FETUSDT": [("FETUSDT", "TAOUSDT")], "TAOUSDT": [("FETUSDT", "TAOUSDT")]}

    rng = np.random.default_rng(7)
    close_a = pd.Series(100 + rng.normal(0, 1, 120).cumsum())
    close_b = pd.Series(100 + rng.normal(0, 4, 120).cumsum())
    frames = {
        ("FETUSDT", "1h"): pd.DataFrame({"close": close_a}),
        ("TAOUSDT", "1h"): pd.DataFrame({"close": close_b}),
    }
    monkeypatch.setattr("agents.pairs_trading_agent.data_store.get_df", lambda s, i: frames.get((s, i)))

    result = agent.analyse("FETUSDT", "1h")
    assert result is None
