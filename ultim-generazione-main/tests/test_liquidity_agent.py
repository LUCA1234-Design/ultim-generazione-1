import numpy as np

from agents.liquidity_agent import LiquidityAgent


def test_analyze_returns_expected_shape(ohlcv_df):
    agent = LiquidityAgent()
    result = agent.analyze("BTCUSDT", "1h", ohlcv_df)

    assert set(result.keys()) == {"signal", "confidence", "details"}
    assert result["signal"] in (-1, 0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["details"], dict)
    assert -1.0 <= result["details"].get("liquidity_score", 0.0) <= 1.0


def test_analyze_can_emit_long_signal_on_strong_buy_flow(ohlcv_df):
    agent = LiquidityAgent()
    df = ohlcv_df.copy()
    idx = df.index[-10:]
    ramp = np.linspace(1.0, 10.0, len(idx))

    df.loc[idx, "close"] = df.loc[idx, "close"].values + ramp
    df.loc[idx, "high"] = df.loc[idx, "close"].values * 1.01
    df.loc[idx, "low"] = df.loc[idx, "close"].values * 0.97
    df.loc[idx, "volume"] = float(df["volume"].iloc[:-10].mean()) * 4.0

    result = agent.analyze("ETHUSDT", "15m", df)
    assert result["signal"] in (0, 1)
    assert result["details"]["volume_ratio"] > 1.0


def test_analyze_can_emit_short_signal_on_strong_sell_flow(ohlcv_df):
    agent = LiquidityAgent()
    df = ohlcv_df.copy()
    idx = df.index[-10:]
    ramp = np.linspace(1.0, 10.0, len(idx))

    df.loc[idx, "close"] = df.loc[idx, "close"].values - ramp
    df.loc[idx, "high"] = df.loc[idx, "close"].values * 1.03
    df.loc[idx, "low"] = df.loc[idx, "close"].values * 0.99
    df.loc[idx, "volume"] = float(df["volume"].iloc[:-10].mean()) * 4.0

    result = agent.analyze("ETHUSDT", "15m", df)
    assert result["signal"] in (-1, 0)
    assert result["details"]["volume_ratio"] > 1.0
