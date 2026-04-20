import numpy as np

from monte_carlo_trainer import (
    build_ohlcv_from_close,
    estimate_drift_vol,
    klines_to_dataframe,
    simulate_close_path,
)


def test_klines_to_dataframe_parses_binance_shape():
    df = klines_to_dataframe(
        [
            [0, "100", "105", "99", "103", "1200"],
            [1, "103", "106", "102", "104", "1500"],
        ]
    )
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert float(df["close"].iloc[-1]) == 104.0


def test_simulate_close_path_and_ohlcv_generation_are_finite():
    rng = np.random.default_rng(1)
    close = simulate_close_path(initial_price=100.0, drift=0.0002, vol=0.01, steps=200, rng=rng)
    assert close.shape[0] == 200
    assert np.isfinite(close).all()
    assert (close > 0).all()

    df = build_ohlcv_from_close(close, rng)
    assert len(df) == 200
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert (df["close"] > 0).all()


def test_estimate_drift_vol_returns_non_negative_vol():
    close = klines_to_dataframe(
        [[i, str(100 + i), str(101 + i), str(99 + i), str(100 + i), "1000"] for i in range(1, 50)]
    )["close"]
    drift, vol = estimate_drift_vol(close)
    assert isinstance(drift, float)
    assert isinstance(vol, float)
    assert vol >= 0.0
