import pandas as pd

from agents.neural_predict_agent import NeuralPredictAgent


def _df_from_closes(closes):
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": pd.Series([100.0] * len(c)),
        }
    )


def test_neural_predict_agent_long_on_uptrend():
    closes = [100 + i for i in range(80)]
    result = NeuralPredictAgent().analyse("BTCUSDT", "1h", _df_from_closes(closes))
    assert result is not None
    assert result.direction == "long"
    assert 0.0 <= result.metadata["prob_bull"] <= 1.0


def test_neural_predict_agent_short_on_downtrend():
    closes = [200 - i for i in range(80)]
    result = NeuralPredictAgent().analyse("BTCUSDT", "1h", _df_from_closes(closes))
    assert result is not None
    assert result.direction == "short"
    assert 0.0 <= result.metadata["prob_bear"] <= 1.0
