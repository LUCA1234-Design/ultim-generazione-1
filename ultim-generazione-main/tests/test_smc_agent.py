import pandas as pd

from agents.smc_agent import SMCAgent


def _df_for_bullish_fvg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100, 101, 110, 112, 114, 116, 117, 118, 120, 121],
            "high": [101, 102, 111, 113, 115, 117, 118, 119, 121, 122],
            "low": [99, 100, 109, 111, 113, 115, 116, 117, 119, 123],
            "close": [100, 101, 110, 112, 114, 116, 117, 118, 120, 121],
            "volume": [1000] * 10,
        }
    )


def test_smc_detects_fvg_and_returns_limit_entry():
    agent = SMCAgent()
    df = _df_for_bullish_fvg()
    result = agent.analyse("SOLUSDT", "1h", df, direction="long")

    assert result is not None
    assert result.agent_name == "smc"
    assert result.direction == "long"
    assert result.metadata.get("setup", "").startswith("fvg_")
    assert result.metadata.get("limit_entry", 0) > 0
