import pandas as pd

from agents.sector_rotation_agent import SectorRotationAgent


def test_sector_rotation_boosts_hot_sector(monkeypatch):
    agent = SectorRotationAgent()
    rising = pd.DataFrame(
        {
            "close": pd.Series(range(100, 130), dtype=float),
            "volume": pd.Series([100.0] * 29 + [300.0], dtype=float),
        }
    )

    monkeypatch.setattr("agents.sector_rotation_agent.data_store.get_df", lambda *_args, **_kwargs: rising)
    result = agent.analyse("FETUSDT", "1h", direction="long")

    assert result.metadata["sector"] == "ai"
    assert result.metadata["confluence_adjustment"] > 0


def test_sector_rotation_penalizes_dead_sector(monkeypatch):
    agent = SectorRotationAgent()
    falling = pd.DataFrame(
        {
            "close": pd.Series(range(130, 100, -1), dtype=float),
            "volume": pd.Series([100.0] * 29 + [220.0], dtype=float),
        }
    )

    monkeypatch.setattr("agents.sector_rotation_agent.data_store.get_df", lambda *_args, **_kwargs: falling)
    result = agent.analyse("FETUSDT", "1h", direction="long")

    assert result.metadata["confluence_adjustment"] < 0
