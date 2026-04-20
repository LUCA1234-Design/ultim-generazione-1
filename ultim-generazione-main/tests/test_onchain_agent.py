from agents.onchain_agent import OnChainAgent


def test_onchain_agent_returns_placeholder_without_api_key(monkeypatch):
    agent = OnChainAgent()
    monkeypatch.setattr(agent, "_fetch_whale_events", lambda _symbol: [])
    result = agent.analyse("BTCUSDT", "1h", None)

    assert result is not None
    assert result.agent_name == "onchain"
    assert result.direction in {"neutral", "long", "short"}
    assert "onchain:" in result.details[0]


def test_onchain_agent_flags_bearish_when_token_moves_to_exchange(monkeypatch):
    agent = OnChainAgent()
    monkeypatch.setattr(
        agent,
        "_fetch_whale_events",
        lambda _symbol: [
            {"token": "BTC", "amount_usd": 5_000_000, "direction": "to_exchange"},
        ],
    )
    result = agent.analyse("BTCUSDT", "1h", None)

    assert result is not None
    assert result.direction == "short"
    assert result.metadata.get("alert_count", 0) >= 1
