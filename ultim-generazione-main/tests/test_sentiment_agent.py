from agents.sentiment_agent import SentimentAgent


class MemoryManagerSpy:
    def __init__(self):
        self.values = {}

    def set_sentiment_score(self, symbol, score, ttl_seconds=1800):
        self.values[symbol] = (float(score), int(ttl_seconds))


def test_sentiment_agent_updates_symbols_once():
    memory = MemoryManagerSpy()
    symbols = ["BTCUSDT", "ETHUSDT"]

    def fetcher(symbol: str) -> float:
        return {"BTCUSDT": -0.7, "ETHUSDT": 0.4}.get(symbol, 0.0)

    agent = SentimentAgent(
        memory_manager=memory,
        symbols_provider=lambda: symbols,
        update_interval_seconds=300,
        sentiment_fetcher=fetcher,
        ttl_seconds=1200,
    )

    agent.update_once()

    assert memory.values["BTCUSDT"] == (-0.7, 1200)
    assert memory.values["ETHUSDT"] == (0.4, 1200)


def test_sentiment_agent_clamps_scores_between_minus_one_and_one():
    memory = MemoryManagerSpy()
    agent = SentimentAgent(
        memory_manager=memory,
        symbols_provider=lambda: ["BTCUSDT"],
        update_interval_seconds=300,
        sentiment_fetcher=lambda _symbol: 4.0,
    )

    agent.update_once()
    assert memory.values["BTCUSDT"][0] == 1.0
