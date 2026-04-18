from agents.sentiment_agent import SentimentAgent
from requests.exceptions import ConnectionError


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


def test_extract_sentiment_score_parses_numeric_text():
    assert SentimentAgent._extract_sentiment_score("0.45") == 0.45
    assert SentimentAgent._extract_sentiment_score("score: -2.7") == -1.0
    assert SentimentAgent._extract_sentiment_score("score: 2.7") == 1.0


def test_sentiment_agent_returns_neutral_if_news_fetch_fails(monkeypatch):
    memory = MemoryManagerSpy()
    agent = SentimentAgent(
        memory_manager=memory,
        symbols_provider=lambda: ["BTCUSDT"],
        update_interval_seconds=300,
    )

    def fail_news_fetch(_symbol: str, limit: int = 3):
        raise RuntimeError("news API unavailable")

    monkeypatch.setattr(agent, "_fetch_news_headlines", fail_news_fetch)
    agent.update_once()

    assert memory.values["BTCUSDT"][0] == 0.0


def test_sentiment_agent_returns_neutral_if_lm_studio_unreachable(monkeypatch):
    memory = MemoryManagerSpy()
    agent = SentimentAgent(
        memory_manager=memory,
        symbols_provider=lambda: ["BTCUSDT"],
        update_interval_seconds=300,
    )

    monkeypatch.setattr(agent, "_fetch_news_headlines", lambda _symbol, limit=3: ["Bitcoin ETF approved"])

    def fail_llm(*_args, **_kwargs):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr("agents.sentiment_agent.requests.post", fail_llm)
    agent.update_once()

    assert memory.values["BTCUSDT"][0] == 0.0
