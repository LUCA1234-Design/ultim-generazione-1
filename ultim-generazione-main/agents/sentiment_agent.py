import logging
import random
import threading
from typing import Callable, Iterable, List, Optional

from engine.memory_manager import RedisMemoryManager

logger = logging.getLogger("SentimentAgent")


class SentimentAgent:
    """Background sentiment updater with pluggable fetcher."""

    def __init__(
        self,
        memory_manager: RedisMemoryManager,
        symbols_provider: Callable[[], Iterable[str]],
        update_interval_seconds: int = 900,
        sentiment_fetcher: Optional[Callable[[str], float]] = None,
        ttl_seconds: int = 1800,
    ):
        self.memory_manager = memory_manager
        self.symbols_provider = symbols_provider
        self.update_interval_seconds = max(int(update_interval_seconds), 60)
        self.sentiment_fetcher = sentiment_fetcher or self._mock_fetch_sentiment
        self.ttl_seconds = max(int(ttl_seconds), 60)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _mock_fetch_sentiment(_symbol: str) -> float:
        return random.uniform(-1.0, 1.0)

    def update_once(self) -> None:
        symbols: List[str] = [str(sym) for sym in self.symbols_provider() if sym]
        if not symbols:
            return

        updated = 0
        for symbol in symbols:
            try:
                score = float(self.sentiment_fetcher(symbol))
                score = max(min(score, 1.0), -1.0)
                self.memory_manager.set_sentiment_score(
                    symbol=symbol,
                    score=score,
                    ttl_seconds=self.ttl_seconds,
                )
                updated += 1
            except Exception as exc:
                logger.debug(f"Sentiment fetch/store error for {symbol}: {exc}")

        logger.info(f"📰 Sentiment updated for {updated}/{len(symbols)} symbols")

    def _run(self) -> None:
        logger.info(
            f"🧠 SentimentAgent started (interval={self.update_interval_seconds}s, ttl={self.ttl_seconds}s)"
        )
        while not self._stop_event.is_set():
            try:
                self.update_once()
            except Exception as exc:
                logger.error(f"SentimentAgent cycle error: {exc}")
            self._stop_event.wait(self.update_interval_seconds)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SentimentAgent",
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
