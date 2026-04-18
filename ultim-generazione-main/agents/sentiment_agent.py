import logging
import os
import re
import threading
import xml.etree.ElementTree as ET
from typing import Callable, Iterable, List, Optional

import requests

from engine.memory_manager import RedisMemoryManager

logger = logging.getLogger("SentimentAgent")
_MIN_SECONDS = 60
_HTTP_TIMEOUT_SECONDS = 8.0
_DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
_DEFAULT_LM_STUDIO_MODEL = "qwen2.5-1.5b-instruct"
_CRYPTO_PANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_RSS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)


class SentimentAgent:
    """Background sentiment updater with pluggable fetcher."""

    def __init__(
        self,
        memory_manager: RedisMemoryManager,
        symbols_provider: Callable[[], Iterable[str]],
        update_interval_seconds: int = 900,
        sentiment_fetcher: Optional[Callable[[str], float]] = None,
        ttl_seconds: int = 1800,
        crypto_panic_api_key: Optional[str] = None,
        lm_studio_url: Optional[str] = None,
        lm_studio_model: Optional[str] = None,
    ):
        self.memory_manager = memory_manager
        self.symbols_provider = symbols_provider
        self.update_interval_seconds = max(int(update_interval_seconds), _MIN_SECONDS)
        self.crypto_panic_api_key = (
            str(crypto_panic_api_key).strip()
            if crypto_panic_api_key is not None
            else str(os.getenv("CRYPTO_PANIC_API_KEY", "")).strip()
        )
        self.lm_studio_url = (
            str(lm_studio_url).strip()
            if lm_studio_url is not None
            else str(os.getenv("LM_STUDIO_URL", _DEFAULT_LM_STUDIO_URL)).strip()
        )
        self.lm_studio_model = (
            str(lm_studio_model).strip()
            if lm_studio_model is not None
            else str(os.getenv("LM_STUDIO_MODEL", _DEFAULT_LM_STUDIO_MODEL)).strip()
        )
        if sentiment_fetcher is None:
            self.sentiment_fetcher = self._fetch_real_sentiment
            logger.info("SentimentAgent using real news + LM Studio sentiment fetcher.")
        else:
            self.sentiment_fetcher = sentiment_fetcher
        self.ttl_seconds = max(int(ttl_seconds), _MIN_SECONDS)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _extract_sentiment_score(raw_text: str) -> float:
        text = str(raw_text or "").strip().replace(",", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            raise ValueError(f"No numeric sentiment score found in: {raw_text!r}")
        return max(min(float(match.group(0)), 1.0), -1.0)

    @staticmethod
    def _symbol_aliases(symbol: str) -> List[str]:
        upper_symbol = str(symbol).upper()
        aliases = {upper_symbol}
        for suffix in ("USDT", "USD", "BUSD", "USDC", "PERP"):
            if upper_symbol.endswith(suffix) and len(upper_symbol) > len(suffix):
                aliases.add(upper_symbol[: -len(suffix)])
        return sorted(aliases)

    @staticmethod
    def _headline_matches_symbol(headline: str, aliases: List[str]) -> bool:
        headline_upper = str(headline or "").upper()
        return any(alias and alias in headline_upper for alias in aliases)

    def _chat_completions_url(self) -> str:
        base = self.lm_studio_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _fetch_news_headlines(self, symbol: str, limit: int = 3) -> List[str]:
        aliases = self._symbol_aliases(symbol)

        if self.crypto_panic_api_key:
            try:
                response = requests.get(
                    _CRYPTO_PANIC_URL,
                    params={"auth_token": self.crypto_panic_api_key, "public": "true"},
                    timeout=_HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                posts = payload.get("results", []) if isinstance(payload, dict) else []
                titles = [
                    str(item.get("title")).strip()
                    for item in posts
                    if isinstance(item, dict) and item.get("title")
                ]
                filtered = [title for title in titles if self._headline_matches_symbol(title, aliases)]
                if filtered:
                    return filtered[:limit]
                if titles:
                    return titles[:limit]
            except Exception as exc:
                logger.warning(f"CryptoPanic fetch failed for {symbol}, falling back to RSS: {exc}")

        rss_titles: List[str] = []
        filtered_rss_titles: List[str] = []
        for feed_url in _RSS_FEEDS:
            try:
                response = requests.get(feed_url, timeout=_HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                root = ET.fromstring(response.text)
                for node in root.iter():
                    if not node.tag.lower().endswith("title"):
                        continue
                    title = str((node.text or "")).strip()
                    if not title:
                        continue
                    rss_titles.append(title)
                    if self._headline_matches_symbol(title, aliases):
                        filtered_rss_titles.append(title)
            except Exception as exc:
                logger.warning(f"RSS fetch failed from {feed_url}: {exc}")

        if filtered_rss_titles:
            return filtered_rss_titles[:limit]
        if rss_titles:
            return rss_titles[:limit]
        raise RuntimeError("Unable to fetch crypto news headlines from CryptoPanic or RSS feeds.")

    def _analyze_headline_with_llm(self, symbol: str, headline: str) -> float:
        prompt = (
            f"Analyze the sentiment of this headline for {symbol}: {headline!r}. "
            "Respond ONLY with a single float number between -1.0 "
            "(extreme negative) and 1.0 (extreme positive)."
        )
        response = requests.post(
            self._chat_completions_url(),
            json={
                "model": self.lm_studio_model,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a sentiment scorer for crypto headlines. "
                            "Return only one float in [-1, 1]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        content = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content", ""))
            else:
                content = str(choices[0].get("text", ""))
        return self._extract_sentiment_score(content)

    def _fetch_real_sentiment(self, symbol: str) -> float:
        try:
            headlines = self._fetch_news_headlines(symbol)
        except Exception as exc:
            logger.warning(f"News fetch failed for {symbol}; using neutral sentiment 0.0. error={exc}")
            return 0.0

        scores: List[float] = []
        for headline in headlines:
            try:
                scores.append(self._analyze_headline_with_llm(symbol, headline))
            except Exception as exc:
                logger.warning(
                    f"LM Studio sentiment analysis failed for {symbol}; headline={headline!r}. "
                    f"Using neutral fallback if needed. error={exc}"
                )

        if not scores:
            return 0.0
        return max(min(sum(scores) / len(scores), 1.0), -1.0)

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
