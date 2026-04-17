"""
Redis-backed memory manager with safe in-process fallback.
"""
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

try:
    import redis
except ImportError:  # pragma: no cover - import exception branch only
    redis = None

logger = logging.getLogger("RedisMemoryManager")


class RedisMemoryManager:
    """Store fast-changing runtime memory in Redis with graceful fallback."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._fallback_agent_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._fallback_recent_fusion: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._redis_client = None
        self._redis_available = False

        if redis is None:
            logger.warning("Redis package not available, using in-memory fallback.")
            return

        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                health_check_interval=30,
            )
            self._redis_client = redis.Redis(connection_pool=pool)
            self._redis_client.ping()
            self._redis_available = True
            logger.info(f"Connected to Redis at {host}:{port} (db={db})")
        except Exception as exc:
            logger.warning(
                f"Redis unavailable at {host}:{port} (db={db}), using in-memory fallback. error={exc}"
            )
            self._redis_client = None
            self._redis_available = False

    @staticmethod
    def _score_key(symbol: str, timeframe: str) -> str:
        return f"memory:agent_scores:{symbol}:{timeframe}"

    @staticmethod
    def _fusion_key(symbol: str, timeframe: str) -> str:
        return f"memory:fusion:{symbol}:{timeframe}:latest"

    @staticmethod
    def _dict_key(symbol: str, timeframe: str) -> Tuple[str, str]:
        return symbol, timeframe

    def set_agent_score(self, symbol: str, timeframe: str, agent_name: str, score: float) -> None:
        fallback_key = self._dict_key(symbol, timeframe)
        if not self._redis_available or self._redis_client is None:
            self._fallback_agent_scores.setdefault(fallback_key, {})[agent_name] = float(score)
            return
        try:
            self._redis_client.hset(
                self._score_key(symbol, timeframe),
                agent_name,
                float(score),
            )
        except Exception as exc:
            logger.error(f"Redis set_agent_score failed, using fallback only. error={exc}")
            self._redis_available = False
            self._fallback_agent_scores.setdefault(fallback_key, {})[agent_name] = float(score)

    def get_agent_scores(self, symbol: str, timeframe: str) -> Dict[str, float]:
        fallback = dict(self._fallback_agent_scores.get(self._dict_key(symbol, timeframe), {}))
        if not self._redis_available or self._redis_client is None:
            return fallback

        try:
            payload = self._redis_client.hgetall(self._score_key(symbol, timeframe))
            if not payload:
                return fallback
            return {k: float(v) for k, v in payload.items()}
        except Exception as exc:
            logger.error(f"Redis get_agent_scores failed, using fallback. error={exc}")
            self._redis_available = False
            return fallback

    def store_fusion_result(self, symbol: str, timeframe: str, result_dict: Dict[str, Any]) -> None:
        payload = dict(result_dict or {})
        payload.setdefault("stored_at", time.time())
        fallback_key = self._dict_key(symbol, timeframe)

        if not self._redis_available or self._redis_client is None:
            self._fallback_recent_fusion[fallback_key] = payload
            return

        try:
            self._redis_client.set(
                self._fusion_key(symbol, timeframe),
                json.dumps(payload),
                ex=300,
            )
        except Exception as exc:
            logger.error(f"Redis store_fusion_result failed, using fallback only. error={exc}")
            self._redis_available = False
            self._fallback_recent_fusion[fallback_key] = payload

    def get_recent_fusion(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        fallback = self._fallback_recent_fusion.get(self._dict_key(symbol, timeframe))
        if not self._redis_available or self._redis_client is None:
            return dict(fallback) if isinstance(fallback, dict) else None

        try:
            payload = self._redis_client.get(self._fusion_key(symbol, timeframe))
            if payload:
                parsed = json.loads(payload)
                return parsed if isinstance(parsed, dict) else fallback
            return dict(fallback) if isinstance(fallback, dict) else None
        except Exception as exc:
            logger.error(f"Redis get_recent_fusion failed, using fallback. error={exc}")
            self._redis_available = False
            return dict(fallback) if isinstance(fallback, dict) else None
