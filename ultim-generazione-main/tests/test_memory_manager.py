from engine.memory_manager import RedisMemoryManager


class InMemoryRedisClient:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = str(value)

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def set(self, key, value, ex=None):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


def test_memory_manager_fallback_stores_and_reads_values():
    manager = RedisMemoryManager()
    manager._redis_available = False
    manager._redis_client = None

    manager.set_agent_score("BTCUSDT", "1h", "pattern", 0.82)
    scores = manager.get_agent_scores("BTCUSDT", "1h")
    assert scores["pattern"] == 0.82

    manager.store_fusion_result("BTCUSDT", "1h", {"decision": "long", "final_score": 0.91})
    fusion = manager.get_recent_fusion("BTCUSDT", "1h")
    assert fusion["decision"] == "long"
    assert fusion["final_score"] == 0.91


def test_memory_manager_uses_redis_client_when_available():
    manager = RedisMemoryManager()
    manager._redis_available = True
    manager._redis_client = InMemoryRedisClient()

    manager.set_agent_score("ETHUSDT", "15m", "regime", 0.64)
    scores = manager.get_agent_scores("ETHUSDT", "15m")
    assert scores["regime"] == 0.64

    payload = {"decision": "hold", "final_score": 0.44}
    manager.store_fusion_result("ETHUSDT", "15m", payload)
    fusion = manager.get_recent_fusion("ETHUSDT", "15m")
    assert fusion["decision"] == "hold"
    assert fusion["final_score"] == 0.44
    assert "stored_at" in fusion
