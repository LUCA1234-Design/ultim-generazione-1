"""
Unit tests for DecisionFusion.
"""
import pytest

from engine.decision_fusion import (
    DecisionFusion, FusionResult, DECISION_LONG, DECISION_SHORT, DECISION_HOLD,
)
from agents.base_agent import AgentResult
from tests.conftest import make_agent_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_results(score=0.65, direction="long", confidence=0.8):
    """Return a minimal set of agent results for fusion."""
    return {
        "pattern": make_agent_result("pattern", score=score, direction=direction, confidence=confidence),
        "regime": make_agent_result("regime", score=score, direction=direction, confidence=confidence),
        "confluence": make_agent_result("confluence", score=score, direction=direction, confidence=confidence),
    }


class MemoryManagerSpy:
    def __init__(self):
        self.scores = []
        self.fusions = []

    def set_agent_score(self, symbol, timeframe, agent_name, score):
        self.scores.append((symbol, timeframe, agent_name, score))

    def get_agent_scores(self, symbol, timeframe):
        return {}

    def store_fusion_result(self, symbol, timeframe, result_dict):
        self.fusions.append((symbol, timeframe, result_dict))

    def get_recent_fusion(self, symbol, timeframe):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecisionFusion:
    @pytest.fixture(autouse=True)
    def fusion(self):
        self.fusion = DecisionFusion(threshold=0.55)

    # ------------------------------------------------------------------
    # Basic happy-path
    # ------------------------------------------------------------------

    def test_fuse_returns_fusion_result(self):
        result = self.fusion.fuse("BTCUSDT", "1h", make_results(score=0.70))
        assert isinstance(result, FusionResult)

    def test_fuse_long_above_threshold(self):
        result = self.fusion.fuse("BTCUSDT", "1h", make_results(score=0.80, direction="long"))
        assert result.decision == DECISION_LONG
        assert result.should_trade() is True

    def test_fuse_short_above_threshold(self):
        result = self.fusion.fuse("BTCUSDT", "1h", make_results(score=0.80, direction="short"))
        assert result.decision == DECISION_SHORT
        assert result.should_trade() is True

    def test_fuse_hold_below_threshold(self):
        result = self.fusion.fuse("BTCUSDT", "1h", make_results(score=0.20))
        assert result.decision == DECISION_HOLD
        assert result.should_trade() is False

    # ------------------------------------------------------------------
    # Score and direction fields
    # ------------------------------------------------------------------

    def test_final_score_within_bounds(self, mock_agent_results):
        result = self.fusion.fuse("BTCUSDT", "1h", mock_agent_results)
        assert 0.0 <= result.final_score <= 1.0

    def test_direction_from_votes(self):
        results = make_results(direction="short", score=0.70)
        result = self.fusion.fuse("BTCUSDT", "1h", results)
        assert result.direction == "short"

    def test_result_symbol_and_interval(self):
        result = self.fusion.fuse("ETHUSDT", "15m", make_results())
        assert result.symbol == "ETHUSDT"
        assert result.interval == "15m"

    # ------------------------------------------------------------------
    # Empty / None agent results
    # ------------------------------------------------------------------

    def test_empty_agent_results_returns_hold(self):
        result = self.fusion.fuse("BTCUSDT", "1h", {})
        assert result.decision == DECISION_HOLD
        assert result.final_score == 0.0

    def test_none_agent_values_are_skipped(self):
        results = {
            "pattern": None,
            "regime": make_agent_result("regime", score=0.70, direction="long"),
        }
        result = self.fusion.fuse("BTCUSDT", "1h", results)
        assert isinstance(result, FusionResult)

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def test_update_weight(self):
        self.fusion.update_weight("pattern", 3.0)
        assert self.fusion._weights["pattern"] == 3.0

    def test_update_weight_clamped(self):
        self.fusion.update_weight("pattern", -100)
        assert self.fusion._weights["pattern"] >= 0.01

    def test_liquidity_weight_is_initialized(self):
        assert "liquidity" in self.fusion._weights

    def test_update_weights_bulk(self):
        self.fusion.update_weights({"pattern": 2.0, "regime": 1.5})
        assert self.fusion._weights["pattern"] == 2.0
        assert self.fusion._weights["regime"] == 1.5

    def test_phase14_weights_present(self):
        assert "onchain" in self.fusion._weights
        assert "neural_predict" in self.fusion._weights

    def test_evaluate_includes_liquidity_in_results(self, ohlcv_df):
        results = make_results(score=0.70, direction="long")
        fusion_result = self.fusion.evaluate("BTCUSDT", "1h", ohlcv_df, agent_results=results)

        assert isinstance(fusion_result, FusionResult)
        assert "liquidity" in results
        assert "liquidity" in fusion_result.agent_results

    # ------------------------------------------------------------------
    # Threshold adaptation
    # ------------------------------------------------------------------

    def test_adapt_threshold_raises_on_poor_accuracy(self):
        initial = self.fusion._threshold
        for _ in range(25):
            self.fusion.adapt_threshold(False, 0.5)
        assert self.fusion._threshold >= initial

    def test_adapt_threshold_lowers_on_good_accuracy(self):
        self.fusion._threshold = 0.75
        for _ in range(25):
            self.fusion.adapt_threshold(True, 0.8)
        assert self.fusion._threshold <= 0.75

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------

    def test_decision_log_appended(self):
        self.fusion.fuse("BTCUSDT", "1h", make_results())
        self.fusion.fuse("BTCUSDT", "1h", make_results())
        log = self.fusion.get_decision_log(limit=10)
        assert len(log) >= 2

    def test_decision_log_limit(self):
        for _ in range(5):
            self.fusion.fuse("BTCUSDT", "1h", make_results())
        log = self.fusion.get_decision_log(limit=3)
        assert len(log) <= 3

    def test_fusion_result_to_dict(self, mock_agent_results):
        result = self.fusion.fuse("BTCUSDT", "1h", mock_agent_results)
        d = result.to_dict()
        assert "decision" in d
        assert "final_score" in d
        assert "reasoning" in d
        assert "threshold" in d

    def test_fuse_persists_scores_and_fusion_to_memory_manager(self):
        memory = MemoryManagerSpy()
        fusion = DecisionFusion(threshold=0.55, memory_manager=memory)
        result = fusion.fuse("BTCUSDT", "1h", make_results(score=0.7))

        assert len(memory.scores) == 3
        assert len(memory.fusions) == 1
        assert {entry[2]: entry[3] for entry in memory.scores} == {
            "pattern": 0.7,
            "regime": 0.7,
            "confluence": 0.7,
        }
        stored = memory.fusions[0][2]
        assert stored["symbol"] == "BTCUSDT"
        assert stored["interval"] == "1h"
        assert stored["decision"] == result.decision
