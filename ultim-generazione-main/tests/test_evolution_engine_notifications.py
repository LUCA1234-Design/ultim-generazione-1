from unittest.mock import MagicMock

from evolution.evolution_engine import EvolutionEngine


class DummyMetaAgent:
    def __init__(self, weight_map=None):
        self._weight_map = weight_map or {}

    def adjust_weights(self):
        return dict(self._weight_map)

    def save_state(self):
        return True

    def load_state(self):
        return False


class DummyFusion:
    def __init__(self, threshold=0.6):
        self._threshold = threshold
        self.updated_weights = None

    def update_weights(self, weight_map):
        self.updated_weights = dict(weight_map)

    def get_threshold_history(self):
        return []

    def set_threshold_history(self, _history):
        return None


class DummyRisk:
    pass


class DummyStrategy:
    def update_strategy_outcome(self, _name, _was_profitable):
        return None

    def prune_and_evolve(self, **_kwargs):
        return []


class DummyConfluence:
    def __init__(self):
        self._tf_weights = {"15m": 0.33, "1h": 0.33, "4h": 0.34}

    def update_tf_weights(self, new_weights):
        self._tf_weights = dict(new_weights)


class DummyTracker:
    def update_risk_agent_win_rates(self, _risk):
        return None

    def get_summary(self):
        return {"max_drawdown": 0.0}


def _build_engine(weight_map=None, threshold=0.6):
    return EvolutionEngine(
        DummyMetaAgent(weight_map=weight_map),
        DummyFusion(threshold=threshold),
        DummyRisk(),
        DummyStrategy(),
        DummyConfluence(),
        DummyTracker(),
    )


def test_tick_sends_telegram_when_weights_are_updated(monkeypatch):
    engine = _build_engine(weight_map={"pattern": 0.7, "risk": 1.234})
    send_mock = MagicMock()
    monkeypatch.setattr("evolution.evolution_engine.send_message", send_mock)

    engine._auto_tune_params = MagicMock()
    engine._save_state = MagicMock()
    engine._check_drawdown = MagicMock()
    engine._confluence_adapter.maybe_adapt = MagicMock()
    engine._last_tune = float("inf")
    engine._last_save = float("inf")

    engine.tick()

    send_mock.assert_called_once()
    msg = send_mock.call_args[0][0]
    assert "🧠 *AUTO-APPRENDIMENTO V17*" in msg
    assert "• pattern: 0.70" in msg
    assert "• risk: 1.23" in msg


def test_auto_tune_params_sends_telegram_when_threshold_changes(monkeypatch):
    engine = _build_engine(threshold=0.6)
    send_mock = MagicMock()
    save_param_mock = MagicMock()
    completed = [
        {"outcome": "win", "pnl": 1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
        {"outcome": "loss", "pnl": -1.0},
    ]
    monkeypatch.setattr("evolution.evolution_engine.send_message", send_mock)
    monkeypatch.setattr(
        "evolution.evolution_engine.experience_db.get_recent_decisions",
        lambda limit=50: completed,
    )
    monkeypatch.setattr(
        "evolution.evolution_engine.experience_db.save_param",
        save_param_mock,
    )

    engine._auto_tune_params()

    assert engine._fusion._threshold == 0.62
    send_mock.assert_called_once()
    msg = send_mock.call_args[0][0]
    assert "🔧 *AUTO-TUNE V17*" in msg
    assert "Win Rate recente: 10.0%" in msg
    assert "Soglia precisione: 0.600 ➡️ 0.620" in msg

