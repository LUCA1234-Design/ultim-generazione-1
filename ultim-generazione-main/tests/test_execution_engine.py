"""
Unit tests for ExecutionEngine TP1 scale-out and dynamic trailing stop behavior.
"""
import pytest

from engine.execution import ExecutionEngine


class TestExecutionEngineScaleOutAndTrailing:
    def test_tp1_scale_out_realizes_pnl_and_moves_sl_to_breakeven_long(self):
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        pos = engine.open_position(
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            entry_price=100.0,
            size=2.0,
            sl=95.0,
            tp1=110.0,
            tp2=120.0,
        )
        assert pos is not None

        closed = engine.check_position_levels("BTCUSDT", 110.0)
        assert closed == []
        stats = engine.get_stats()
        assert stats["balance"] == pytest.approx(1010.0)
        assert stats["total_pnl"] == pytest.approx(10.0)
        assert stats["daily_pnl"] == pytest.approx(10.0)
        assert pos.tp1_hit is True
        assert pos.sl == pytest.approx(100.0)
        assert pos.size == pytest.approx(1.0)
        assert pos.realized_pnl == pytest.approx(10.0)

    def test_tp1_scale_out_realizes_pnl_and_moves_sl_to_breakeven_short(self):
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        pos = engine.open_position(
            symbol="BTCUSDT",
            interval="1h",
            direction="short",
            entry_price=100.0,
            size=2.0,
            sl=105.0,
            tp1=90.0,
            tp2=80.0,
        )
        assert pos is not None

        closed = engine.check_position_levels("BTCUSDT", 90.0)
        assert closed == []
        stats = engine.get_stats()
        assert stats["balance"] == pytest.approx(1010.0)
        assert stats["total_pnl"] == pytest.approx(10.0)
        assert stats["daily_pnl"] == pytest.approx(10.0)
        assert pos.tp1_hit is True
        assert pos.sl == pytest.approx(100.0)
        assert pos.size == pytest.approx(1.0)
        assert pos.realized_pnl == pytest.approx(10.0)

    def test_dynamic_trailing_tightens_without_moving_backwards_long(self):
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        pos = engine.open_position(
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            entry_price=100.0,
            size=2.0,
            sl=95.0,
            tp1=110.0,
            tp2=120.0,
        )
        assert pos is not None

        engine.check_position_levels("BTCUSDT", 110.0)  # hit TP1
        sl_after_tp1 = pos.sl

        engine.check_position_levels("BTCUSDT", 112.0)
        sl_after_112 = pos.sl
        dist_112 = engine._dynamic_trail_distance(pos, 112.0)
        dist_118 = engine._dynamic_trail_distance(pos, 118.0)
        engine.check_position_levels("BTCUSDT", 111.0)  # lower price should not lower SL
        sl_after_pullback = pos.sl
        engine.check_position_levels("BTCUSDT", 118.0)
        sl_after_118 = pos.sl

        assert sl_after_112 > sl_after_tp1
        assert sl_after_pullback == pytest.approx(sl_after_112)
        assert sl_after_118 >= sl_after_112
        assert dist_118 < dist_112

    def test_dynamic_trailing_tightens_without_moving_backwards_short(self):
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        pos = engine.open_position(
            symbol="BTCUSDT",
            interval="1h",
            direction="short",
            entry_price=100.0,
            size=2.0,
            sl=105.0,
            tp1=90.0,
            tp2=80.0,
        )
        assert pos is not None

        engine.check_position_levels("BTCUSDT", 90.0)  # hit TP1
        sl_after_tp1 = pos.sl

        engine.check_position_levels("BTCUSDT", 88.0)
        sl_after_88 = pos.sl
        dist_88 = engine._dynamic_trail_distance(pos, 88.0)
        dist_82 = engine._dynamic_trail_distance(pos, 82.0)
        engine.check_position_levels("BTCUSDT", 89.0)  # rebound should not raise SL
        sl_after_rebound = pos.sl
        engine.check_position_levels("BTCUSDT", 82.0)
        sl_after_82 = pos.sl

        assert sl_after_88 < sl_after_tp1
        assert sl_after_rebound == pytest.approx(sl_after_88)
        assert sl_after_82 <= sl_after_88
        assert dist_82 < dist_88
