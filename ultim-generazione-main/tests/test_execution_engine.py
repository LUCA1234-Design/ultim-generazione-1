"""
Unit tests for ExecutionEngine TP1 scale-out and dynamic trailing stop behavior.
"""
import pytest

from engine.execution import ExecutionEngine


class TestExecutionEngineScaleOutAndTrailing:
    def test_restores_paper_trade_count_from_experience_db(self, monkeypatch):
        monkeypatch.setattr(
            "engine.execution.experience_db.get_completed_trade_count",
            lambda paper_only=None: 45 if paper_only else 99,
        )
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        assert engine.get_stats()["trade_count"] == 45

    def test_restores_live_trade_count_from_experience_db(self, monkeypatch):
        monkeypatch.setattr(
            "engine.execution.experience_db.get_completed_trade_count",
            lambda paper_only=None: 45 if paper_only else 99,
        )
        engine = ExecutionEngine(paper_trading=False, initial_balance=1000.0)
        assert engine.get_stats()["trade_count"] == 99

    def test_position_tracks_is_tp1_hit_and_current_size(self):
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
        assert pos.is_tp1_hit is False
        assert pos.size == pytest.approx(2.0)
        assert pos.initial_size == pytest.approx(2.0)

        engine.check_position_levels("BTCUSDT", 110.0)
        assert pos.tp1_hit is True
        assert pos.is_tp1_hit is True
        assert pos.size == pytest.approx(1.0)

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
        engine.check_position_levels("BTCUSDT", 111.6)  # pullback above SL: no close
        sl_after_pullback = pos.sl
        engine.check_position_levels("BTCUSDT", 118.0)
        sl_after_118 = pos.sl

        assert sl_after_112 > sl_after_tp1
        assert sl_after_pullback == pytest.approx(sl_after_112)
        assert sl_after_118 > sl_after_112

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
        engine.check_position_levels("BTCUSDT", 88.4)  # rebound below SL: no close
        sl_after_rebound = pos.sl
        engine.check_position_levels("BTCUSDT", 82.0)
        sl_after_82 = pos.sl

        assert sl_after_88 < sl_after_tp1
        assert sl_after_rebound == pytest.approx(sl_after_88)
        assert sl_after_82 < sl_after_88

    def test_dynamic_trailing_widens_with_higher_initial_atr(self):
        engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
        low_atr = engine.open_position(
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            entry_price=100.0,
            size=1.0,
            sl=95.0,
            tp1=110.0,
            tp2=120.0,
            initial_atr=1.0,
        )
        high_atr = engine.open_position(
            symbol="ETHUSDT",
            interval="1h",
            direction="long",
            entry_price=100.0,
            size=1.0,
            sl=95.0,
            tp1=110.0,
            tp2=120.0,
            initial_atr=6.0,
        )
        assert low_atr is not None and high_atr is not None

        d_low = engine._dynamic_trail_distance(low_atr, 112.0)
        d_high = engine._dynamic_trail_distance(high_atr, 112.0)
        assert d_high > d_low
