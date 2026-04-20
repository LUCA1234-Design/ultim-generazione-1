"""
Phase 13 "Dream Generator":
Monte Carlo synthetic-path training for rapid MetaAgent calibration.
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from agents.confluence_agent import ConfluenceAgent
from agents.meta_agent import MetaAgent
from agents.pattern_agent import PatternAgent
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.smc_agent import SMCAgent
from agents.strategy_agent import StrategyAgent
from config.settings import PAIRS_TRADING_CANDIDATE_PAIRS
from data.binance_client import fetch_futures_klines
from engine.decision_fusion import DecisionFusion


def klines_to_dataframe(klines: List[list]) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    for k in klines:
        rows.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return pd.DataFrame(rows)


def estimate_drift_vol(close: pd.Series) -> tuple[float, float]:
    returns = np.log(close / close.shift(1)).dropna()
    if returns.empty:
        return 0.0, 0.01
    return float(returns.mean()), float(returns.std(ddof=0) or 0.01)


def simulate_close_path(
    initial_price: float,
    drift: float,
    vol: float,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if steps <= 1:
        return np.array([initial_price], dtype=float)
    shocks = rng.normal(loc=drift, scale=max(vol, 1e-8), size=steps - 1)
    log_prices = np.concatenate([[np.log(max(initial_price, 1e-8))], np.log(max(initial_price, 1e-8)) + np.cumsum(shocks)])
    return np.exp(log_prices)


def build_ohlcv_from_close(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    jitter = np.maximum(close * 0.002, 1e-6)
    open_ = close + rng.normal(0.0, jitter, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, jitter, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, jitter, size=n))
    low = np.clip(low, 1e-8, None)
    volume = np.abs(rng.normal(1_000_000.0, 250_000.0, size=n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _train_on_path(
    symbol: str,
    interval: str,
    df: pd.DataFrame,
    agents: Dict[str, object],
    meta: MetaAgent,
    fusion: DecisionFusion,
) -> int:
    processed = 0
    if len(df) < 90:
        return processed

    pattern: PatternAgent = agents["pattern"]  # type: ignore[assignment]
    regime: RegimeAgent = agents["regime"]  # type: ignore[assignment]
    confluence: ConfluenceAgent = agents["confluence"]  # type: ignore[assignment]
    risk: RiskAgent = agents["risk"]  # type: ignore[assignment]
    strategy: StrategyAgent = agents["strategy"]  # type: ignore[assignment]
    smc: SMCAgent = agents["smc"]  # type: ignore[assignment]

    for idx in range(80, len(df) - 1):
        sub_df = df.iloc[: idx + 1].copy()
        pattern_result = pattern.safe_analyse(symbol, interval, sub_df, sub_df)
        direction_hint = pattern_result.direction if pattern_result is not None else "neutral"

        regime_result = regime.safe_analyse(symbol, interval, sub_df)
        current_regime = "unknown"
        if regime_result is not None and regime_result.metadata:
            current_regime = regime_result.metadata.get("regime", "unknown")

        agent_results = {}
        if pattern_result is not None:
            agent_results["pattern"] = pattern_result
        if regime_result is not None:
            agent_results["regime"] = regime_result
        confluence_result = confluence.safe_analyse(symbol, interval, sub_df, direction_hint)
        if confluence_result is not None:
            agent_results["confluence"] = confluence_result
        risk_result = risk.safe_analyse(symbol, interval, sub_df, direction_hint, regime=current_regime)
        if risk_result is not None:
            agent_results["risk"] = risk_result
        strategy_result = strategy.safe_analyse(symbol, interval, sub_df, direction_hint)
        if strategy_result is not None:
            agent_results["strategy"] = strategy_result
        smc_result = smc.safe_analyse(symbol, interval, sub_df, direction=direction_hint)
        if smc_result is not None:
            agent_results["smc"] = smc_result

        if len(agent_results) < 3:
            continue

        fusion_result = fusion.fuse(symbol, interval, agent_results, regime=current_regime)
        if fusion_result.decision not in {"long", "short"}:
            continue

        next_ret = float((df["close"].iloc[idx + 1] / df["close"].iloc[idx]) - 1.0)
        was_correct = (fusion_result.decision == "long" and next_ret > 0) or (
            fusion_result.decision == "short" and next_ret < 0
        )
        meta.record_outcome(
            decision_id=f"mc-{symbol}-{idx}",
            agent_results=agent_results,
            was_correct=was_correct,
            regime=current_regime,
        )
        processed += 1
    return processed


def _default_symbols() -> List[str]:
    syms = set()
    for a, b in PAIRS_TRADING_CANDIDATE_PAIRS:
        syms.add(str(a).upper())
        syms.add(str(b).upper())
    return sorted(syms) or ["BTCUSDT", "ETHUSDT"]


def run_training(
    symbols: Iterable[str],
    interval: str = "1h",
    history_limit: int = 500,
    paths_per_symbol: int = 250,
    random_seed: int = 42,
    state_path: str = "data/meta_agent_state.json",
) -> dict:
    rng = np.random.default_rng(random_seed)

    pattern = PatternAgent()
    regime = RegimeAgent()
    confluence = ConfluenceAgent()
    risk = RiskAgent()
    strategy = StrategyAgent()
    smc = SMCAgent()
    meta = MetaAgent(agents=[pattern, regime, confluence, risk, strategy, smc])
    meta.load_state(state_path)
    fusion = DecisionFusion()

    agents = {
        "pattern": pattern,
        "regime": regime,
        "confluence": confluence,
        "risk": risk,
        "strategy": strategy,
        "smc": smc,
    }

    total_decisions = 0
    for symbol in symbols:
        klines = fetch_futures_klines(str(symbol).upper(), interval, limit=history_limit)
        df_hist = klines_to_dataframe(klines)
        if df_hist.empty or len(df_hist) < 100:
            continue

        drift, vol = estimate_drift_vol(df_hist["close"])
        initial_price = float(df_hist["close"].iloc[-1])
        sim_steps = len(df_hist)

        for _ in range(paths_per_symbol):
            close_path = simulate_close_path(initial_price, drift, vol, sim_steps, rng)
            df_sim = build_ohlcv_from_close(close_path, rng)
            total_decisions += _train_on_path(symbol, interval, df_sim, agents, meta, fusion)
        meta.adjust_weights()

    weight_map = meta.adjust_weights()
    meta.save_state(state_path)
    return {"trained_decisions": total_decisions, "weights": weight_map, "state_path": state_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 13 Monte Carlo synthetic training for MetaAgent.")
    parser.add_argument("--symbols", nargs="*", default=_default_symbols(), help="Symbols to train on")
    parser.add_argument("--interval", default="1h", help="Binance kline interval (default: 1h)")
    parser.add_argument("--history-limit", type=int, default=500, help="Historical kline rows")
    parser.add_argument("--paths-per-symbol", type=int, default=250, help="Synthetic paths per symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--state-path", default="data/meta_agent_state.json", help="MetaAgent state output path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = run_training(
        symbols=args.symbols,
        interval=args.interval,
        history_limit=args.history_limit,
        paths_per_symbol=args.paths_per_symbol,
        random_seed=args.seed,
        state_path=args.state_path,
    )
    print(
        "Monte Carlo training complete | "
        f"trained_decisions={report['trained_decisions']} | "
        f"state={report['state_path']}"
    )


if __name__ == "__main__":
    main()
