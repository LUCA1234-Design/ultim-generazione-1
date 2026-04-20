# ultim-generazione — V17 Agentic AI Trading System

Evolution of the V16 "Cecchino Istituzionale" monolithic bot into a modular,
adaptive, self-learning multi-agent trading system for Binance Futures.

## Architecture

```
ultim-generazione/
├── README.md
├── requirements.txt
├── legacy/
│   └── bot_v16.py                  # Complete original V16 bot preserved
├── config/
│   ├── __init__.py
│   └── settings.py                 # All settings (V16-style os.getenv fallbacks)
├── data/
│   ├── __init__.py
│   ├── binance_client.py           # Exchange connection wrapper
│   ├── data_store.py               # Thread-safe DataFrame store
│   └── websocket_manager.py        # Multi-WS + REST fallback + exponential backoff
├── indicators/
│   ├── __init__.py
│   ├── technical.py                # RSI, ATR, MACD, OBV, BBands, Keltner, ADX, Z-Score
│   └── smart_money.py              # CVD, Volume Delta, Liquidity Sweep, Order Blocks
├── agents/
│   ├── __init__.py
│   ├── base_agent.py               # Abstract BaseAgent with AgentResult dataclass
│   ├── regime_agent.py             # GaussianMixture regime detection (sklearn)
│   ├── pattern_agent.py            # All V16 detectors + auto-calibrating thresholds
│   ├── confluence_agent.py         # Probabilistic MTF confluence (replaces Muro di Berlino)
│   ├── risk_agent.py               # Kelly with real win rates, adaptive sizing
│   ├── strategy_agent.py           # Generates & evaluates trading strategies
│   └── meta_agent.py               # Monitors & adjusts other agent weights
│   └── pairs_trading_agent.py      # Phase 13 statistical arbitrage (delta-neutral pairs)
├── engine/
│   ├── __init__.py
│   ├── decision_fusion.py          # Weighted vote fusion (replaces IF/RETURN cascade)
│   ├── execution.py                # Paper trading (default) + real Binance Futures execution
│   └── event_processor.py         # Routes candle close events through agent pipeline
├── memory/
│   ├── __init__.py
│   ├── experience_db.py            # SQLite: decisions, agent_performance, optimal_params, trade_outcomes
│   └── performance_tracker.py     # Real win rates, P&L, Sharpe ratio
├── notifications/
│   ├── __init__.py
│   └── telegram_service.py        # Enhanced Telegram with regime probs & agent reasoning
├── monte_carlo_trainer.py          # Phase 13 synthetic-path training for MetaAgent
└── main.py                         # Orchestrator
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

When running locally, the Phase 9 control room dashboard is available at:

```text
http://127.0.0.1:5000
```

Set environment variables (or use the hardcoded V16-style fallbacks):
```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
export CRYPTO_PANIC_API_KEY=your_cryptopanic_key   # optional, RSS fallback if omitted
export LM_STUDIO_URL=http://localhost:1234/v1
```

## Key Improvements Over V16

| Feature | V16 | V17 |
|---|---|---|
| Architecture | 1500-line monolith | Modular multi-agent |
| Market regime | None | GaussianMixture (3 regimes) |
| Decision making | Binary IF/RETURN cascade | Probabilistic weighted voting |
| Win rate | Hardcoded 0.65 | Real win rates from SQLite |
| Position sizing | Static Kelly | Adaptive Kelly (real win rates) |
| Execution | Signal-only | Paper + Real Binance Futures |
| Memory | Limited | Full SQLite experience DB |
| Agent feedback | None | MetaAgent weight adjustment loop |
| Telegram | Basic signal | Agent reasoning + regime probs |

## Execution Modes

- **Paper Trading** (default, `PAPER_TRADING=True` in `config/settings.py`):
  Simulates orders, tracks P&L, no real money at risk.
- **Live Trading** (`PAPER_TRADING=False`):
  Places real Binance Futures orders via `futures_create_order()`.
