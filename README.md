# Carner-TB Trading Bot (Ross Cameron Momentum RL)

This repository provides a reinforcement-learning (RL) training loop, a market-simulator that replays historical bars, and Alpha Vantage ingestion utilities. The system is designed to:

* Replay historical data as a "market" for training.
* Learn a momentum-focused, Ross Cameron–style day-trading policy.
* Run multiple training instances concurrently via vectorized environments.

## Quick start

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Alpha Vantage**

   ```bash
   export ALPHAVANTAGE_API_KEY="YOUR_KEY"
   ```

3. **Pull data**

   ```bash
   python -m src.data.alpha_vantage
   ```

4. **Train a policy**

   ```bash
   python -m src.rl.train
   ```

   By default this uses a GPU-enabled DQN agent (if CUDA is available). Set `use_dqn = False` in `src/config.py` to revert to the lightweight tabular Q-learning loop.

5. **Launch the GUI**

   ```bash
   python -m src.gui.app
   ```

   Use the control panel to pull data, run the baseline, train the model, and monitor the latest training metrics.

6. **Update and organize data**

   ```bash
   python -c "from src.data.alpha_vantage import update; from src.config import AlphaVantageConfig; update(AlphaVantageConfig())"
   python -m src.data.organize
   ```

## Data coverage

The loader is configured for:

* 1-minute bars
* 5-minute bars
* Daily bars

Date range defaults to **2025-10-01** through **today**. Update `src/config.py` to change symbols or dates.

## Notes on Ross Cameron style

Ross Cameron–style trading focuses on momentum, relative volume, and fast exits. The environment exposes:

* 1m / 5m / daily momentum features.
* Relative volume (current vs. rolling 1m average).
* VWAP distance.
* Recent 1m candle range.

The reward function emphasizes small, frequent gains and penalizes holding too long, aligning with day-trading rules.

## Project layout

```
src/
  config.py               # symbols, API config, training defaults
  data/
    alpha_vantage.py       # data ingestion + caching
    ingest.py              # CSV loaders
    organize.py            # per-day/per-interval export
  market/
    features.py            # multi-interval feature engineering
    simulator.py           # historical replay environment
  analysis/
    baseline.py            # Ross-style baseline trade logger
  gui/
    app.py                 # desktop control panel
  rl/
    env.py                 # trading RL environment
    vector_env.py          # parallel env manager
    agent.py               # lightweight Q-learning agent
    dqn.py                 # GPU-accelerated DQN agent
    train.py               # training loop
```
