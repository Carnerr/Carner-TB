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

   Use the control panel to pull data, pull news, run the scanner, generate baseline trades, train the model, and monitor the latest metrics.

6. **Update and organize data**

   ```bash
   python -c "from src.data.alpha_vantage import update; from src.config import AlphaVantageConfig; update(AlphaVantageConfig())"
   python -m src.data.organize
   ```

7. **Pull news and run scanner**

   ```bash
   python -m src.data.news_alpha_vantage
   python -m src.analysis.scanner
   ```

8. **Generate performance reports**

   ```bash
   python -m src.analysis.performance
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

## Ross-like workflow in this repo

This project mimics a public Ross-style process (not proprietary execution):

1. **Scanner**: filter US stocks by price ($2–$20), % up on day, high relative volume, and news catalysts.
2. **Top movers**: rank by % gain and select the top 3; trade only the top candidate.
3. **Morning window only**: trade 7:00–10:00 ET with no new entries outside the window.
4. **Momentum patterns**: focus on first pullbacks, bull-flag-like continuation, parabolic proxies, and ABCD-style proxies.
5. **Risk plan**: hard daily profit target and max loss with one trade per day.

## Project layout

```
src/
  config.py               # symbols, API config, training defaults
  data/
    alpha_vantage.py       # data ingestion + caching
    ingest.py              # CSV loaders
    organize.py            # per-day/per-interval export
    news_alpha_vantage.py  # news sentiment ingestion
  market/
    features.py            # multi-interval feature engineering
    simulator.py           # historical replay environment
  analysis/
    baseline.py            # Ross-style baseline trade logger
    scanner.py             # Ross-like scanner
    performance.py         # performance reports
  gui/
    app.py                 # desktop control panel
  rl/
    env.py                 # trading RL environment
    vector_env.py          # parallel env manager
    agent.py               # lightweight Q-learning agent
    dqn.py                 # GPU-accelerated DQN agent
    train.py               # training loop
```
