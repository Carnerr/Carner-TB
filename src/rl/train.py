from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import csv
from datetime import datetime

import numpy as np
import pandas as pd

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from src.config import AlphaVantageConfig, RossLikeConfig, TrainingConfig
from src.data.ingest import load_symbol_bundle
from src.data.news_alpha_vantage import load_cached_news
from src.analysis.scanner import scan_day
from src.market.features import build_feature_frame
from src.market.simulator import MarketSimulator
from src.rl.agent import QAgent, QAgentConfig
from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.env import TradingEnv
from src.rl.vector_env import run_parallel_episodes


def _slice_trade_window(df: pd.DataFrame, ross_config: RossLikeConfig) -> pd.DataFrame:
    tz = ZoneInfo(ross_config.timezone)
    start_time = _parse_time(ross_config.trade_start)
    end_time = _parse_time(ross_config.trade_end)
    timestamps = df.index
    localized = []
    for ts in timestamps:
        if ts.tzinfo is None:
            localized.append(ts.replace(tzinfo=tz))
        else:
            localized.append(ts.astimezone(tz))
    df = df.copy()
    df["__local_time"] = [t.time() for t in localized]
    window = df[(df["__local_time"] >= start_time) & (df["__local_time"] <= end_time)].drop(columns="__local_time")
    return window


def _parse_time(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(hour=int(hour), minute=int(minute))


def _rollout_single(env: TradingEnv, agent: QAgent) -> float:
    state = env.reset()
    total_reward = 0.0
    try:
        while True:
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
    except StopIteration:
        pass
    return total_reward


def _rollout_dqn(env: TradingEnv, agent: DQNAgent) -> tuple[float, float]:
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []
    try:
        while True:
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)
            agent.push(state, action, reward, next_state, False)
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            total_reward += reward
            state = next_state
    except StopIteration:
        pass
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return total_reward, avg_loss


def _init_metrics_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "episode", "epsilon", "avg_reward", "avg_loss", "algo"])


def train(config: TrainingConfig, data_config: AlphaVantageConfig) -> None:
    data_dir = Path(data_config.output_dir)
    ross_config = RossLikeConfig()
    news_df = load_cached_news("data/raw/news")

    symbol_bundle: dict[str, dict[str, pd.DataFrame]] = {}
    daily_bundle: dict[str, pd.DataFrame] = {}
    all_days: set[pd.Timestamp] = set()
    for symbol in data_config.symbols:
        bundle = load_symbol_bundle(str(data_dir), symbol)
        symbol_bundle[symbol] = bundle
        daily_bundle[symbol] = bundle["daily"]
        all_days.update(bundle["1min"].index.normalize().unique())

    if not all_days:
        raise ValueError("No data available to train.")

    sample_symbol = data_config.symbols[0]
    sample_bundle = symbol_bundle[sample_symbol]
    sample_features = build_feature_frame(
        sample_bundle["1min"], sample_bundle["5min"], sample_bundle["daily"], config.lookback
    )
    sample_window = _slice_trade_window(sample_features, ross_config)
    if sample_window.empty:
        raise ValueError("No sample window available for sizing.")
    sample_env = TradingEnv(
        MarketSimulator(sample_window, config.lookback),
        config.max_position,
        config.cash_start,
        config.max_hold_steps,
        ross_config,
    )
    obs_dim = int(sample_env.reset().shape[0])

    agent_config = QAgentConfig(
        learning_rate=config.learning_rate,
        discount=config.discount,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
    )
    dqn_config = DQNConfig(
        learning_rate=config.learning_rate,
        discount=config.discount,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        batch_size=config.batch_size,
        replay_capacity=config.replay_capacity,
        target_sync_steps=config.target_sync_steps,
        device=config.device,
    )
    agent = QAgent(agent_config)
    dqn_agent = DQNAgent(dqn_config, input_dim=obs_dim)

    metrics_path = Path("logs/training_metrics.csv")
    _init_metrics_log(metrics_path)

    for episode in range(config.episodes):
        attempt = 0
        selected_window = None
        selected_symbol = None
        while attempt < 10 and selected_window is None:
            day = random.choice(list(all_days)).to_pydatetime()
            symbol_minute = {symbol: data["1min"] for symbol, data in symbol_bundle.items()}
            scan_result = scan_day(symbol_minute, daily_bundle, news_df, ross_config, day)
            if not scan_result.candidates:
                attempt += 1
                continue
            selected_symbol = scan_result.candidates[0].symbol
            bundle = symbol_bundle[selected_symbol]
            df = build_feature_frame(bundle["1min"], bundle["5min"], bundle["daily"], config.lookback)
            df_day = df[df.index.date == day.date()]
            window = _slice_trade_window(df_day, ross_config)
            if window.empty:
                attempt += 1
                continue
            selected_window = window

        if selected_window is None:
            continue

        if config.use_dqn:
            simulator = MarketSimulator(selected_window, config.lookback)
            env = TradingEnv(simulator, config.max_position, config.cash_start, config.max_hold_steps, ross_config)
            avg_reward, avg_loss = _rollout_dqn(env, dqn_agent)
        elif config.num_envs > 1:
            agent_state = {"config": asdict(agent_config), "q_table": agent.q_table, "epsilon": agent.epsilon}
            results = run_parallel_episodes(
                [selected_window for _ in range(config.num_envs)],
                config.lookback,
                config.max_position,
                config.cash_start,
                config.max_hold_steps,
                agent_state,
                config.num_envs,
            )
            for result in results:
                for state, action, reward, next_state in result.transitions:
                    agent.learn(state, action, reward, next_state)
            avg_reward = float(np.mean([r.total_reward for r in results]))
            avg_loss = 0.0
        else:
            simulator = MarketSimulator(selected_window, config.lookback)
            env = TradingEnv(simulator, config.max_position, config.cash_start, config.max_hold_steps, ross_config)
            avg_reward = _rollout_single(env, agent)
            avg_loss = 0.0

        if config.use_dqn:
            dqn_agent.decay()
            epsilon = dqn_agent.epsilon
            algo = "dqn"
        else:
            agent.decay()
            epsilon = agent.epsilon
            algo = "q_learning"
        print(f"Episode {episode + 1}/{config.episodes} epsilon={epsilon:.3f} avg_reward={avg_reward:.2f}")

        with metrics_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    episode + 1,
                    f"{epsilon:.6f}",
                    f"{avg_reward:.6f}",
                    f"{avg_loss:.6f}",
                    algo,
                ]
            )


if __name__ == "__main__":
    train(TrainingConfig(), AlphaVantageConfig())
