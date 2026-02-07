from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import csv
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import AlphaVantageConfig, TrainingConfig
from src.data.ingest import load_symbol_bundle
from src.market.features import build_feature_frame
from src.market.simulator import MarketSimulator
from src.rl.agent import QAgent, QAgentConfig
from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.env import TradingEnv
from src.rl.vector_env import run_parallel_episodes


def _sample_windows(df: pd.DataFrame, lookback: int, episode_length: int, num_envs: int) -> list[pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    max_start = len(df) - episode_length - 1
    if max_start <= lookback:
        raise ValueError("Not enough data to sample training windows.")
    for _ in range(num_envs):
        start = random.randint(lookback, max_start)
        window = df.iloc[start - lookback : start + episode_length].copy()
        windows.append(window)
    return windows


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
    symbol = data_config.symbols[0]
    bundle = load_symbol_bundle(str(data_dir), symbol)
    df = build_feature_frame(bundle["1min"], bundle["5min"], bundle["daily"], config.lookback)

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
    dqn_agent = DQNAgent(dqn_config, input_dim=df.shape[1] + 1)

    metrics_path = Path("logs/training_metrics.csv")
    _init_metrics_log(metrics_path)

    for episode in range(config.episodes):
        windows = _sample_windows(df, config.lookback, config.episode_length, config.num_envs)
        if config.use_dqn:
            simulator = MarketSimulator(windows[0], config.lookback)
            env = TradingEnv(simulator, config.max_position, config.cash_start, config.max_hold_steps)
            avg_reward, avg_loss = _rollout_dqn(env, dqn_agent)
        elif config.num_envs > 1:
            agent_state = {"config": asdict(agent_config), "q_table": agent.q_table, "epsilon": agent.epsilon}
            results = run_parallel_episodes(
                windows,
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
            simulator = MarketSimulator(windows[0], config.lookback)
            env = TradingEnv(simulator, config.max_position, config.cash_start, config.max_hold_steps)
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
