from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.market.simulator import MarketSimulator
from src.rl.env import TradingEnv


@dataclass
class EpisodeResult:
    transitions: list[tuple[np.ndarray, int, float, np.ndarray]]
    total_reward: float


def _rollout_episode(args: tuple[pd.DataFrame, int, int, float, int, dict]) -> EpisodeResult:
    from src.rl.agent import QAgent, QAgentConfig

    bars, lookback, max_position, cash_start, max_hold, agent_state = args
    simulator = MarketSimulator(bars, lookback)
    env = TradingEnv(simulator, max_position, cash_start, max_hold)
    config = QAgentConfig(**agent_state["config"])
    agent = QAgent(config)
    agent.q_table = agent_state["q_table"]
    agent.epsilon = agent_state["epsilon"]

    state = env.reset()
    transitions: list[tuple[np.ndarray, int, float, np.ndarray]] = []
    total_reward = 0.0
    try:
        while True:
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)
            transitions.append((state, action, reward, next_state))
            total_reward += reward
            state = next_state
    except StopIteration:
        pass
    return EpisodeResult(transitions=transitions, total_reward=total_reward)


def run_parallel_episodes(
    bars_list: Iterable[pd.DataFrame],
    lookback: int,
    max_position: int,
    cash_start: float,
    max_hold: int,
    agent_state: dict,
    num_envs: int,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    with ProcessPoolExecutor(max_workers=num_envs) as executor:
        futures = [
            executor.submit(
                _rollout_episode,
                (bars, lookback, max_position, cash_start, max_hold, agent_state),
            )
            for bars in bars_list
        ]
        for future in futures:
            results.append(future.result())
    return results
