from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.market.simulator import MarketSimulator, compute_features


@dataclass
class Position:
    size: int = 0
    entry_price: float = 0.0
    hold_steps: int = 0


class TradingEnv:
    def __init__(self, simulator: MarketSimulator, max_position: int, cash_start: float, max_hold: int):
        self.simulator = simulator
        self.max_position = max_position
        self.cash_start = cash_start
        self.max_hold = max_hold
        self.position = Position()
        self.cash = cash_start

    def reset(self) -> np.ndarray:
        self.position = Position()
        self.cash = self.cash_start
        self.simulator.reset()
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        state = self.simulator.step()

        if self.position.size != 0:
            self.position.hold_steps += 1

        if action == 1 and self.position.size == 0:
            self.position = Position(size=1, entry_price=state.price, hold_steps=0)
        elif action == 2 and self.position.size != 0:
            pnl = (state.price - self.position.entry_price) * self.position.size
            reward += pnl
            self.cash += pnl
            self.position = Position()

        if self.position.size != 0:
            reward += (state.price - self.position.entry_price) * 0.01
            if self.position.hold_steps > self.max_hold:
                reward -= abs(state.price - self.position.entry_price) * 0.5

        done = False
        info = {"cash": self.cash, "position": self.position.size}
        return self._obs(), reward, done, info

    def _obs(self) -> np.ndarray:
        window = self.simulator.window()
        features = compute_features(window)
        position_flag = np.array([self.position.size], dtype=float)
        return np.concatenate([features, position_flag], axis=0)
