from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QAgentConfig:
    learning_rate: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float


class QAgent:
    def __init__(self, config: QAgentConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)
        q_values = self._q_values(state)
        return int(np.argmax(q_values))

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        self.q_table.setdefault(key, np.zeros(3))
        self.q_table.setdefault(next_key, np.zeros(3))
        best_next = np.max(self.q_table[next_key])
        td_target = reward + self.config.discount * best_next
        td_error = td_target - self.q_table[key][action]
        self.q_table[key][action] += self.config.learning_rate * td_error

    def decay(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        key = self._discretize(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(3)
        return self.q_table[key]

    def _discretize(self, state: np.ndarray) -> tuple[int, ...]:
        bins = np.array([-0.02, -0.005, 0.0, 0.005, 0.02])
        discretized = np.digitize(state, bins)
        return tuple(int(x) for x in discretized)
