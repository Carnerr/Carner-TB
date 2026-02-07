from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    learning_rate: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    batch_size: int
    replay_capacity: int
    target_sync_steps: int
    device: str


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.index = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, config: DQNConfig, input_dim: int, action_dim: int = 3) -> None:
        self.config = config
        self.device = torch.device(self._resolve_device(config.device))
        self.epsilon = config.epsilon_start
        self.action_dim = action_dim
        self.policy = QNetwork(input_dim, action_dim).to(self.device)
        self.target = QNetwork(input_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.replay_capacity)
        self.steps = 0

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        if len(self.buffer) < self.config.batch_size:
            return 0.0
        batch = self.buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.config.discount * next_q_values * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def decay(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
