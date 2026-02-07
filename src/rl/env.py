from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import RossLikeConfig
from src.market.simulator import MarketSimulator, compute_features


@dataclass
class Position:
    size: int = 0
    entry_price: float = 0.0
    hold_steps: int = 0


class TradingEnv:
    def __init__(
        self,
        simulator: MarketSimulator,
        max_position: int,
        cash_start: float,
        max_hold: int,
        ross_config: RossLikeConfig | None = None,
    ):
        self.simulator = simulator
        self.max_position = max_position
        self.cash_start = cash_start
        self.max_hold = max_hold
        self.ross_config = ross_config or RossLikeConfig()
        self.position = Position()
        self.cash = cash_start
        self.trades_today = 0
        self.start_equity = cash_start

    def reset(self) -> np.ndarray:
        self.position = Position()
        self.cash = self.cash_start
        self.trades_today = 0
        self.start_equity = self.cash_start
        self.simulator.reset()
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        state = self.simulator.step()
        now = self._to_et(state.timestamp)

        if not self._within_trade_window(now):
            if action == 1:
                reward -= 1.0
                action = 0

        if self.position.size != 0:
            self.position.hold_steps += 1

        if action == 1 and self.position.size == 0:
            fill_price = self._apply_slippage(state.price, side="buy")
            self.position = Position(size=1, entry_price=fill_price, hold_steps=0)
        elif action == 2 and self.position.size != 0:
            fill_price = self._apply_slippage(state.price, side="sell")
            pnl = (fill_price - self.position.entry_price) * self.position.size
            reward += pnl
            self.cash += pnl
            self.position = Position()
            self.trades_today += 1

        if self.position.size != 0:
            reward += (state.price - self.position.entry_price) * 0.01
            if self.position.hold_steps > self.max_hold:
                reward -= abs(state.price - self.position.entry_price) * 0.5

        done = False
        equity = self.cash + (state.price - self.position.entry_price) * self.position.size
        if equity >= self.start_equity * (1 + self.ross_config.daily_profit_target_pct):
            done = True
        if equity <= self.start_equity * (1 - self.ross_config.daily_max_loss_pct):
            done = True
        if self.trades_today >= self.ross_config.max_trades_per_day and self.position.size == 0:
            done = True

        if now and now.time() >= self._parse_time(self.ross_config.trade_end):
            if self.position.size != 0:
                fill_price = self._apply_slippage(state.price, side="sell")
                pnl = (fill_price - self.position.entry_price) * self.position.size
                reward += pnl
                self.cash += pnl
                self.position = Position()
                self.trades_today += 1
            done = True

        info = {"cash": self.cash, "position": self.position.size}
        return self._obs(), reward, done, info

    def _obs(self) -> np.ndarray:
        window = self.simulator.window()
        features = compute_features(window)
        position_flag = np.array([self.position.size], dtype=float)
        return np.concatenate([features, position_flag], axis=0)

    def _to_et(self, timestamp) -> datetime | None:
        if timestamp is None:
            return None
        tz = ZoneInfo(self.ross_config.timezone)
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=tz).to_pydatetime()
        return timestamp.astimezone(tz).to_pydatetime()

    def _within_trade_window(self, now: datetime | None) -> bool:
        if now is None:
            return True
        start = self._parse_time(self.ross_config.trade_start)
        end = self._parse_time(self.ross_config.trade_end)
        return start <= now.time() <= end

    def _parse_time(self, value: str) -> dt_time:
        hour, minute = value.split(":")
        return dt_time(hour=int(hour), minute=int(minute))

    def _apply_slippage(self, price: float, side: str) -> float:
        bps = self.ross_config.slippage_bps
        if bps <= 0:
            return price
        multiplier = 1 + (bps / 10_000) if side == "buy" else 1 - (bps / 10_000)
        return price * multiplier
