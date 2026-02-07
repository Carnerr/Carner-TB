from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketState:
    step: int
    price: float
    volume: float
    vwap: float
    high: float
    low: float
    close: float
    open: float
    timestamp: pd.Timestamp | None = None


class MarketSimulator:
    def __init__(self, bars: pd.DataFrame, lookback: int) -> None:
        self.bars = bars.reset_index()
        self.lookback = lookback
        self._index = lookback

    def reset(self) -> MarketState:
        self._index = self.lookback
        return self._state()

    def step(self) -> MarketState:
        self._index += 1
        if self._index >= len(self.bars):
            raise StopIteration
        return self._state()

    def window(self) -> pd.DataFrame:
        start = max(0, self._index - self.lookback)
        return self.bars.iloc[start : self._index + 1]

    def _state(self) -> MarketState:
        row = self.bars.iloc[self._index]
        price = float(row.get("close", 0.0))
        volume = float(row.get("volume", 0.0))
        vwap = float(row.get("vwap", price))
        return MarketState(
            step=self._index,
            price=price,
            volume=volume,
            vwap=vwap,
            high=float(row.get("high", price)),
            low=float(row.get("low", price)),
            close=price,
            open=float(row.get("open", price)),
            timestamp=row.get("timestamp", row.get("ts", row.get("index"))),
        )


def compute_features(window: pd.DataFrame) -> np.ndarray:
    row = window.iloc[-1]
    return np.array(
        [
            float(row.get("ret_1m", 0.0)),
            float(row.get("ret_5m", 0.0)),
            float(row.get("ret_1d", 0.0)),
            float(row.get("rel_vol_1m", 0.0)),
            float(row.get("vwap_dist", 0.0)),
            float(row.get("range_1m", 0.0)),
            float(row.get("macd_line", 0.0)),
            float(row.get("macd_signal", 0.0)),
            float(row.get("macd_hist", 0.0)),
            float(row.get("macd_positive", 0.0)),
            float(row.get("bull_flag_like", 0.0)),
            float(row.get("first_pullback_like", 0.0)),
            float(row.get("parabolic_proxy", 0.0)),
            float(row.get("abcd_proxy", 0.0)),
        ],
        dtype=float,
    )
