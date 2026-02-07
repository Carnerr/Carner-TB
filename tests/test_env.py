import pandas as pd

from src.config import RossLikeConfig
from src.market.simulator import MarketSimulator
from src.rl.env import TradingEnv


def test_trade_window_blocks_entry():
    ross = RossLikeConfig(trade_start="07:00", trade_end="10:00")
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-10-02 06:00:00", "2025-10-02 06:01:00"]),
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [100, 100],
        }
    )
    simulator = MarketSimulator(bars, lookback=0)
    env = TradingEnv(simulator, max_position=1, cash_start=1000.0, max_hold=5, ross_config=ross)
    env.reset()
    _, _, _, info = env.step(1)
    assert info["position"] == 0
