from dataclasses import dataclass, field
from datetime import date
import os


@dataclass
class AlphaVantageConfig:
    api_key: str = field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY", ""))
    symbols: tuple[str, ...] = ("AAPL", "TSLA", "NVDA")
    start_date: date = date(2025, 10, 1)
    end_date: date = field(default_factory=date.today)
    intervals: tuple[str, ...] = ("1min", "5min", "daily")
    output_dir: str = "data/raw"


@dataclass
class TrainingConfig:
    episode_length: int = 390
    lookback: int = 20
    max_position: int = 1
    cash_start: float = 25_000.0
    risk_penalty: float = 0.1
    max_hold_steps: int = 15
    num_envs: int = 4
    episodes: int = 30
    learning_rate: float = 0.15
    discount: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    use_dqn: bool = True
    batch_size: int = 256
    replay_capacity: int = 50_000
    target_sync_steps: int = 250
    device: str = "auto"
