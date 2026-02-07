from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AlphaVantageConfig
from src.data.ingest import normalize_columns


@dataclass
class OrganizeConfig:
    output_root: str = "data/organized"


def _write_daily_groups(df: pd.DataFrame, output_dir: Path, symbol: str, interval: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(df.index.date)
    for day, group in grouped:
        day_str = str(day)
        filename = f"{symbol}_{interval}_{day_str}.csv"
        group.to_csv(output_dir / filename, index_label="timestamp")


def organize(config: AlphaVantageConfig, organize_config: OrganizeConfig) -> None:
    output_root = Path(organize_config.output_root)
    for symbol in config.symbols:
        for interval in config.intervals:
            pattern = f"{symbol}_{interval}_*.csv"
            files = sorted(Path(config.output_dir).glob(pattern))
            if not files:
                continue
            frames = [pd.read_csv(file, parse_dates=["timestamp"]) for file in files]
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df = normalize_columns(df).dropna()
            interval_dir = output_root / symbol / interval
            _write_daily_groups(df, interval_dir, symbol, interval)


if __name__ == "__main__":
    organize(AlphaVantageConfig(), OrganizeConfig())
