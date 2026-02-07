from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import pandas as pd

from src.config import AlphaVantageConfig, TrainingConfig
from src.data.ingest import load_symbol_bundle
from src.market.features import build_feature_frame


@dataclass
class BaselineConfig:
    min_momentum_1m: float = 0.002
    min_rel_volume: float = 1.5
    max_hold_bars: int = 5
    stop_loss_1m: float = -0.002
    take_profit_1m: float = 0.004
    top_trades_per_day: int = 5


def _run_baseline(df: pd.DataFrame, config: BaselineConfig) -> list[dict]:
    trades: list[dict] = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    hold = 0

    for timestamp, row in df.iterrows():
        if not in_trade:
            if (
                row["ret_1m"] >= config.min_momentum_1m
                and row["rel_vol_1m"] >= config.min_rel_volume
                and row["vwap_dist"] >= 0.0
            ):
                in_trade = True
                entry_price = row["close"]
                entry_time = timestamp
                hold = 0
        else:
            hold += 1
            pnl = row["close"] - entry_price
            ret = pnl / entry_price
            if ret <= config.stop_loss_1m or ret >= config.take_profit_1m or hold >= config.max_hold_bars:
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                        "pnl": pnl,
                        "return": ret,
                        "hold_bars": hold,
                    }
                )
                in_trade = False

    return trades


def _write_best_trades(trades: list[dict], output_path: Path, top_n: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_by_day: dict[str, list[dict]] = {}
    for trade in trades:
        day = str(trade["entry_time"].date())
        trades_by_day.setdefault(day, []).append(trade)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trade_day",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl",
                "return",
                "hold_bars",
            ]
        )
        for day, day_trades in sorted(trades_by_day.items()):
            top = sorted(day_trades, key=lambda t: t["pnl"], reverse=True)[:top_n]
            for trade in top:
                writer.writerow(
                    [
                        day,
                        trade["entry_time"],
                        trade["exit_time"],
                        f"{trade['entry_price']:.4f}",
                        f"{trade['exit_price']:.4f}",
                        f"{trade['pnl']:.4f}",
                        f"{trade['return']:.6f}",
                        trade["hold_bars"],
                    ]
                )


def run(
    data_config: AlphaVantageConfig,
    training_config: TrainingConfig,
    baseline_config: BaselineConfig,
) -> Path:
    data_dir = Path(data_config.output_dir)
    symbol = data_config.symbols[0]
    bundle = load_symbol_bundle(str(data_dir), symbol)
    df = build_feature_frame(bundle["1min"], bundle["5min"], bundle["daily"], training_config.lookback)
    trades = _run_baseline(df, baseline_config)
    output_path = Path("logs") / f"baseline_best_trades_{symbol}.csv"
    _write_best_trades(trades, output_path, baseline_config.top_trades_per_day)
    return output_path


if __name__ == "__main__":
    output = run(AlphaVantageConfig(), TrainingConfig(), BaselineConfig())
    print(f"Wrote baseline trades to {output}")
