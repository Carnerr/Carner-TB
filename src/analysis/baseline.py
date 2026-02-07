from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import AlphaVantageConfig, RossLikeConfig, TrainingConfig
from src.data.ingest import load_symbol_bundle
from src.market.features import build_feature_frame
from src.analysis.scanner import scan_day
from src.data.news_alpha_vantage import load_cached_news


@dataclass
class BaselineConfig:
    min_momentum_1m: float = 0.002
    min_rel_volume: float = 1.5
    max_hold_bars: int = 5
    stop_loss_1m: float = -0.002
    take_profit_1m: float = 0.004
    top_trades_per_day: int = 5
    stop_loss_pct: float = -0.03
    take_profit_pct: float = 0.05


def _run_baseline(
    df: pd.DataFrame,
    config: BaselineConfig,
    ross_config: RossLikeConfig,
    symbol: str,
    rvol: float,
    pct_up: float,
    had_news: bool,
) -> list[dict]:
    trades: list[dict] = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    hold = 0
    trades_today = 0

    for timestamp, row in df.iterrows():
        if not in_trade:
            if row["first_pullback_like"] and row["macd_positive"]:
                in_trade = True
                entry_price = row["close"]
                entry_time = timestamp
                hold = 0
        else:
            hold += 1
            pnl = row["close"] - entry_price
            ret = pnl / entry_price
            if ret <= config.stop_loss_pct or ret >= config.take_profit_pct or hold >= config.max_hold_bars:
                trades.append(
                    {
                        "day": str(entry_time.date()) if entry_time else "",
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                        "pnl": pnl,
                        "return_pct": ret,
                        "hold_bars": hold,
                        "rvol": rvol,
                        "pct_up_at_entry": pct_up,
                        "had_news": had_news,
                        "pattern_flags": {
                            "first_pullback_like": bool(row["first_pullback_like"]),
                            "macd_positive": bool(row["macd_positive"]),
                            "bull_flag_like": bool(row["bull_flag_like"]),
                            "parabolic_proxy": bool(row["parabolic_proxy"]),
                            "abcd_proxy": bool(row["abcd_proxy"]),
                        },
                    }
                )
                in_trade = False
                trades_today += 1
                if trades_today >= ross_config.max_trades_per_day:
                    break

    return trades


def _write_best_trades(trades: list[dict], output_path: Path, top_n: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_by_day: dict[str, list[dict]] = {}
    for trade in trades:
        day = trade["day"]
        trades_by_day.setdefault(day, []).append(trade)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trade_day",
                "symbol",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl",
                "return_pct",
                "hold_bars",
                "rvol",
                "pct_up_at_entry",
                "had_news",
                "pattern_flags",
            ]
        )
        for day, day_trades in sorted(trades_by_day.items()):
            top = sorted(day_trades, key=lambda t: t["pnl"], reverse=True)[:top_n]
            for trade in top:
                writer.writerow(
                    [
                        day,
                        trade["symbol"],
                        trade["entry_time"],
                        trade["exit_time"],
                        f"{trade['entry_price']:.4f}",
                        f"{trade['exit_price']:.4f}",
                        f"{trade['pnl']:.4f}",
                        f"{trade['return_pct']:.6f}",
                        trade["hold_bars"],
                        f"{trade['rvol']:.4f}",
                        f"{trade['pct_up_at_entry']:.4f}",
                        trade["had_news"],
                        trade["pattern_flags"],
                    ]
                )


def run(
    data_config: AlphaVantageConfig,
    training_config: TrainingConfig,
    baseline_config: BaselineConfig,
) -> Path:
    data_dir = Path(data_config.output_dir)
    ross_config = RossLikeConfig()
    news_df = load_cached_news("data/raw/news")
    tz = ZoneInfo(ross_config.timezone)

    symbol_bundle: dict[str, pd.DataFrame] = {}
    daily_bundle: dict[str, pd.DataFrame] = {}
    all_days: set[pd.Timestamp] = set()
    for symbol in data_config.symbols:
        bundle = load_symbol_bundle(str(data_dir), symbol)
        symbol_bundle[symbol] = bundle["1min"]
        daily_bundle[symbol] = bundle["daily"]
        all_days.update(bundle["1min"].index.normalize().unique())

    trades: list[dict] = []
    for day in sorted(all_days):
        scan_result = scan_day(symbol_bundle, daily_bundle, news_df, ross_config, day.to_pydatetime())
        if not scan_result.candidates:
            continue
        selected = scan_result.candidates[0]
        selected_symbol = selected.symbol
        bundle = load_symbol_bundle(str(data_dir), selected_symbol)
        features = build_feature_frame(bundle["1min"], bundle["5min"], bundle["daily"], training_config.lookback)
        day_df = features[features.index.date == day.date()]
        day_df = day_df[
            day_df.index.map(lambda ts: (ts.tzinfo and ts.astimezone(tz) or ts.replace(tzinfo=tz)).time())
            >= datetime.strptime(ross_config.trade_start, "%H:%M").time()
        ]
        day_df = day_df[
            day_df.index.map(lambda ts: (ts.tzinfo and ts.astimezone(tz) or ts.replace(tzinfo=tz)).time())
            <= datetime.strptime(ross_config.trade_end, "%H:%M").time()
        ]
        trades.extend(_run_baseline(day_df, baseline_config, ross_config, selected_symbol, selected.daily_rvol, selected.pct_up, selected.had_news))

    output_path = Path("logs") / "trades.csv"
    _write_best_trades(trades, output_path, baseline_config.top_trades_per_day)
    return output_path


if __name__ == "__main__":
    output = run(AlphaVantageConfig(), TrainingConfig(), BaselineConfig())
    print(f"Wrote baseline trades to {output}")
