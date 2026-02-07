from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PerformanceConfig:
    trades_path: str = "logs/trades.csv"
    output_dir: str = "logs/reports"


def _summary(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = df.set_index("entry_time").groupby(pd.Grouper(freq=freq))
    summary = grouped["pnl"].agg(["count", "sum", "mean"])
    wins = grouped.apply(lambda x: (x["pnl"] > 0).mean())
    avg_win = grouped.apply(lambda x: x.loc[x["pnl"] > 0, "pnl"].mean())
    avg_loss = grouped.apply(lambda x: x.loc[x["pnl"] < 0, "pnl"].mean())
    profit_factor = grouped.apply(lambda x: x.loc[x["pnl"] > 0, "pnl"].sum() / abs(x.loc[x["pnl"] < 0, "pnl"].sum() or 1))
    hold_time = grouped["hold_bars"].mean()
    max_drawdown = grouped.apply(_max_drawdown)
    summary["win_rate"] = wins
    summary["avg_win"] = avg_win
    summary["avg_loss"] = avg_loss
    summary["profit_factor"] = profit_factor
    summary["avg_hold_bars"] = hold_time
    summary["max_drawdown"] = max_drawdown
    return summary.reset_index()


def _max_drawdown(group: pd.DataFrame) -> float:
    if group.empty:
        return 0.0
    equity = group["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak).min()
    return float(drawdown)


def run(config: PerformanceConfig) -> Path | None:
    trades_path = Path(config.trades_path)
    if not trades_path.exists():
        return None
    df = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    if df.empty:
        return None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = _summary(df, "D")
    weekly = _summary(df, "W")
    monthly = _summary(df, "M")

    daily.to_csv(output_dir / "daily_summary.csv", index=False)
    weekly.to_csv(output_dir / "weekly_summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly_summary.csv", index=False)

    report_path = output_dir / "performance_report.md"
    with report_path.open("w") as handle:
        handle.write("# Performance Report\n\n")
        handle.write("## Daily Summary\n\n")
        handle.write(daily.to_string(index=False))
        handle.write("\n\n## Weekly Summary\n\n")
        handle.write(weekly.to_string(index=False))
        handle.write("\n\n## Monthly Summary\n\n")
        handle.write(monthly.to_string(index=False))
    return report_path


if __name__ == "__main__":
    output = run(PerformanceConfig())
    if output:
        print(f"Wrote report to {output}")
