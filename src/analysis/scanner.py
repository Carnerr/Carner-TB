from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import AlphaVantageConfig, RossLikeConfig
from src.data.ingest import load_symbol_bundle
from src.data.news_alpha_vantage import load_cached_news, has_recent_news


@dataclass
class Candidate:
    symbol: str
    pct_up: float
    daily_rvol: float
    price: float
    had_news: bool
    reasons: dict[str, bool]


@dataclass
class ScanResult:
    day: str
    candidates: list[Candidate]


def _to_et(timestamp: pd.Timestamp, tz: ZoneInfo) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=tz).to_pydatetime()
    return timestamp.astimezone(tz).to_pydatetime()


def _parse_time(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(hour=int(hour), minute=int(minute))


def _prior_close(daily_df: pd.DataFrame, day: datetime) -> float | None:
    prior = daily_df[daily_df.index.date < day.date()]
    if prior.empty:
        return None
    return float(prior["close"].iloc[-1])


def _avg_daily_volume(daily_df: pd.DataFrame, day: datetime, window: int = 30) -> float | None:
    prior = daily_df[daily_df.index.date < day.date()]
    if prior.empty:
        return None
    return float(prior["volume"].tail(window).mean())


def scan_day(
    symbol_bundle: dict[str, pd.DataFrame],
    daily_bundle: dict[str, pd.DataFrame],
    news_df: pd.DataFrame,
    ross_config: RossLikeConfig,
    day: datetime,
) -> ScanResult:
    tz = ZoneInfo(ross_config.timezone)
    asof_time = datetime.combine(day.date(), _parse_time(ross_config.trade_start), tzinfo=tz)
    candidates: list[Candidate] = []
    float_limits = _load_float_metadata()

    for symbol, minute_df in symbol_bundle.items():
        daily_df = daily_bundle[symbol]
        minute = minute_df.copy()
        minute = minute.sort_index()
        minute_day = minute[minute.index.date == day.date()]
        if minute_day.empty:
            continue

        minute_day = minute_day.loc[minute_day.index <= asof_time]
        if minute_day.empty:
            continue

        price_at_asof = float(minute_day["close"].iloc[-1])
        prior_close = _prior_close(daily_df, asof_time)
        if prior_close is None or prior_close == 0:
            continue
        pct_up = (price_at_asof - prior_close) / prior_close

        avg_daily_vol = _avg_daily_volume(daily_df, asof_time)
        if avg_daily_vol is None or avg_daily_vol == 0:
            continue
        day_volume = float(minute_day["volume"].sum())
        daily_rvol = day_volume / avg_daily_vol

        had_news = has_recent_news(symbol, asof_time, ross_config.news_lookback_hours, news_df)

        float_ok = True
        if float_limits and symbol in float_limits:
            float_ok = float_limits[symbol] <= ross_config.float_max

        reasons = {
            "price_range": ross_config.price_min <= price_at_asof <= ross_config.price_max,
            "pct_up": pct_up >= ross_config.min_pct_up,
            "rvol": daily_rvol >= ross_config.min_rvol_daily,
            "news": had_news if ross_config.require_news else True,
            "float": float_ok,
        }

        if all(reasons.values()):
            candidates.append(
                Candidate(
                    symbol=symbol,
                    pct_up=pct_up,
                    daily_rvol=daily_rvol,
                    price=price_at_asof,
                    had_news=had_news,
                    reasons=reasons,
                )
            )

    ranked = sorted(candidates, key=lambda c: c.pct_up, reverse=True)[: ross_config.top_n]
    return ScanResult(day=str(day.date()), candidates=ranked)


def _load_float_metadata() -> dict[str, float]:
    path = Path("data/meta/float.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "symbol" not in df.columns or "float" not in df.columns:
        return {}
    return {row["symbol"]: float(row["float"]) for _, row in df.iterrows()}


def run_latest_scan() -> Path | None:
    alpha_config = AlphaVantageConfig()
    ross_config = RossLikeConfig()
    news_df = load_cached_news("data/raw/news")

    symbol_bundle: dict[str, pd.DataFrame] = {}
    daily_bundle: dict[str, pd.DataFrame] = {}
    days: set[pd.Timestamp] = set()
    for symbol in alpha_config.symbols:
        bundle = load_symbol_bundle(alpha_config.output_dir, symbol)
        symbol_bundle[symbol] = bundle["1min"]
        daily_bundle[symbol] = bundle["daily"]
        days.update(bundle["1min"].index.normalize().unique())

    if not days:
        return None
    latest_day = max(days)
    result = scan_day(symbol_bundle, daily_bundle, news_df, ross_config, latest_day.to_pydatetime())
    output_path = Path("logs/scanner_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in result.candidates:
        rows.append(
            {
                "day": result.day,
                "symbol": candidate.symbol,
                "pct_up": candidate.pct_up,
                "daily_rvol": candidate.daily_rvol,
                "price": candidate.price,
                "had_news": candidate.had_news,
                "reasons": candidate.reasons,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output = run_latest_scan()
    if output:
        print(f"Wrote scanner output to {output}")
