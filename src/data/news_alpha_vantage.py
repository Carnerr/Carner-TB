from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import time

import pandas as pd
import requests

from src.config import AlphaVantageConfig


BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class NewsConfig:
    output_dir: str = "data/raw/news"
    limit: int = 200


def fetch_news_sentiment(tickers: list[str], api_key: str, limit: int = 200) -> pd.DataFrame:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "apikey": api_key,
        "limit": str(limit),
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    feed = payload.get("feed", [])
    rows: list[dict] = []
    for item in feed:
        tickers_list = [t.get("ticker") for t in item.get("ticker_sentiment", []) if t.get("ticker")]
        rows.append(
            {
                "time_published": item.get("time_published"),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "url": item.get("url"),
                "tickers": ",".join(tickers_list),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time_published"] = pd.to_datetime(df["time_published"], errors="coerce")
    return df


def cache_news(df: pd.DataFrame, output_dir: str) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = Path(output_dir) / f"news_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def load_cached_news(output_dir: str) -> pd.DataFrame:
    path = Path(output_dir)
    files = sorted(path.glob("news_*.csv"))
    if not files:
        return pd.DataFrame(columns=["time_published", "title", "summary", "source", "url", "tickers"])
    frames = [pd.read_csv(file, parse_dates=["time_published"]) for file in files]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def has_recent_news(symbol: str, asof_time: datetime, lookback_hours: int, news_df: pd.DataFrame) -> bool:
    if news_df.empty:
        return False
    cutoff = asof_time - timedelta(hours=lookback_hours)
    subset = news_df[news_df["time_published"] >= cutoff]
    if subset.empty:
        return False
    return subset["tickers"].str.contains(fr"\b{symbol}\b", regex=True).any()


def run(alpha_config: AlphaVantageConfig, news_config: NewsConfig) -> Path | None:
    if not alpha_config.api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not set.")
    df = fetch_news_sentiment(list(alpha_config.symbols), alpha_config.api_key, limit=news_config.limit)
    if df.empty:
        return None
    path = cache_news(df, news_config.output_dir)
    time.sleep(12)
    return path


if __name__ == "__main__":
    output = run(AlphaVantageConfig(), NewsConfig())
    if output:
        print(f"Wrote news to {output}")
