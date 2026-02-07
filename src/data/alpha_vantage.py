from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
import time

import pandas as pd
import requests

from src.config import AlphaVantageConfig


BASE_URL = "https://www.alphavantage.co/query"


def _build_params(symbol: str, interval: str, api_key: str) -> dict[str, str]:
    if interval == "daily":
        return {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
    return {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "apikey": api_key,
    }


def fetch_alpha_vantage(symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    params = _build_params(symbol, interval, api_key)
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    key = next((k for k in payload if "Time Series" in k), None)
    if not key:
        message = payload.get("Note") or payload.get("Error Message") or "Unknown error"
        raise ValueError(f"Alpha Vantage error: {message}")
    df = pd.DataFrame.from_dict(payload[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=lambda x: x.split(". ")[-1])
    df = df.sort_index()
    df["symbol"] = symbol
    df["interval"] = interval
    return df


def filter_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    return df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]


def store_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index_label="timestamp")


def _latest_cached_timestamp(output_dir: str, symbol: str, interval: str) -> date | None:
    path = Path(output_dir)
    files = sorted(path.glob(f"{symbol}_{interval}_*.csv"))
    if not files:
        return None
    frames = [pd.read_csv(file, parse_dates=["timestamp"]) for file in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    if df.empty:
        return None
    return df["timestamp"].max().date()


def run(config: AlphaVantageConfig) -> None:
    if not config.api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not set.")
    for symbol in config.symbols:
        for interval in config.intervals:
            df = fetch_alpha_vantage(symbol, interval, config.api_key)
            df = filter_date_range(df, config.start_date, config.end_date)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            filename = f"{symbol}_{interval}_{timestamp}.csv"
            output_path = Path(config.output_dir) / filename
            store_csv(df, output_path)
            time.sleep(12)


def update(config: AlphaVantageConfig) -> None:
    if not config.api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not set.")
    for symbol in config.symbols:
        for interval in config.intervals:
            last_date = _latest_cached_timestamp(config.output_dir, symbol, interval)
            start_date = last_date or config.start_date
            df = fetch_alpha_vantage(symbol, interval, config.api_key)
            df = filter_date_range(df, start_date, config.end_date)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            filename = f"{symbol}_{interval}_{timestamp}.csv"
            output_path = Path(config.output_dir) / filename
            store_csv(df, output_path)
            time.sleep(12)


if __name__ == "__main__":
    run(AlphaVantageConfig())
