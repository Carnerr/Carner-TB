from pathlib import Path

import pandas as pd


def load_csvs(data_dir: str, symbol: str, interval: str) -> pd.DataFrame:
    path = Path(data_dir)
    files = sorted(path.glob(f"{symbol}_{interval}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No cached data for {symbol} {interval} in {data_dir}")
    frames = [pd.read_csv(file, parse_dates=["timestamp"]) for file in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")
    return df


def load_multi_interval(data_dir: str, symbol: str) -> dict[str, pd.DataFrame]:
    return {
        "1min": load_csvs(data_dir, symbol, "1min"),
        "5min": load_csvs(data_dir, symbol, "5min"),
        "daily": load_csvs(data_dir, symbol, "daily"),
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )


def load_symbol_bundle(data_dir: str, symbol: str) -> dict[str, pd.DataFrame]:
    bundle = load_multi_interval(data_dir, symbol)
    return {interval: normalize_columns(df).dropna() for interval, df in bundle.items()}
