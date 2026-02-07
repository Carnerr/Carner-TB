from __future__ import annotations

import pandas as pd


def _rolling_relative(series: pd.Series, window: int) -> pd.Series:
    return series / (series.rolling(window).mean() + 1e-8)


def build_feature_frame(
    minute_df: pd.DataFrame,
    five_min_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    minute = minute_df.copy()
    five = five_min_df.copy()
    daily = daily_df.copy()

    minute = minute.sort_index()
    five = five.sort_index()
    daily = daily.sort_index()

    minute["ret_1m"] = minute["close"].pct_change()
    minute["rel_vol_1m"] = _rolling_relative(minute["volume"], window=lookback)
    minute["range_1m"] = (minute["high"] - minute["low"]) / (minute["close"] + 1e-8)

    five["ret_5m"] = five["close"].pct_change()
    daily["ret_1d"] = daily["close"].pct_change()

    merged = pd.merge_asof(
        minute.reset_index().rename(columns={"timestamp": "ts"}).sort_values("ts"),
        five.reset_index().rename(columns={"timestamp": "ts"}).sort_values("ts")[["ts", "ret_5m"]],
        on="ts",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged,
        daily.reset_index().rename(columns={"timestamp": "ts"}).sort_values("ts")[["ts", "ret_1d"]],
        on="ts",
        direction="backward",
    )
    merged = merged.set_index("ts")
    merged["vwap"] = merged.get("vwap", merged["close"])
    merged["vwap_dist"] = (merged["close"] - merged["vwap"]) / (merged["vwap"] + 1e-8)

    features = merged[
        ["open", "high", "low", "close", "volume", "ret_1m", "ret_5m", "ret_1d", "rel_vol_1m", "range_1m", "vwap_dist"]
    ].dropna()
    return features
