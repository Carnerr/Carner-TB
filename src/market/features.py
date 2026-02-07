from __future__ import annotations

import pandas as pd


def _rolling_relative(series: pd.Series, window: int) -> pd.Series:
    return series / (series.rolling(window).mean() + 1e-8)


def _to_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    reset = df.reset_index()
    if "timestamp" in reset.columns:
        return reset.rename(columns={"timestamp": "ts"})
    if "index" in reset.columns:
        return reset.rename(columns={"index": "ts"})
    return reset

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    macd_positive = (hist > 0) | (macd_line > signal_line)
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
            "macd_positive": macd_positive,
        }
    )


def _bull_flag_like(prices: pd.Series, ranges: pd.Series) -> pd.Series:
    impulse = prices.pct_change(10)
    tight = ranges.rolling(3).mean() < 0.01
    breakout = prices > prices.rolling(10).max().shift(1)
    return (impulse > 0.03) & tight & breakout


def _first_pullback_like(prices: pd.Series) -> pd.Series:
    breakout = prices > prices.rolling(20).max().shift(1)
    pullback = prices.pct_change(3) < -0.01
    reclaim = prices.pct_change(1) > 0.002
    return breakout.shift(1).fillna(False) & pullback.shift(1).fillna(False) & reclaim


def _parabolic_proxy(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    green = prices.diff() > 0
    increasing_range = prices.diff().rolling(3).mean() > 0
    vol_up = volumes.diff().rolling(3).mean() > 0
    return (green.rolling(3).sum() == 3) & increasing_range & vol_up


def _abcd_proxy(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change()
    pattern = (returns.shift(3) > 0.01) & (returns.shift(2) < -0.005) & (returns.shift(1) > 0.008)
    return pattern


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

    macd = _macd(minute["close"])
    minute = pd.concat([minute, macd], axis=1)
    minute["bull_flag_like"] = _bull_flag_like(minute["close"], minute["range_1m"])
    minute["first_pullback_like"] = _first_pullback_like(minute["close"])
    minute["parabolic_proxy"] = _parabolic_proxy(minute["close"], minute["volume"])
    minute["abcd_proxy"] = _abcd_proxy(minute["close"])

    five["ret_5m"] = five["close"].pct_change()
    daily["ret_1d"] = daily["close"].pct_change()

    merged = pd.merge_asof(
        _to_timestamp_column(minute).sort_values("ts"),
        _to_timestamp_column(five).sort_values("ts")[["ts", "ret_5m"]],
        on="ts",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged,
        _to_timestamp_column(daily).sort_values("ts")[["ts", "ret_1d"]],
        on="ts",
        direction="backward",
    )
    merged = merged.set_index("ts")
    merged["vwap"] = merged.get("vwap", merged["close"])
    merged["vwap_dist"] = (merged["close"] - merged["vwap"]) / (merged["vwap"] + 1e-8)

    features = merged[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ret_1m",
            "ret_5m",
            "ret_1d",
            "rel_vol_1m",
            "range_1m",
            "vwap_dist",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "macd_positive",
            "bull_flag_like",
            "first_pullback_like",
            "parabolic_proxy",
            "abcd_proxy",
        ]
    ].dropna()
    return features
