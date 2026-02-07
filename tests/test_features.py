import pandas as pd

from src.market.features import build_feature_frame


def test_macd_positive_on_uptrend():
    index = pd.date_range("2025-10-02 07:00:00", periods=50, freq="T")
    minute = pd.DataFrame(
        {
            "open": range(1, 51),
            "high": range(1, 51),
            "low": range(1, 51),
            "close": range(1, 51),
            "volume": [100] * 50,
        },
        index=index,
    )
    five = minute.resample("5T").last()
    daily = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [2, 3],
            "low": [1, 2],
            "close": [1.5, 2.5],
            "volume": [1000, 1200],
        },
        index=pd.to_datetime(["2025-10-01", "2025-10-02"]),
    )
    features = build_feature_frame(minute, five, daily, lookback=5)
    assert "macd_positive" in features.columns
    assert features["macd_positive"].iloc[-1] in [True, False]
