from datetime import datetime

import pandas as pd

from src.analysis.scanner import scan_day
from src.config import RossLikeConfig


def test_scanner_pct_up_and_rvol():
    ross = RossLikeConfig(require_news=False)
    day = datetime(2025, 10, 2)
    minutes = pd.DataFrame(
        {
            "close": [10.0, 11.5],
            "volume": [1000, 2000],
        },
        index=pd.to_datetime(["2025-10-02 07:00:00", "2025-10-02 07:01:00"]),
    )
    daily = pd.DataFrame(
        {
            "close": [9.0],
            "volume": [1000],
        },
        index=pd.to_datetime(["2025-10-01"]),
    )

    symbol_bundle = {"TEST": minutes}
    daily_bundle = {"TEST": daily}
    news_df = pd.DataFrame(columns=["time_published", "tickers"])

    result = scan_day(symbol_bundle, daily_bundle, news_df, ross, day)
    assert result.candidates
    assert result.candidates[0].pct_up > 0.1
    assert result.candidates[0].daily_rvol > 1.0


def test_scanner_no_lookahead():
    ross = RossLikeConfig(require_news=False)
    day = datetime(2025, 10, 2)
    minutes = pd.DataFrame(
        {
            "close": [10.0, 20.0],
            "volume": [1000, 2000],
        },
        index=pd.to_datetime(["2025-10-02 06:59:00", "2025-10-02 12:00:00"]),
    )
    daily = pd.DataFrame(
        {
            "close": [9.0],
            "volume": [1000],
        },
        index=pd.to_datetime(["2025-10-01"]),
    )
    symbol_bundle = {"TEST": minutes}
    daily_bundle = {"TEST": daily}
    news_df = pd.DataFrame(columns=["time_published", "tickers"])
    result = scan_day(symbol_bundle, daily_bundle, news_df, ross, day)
    assert result.candidates
    assert result.candidates[0].price == 10.0
