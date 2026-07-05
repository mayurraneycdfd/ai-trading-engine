"""
Feature engineering: converts raw data into one row PER STOCK PER DAY,
where every column is a candidate factor from the master factor list
(see DATA_CATALOG.md). All factors are known BY the market open of that
day -- no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_loader as dl


# ------------------------------------------------------------------------
# Category 1: own price/volume factors (computed from 1-min + daily bars)
# ------------------------------------------------------------------------
def price_volume_features(minute: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=daily.index)
    prev_close = daily["close"].shift(1)

    # Gap
    f["gap_pct"] = (daily["open"] - prev_close) / prev_close * 100
    f["gap_abs"] = f["gap_pct"].abs()
    f["gap_dir"] = np.sign(f["gap_pct"])

    # Momentum horizons (all shifted -> known at today's open)
    f["ret_1d"] = daily["close"].pct_change(1).shift(1) * 100
    f["ret_5d"] = daily["close"].pct_change(5).shift(1) * 100
    f["ret_20d"] = daily["close"].pct_change(20).shift(1) * 100

    # Volatility
    tr = np.maximum(daily["high"] - daily["low"],
                    np.maximum((daily["high"] - prev_close).abs(),
                               (daily["low"] - prev_close).abs()))
    f["atr14_pct"] = (tr.rolling(14).mean() / daily["close"]).shift(1) * 100
    f["rv20"] = (daily["close"].pct_change().rolling(20).std() * np.sqrt(252)).shift(1) * 100
    f["gap_vs_atr"] = f["gap_abs"] / f["atr14_pct"].replace(0, np.nan)

    # Volatility compression: range vs its own history
    rng = (daily["high"] - daily["low"]) / daily["close"] * 100
    f["range_compression"] = (rng.rolling(5).mean() / rng.rolling(60).mean()).shift(1)

    # Volume
    f["rvol_20d"] = (daily["volume"] / daily["volume"].rolling(20).mean().shift(1))
    # first-30-min relative volume (vs same window 20d avg)
    m = minute.between_time("09:15", "09:45")
    v30 = m.groupby(m.index.date)["volume"].sum()
    v30.index = pd.to_datetime(v30.index)
    f["rvol_open30"] = v30 / v30.rolling(20).mean().shift(1)

    # Levels
    f["dist_prev_high"] = (daily["open"] - daily["high"].shift(1)) / daily["close"].shift(1) * 100
    f["dist_prev_low"] = (daily["open"] - daily["low"].shift(1)) / daily["close"].shift(1) * 100
    f["dist_52w_high"] = (daily["open"] / daily["high"].rolling(252).max().shift(1) - 1) * 100
    f["dist_52w_low"] = (daily["open"] / daily["low"].rolling(252).min().shift(1) - 1) * 100

    # Consecutive closes in same direction before today
    up = (daily["close"].diff() > 0).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumsum()
    f["up_streak"] = streak.where(up == 1, -streak).shift(1)

    # Distance from 20-DMA at open (mean-reversion anchor)
    sma20 = daily["close"].rolling(20).mean().shift(1)
    sd20 = daily["close"].rolling(20).std().shift(1)
    f["zscore_20d"] = (daily["open"] - sma20) / sd20

    # Calendar
    f["day_of_week"] = f.index.dayofweek
    f["is_month_end"] = (f.index.is_month_end | (f.index.day >= 28)).astype(int)

    return f


# ------------------------------------------------------------------------
# Category 3: index / market-wide factors
# ------------------------------------------------------------------------
def market_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)

    nifty = dl.load_daily_series("nifty")
    if nifty is not None:
        nifty = nifty.resample("D").last().dropna()
        f["nifty_ret_1d"] = nifty.pct_change().shift(1).reindex(dates) * 100
        f["nifty_ret_5d"] = nifty.pct_change(5).shift(1).reindex(dates) * 100
        sma50 = nifty.rolling(50).mean()
        f["nifty_above_50dma"] = (nifty > sma50).astype(int).shift(1).reindex(dates)

    vix = dl.load_daily_series("india_vix")
    if vix is not None:
        f["vix_level"] = vix.shift(1).reindex(dates)
        f["vix_chg_1d"] = vix.pct_change().shift(1).reindex(dates) * 100
        f["vix_pctile_1y"] = vix.rolling(252).rank(pct=True).shift(1).reindex(dates)

    sp500 = dl.load_daily_series("sp500")
    if sp500 is not None:
        # US closes before India opens: yesterday's US return IS overnight info
        f["sp500_overnight"] = sp500.pct_change().reindex(dates, method="ffill") * 100

    crude = dl.load_daily_series("crude")
    if crude is not None:
        f["crude_ret_1d"] = crude.pct_change().reindex(dates, method="ffill") * 100
        f["crude_ret_5d"] = crude.pct_change(5).reindex(dates, method="ffill") * 100

    usdinr = dl.load_daily_series("usdinr")
    if usdinr is not None:
        f["usdinr_ret_1d"] = usdinr.pct_change().reindex(dates, method="ffill") * 100

    fii = dl.load_symbol_table("fii_dii")
    if fii is not None and "fii_net" in fii.columns:
        s = fii.set_index("date")["fii_net"]
        f["fii_net_prev"] = s.shift(1).reindex(dates)
        f["fii_net_5d"] = s.rolling(5).sum().shift(1).reindex(dates)

    return f


# ------------------------------------------------------------------------
# Category 4: derivatives factors
# ------------------------------------------------------------------------
def derivative_features(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    opt = dl.load_options(symbol)
    if opt is not None:
        for col in ("pcr", "iv", "iv_percentile", "oi_change", "max_pain"):
            if col in opt.columns:
                f[f"opt_{col}"] = opt[col].shift(1).reindex(dates)
    fut = dl.load_futures(symbol)
    if fut is not None:
        for col in ("basis", "oi", "oi_change", "rollover"):
            if col in fut.columns:
                f[f"fut_{col}"] = fut[col].shift(1).reindex(dates)
        if "oi_change" in fut.columns and "basis" in fut.columns:
            # long buildup = OI up + price premium widening
            f["fut_long_buildup"] = ((fut["oi_change"] > 0) & (fut["basis"].diff() > 0)) \
                .astype(int).shift(1).reindex(dates)
    return f


# ------------------------------------------------------------------------
# Categories 5 & 8: events and news
# ------------------------------------------------------------------------
def event_features(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)

    earn = dl.load_symbol_table("earnings_dates")
    if earn is not None:
        e = earn[earn["symbol"] == symbol]["date"]
        f["earnings_today"] = dates.normalize().isin(set(e)).astype(int)
        f["earnings_yesterday"] = dates.normalize().isin(set(e + pd.Timedelta(days=1))).astype(int)

    ca = dl.load_symbol_table("corporate_actions")
    if ca is not None:
        c = ca[ca["symbol"] == symbol]
        f["exdate_today"] = dates.normalize().isin(
            set(c.loc[c.get("type", pd.Series(dtype=str)).isin(["dividend", "split", "bonus"]), "date"])
        ).astype(int)

    news = dl.load_symbol_table("news")
    if news is not None:
        n = news[news["symbol"] == symbol]
        if "sentiment" in n.columns:
            s = n.groupby("date")["sentiment"].mean()
            f["news_sentiment"] = s.reindex(dates.normalize()).values
        cnt = n.groupby("date").size()
        f["news_count"] = cnt.reindex(dates.normalize()).fillna(0).values
        f["has_news"] = (f["news_count"] > 0).astype(int)

    return f


# ------------------------------------------------------------------------
# Category 2: cross-sectional (needs all-stocks daily panel)
# ------------------------------------------------------------------------
def cross_sectional_features(symbol: str, daily: pd.DataFrame,
                             panel_gaps: pd.DataFrame | None,
                             sector_map: dict[str, str]) -> pd.DataFrame:
    """panel_gaps: DataFrame [date x symbol] of gap_pct for all stocks."""
    f = pd.DataFrame(index=daily.index)
    if panel_gaps is None or symbol not in panel_gaps.columns:
        return f
    own = panel_gaps[symbol]
    others = panel_gaps.drop(columns=[symbol])
    f["mkt_gap_median"] = others.median(axis=1).reindex(daily.index)
    f["gap_idiosyncratic"] = (own - f["mkt_gap_median"]).reindex(daily.index)

    sector = sector_map.get(symbol)
    if sector:
        peers = [s for s, sec in sector_map.items() if sec == sector and s != symbol
                 and s in panel_gaps.columns]
        if peers:
            f["sector_gap_median"] = panel_gaps[peers].median(axis=1).reindex(daily.index)
            f["gap_vs_sector"] = (own - f["sector_gap_median"]).reindex(daily.index)
    return f


# ------------------------------------------------------------------------
# assemble
# ------------------------------------------------------------------------
def build_features(symbol: str, minute: pd.DataFrame,
                   panel_gaps: pd.DataFrame | None = None,
                   sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    daily = dl.to_daily(minute)
    parts = [
        price_volume_features(minute, daily),
        market_features(daily.index),
        derivative_features(symbol, daily.index),
        event_features(symbol, daily.index),
        cross_sectional_features(symbol, daily, panel_gaps, sector_map or {}),
    ]
    feats = pd.concat(parts, axis=1)
    feats["symbol"] = symbol
    return feats
