"""
Market-microstructure features from 1-minute bars.

These capture INFORMED TRADING and LIQUIDITY -- the mechanics behind why
gaps continue or fade -- and are invisible to ordinary OHLCV factors:

- Amihud illiquidity  : price impact per rupee traded (Amihud 2002)
- Kyle lambda         : slope of |return| on signed volume (Kyle 1985)
- Roll spread         : effective spread from serial covariance (Roll 1984)
- VPIN proxy          : order-flow toxicity from bulk-classified volume
                        (Easley, Lopez de Prado, O'Hara 2012; bar-level proxy)
- CLV pressure        : where closes sit inside bar ranges (buy/sell pressure)
- Run entropy         : Shannon entropy of up/down bar runs -- low entropy =
                        one-sided informed flow, high = noise churn

ALL features are computed from day T-1's bars (or a trailing window ending
T-1) and describe the state KNOWN at day T's open. No same-day data leaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_amihud(day: pd.DataFrame) -> float:
    """Mean |1-min return| / rupee volume, scaled. Higher = more illiquid."""
    ret = day["close"].pct_change().abs()
    rupee = (day["close"] * day["volume"]).replace(0, np.nan)
    v = (ret / rupee).mean()
    return float(v * 1e9) if np.isfinite(v) else np.nan


def _daily_kyle_lambda(day: pd.DataFrame) -> float:
    """OLS slope of |return| on signed sqrt rupee volume (tick-rule sign)."""
    ret = day["close"].pct_change()
    sign = np.sign(ret).replace(0, np.nan).ffill().fillna(0)
    x = sign * np.sqrt((day["close"] * day["volume"]).clip(lower=0))
    y = ret.abs()
    ok = x.notna() & y.notna() & (x != 0)
    if ok.sum() < 30:
        return np.nan
    xv, yv = x[ok].values, y[ok].values
    denom = (xv ** 2).sum()
    return float((xv * yv).sum() / denom * 1e6) if denom > 0 else np.nan


def _daily_roll_spread(day: pd.DataFrame) -> float:
    """Roll (1984): 2*sqrt(-cov(dp_t, dp_{t-1})) as % of price."""
    dp = day["close"].diff().dropna()
    if len(dp) < 30:
        return np.nan
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    if cov >= 0:
        return 0.0                      # positive autocov -> no spread estimate
    mid = float(day["close"].median())
    return float(2 * np.sqrt(-cov) / mid * 100) if mid > 0 else np.nan


def _daily_vpin_proxy(day: pd.DataFrame, n_buckets: int = 20) -> float:
    """Bar-level VPIN proxy: split the day's volume into equal buckets,
    classify each bar's volume buy/sell via the bulk classification rule
    (normal CDF of standardized return), then average |buy - sell| / total
    per bucket. Higher = more toxic/one-sided flow."""
    from scipy.stats import norm
    ret = day["close"].pct_change()
    sd = ret.std()
    if not np.isfinite(sd) or sd == 0 or day["volume"].sum() <= 0:
        return np.nan
    z = (ret / sd).clip(-6, 6)
    buy_frac = pd.Series(norm.cdf(z), index=day.index).fillna(0.5)
    vol = day["volume"].astype(float)
    cum = vol.cumsum()
    total = cum.iloc[-1]
    bucket = np.minimum((cum / total * n_buckets).astype(int), n_buckets - 1)
    b = pd.DataFrame({"vol": vol, "buy": buy_frac * vol, "bucket": bucket})
    g = b.groupby("bucket").sum()
    imb = (g["buy"] - (g["vol"] - g["buy"])).abs()
    return float((imb / g["vol"].replace(0, np.nan)).mean())


def _daily_clv_pressure(day: pd.DataFrame) -> float:
    """Volume-weighted Close Location Value: +1 = every close at bar high
    (relentless buying), -1 = at lows. Persistent pressure precedes moves."""
    rng = (day["high"] - day["low"]).replace(0, np.nan)
    clv = ((day["close"] - day["low"]) - (day["high"] - day["close"])) / rng
    w = day["volume"] / day["volume"].sum() if day["volume"].sum() > 0 else None
    if w is None:
        return np.nan
    return float((clv * w).sum())


def _daily_run_entropy(day: pd.DataFrame) -> float:
    """Shannon entropy (bits) of the distribution of up/down run lengths.
    Low entropy = long one-sided runs (informed flow)."""
    direction = np.sign(day["close"].diff()).replace(0, np.nan).dropna()
    if len(direction) < 30:
        return np.nan
    changes = (direction != direction.shift()).cumsum()
    run_lengths = direction.groupby(changes).size()
    counts = run_lengths.value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def microstructure_features(minute: pd.DataFrame) -> pd.DataFrame:
    """One row per day. Every column describes day T-1 (shifted), so it is
    known at day T's open. Also includes 5-day z-scores to flag ABNORMAL
    microstructure states, which matter more than levels."""
    rows = {}
    for date, day in minute.groupby(minute.index.date):
        if len(day) < 60:               # need a reasonably complete session
            continue
        rows[pd.Timestamp(date)] = {
            "ms_amihud": _daily_amihud(day),
            "ms_kyle_lambda": _daily_kyle_lambda(day),
            "ms_roll_spread": _daily_roll_spread(day),
            "ms_vpin": _daily_vpin_proxy(day),
            "ms_clv_pressure": _daily_clv_pressure(day),
            "ms_run_entropy": _daily_run_entropy(day),
        }
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # shift(1): day T's row describes day T-1's microstructure
    f = f.shift(1)
    # abnormality z-scores vs the trailing 20 sessions
    for c in list(f.columns):
        mu = f[c].rolling(20).mean()
        sd = f[c].rolling(20).std()
        f[c + "_z20"] = (f[c] - mu) / sd.replace(0, np.nan)
    return f
