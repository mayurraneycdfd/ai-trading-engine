"""
Outcome labels measured strictly AFTER the open (no look-ahead into features).

Gap & Go   : did the gap CONTINUE in the gap direction over N minutes?
MeanRev    : after an extension event, did price revert toward the anchor?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import GAP_MIN_PCT, GAP_GO_HORIZONS, MR_ZSCORE_ENTRY, MR_HORIZONS


def _open_price_and_bars(minute: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return minute.groupby(minute.index.date)


def gap_go_labels(minute: pd.DataFrame, gap_pct: pd.Series) -> pd.DataFrame:
    """
    For every day with |gap| >= GAP_MIN_PCT:
      cont_{h}m  : signed return in GAP DIRECTION from open to open+h minutes (%)
      go_{h}m    : 1 if continuation positive, else 0
      filled_gap : 1 if price touched previous close during the day (gap fill)
    """
    out = {}
    grouped = _open_price_and_bars(minute)
    for date, day in grouped:
        d = pd.Timestamp(date)
        g = gap_pct.get(d, np.nan)
        if pd.isna(g) or abs(g) < GAP_MIN_PCT:
            continue
        day = day.between_time("09:15", "15:30")
        if len(day) < 5:
            continue
        o = day["open"].iloc[0]
        sign = np.sign(g)
        row = {"gap_pct": g}
        for h in GAP_GO_HORIZONS:
            upto = day.iloc[: max(h, 1)]
            px = upto["close"].iloc[-1]
            row[f"cont_{h}m"] = sign * (px - o) / o * 100
            row[f"go_{h}m"] = int(row[f"cont_{h}m"] > 0)
        prev_close = o / (1 + g / 100)
        touched = ((day["low"] <= prev_close) & (day["high"] >= prev_close)).any()
        row["filled_gap"] = int(touched)
        row["cont_close"] = sign * (day["close"].iloc[-1] - o) / o * 100
        out[d] = row
    return pd.DataFrame.from_dict(out, orient="index")


def mean_reversion_labels(minute: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday VWAP mean-reversion events:
      trigger = |price - vwap| / rolling_std >= MR_ZSCORE_ENTRY, checked each minute
      revert_{h}m : % move BACK TOWARD vwap over h minutes after trigger (positive = reverted)
    One event max per day (first trigger) to keep events independent.
    """
    out = {}
    grouped = _open_price_and_bars(minute)
    for date, day in grouped:
        day = day.between_time("09:15", "15:00")  # leave room for horizon
        if len(day) < 60:
            continue
        tp = (day["high"] + day["low"] + day["close"]) / 3
        cum_v = day["volume"].cumsum().replace(0, np.nan)
        vwap = (tp * day["volume"]).cumsum() / cum_v
        dev = day["close"] - vwap
        sd = dev.expanding(30).std()
        z = dev / sd
        trig = z.abs() >= MR_ZSCORE_ENTRY
        if not trig.any():
            continue
        i = int(np.argmax(trig.values))
        if i < 30 or i > len(day) - max(MR_HORIZONS):
            continue
        t0 = day.index[i]
        p0, v0 = day["close"].iloc[i], vwap.iloc[i]
        direction = -np.sign(p0 - v0)  # trade toward vwap
        row = {"mr_z": float(z.iloc[i]), "mr_time": t0.strftime("%H:%M")}
        for h in MR_HORIZONS:
            px = day["close"].iloc[min(i + h, len(day) - 1)]
            row[f"revert_{h}m"] = direction * (px - p0) / p0 * 100
            row[f"rev_{h}m"] = int(row[f"revert_{h}m"] > 0)
        out[pd.Timestamp(date)] = row
    return pd.DataFrame.from_dict(out, orient="index")


def build_labels(minute: pd.DataFrame, gap_pct: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    return gap_go_labels(minute, gap_pct), mean_reversion_labels(minute)
