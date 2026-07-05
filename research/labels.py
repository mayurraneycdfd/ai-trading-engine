"""
Outcome labels measured strictly AFTER the open (no look-ahead into features).

Gap & Go   : did the gap CONTINUE in the gap direction over N minutes?
MeanRev    : after an extension event, did price revert toward the anchor?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (GAP_MIN_PCT, GAP_GO_HORIZONS, MR_ZSCORE_ENTRY, MR_HORIZONS,
                    TB_UP_MULT, TB_DN_MULT, TB_MAX_MIN)


def triple_barrier(day: pd.DataFrame, entry_idx: int, direction: float,
                   atr_pct: float) -> dict:
    """Triple-barrier outcome (Lopez de Prado): from the entry bar, which is
    hit first -- profit barrier (+TB_UP_MULT*ATR%), stop (-TB_DN_MULT*ATR%),
    or the vertical time barrier (TB_MAX_MIN minutes)?

    Returns tb_ret (realised % at whichever barrier hit, signed in trade
    direction) and tb_hit (1 profit / -1 stop / 0 time). This is
    path-AWARE: a trade that ends positive but got stopped first is a LOSS,
    which fixed-horizon labels get wrong."""
    entry = day["close"].iloc[entry_idx]
    up = entry * (1 + direction * TB_UP_MULT * atr_pct / 100)
    dn = entry * (1 - direction * TB_DN_MULT * atr_pct / 100)
    path = day.iloc[entry_idx + 1: entry_idx + 1 + TB_MAX_MIN]
    for _, bar in path.iterrows():
        hi, lo = bar["high"], bar["low"]
        hit_up = hi >= up if direction > 0 else lo <= up
        hit_dn = lo <= dn if direction > 0 else hi >= dn
        if hit_up and hit_dn:   # both inside one bar -> assume stop first (conservative)
            return {"tb_ret": -TB_DN_MULT * atr_pct, "tb_hit": -1}
        if hit_dn:
            return {"tb_ret": -TB_DN_MULT * atr_pct, "tb_hit": -1}
        if hit_up:
            return {"tb_ret": TB_UP_MULT * atr_pct, "tb_hit": 1}
    if len(path) == 0:
        return {"tb_ret": np.nan, "tb_hit": np.nan}
    exit_px = path["close"].iloc[-1]
    return {"tb_ret": direction * (exit_px - entry) / entry * 100, "tb_hit": 0}


def _day_atr_pct(day: pd.DataFrame, upto_idx: int) -> float:
    """Simple intraday ATR% estimate from bars up to the entry (no lookahead)."""
    upto = day.iloc[: max(upto_idx, 5)]
    tr = (upto["high"] - upto["low"]) / upto["close"] * 100
    v = float(tr.mean()) * 10  # scale bar-range to a day-magnitude barrier
    return max(v, 0.3)


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
        # path-aware triple-barrier outcome for a gap-direction trade at open
        atr = _day_atr_pct(day, 5)
        row.update(triple_barrier(day, 0, sign, atr))  # keys tb_ret / tb_hit
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
        atr = _day_atr_pct(day, i)
        row.update(triple_barrier(day, i, direction, atr))  # keys tb_ret / tb_hit
        out[pd.Timestamp(date)] = row
    return pd.DataFrame.from_dict(out, orient="index")


def build_labels(minute: pd.DataFrame, gap_pct: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    return gap_go_labels(minute, gap_pct), mean_reversion_labels(minute)
