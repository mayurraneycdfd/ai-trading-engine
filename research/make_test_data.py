"""
Generates synthetic 1-min parquet data with a PLANTED edge so you can verify
the discovery engine end-to-end before pointing it at real data:

  planted edge: gaps that come WITH high open-30min relative volume continue;
                gaps on low volume fade (classic gap-and-go edge).

Run:  python make_test_data.py   then   python run_discovery.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PATHS

rng = np.random.default_rng(7)

N_STOCKS = 8
YEARS = 4
MINUTES = pd.date_range("09:15", "15:29", freq="1min").time


def make_stock(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    px = 1000.0 * (1 + rng.random())
    for d in dates:
        # decide today's gap and whether volume is high
        gap = rng.normal(0, 0.9)
        high_vol = rng.random() < 0.5
        # PLANTED EDGE: high-volume gaps continue ~0.4%, low-volume gaps fade
        drift_total = np.sign(gap) * (0.004 if high_vol else -0.003) if abs(gap) > 0.5 else 0.0
        o = px * (1 + gap / 100)
        n = len(MINUTES)
        steps = rng.normal(drift_total / n, 0.0006, n)
        closes = o * np.cumprod(1 + steps)
        opens = np.concatenate([[o], closes[:-1]])
        hi = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 3e-4, n)))
        lo = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 3e-4, n)))
        base_v = rng.integers(5_000, 15_000, n).astype(float)
        if high_vol:
            base_v[:30] *= 4  # first 30 min volume spike
        for i, t in enumerate(MINUTES):
            rows.append((pd.Timestamp.combine(d.date(), t), opens[i], hi[i], lo[i], closes[i], base_v[i]))
        px = closes[-1]
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def main():
    dates = pd.bdate_range(end="2024-12-31", periods=YEARS * 250)
    PATHS["minute_bars"].mkdir(parents=True, exist_ok=True)
    for i in range(N_STOCKS):
        sym = f"TEST{i:02d}"
        df = make_stock(sym, dates)
        df.to_parquet(PATHS["minute_bars"] / f"{sym}.parquet", index=False)
        print(f"wrote {sym}: {len(df):,} bars")

    # simple daily nifty + vix so market features activate
    daily = pd.DataFrame(index=dates)
    daily["close"] = 20000 * np.cumprod(1 + rng.normal(2e-4, 0.01, len(dates)))
    daily.reset_index(names="date").to_parquet(PATHS["nifty"])
    vix = pd.DataFrame({"date": dates, "close": np.clip(rng.normal(15, 4, len(dates)), 9, 40)})
    vix.to_parquet(PATHS["india_vix"])
    print("wrote nifty + india_vix")


if __name__ == "__main__":
    main()
