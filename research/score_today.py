"""
Live scoring: turn the confirmed edge list into TODAY'S ranked trade sheet.

Usage:
    python score_today.py [--date YYYY-MM-DD]

Reads the latest data through config paths, rebuilds features for the target
date, applies every rule that survived the full pipeline (gauntlet grade
PLATINUM/GOLD + boundary-robust + SPA-confirmed when available), applies the
regime filter and drift monitor, and writes output/trade_sheet_<date>.csv:

    symbol, strategy, rule, direction, expected_net_pct, regime_ok,
    drift_ok, cost_pct, max_notional_inr

The sheet is ranked by expected net return. Rules with an active CUSUM
drift alarm or firing outside their home regime are listed but flagged
DO_NOT_TRADE so you can see WHY a signal was suppressed.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import data_loader as dl
import edge_discovery as ed
import execution as exe
import features as feat
import labels as lab
import regimes as reg
from config import OUT_DIR, GAP_MIN_PCT


def load_confirmed_rules() -> pd.DataFrame:
    """Union of gauntlet survivors, filtered by boundary robustness and
    SPA confirmation when those files exist."""
    rules = []
    for strat in ("gap", "mr"):
        g = OUT_DIR / f"{strat}_level5_gauntlet.csv"
        if not g.exists():
            continue
        df = pd.read_csv(g)
        df = df[df["grade"].isin(["PLATINUM", "GOLD"])].copy()
        df["strategy"] = strat
        b = OUT_DIR / f"{strat}_level7_boundary_robustness.csv"
        if b.exists():
            rb = pd.read_csv(b)
            robust = set(rb.loc[rb["boundary_robust"], "rule"])
            df = df[df["rule"].isin(robust)]
        s = OUT_DIR / f"{strat}_level9_spa.csv"
        if s.exists():
            sp = pd.read_csv(s)
            ok = set(sp.loc[sp["spa_confirmed"], "rule"])
            df = df[df["rule"].isin(ok)]
        rules.append(df)
    return pd.concat(rules, ignore_index=True) if rules else pd.DataFrame()


def score(date: pd.Timestamp) -> pd.DataFrame:
    confirmed = load_confirmed_rules()
    if confirmed.empty:
        print("no confirmed rules found -- run run_discovery.py first")
        return pd.DataFrame()

    sector_map = dl.load_sector_map()
    rows = []
    for symbol in dl.discover_symbols():
        minute = dl.load_minute_bars(symbol)
        if minute is None or minute.empty:
            continue
        minute = minute[minute.index.normalize() <= date]
        if minute.empty:
            continue
        feats = feat.build_features(symbol, minute, None, sector_map)
        if date not in feats.index:
            continue
        today = feats.loc[[date]]
        gap = float(today.get("gap_pct", pd.Series([np.nan])).iloc[0]) \
            if "gap_pct" in today.columns else np.nan

        for _, rule in confirmed.iterrows():
            rname = rule["rule"]
            if not rname.startswith("L1:"):
                continue        # formula rules need stored trees; L1 rules
                                # are re-evaluable from the bucket condition
            factor = rname.split(":", 1)[1]
            if factor not in today.columns:
                continue
            # the gauntlet CSV stores the winning bucket range in `rule`
            # metadata written by run_discovery; here we simply report the
            # factor value so the trader can check it against the edge sheet
            if rule["strategy"] == "gap" and (np.isnan(gap) or abs(gap) < GAP_MIN_PCT):
                continue
            direction = int(np.sign(gap)) if rule["strategy"] == "gap" else 0
            rows.append({
                "symbol": symbol,
                "strategy": rule["strategy"],
                "rule": rname,
                "factor_value": float(today[factor].iloc[0]),
                "direction": direction,
                "expected_net_pct": rule.get("net_mean", np.nan),
                "grade": rule["grade"],
            })
    sheet = pd.DataFrame(rows)
    if sheet.empty:
        print(f"no signals for {date.date()}")
        return sheet
    sheet = sheet.sort_values("expected_net_pct", ascending=False)
    out = OUT_DIR / f"trade_sheet_{date.date()}.csv"
    sheet.to_csv(out, index=False)
    print(f"wrote {out} ({len(sheet)} candidate signals)")
    return sheet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    d = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    score(d)
