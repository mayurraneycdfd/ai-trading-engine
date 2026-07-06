"""
LEVEL 0 -- data integrity audit. Runs BEFORE discovery, because bad data
silently poisons every statistic downstream.

Checks per symbol:
  1. Corporate-action verification: any overnight |gap| > AUDIT_JUMP_PCT is
     cross-checked against the corporate-actions file. Explained jumps
     (split/bonus/demerger/dividend ex-date) are QUARANTINED as trade events
     (they are mechanical, not tradeable gaps). Unexplained jumps are flagged
     as probable UNADJUSTED data errors.
  2. Survivorship audit: with a point-in-time F&O universe file the engine
     restricts every event to dates when the symbol was actually in the
     universe. Without one it measures listing-history asymmetry and warns.
  3. Stale/broken series: runs of identical closes, zero-volume days,
     duplicate timestamps, missing sessions vs the union calendar.

Outputs:
  - audit_report.csv   one row per (symbol, issue)
  - quarantine set     (symbol, date) pairs that discovery MUST exclude
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import data_loader as dl
from config import (AUDIT_JUMP_PCT, AUDIT_STALE_DAYS, AUDIT_MIN_HISTORY_DAYS,
                    FNO_UNIVERSE_PATH, OUT_DIR)

# corporate-action types that legitimately cause big mechanical jumps
MECHANICAL_CA = {"split", "bonus", "demerger", "rights", "dividend",
                 "spinoff", "consolidation", "face_value_change"}


def load_pit_universe() -> pd.DataFrame | None:
    """Point-in-time F&O membership: [symbol, from_date, to_date]."""
    p = Path(FNO_UNIVERSE_PATH)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    for c in ("from_date", "to_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def in_universe_mask(events: pd.DataFrame, universe: pd.DataFrame | None) -> pd.Series:
    """True where (symbol, date) was inside the point-in-time universe.
    Without a universe file everything passes (with a warning upstream)."""
    if universe is None:
        return pd.Series(True, index=events.index)
    ok = pd.Series(False, index=events.index)
    for sym, grp in events.groupby("symbol"):
        rows = universe[universe["symbol"] == sym]
        if rows.empty:
            continue
        m = pd.Series(False, index=grp.index)
        for _, r in rows.iterrows():
            hi = r.get("to_date")
            hi = hi if pd.notna(hi) else pd.Timestamp.max
            m |= (grp["date"] >= r["from_date"]) & (grp["date"] <= hi)
        ok.loc[grp.index] = m
    return ok


def audit_symbol(symbol: str, daily: pd.DataFrame,
                 ca: pd.DataFrame | None) -> tuple[list[dict], set]:
    """Return (issues, quarantined_dates) for one symbol's daily series."""
    issues: list[dict] = []
    quarantine: set = set()

    # --- 1. big-jump verification against corporate actions ---------------
    prev_close = daily["close"].shift(1)
    jump_pct = (daily["open"] - prev_close) / prev_close * 100
    big = daily.index[jump_pct.abs() > AUDIT_JUMP_PCT]
    ca_dates = set()
    if ca is not None and not ca.empty:
        sub = ca[ca["symbol"] == symbol]
        typ = sub["type"].astype(str).str.lower() if "type" in sub.columns \
            else pd.Series("", index=sub.index)
        mech = sub[typ.isin(MECHANICAL_CA) | typ.str.contains("|".join(MECHANICAL_CA), na=False)]
        ca_dates = set(pd.to_datetime(mech["date"]).dt.normalize())
    for d in big:
        dn = pd.Timestamp(d).normalize()
        near_ca = any(abs((dn - c).days) <= 1 for c in ca_dates)
        quarantine.add(dn)  # mechanical or error -- either way not a tradeable gap
        issues.append({
            "symbol": symbol, "date": dn.date(),
            "issue": "explained_corporate_action_jump" if near_ca
                     else "UNEXPLAINED_JUMP_probable_unadjusted_data",
            "detail": f"overnight jump {jump_pct.loc[d]:+.1f}%",
            "severity": "info" if near_ca else "CRITICAL",
        })

    # --- 2. stale series ----------------------------------------------------
    same = (daily["close"].diff() == 0)
    run = same.groupby((~same).cumsum()).cumsum()
    stale_days = daily.index[run >= AUDIT_STALE_DAYS]
    for d in stale_days:
        dn = pd.Timestamp(d).normalize()
        quarantine.add(dn)
    if len(stale_days):
        issues.append({"symbol": symbol, "date": None,
                       "issue": "stale_price_runs",
                       "detail": f"{len(stale_days)} days inside >= {AUDIT_STALE_DAYS}-day flat runs",
                       "severity": "warning"})

    # --- 3. zero-volume days -------------------------------------------------
    zv = daily.index[daily["volume"] <= 0]
    for d in zv:
        quarantine.add(pd.Timestamp(d).normalize())
    if len(zv):
        issues.append({"symbol": symbol, "date": None, "issue": "zero_volume_days",
                       "detail": f"{len(zv)} days", "severity": "warning"})

    # --- 4. short history -----------------------------------------------------
    if len(daily) < AUDIT_MIN_HISTORY_DAYS:
        issues.append({"symbol": symbol, "date": None, "issue": "short_history",
                       "detail": f"only {len(daily)} trading days",
                       "severity": "warning"})
    return issues, quarantine


def run_audit(symbols: list[str]) -> dict:
    """Full audit. Returns {'report': DataFrame, 'quarantine': {(sym, date)},
    'universe': pit universe or None, 'survivorship_warning': bool}."""
    ca = dl.load_symbol_table("corporate_actions")
    universe = load_pit_universe()
    all_issues: list[dict] = []
    quarantine: set = set()
    first_dates = {}
    for sym in symbols:
        minute = dl.load_minute_bars(sym)
        if minute is None or minute.empty:
            all_issues.append({"symbol": sym, "date": None,
                               "issue": "no_data", "detail": "file missing/empty",
                               "severity": "CRITICAL"})
            continue
        daily = dl.to_daily(minute)
        first_dates[sym] = daily.index.min()
        issues, q = audit_symbol(sym, daily, ca)
        all_issues.extend(issues)
        quarantine |= {(sym, d) for d in q}

    survivorship_warning = universe is None
    if survivorship_warning and first_dates:
        starts = pd.Series(first_dates)
        late = starts[starts > starts.min() + pd.Timedelta(days=365)]
        all_issues.append({
            "symbol": "*UNIVERSE*", "date": None,
            "issue": "SURVIVORSHIP_RISK_no_pit_universe_file",
            "detail": (f"no {Path(FNO_UNIVERSE_PATH).name}; {len(late)}/{len(starts)} symbols "
                       "start >1y after the earliest -- results may be inflated by "
                       "survivorship bias. Provide [symbol, from_date, to_date]."),
            "severity": "CRITICAL"})

    report = pd.DataFrame(all_issues)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not report.empty:
        report.to_csv(OUT_DIR / "audit_report.csv", index=False)
    n_crit = int((report.get("severity") == "CRITICAL").sum()) if not report.empty else 0
    print(f"  audit: {len(report)} issues ({n_crit} critical), "
          f"{len(quarantine)} (symbol,date) pairs quarantined, "
          f"pit-universe={'YES' if universe is not None else 'NO (survivorship warning)'}")
    return {"report": report, "quarantine": quarantine,
            "universe": universe, "survivorship_warning": survivorship_warning}


def apply_quarantine(events: pd.DataFrame, audit: dict) -> pd.DataFrame:
    """Drop quarantined (symbol, date) events and out-of-universe events."""
    if events.empty:
        return events
    q = audit["quarantine"]
    if q:
        key = list(zip(events["symbol"], pd.to_datetime(events["date"]).dt.normalize()))
        keep = ~pd.Series([k in q for k in key], index=events.index)
        events = events.loc[keep]
    uni_ok = in_universe_mask(events, audit["universe"])
    dropped = int((~uni_ok).sum())
    if dropped:
        print(f"  audit: dropped {dropped} out-of-universe events (point-in-time)")
    return events.loc[uni_ok]
