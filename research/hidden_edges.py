"""
LEVEL 4 -- hidden-edge detectors.

These find the edge classes that quantile-bucket tests and simple AND-combos
structurally CANNOT see:

  A. optimal_thresholds      : exact breakpoint search (non-linear cliffs)
  B. regime_conditional      : edges that only exist inside a market regime
  C. edge_decay              : rolling-window stability -- flags dead edges
                               that still look "significant" on the full sample
  D. lead_lag_scanner        : cross-asset predictors at lags 1..N days
                               (crude -> OMCs, S&P -> IT, large-cap -> peers)
  E. sequence_patterns       : path-dependent edges (what happened the last
                               k event-days changes today's odds)
  F. anomaly_precursors      : abnormal factor readings 1-3 days BEFORE the
                               event that shift outcomes
  G. hidden_regimes          : unsupervised clustering of market state --
                               regimes no single indicator defines
  H. calendar_scan           : exhaustive micro-seasonality
                               (expiry day, day-of-month, pre/post holiday...)

All discovery runs on the TRAIN window only; every candidate is re-measured
on the untouched OOS window through stats.oos_confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FDR_ALPHA, MIN_EVENTS, TRAIN_END
from stats import (benjamini_hochberg, edge_metrics, oos_confirmation,
                   train_test_split_by_date)

META = {"symbol", "date", "mr_time"}


def _factors(df: pd.DataFrame, targets: list[str]) -> list[str]:
    skip = META | set(targets)
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# ======================================================================
# A. optimal threshold search (breakpoint / cliff edges)
# ======================================================================
def optimal_thresholds(panel: pd.DataFrame, target: str,
                       targets: list[str]) -> pd.DataFrame:
    """For each factor, find the single cutoff that maximizes the mean-return
    difference between the two sides (like a 1-split decision tree), then
    confirm the winning side OOS. Catches edges that only exist beyond a
    specific level (e.g. gaps > 4.7%) which fixed quantiles blur away."""
    train, test = train_test_split_by_date(panel, TRAIN_END)
    rows = []
    for f in _factors(panel, targets):
        sub = train[[f, target]].dropna()
        if len(sub) < MIN_EVENTS * 2 or sub[f].nunique() < 10:
            continue
        cands = sub[f].quantile(np.arange(0.05, 1.0, 0.05)).unique()
        best = None
        for c in cands:
            hi, lo = sub.loc[sub[f] > c, target], sub.loc[sub[f] <= c, target]
            if len(hi) < MIN_EVENTS or len(lo) < MIN_EVENTS:
                continue
            diff = hi.mean() - lo.mean()
            if best is None or abs(diff) > abs(best[1]):
                best = (c, diff)
        if best is None:
            continue
        cut, diff = best
        side = ">" if diff > 0 else "<="
        m_tr = sub[f] > cut if diff > 0 else sub[f] <= cut
        te = test[[f, target]].dropna()
        m_te = te[f] > cut if diff > 0 else te[f] <= cut
        tr_m = edge_metrics(sub.loc[m_tr, target])
        te_m = edge_metrics(te.loc[m_te, target])
        row = {"factor": f, "rule": f"{f} {side} {cut:.4g}",
               "is_mean": tr_m["mean_ret"], "is_p": tr_m["p_value"],
               "is_n": tr_m["n"]}
        row.update(oos_confirmation(tr_m, te_m))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows).set_index("factor")
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", key=abs, ascending=False)


# ======================================================================
# B. regime-conditional edges
# ======================================================================
DEFAULT_REGIME_DEFS = {
    # regime column -> (name, condition builder)
    "vix_pctile_1y": [("low_vol", lambda s: s <= 0.3),
                      ("high_vol", lambda s: s >= 0.7)],
    "nifty_above_50dma": [("uptrend", lambda s: s == 1),
                          ("downtrend", lambda s: s == 0)],
    "nifty_ret_5d": [("mkt_weak", lambda s: s < s.quantile(0.3)),
                     ("mkt_strong", lambda s: s > s.quantile(0.7))],
}


def regime_conditional(panel: pd.DataFrame, target: str,
                       targets: list[str],
                       top_factors: list[str] | None = None) -> pd.DataFrame:
    """Re-test factors INSIDE each regime. An edge averaging to zero on the
    full sample (e.g. mean reversion that only works in low VIX) shows up here."""
    train, test = train_test_split_by_date(panel, TRAIN_END)
    factors = top_factors or _factors(panel, targets)[:30]
    rows = []
    for reg_col, regimes in DEFAULT_REGIME_DEFS.items():
        if reg_col not in panel.columns:
            continue
        for reg_name, cond in regimes:
            m_tr, m_te = cond(train[reg_col]), cond(test[reg_col])
            if m_tr.sum() < MIN_EVENTS * 2:
                continue
            tr_r, te_r = train[m_tr.fillna(False)], test[m_te.fillna(False)]
            # baseline: does the TARGET itself behave differently in-regime?
            base_tr, base_te = edge_metrics(tr_r[target]), edge_metrics(te_r[target])
            row = {"regime": f"{reg_col}:{reg_name}", "factor": "(baseline)",
                   "is_mean": base_tr["mean_ret"], "is_p": base_tr["p_value"],
                   "is_n": base_tr["n"]}
            row.update(oos_confirmation(base_tr, base_te))
            rows.append(row)
            # factor edges inside the regime (median split for speed)
            for f in factors:
                s = tr_r[f].dropna()
                if len(s) < MIN_EVENTS * 2:
                    continue
                med = s.median()
                hi_tr = edge_metrics(tr_r.loc[tr_r[f] > med, target])
                lo_tr = edge_metrics(tr_r.loc[tr_r[f] <= med, target])
                if hi_tr["n"] < MIN_EVENTS or lo_tr["n"] < MIN_EVENTS:
                    continue
                pick_hi = abs(hi_tr["mean_ret"]) > abs(lo_tr["mean_ret"])
                side, tr_m = (">", hi_tr) if pick_hi else ("<=", lo_tr)
                te_mask = te_r[f] > med if pick_hi else te_r[f] <= med
                te_m = edge_metrics(te_r.loc[te_mask, target])
                row = {"regime": f"{reg_col}:{reg_name}",
                       "factor": f"{f} {side} {med:.4g}",
                       "is_mean": tr_m["mean_ret"], "is_p": tr_m["p_value"],
                       "is_n": tr_m["n"]}
                row.update(oos_confirmation(tr_m, te_m))
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", key=abs, ascending=False)


# ======================================================================
# C. edge decay / stability
# ======================================================================
def edge_decay(panel: pd.DataFrame, target: str, rules: pd.DataFrame,
               window_days: int = 252) -> pd.DataFrame:
    """Rolling yearly re-measurement of confirmed rules. Flags edges that
    were strong early and are now flat/negative (crowded-out edges)."""
    if rules.empty:
        return pd.DataFrame()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    years = panel["date"].dt.year
    rows = []
    for name, r in rules.head(20).iterrows():
        rule = r.get("rule") or r.get("combo") or str(name)
        mask = _eval_rule(panel, rule)
        if mask is None:
            continue
        yearly = panel[mask].groupby(years[mask])[target].agg(["mean", "count"])
        yearly = yearly[yearly["count"] >= 20]
        if len(yearly) < 4:
            continue
        first_half = yearly["mean"].iloc[: len(yearly) // 2].mean()
        second_half = yearly["mean"].iloc[len(yearly) // 2:].mean()
        trend = np.polyfit(range(len(yearly)), yearly["mean"], 1)[0]
        rows.append({
            "rule": rule,
            "years_active": len(yearly),
            "early_mean": first_half, "recent_mean": second_half,
            "yearly_trend": trend,
            "pct_years_positive": (yearly["mean"] > 0).mean(),
            "status": ("DECAYING" if second_half < 0.5 * first_half and first_half > 0
                       else "STRENGTHENING" if second_half > 1.5 * first_half
                       else "STABLE"),
        })
    return pd.DataFrame(rows)


def _eval_rule(df: pd.DataFrame, rule: str) -> pd.Series | None:
    """Evaluate simple 'factor > x' / 'factor <= x' / AND-joined rules."""
    mask = pd.Series(True, index=df.index)
    for part in rule.split(" AND "):
        part = part.strip()
        for op in (" <= ", " > ", " in "):
            if op in part:
                f, val = part.split(op)
                f = f.strip()
                if f not in df.columns:
                    return None
                if op == " in ":  # bucket label
                    lo, hi = val.strip("([])").split(",")
                    mask &= (df[f] > float(lo)) & (df[f] <= float(hi))
                elif op == " > ":
                    mask &= df[f] > float(val)
                else:
                    mask &= df[f] <= float(val)
                break
        else:
            return None
    return mask


# ======================================================================
# D. lead-lag scanner (cross-asset, lagged predictors)
# ======================================================================
def lead_lag_scanner(panel: pd.DataFrame, target: str,
                     targets: list[str], max_lag: int = 5) -> pd.DataFrame:
    """Shift every market-wide factor by 1..max_lag additional days and
    measure rank-IC with the target on train, confirm on OOS.
    Finds delayed transmission: crude oil move 3 days ago still predicting
    OMC gap behavior today, S&P weakness bleeding into IT for 2 days, etc."""
    mkt_factors = [c for c in _factors(panel, targets)
                   if any(k in c for k in
                          ("nifty", "sp500", "crude", "usdinr", "vix", "fii",
                           "sector", "mkt_"))]
    panel = panel.sort_values(["symbol", "date"])
    train, test = train_test_split_by_date(panel, TRAIN_END)
    rows = []
    for f in mkt_factors:
        for lag in range(1, max_lag + 1):
            tr_f = train.groupby("symbol")[f].shift(lag)
            te_f = test.groupby("symbol")[f].shift(lag)
            ic_tr = tr_f.corr(train[target], method="spearman")
            ic_te = te_f.corr(test[target], method="spearman")
            n = tr_f.notna().sum()
            if n < MIN_EVENTS * 2 or pd.isna(ic_tr):
                continue
            # t-stat of rank IC
            t = ic_tr * np.sqrt((n - 2) / max(1e-9, 1 - ic_tr ** 2))
            from scipy import stats as sps
            p = 2 * sps.t.sf(abs(t), n - 2)
            rows.append({"factor": f, "extra_lag_days": lag,
                         "is_rank_ic": ic_tr, "oos_rank_ic": ic_te,
                         "is_p": p, "n": int(n),
                         "confirmed": bool(np.sign(ic_tr) == np.sign(ic_te)
                                           and abs(ic_te) >= 0.5 * abs(ic_tr))})
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_rank_ic", key=abs, ascending=False)


# ======================================================================
# E. sequence / path-dependent patterns
# ======================================================================
def sequence_patterns(panel: pd.DataFrame, target: str,
                      max_len: int = 3) -> pd.DataFrame:
    """Encode each event-day per symbol as W (target>0) / L (target<=0) and
    test whether the last k outcomes change today's odds.
    e.g. 'LLL' before a gap day -> gaps fade harder after 3 failed days."""
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["_o"] = np.where(panel[target] > 0, "W", "L")
    hist_cols = []
    for k in range(1, max_len + 1):
        col = f"_h{k}"
        panel[col] = panel.groupby("symbol")["_o"].shift(k)
        hist_cols.append(col)
    train, test = train_test_split_by_date(panel, TRAIN_END)
    rows = []
    for k in range(1, max_len + 1):
        cols = hist_cols[:k][::-1]  # oldest first
        tr_seq = train[cols].fillna("?").agg("".join, axis=1)
        te_seq = test[cols].fillna("?").agg("".join, axis=1)
        valid_tr = ~train[cols].isna().any(axis=1)
        for seq in tr_seq[valid_tr].unique():
            m_tr = valid_tr & (tr_seq == seq)
            if m_tr.sum() < MIN_EVENTS:
                continue
            tr_m = edge_metrics(train.loc[m_tr, target])
            te_m = edge_metrics(test.loc[te_seq == seq, target])
            row = {"pattern": f"last {k} outcomes = {seq}",
                   "is_mean": tr_m["mean_ret"], "is_win": tr_m["win_rate"],
                   "is_p": tr_m["p_value"], "is_n": tr_m["n"]}
            row.update(oos_confirmation(tr_m, te_m))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", key=abs, ascending=False)


# ======================================================================
# F. anomaly precursors
# ======================================================================
ANOMALY_CANDIDATES = ["rvol_20d", "opt_oi_change", "fut_oi_change",
                      "opt_iv", "news_count", "opt_pcr"]


def anomaly_precursors(panel: pd.DataFrame, target: str,
                       z_cut: float = 2.0) -> pd.DataFrame:
    """Flag events where a factor was ANOMALOUS (z>=2 vs the symbol's own
    trailing 60-event distribution) 1-3 days before, and test outcome shift.
    Catches informed positioning: OI/volume/IV spikes preceding moves."""
    panel = panel.sort_values(["symbol", "date"]).copy()
    train_rows = []
    cols = [c for c in ANOMALY_CANDIDATES if c in panel.columns]
    for f in cols:
        g = panel.groupby("symbol")[f]
        mu = g.transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
        sd = g.transform(lambda s: s.rolling(60, min_periods=20).std().shift(1))
        z = (panel[f] - mu) / sd
        for lag in (1, 2, 3):
            zl = z.groupby(panel["symbol"]).shift(lag)
            panel[f"_anom_{f}_lag{lag}"] = (zl.abs() >= z_cut).astype(float) \
                .where(zl.notna())
    anom_cols = [c for c in panel.columns if c.startswith("_anom_")]
    train, test = train_test_split_by_date(panel, TRAIN_END)
    for c in anom_cols:
        m_tr, m_te = train[c] == 1, test[c] == 1
        if m_tr.sum() < MIN_EVENTS:
            continue
        tr_m = edge_metrics(train.loc[m_tr, target])
        base = edge_metrics(train.loc[train[c] == 0, target])
        te_m = edge_metrics(test.loc[m_te, target])
        row = {"anomaly": c.replace("_anom_", ""),
               "is_mean": tr_m["mean_ret"], "base_mean": base["mean_ret"],
               "lift": tr_m["mean_ret"] - base["mean_ret"],
               "is_p": tr_m["p_value"], "is_n": tr_m["n"]}
        row.update(oos_confirmation(tr_m, te_m))
        train_rows.append(row)
    if not train_rows:
        return pd.DataFrame()
    res = pd.DataFrame(train_rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("lift", key=abs, ascending=False)


# ======================================================================
# G. hidden regimes via clustering
# ======================================================================
def hidden_regimes(panel: pd.DataFrame, target: str,
                   targets: list[str], n_clusters: int = 4) -> pd.DataFrame:
    """KMeans on market-state factors -> regimes no single indicator defines.
    Then measure the target inside each discovered regime, OOS-confirmed."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return pd.DataFrame()
    state_cols = [c for c in _factors(panel, targets)
                  if any(k in c for k in ("nifty", "vix", "sp500", "crude",
                                          "usdinr", "fii", "mkt_"))]
    # keep only well-covered state columns so one short-history factor
    # (e.g. a 1y-rolling percentile) doesn't wipe out the training window
    state_cols = [c for c in state_cols
                  if panel[c].notna().mean() >= 0.8]
    if len(state_cols) < 3:
        return pd.DataFrame()
    sub = panel.dropna(subset=state_cols + [target])
    train, test = train_test_split_by_date(sub, TRAIN_END)
    if len(train) < 1000:
        return pd.DataFrame()
    scaler = StandardScaler().fit(train[state_cols])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) \
        .fit(scaler.transform(train[state_cols]))
    tr_lab = km.labels_
    te_lab = km.predict(scaler.transform(test[state_cols]))
    rows = []
    for k in range(n_clusters):
        tr_m = edge_metrics(train.loc[tr_lab == k, target])
        te_m = edge_metrics(test.loc[te_lab == k, target])
        centroid = pd.Series(scaler.inverse_transform(
            km.cluster_centers_)[k], index=state_cols)
        desc = ", ".join(f"{c}={v:.2f}"
                         for c, v in centroid.nlargest(3).items())
        row = {"regime": f"cluster_{k}", "profile": desc,
               "is_mean": tr_m["mean_ret"], "is_win": tr_m["win_rate"],
               "is_p": tr_m["p_value"], "is_n": tr_m["n"]}
        row.update(oos_confirmation(tr_m, te_m))
        rows.append(row)
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", key=abs, ascending=False)


# ======================================================================
# H. calendar micro-seasonality scan
# ======================================================================
def calendar_scan(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    """Exhaustive calendar conditions: weekday, day-of-month bands, month,
    expiry week (last Thu of month proxy), first/last trading day of month."""
    p = panel.copy()
    d = pd.to_datetime(p["date"])
    conds = {
        **{f"weekday_{n}": d.dt.dayofweek == i
           for i, n in enumerate(["mon", "tue", "wed", "thu", "fri"])},
        "month_start_d1_3": d.dt.day <= 3,
        "month_end_d28p": d.dt.day >= 28,
        "expiry_week": (d.dt.day >= 22) & (d.dt.dayofweek == 3),
        **{f"month_{m}": d.dt.month == m for m in range(1, 13)},
    }
    train_mask = d <= pd.Timestamp(TRAIN_END)
    rows = []
    for name, m in conds.items():
        m_tr, m_te = m & train_mask, m & ~train_mask
        if m_tr.sum() < MIN_EVENTS:
            continue
        tr_m = edge_metrics(p.loc[m_tr, target])
        te_m = edge_metrics(p.loc[m_te, target])
        row = {"condition": name, "is_mean": tr_m["mean_ret"],
               "is_win": tr_m["win_rate"], "is_p": tr_m["p_value"],
               "is_n": tr_m["n"]}
        row.update(oos_confirmation(tr_m, te_m))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", key=abs, ascending=False)


# ======================================================================
# orchestrator
# ======================================================================
def run_all(panel: pd.DataFrame, target: str, targets: list[str],
            level1: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    top = (list(level1[level1["fdr_pass"]].head(15).index)
           if level1 is not None and not level1.empty else None)
    out = {
        "A_optimal_thresholds": optimal_thresholds(panel, target, targets),
        "B_regime_conditional": regime_conditional(panel, target, targets, top),
        "D_lead_lag": lead_lag_scanner(panel, target, targets),
        "E_sequence_patterns": sequence_patterns(panel, target),
        "F_anomaly_precursors": anomaly_precursors(panel, target),
        "G_hidden_regimes": hidden_regimes(panel, target, targets),
        "H_calendar": calendar_scan(panel, target),
    }
    # C: decay check on whatever A found
    thr = out["A_optimal_thresholds"]
    if not thr.empty:
        out["C_edge_decay"] = edge_decay(
            panel, target, thr[thr.get("confirmed", False) == True])  # noqa: E712
    else:
        out["C_edge_decay"] = pd.DataFrame()
    return out
