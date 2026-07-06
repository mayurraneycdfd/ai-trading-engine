"""
Regime detection, edge-drift monitoring, and Bayesian shrinkage.

1. GMM market regimes: fits a Gaussian Mixture on market-state variables
   (trend, vol, breadth proxies) over the TRAIN window only, then assigns
   every day a regime id + probabilities. Edges are re-tested per regime;
   the live scorer can refuse to trade an edge outside its home regime.

2. CUSUM drift monitor: a sequential Page-Hinkley style test on each edge's
   event returns. Detects the point where an edge's mean degrades from its
   in-sample level -- catches edge death FAST instead of at the yearly review.

3. James-Stein shrinkage of per-symbol edge means toward the pooled mean.
   A rule that made 40% of its money on one stock is one delisting away
   from nothing; shrinkage reveals how much of the edge is broad vs narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from config import TRAIN_END

N_REGIMES = 3
CUSUM_THRESHOLD = 8.0      # detection threshold (in units of return std)
CUSUM_DRIFT = 0.25         # allowed slack per observation (fraction of std)


# --------------------------------------------------------------- regimes ---
def fit_market_regimes(panel: pd.DataFrame,
                       n_regimes: int = N_REGIMES) -> pd.DataFrame | None:
    """Assign each date a market regime. State variables are chosen from
    what the panel actually has; fit on train dates only (no lookahead)."""
    candidates = [c for c in ("nifty_ret_5d", "vix_level", "vix_pctile_1y",
                              "nifty_above_50dma", "sp500_overnight",
                              "mkt_gap_median", "fii_net_5d")
                  if c in panel.columns and panel[c].notna().mean() > 0.6]
    if len(candidates) < 3:
        return None
    day = panel.groupby("date")[candidates].median().dropna()
    if len(day) < 200:
        return None
    tr = day[day.index <= pd.Timestamp(TRAIN_END)]
    if len(tr) < 100:
        return None
    mu, sd = tr.mean(), tr.std().replace(0, np.nan)
    # drop state columns that are constant (sd NaN) in the train window --
    # they would NaN-out every z-scored row and empty the fit sample
    good = sd.dropna().index.tolist()
    if len(good) < 3:
        return None
    day, tr, mu, sd = day[good], tr[good], mu[good], sd[good]
    z_tr = ((tr - mu) / sd).dropna()
    if len(z_tr) < 100:
        return None
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                          random_state=42, n_init=3)
    gmm.fit(z_tr)
    z_all = ((day - mu) / sd).dropna()
    out = pd.DataFrame(index=z_all.index)
    out["regime"] = gmm.predict(z_all)
    proba = gmm.predict_proba(z_all)
    out["regime_confidence"] = proba.max(axis=1)
    return out


def edge_by_regime(panel: pd.DataFrame, mask: pd.Series, target: str,
                   regimes: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Net mean of the rule inside each regime, train and OOS separately.
    An edge concentrated in one regime is a REGIME BET, not a stock edge."""
    sel = panel.loc[mask.fillna(False)].copy()
    costs = sel["cost_pct"] if "cost_pct" in sel.columns else 0.0
    sel["net"] = sel[target] - costs
    sel["regime"] = sel["date"].map(regimes["regime"])
    sel = sel.dropna(subset=["net", "regime"])
    rows = []
    for reg, g in sel.groupby("regime"):
        tr = g[g["date"] <= pd.Timestamp(TRAIN_END)]["net"]
        te = g[g["date"] > pd.Timestamp(TRAIN_END)]["net"]
        rows.append({
            "rule": rule_name, "regime": int(reg),
            "n_train": len(tr), "n_oos": len(te),
            "train_mean": round(float(tr.mean()), 4) if len(tr) else np.nan,
            "oos_mean": round(float(te.mean()), 4) if len(te) else np.nan,
            "tradeable_in_regime": bool(len(tr) >= 50 and len(te) >= 20
                                        and tr.mean() > 0 and te.mean() > 0),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- CUSUM ---
def cusum_drift(dates: pd.Series, net_returns: pd.Series,
                rule_name: str) -> dict:
    """Page-Hinkley CUSUM on the chronological net returns of an edge.
    Reference mean = train-period mean. Alarm when the cumulative
    shortfall exceeds CUSUM_THRESHOLD * std."""
    df = pd.DataFrame({"d": pd.to_datetime(dates.values),
                       "r": net_returns.values}).dropna().sort_values("d")
    tr = df[df["d"] <= pd.Timestamp(TRAIN_END)]["r"]
    if len(tr) < 50:
        return {"rule": rule_name, "error": "too little train history"}
    mu0, sd = float(tr.mean()), float(tr.std())
    if sd <= 0:
        return {"rule": rule_name, "error": "zero variance"}
    slack = CUSUM_DRIFT * sd
    cum, alarm_date, worst = 0.0, None, 0.0
    for _, row in df[df["d"] > pd.Timestamp(TRAIN_END)].iterrows():
        cum = max(0.0, cum + (mu0 - row["r"]) - slack)
        worst = max(worst, cum)
        if cum > CUSUM_THRESHOLD * sd and alarm_date is None:
            alarm_date = row["d"]
    return {
        "rule": rule_name,
        "train_mean": round(mu0, 4),
        "cusum_peak_sigma": round(worst / sd, 2),
        "drift_alarm": alarm_date is not None,
        "alarm_date": str(alarm_date.date()) if alarm_date is not None else "",
    }


# ------------------------------------------------------------- shrinkage ---
def james_stein_by_symbol(panel: pd.DataFrame, mask: pd.Series, target: str,
                          rule_name: str) -> dict:
    """Shrink per-symbol edge means toward the pooled mean; report how much
    of the edge survives shrinkage and its concentration (HHI)."""
    sel = panel.loc[mask.fillna(False)].copy()
    costs = sel["cost_pct"] if "cost_pct" in sel.columns else 0.0
    sel["net"] = sel[target] - costs
    sel = sel.dropna(subset=["net"])
    g = sel.groupby("symbol")["net"]
    means, counts = g.mean(), g.size()
    keep = counts >= 10
    means, counts = means[keep], counts[keep]
    if len(means) < 5:
        return {"rule": rule_name, "error": "too few symbols"}
    grand = float(sel["net"].mean())
    var_within = float(sel["net"].var())
    # James-Stein factor per symbol: shrink more when n small / variance high
    shrunk = grand + (1 - (var_within / counts) /
                      ((means - grand) ** 2 + var_within / counts)) * (means - grand)
    pos_frac = float((shrunk > 0).mean())
    # concentration: HHI of positive contribution across symbols
    contrib = (means * counts).clip(lower=0)
    total = contrib.sum()
    hhi = float(((contrib / total) ** 2).sum()) if total > 0 else np.nan
    return {
        "rule": rule_name,
        "n_symbols": int(len(means)),
        "pooled_mean": round(grand, 4),
        "shrunk_pos_frac": round(pos_frac, 3),
        "concentration_hhi": round(hhi, 3) if not np.isnan(hhi) else np.nan,
        "broad_edge": bool(pos_frac >= 0.6 and (np.isnan(hhi) or hhi <= 0.15)),
    }
