"""
Meta-labeling and bet sizing (Lopez de Prado, "Advances in Financial ML" ch.3).

The edge-discovery levels answer WHICH events to trade (the primary signal).
Meta-labeling answers a different question: given that the primary rule fired,
WHAT IS THE PROBABILITY this particular instance wins? A secondary classifier
is trained on {primary rule fired} events with label = {trade was profitable
net of costs}, using ALL panel features as inputs. Its predicted probability:

1. filters low-confidence instances (skip if p < threshold), and
2. sizes the bet via fractional Kelly.

This typically raises the Sharpe of a validated edge substantially because it
removes the worst-conditioned instances without touching the discovery logic.

Sample-uniqueness weights: events of the SAME day are highly overlapping bets;
each event is weighted 1 / (# events that day) during training so a single
market-wide gap day cannot dominate the classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from config import COST_PCT, TRAIN_END

META_MIN_EVENTS = 300          # need enough fired events to fit the model
META_PROB_THRESHOLD = 0.55     # skip instances below this predicted P(win)
KELLY_FRACTION = 0.25          # quarter-Kelly (full Kelly is too aggressive)


def _uniqueness_weights(dates: pd.Series) -> np.ndarray:
    """Weight = 1 / concurrency: events sharing a day share one bet."""
    d = pd.to_datetime(dates).dt.normalize()
    counts = d.map(d.value_counts())
    return (1.0 / counts).values


def meta_label_edge(panel: pd.DataFrame, mask: pd.Series, target: str,
                    factor_cols: list[str], rule_name: str) -> dict:
    """Train the meta-model on the rule's TRAIN-period events, evaluate the
    filtered strategy OOS. Returns metrics + the fitted model artifacts."""
    fired = panel.loc[mask.fillna(False)].dropna(subset=[target])
    if len(fired) < META_MIN_EVENTS:
        return {"rule": rule_name, "error": "too few events for meta-labeling"}

    costs = fired["cost_pct"] if "cost_pct" in fired.columns else COST_PCT
    net = fired[target] - costs
    y = (net > 0).astype(int)

    cols = [c for c in factor_cols if c in fired.columns
            and fired[c].notna().mean() > 0.5]
    X = fired[cols].fillna(fired[cols].median())

    tr = fired["date"] <= pd.Timestamp(TRAIN_END)
    te = ~tr
    if tr.sum() < META_MIN_EVENTS // 2 or te.sum() < 50:
        return {"rule": rule_name, "error": "insufficient train/test split"}

    w = _uniqueness_weights(fired.loc[tr, "date"])
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                     subsample=0.7, random_state=42)
    clf.fit(X[tr], y[tr], sample_weight=w)

    # isotonic calibration on train predictions so probabilities are honest
    p_tr_raw = clf.predict_proba(X[tr])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr_raw, y[tr])

    p_te = iso.predict(clf.predict_proba(X[te])[:, 1])
    net_te = net[te].values

    base_mean = float(np.mean(net_te))
    keep = p_te >= META_PROB_THRESHOLD
    filt_mean = float(np.mean(net_te[keep])) if keep.sum() >= 30 else np.nan

    # fractional Kelly sizing: f = kelly_frac * (2p - 1), clipped to [0, 1].
    # PnL is per unit capital; unsized instances contribute 0.
    kelly = np.clip(KELLY_FRACTION * (2 * p_te - 1), 0, 1)
    sized_pnl = float(np.mean(kelly * net_te))
    flat_pnl = float(np.mean(np.where(keep, net_te, 0.0)))

    def _sharpe(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan

    return {
        "rule": rule_name,
        "n_train": int(tr.sum()),
        "n_oos": int(te.sum()),
        "oos_base_mean": round(base_mean, 4),
        "oos_filtered_mean": round(filt_mean, 4) if not np.isnan(filt_mean) else np.nan,
        "oos_kept_frac": round(float(keep.mean()), 3),
        "oos_base_sharpe": round(_sharpe(net_te), 2),
        "oos_filtered_sharpe": round(_sharpe(np.where(keep, net_te, 0.0)), 2),
        "oos_kelly_mean": round(sized_pnl, 4),
        "improves": bool(not np.isnan(filt_mean) and filt_mean > base_mean
                         and flat_pnl >= base_mean * 0.8),
        "top_features": list(pd.Series(clf.feature_importances_, index=cols)
                             .nlargest(6).index),
    }


def run_meta_labeling(panel: pd.DataFrame, masks: dict[str, pd.Series],
                      target: str, factor_cols: list[str]) -> pd.DataFrame:
    rows = []
    for rule_name, mask in masks.items():
        res = meta_label_edge(panel, mask, target, factor_cols, rule_name)
        rows.append(res)
    return pd.DataFrame(rows)
