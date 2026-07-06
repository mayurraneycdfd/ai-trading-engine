"""
Three-level edge discovery on the (events x factors) panel.

  LEVEL 1  single factors      : bucket every factor vs outcome, FDR-corrected
  LEVEL 2  factor combinations : 2- and 3-factor AND-conditions built from the
                                 top single factors, validated out-of-sample
  LEVEL 3  cumulative model    : gradient boosting on ALL factors at once,
                                 walk-forward, with feature importances --
                                 finds non-linear multi-factor structure the
                                 bucket tests miss.

Discovery happens ONLY on the training window; every candidate edge must be
confirmed on the untouched out-of-sample window to be reported.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from config import (COMBO_MAX_FACTORS, COMBO_TOP_SINGLE, FDR_ALPHA,
                    MIN_EVENTS, N_BUCKETS, TRAIN_END)
from stats import (benjamini_hochberg, bucket_analysis, edge_metrics,
                   factor_score, oos_confirmation, train_test_split_by_date)

META_COLS = {"symbol", "date", "mr_time", "cost_pct"}

# outcome-label prefixes that must NEVER appear as predictors (lookahead guard)
OUTCOME_PREFIXES = ("cont_", "go_", "revert_", "rev_", "tb_", "filled_")


def _factor_cols(df: pd.DataFrame, target: str, all_targets: list[str]) -> list[str]:
    skip = META_COLS | set(all_targets)
    return [c for c in df.columns
            if c not in skip
            and not c.startswith(OUTCOME_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]


# ======================================================================
# LEVEL 1 -- single factors
# ======================================================================
def level1_single_factors(panel: pd.DataFrame, target: str,
                          all_targets: list[str]) -> pd.DataFrame:
    train, test = train_test_split_by_date(panel, TRAIN_END)
    rows = []
    for factor in _factor_cols(panel, target, all_targets):
        b = bucket_analysis(train, factor, target, N_BUCKETS, MIN_EVENTS)
        if b is None:
            continue
        score = factor_score(b)
        score["factor"] = factor
        # OOS check on the best bucket condition
        best_mask_train = _bucket_mask(train, factor, score["best_bucket"])
        best_mask_test = _bucket_mask(test, factor, score["best_bucket"])
        tr_m = edge_metrics(train.loc[best_mask_train, target])
        te_m = edge_metrics(test.loc[best_mask_test, target])
        score.update(oos_confirmation(tr_m, te_m))
        rows.append(score)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows).set_index("factor")
    res["fdr_pass"] = benjamini_hochberg(res["best_p"], FDR_ALPHA)
    return res.sort_values("spread", ascending=False)


def _bucket_mask(df: pd.DataFrame, factor: str, bucket_label: str) -> pd.Series:
    """Recreate the boolean condition for a bucket label like '(1.2, 3.4]' or '1'."""
    s = df[factor]
    if bucket_label.startswith("(") or bucket_label.startswith("["):
        lo, hi = bucket_label.strip("([])").split(",")
        return (s > float(lo)) & (s <= float(hi))
    try:
        return s == float(bucket_label)
    except ValueError:
        return pd.Series(False, index=df.index)


# ======================================================================
# LEVEL 2 -- factor combinations (cumulative of a few factors)
# ======================================================================
def level2_combinations(panel: pd.DataFrame, target: str,
                        level1: pd.DataFrame) -> pd.DataFrame:
    """AND together the best-bucket conditions of the top single factors."""
    if level1.empty:
        return pd.DataFrame()
    top = level1[level1["fdr_pass"]].head(COMBO_TOP_SINGLE)
    if len(top) < 2:
        top = level1.head(COMBO_TOP_SINGLE)
    train, test = train_test_split_by_date(panel, TRAIN_END)

    conditions = {f: (str(row["best_bucket"])) for f, row in top.iterrows()}
    rows = []
    factors = list(conditions)
    for k in range(2, COMBO_MAX_FACTORS + 1):
        for combo in combinations(factors, k):
            m_tr = pd.Series(True, index=train.index)
            m_te = pd.Series(True, index=test.index)
            for f in combo:
                m_tr &= _bucket_mask(train, f, conditions[f])
                m_te &= _bucket_mask(test, f, conditions[f])
            if m_tr.sum() < MIN_EVENTS:
                continue
            tr_m = edge_metrics(train.loc[m_tr, target])
            te_m = edge_metrics(test.loc[m_te, target])
            row = {
                "combo": " AND ".join(f"{f} in {conditions[f]}" for f in combo),
                "n_factors": k,
                "is_n": tr_m["n"], "is_mean": tr_m["mean_ret"],
                "is_win": tr_m["win_rate"], "is_p": tr_m["p_value"],
            }
            row.update({f"oos_{kk}": v for kk, v in oos_confirmation(tr_m, te_m).items()})
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["fdr_pass"] = benjamini_hochberg(res["is_p"], FDR_ALPHA)
    return res.sort_values("is_mean", ascending=False)


# ======================================================================
# LEVEL 3 -- cumulative model on ALL factors
# ======================================================================
def level3_cumulative(panel: pd.DataFrame, target: str,
                      all_targets: list[str]) -> dict:
    """Gradient boosting on every factor simultaneously.
    Reports OOS hit rate of top-decile predictions + feature importances."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
    except ImportError:
        return {"error": "scikit-learn not installed"}

    cols = _factor_cols(panel, target, all_targets)
    sub = panel.dropna(subset=[target])
    train, test = train_test_split_by_date(sub, TRAIN_END)
    # drop columns that are constant or (near-)all-NaN in EITHER split:
    # they break tree binning and carry no signal
    cols = [c for c in cols
            if train[c].notna().sum() >= 50 and train[c].nunique(dropna=True) >= 2
            and test[c].nunique(dropna=True) >= 1]
    if len(train) < 500 or len(test) < 200:
        return {"error": "not enough events for cumulative model"}

    Xtr, ytr = train[cols], train[target]
    Xte, yte = test[cols], test[target]

    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=42)
    model.fit(Xtr, ytr)

    pred = pd.Series(model.predict(Xte), index=Xte.index)
    decile = pred.rank(pct=True)
    top = yte[decile >= 0.9]
    bottom = yte[decile <= 0.1]

    imp = permutation_importance(model, Xte.fillna(0), yte,
                                 n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(imp.importances_mean, index=cols) \
        .sort_values(ascending=False)

    return {
        "oos_top_decile": edge_metrics(top),
        "oos_bottom_decile": edge_metrics(bottom),
        "oos_long_short_spread": top.mean() - bottom.mean(),
        "oos_rank_ic": pred.corr(yte, method="spearman"),
        "feature_importance": importances.head(25),
        "n_train": len(train), "n_test": len(test),
    }
