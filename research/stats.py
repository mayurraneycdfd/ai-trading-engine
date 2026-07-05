"""
Statistical machinery: edge metrics, significance tests, and
multiple-hypothesis (false discovery) control. Testing hundreds of factors
WILL produce fake edges by chance -- Benjamini-Hochberg keeps us honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def edge_metrics(returns: pd.Series) -> dict:
    """Summary stats for a set of event returns (in %)."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    win = (r > 0).mean()
    t, p = sps.ttest_1samp(r, 0.0)
    return {
        "n": n,
        "mean_ret": r.mean(),
        "median_ret": r.median(),
        "win_rate": win,
        "t_stat": t,
        "p_value": p,
        "sharpe_like": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "profit_factor": r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf,
    }


def bucket_analysis(df: pd.DataFrame, factor: str, target: str,
                    n_buckets: int = 5, min_events: int = 100) -> pd.DataFrame | None:
    """Quantile-bucket a factor and measure the target return in each bucket.
    Monotonicity across buckets = real relationship, not noise."""
    sub = df[[factor, target]].dropna()
    if len(sub) < min_events * 2:
        return None
    nunique = sub[factor].nunique()
    if nunique <= 2:  # binary / categorical factor
        groups = sub.groupby(sub[factor])
    else:
        try:
            buckets = pd.qcut(sub[factor], n_buckets, duplicates="drop")
        except ValueError:
            return None
        groups = sub.groupby(buckets, observed=True)
    rows = []
    for name, g in groups:
        m = edge_metrics(g[target])
        m["bucket"] = str(name)
        rows.append(m)
    res = pd.DataFrame(rows)
    if res["n"].min() < min_events // n_buckets:
        return None
    # spearman rank correlation between bucket order and mean return
    res["bucket_rank"] = range(len(res))
    if len(res) >= 3:
        rho, _ = sps.spearmanr(res["bucket_rank"], res["mean_ret"])
        res["monotonicity"] = rho
    return res


def factor_score(bucket_df: pd.DataFrame) -> dict:
    """Collapse a bucket analysis into a single edge score:
    spread between best and worst bucket + significance of best bucket."""
    best = bucket_df.loc[bucket_df["mean_ret"].idxmax()]
    worst = bucket_df.loc[bucket_df["mean_ret"].idxmin()]
    return {
        "spread": best["mean_ret"] - worst["mean_ret"],
        "best_bucket": best["bucket"],
        "best_mean": best["mean_ret"],
        "best_win_rate": best["win_rate"],
        "best_n": best["n"],
        "best_p": best["p_value"],
        "monotonicity": bucket_df.get("monotonicity", pd.Series([np.nan])).iloc[0],
    }


def benjamini_hochberg(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Return boolean mask of hypotheses that survive FDR control."""
    p = pvals.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p.values <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    survivors = set(p.index[:k])
    return pvals.index.to_series().isin(survivors)


def train_test_split_by_date(df: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(train_end)
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else df.index
    return df[dates <= cutoff], df[dates > cutoff]


def oos_confirmation(train_metrics: dict, test_metrics: dict) -> dict:
    """An edge is CONFIRMED only if it holds out-of-sample with same sign
    and at least half the in-sample magnitude."""
    if test_metrics.get("n", 0) < 30:
        return {"confirmed": False, "reason": "too few OOS events"}
    same_sign = np.sign(train_metrics["mean_ret"]) == np.sign(test_metrics["mean_ret"])
    holds = abs(test_metrics["mean_ret"]) >= 0.5 * abs(train_metrics["mean_ret"])
    return {
        "confirmed": bool(same_sign and holds),
        "is_mean": train_metrics["mean_ret"],
        "oos_mean": test_metrics["mean_ret"],
        "oos_win_rate": test_metrics["win_rate"],
        "oos_n": test_metrics["n"],
    }
