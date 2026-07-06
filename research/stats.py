"""
Statistical machinery: edge metrics, significance tests, and
multiple-hypothesis (false discovery) control. Testing hundreds of factors
WILL produce fake edges by chance -- Benjamini-Hochberg keeps us honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def kish_effective_n(returns: pd.Series, dates: pd.Series | None) -> float:
    """Kish effective sample size for cross-sectionally correlated events.
    50 stocks gapping on the SAME day due to one overnight move are ~1 bet,
    not 50. n_eff = n / (1 + (n_bar - 1) * rho) where rho is the mean
    within-day pairwise correlation proxy and n_bar the mean events/day."""
    r = returns.dropna()
    if dates is None or len(r) < 30:
        return float(len(r))
    d = pd.to_datetime(dates.reindex(r.index))
    day_groups = r.groupby(d.dt.normalize())
    n_bar = float(day_groups.size().mean())
    if n_bar <= 1.5:
        return float(len(r))
    # ANOVA-style within-day correlation proxy: between-day variance share
    day_means = day_groups.mean()
    day_sizes = day_groups.size()
    grand = r.mean()
    between = float((day_sizes * (day_means - grand) ** 2).sum())
    total = float(((r - grand) ** 2).sum())
    if total <= 0:
        return float(len(r))
    icc = max(0.0, min(0.99, (between / total - 1.0 / n_bar) / (1 - 1.0 / n_bar)))
    n_eff = len(r) / (1 + (n_bar - 1) * icc)
    return float(max(min(n_eff, len(r)), day_groups.ngroups))


def edge_metrics(returns: pd.Series, dates: pd.Series | None = None) -> dict:
    """Summary stats for a set of event returns (in %). When `dates` is
    provided, the t-stat and p-value use the Kish EFFECTIVE sample size so
    same-day correlated events don't fake significance."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    win = (r > 0).mean()
    n_eff = kish_effective_n(r, dates)
    # t-stat with the effective, not nominal, sample size
    sd = r.std()
    if sd > 0 and n_eff > 1:
        t = float(r.mean() / (sd / np.sqrt(n_eff)))
        p = float(2 * sps.t.sf(abs(t), df=max(int(n_eff) - 1, 1)))
    else:
        t, p = np.nan, np.nan
    return {
        "n": n,
        "n_eff": round(n_eff, 1),
        "mean_ret": r.mean(),
        "median_ret": r.median(),
        "win_rate": win,
        "t_stat": t,
        "p_value": p,
        "sharpe_like": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "profit_factor": r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf,
    }


def bucket_analysis(df: pd.DataFrame, factor: str, target: str,
                    n_buckets: int = 5, min_events: int = 100) -> pd.DataFrame | None:
    """Quantile-bucket a factor and measure the target return in each bucket.
    Monotonicity across buckets = real relationship, not noise.
    ALSO tests the extreme tails (top/bottom 5% and 1%) separately, because
    equal-population quintiles dilute cliff edges concentrated in extremes
    (e.g. |gap| > 4% behaves nothing like |gap| 1-2%)."""
    cols = [factor, target] + (["date"] if "date" in df.columns else [])
    sub = df[cols].dropna(subset=[factor, target])
    dts = sub["date"] if "date" in sub.columns else None
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
        m = edge_metrics(g[target], g["date"] if "date" in g.columns else None)
        m["bucket"] = str(name)
        rows.append(m)
    # tail-focused tests: extreme 5% and 1% of the factor, both sides
    if nunique > 20:
        for q, label in ((0.95, "tail_top5"), (0.99, "tail_top1"),
                         (0.05, "tail_bot5"), (0.01, "tail_bot1")):
            thr = sub[factor].quantile(q)
            tail = sub[sub[factor] >= thr] if q > 0.5 else sub[sub[factor] <= thr]
            if len(tail) >= 30:
                m = edge_metrics(tail[target],
                                 tail["date"] if "date" in tail.columns else None)
                m["bucket"] = label
                rows.append(m)
    res = pd.DataFrame(rows)
    core = res[~res["bucket"].str.startswith("tail_")]
    if core["n"].min() < min_events // n_buckets:
        return None
    # spearman rank correlation between bucket order and mean return
    # (computed on the core quantile buckets only, tails excluded)
    res["bucket_rank"] = range(len(res))
    if len(core) >= 3:
        rho, _ = sps.spearmanr(range(len(core)), core["mean_ret"])
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
    """An edge is CONFIRMED only if it holds out-of-sample with:
      1. the same sign as in-sample,
      2. at least half the in-sample magnitude,
      3. a minimum absolute OOS mean (0.03%) so trivial IS edges cannot
         pass with near-zero OOS values, and
      4. an OOS t-stat >= 1.5 (using the effective-N t-stat when present)."""
    if test_metrics.get("n", 0) < 30:
        return {"confirmed": False, "reason": "too few OOS events"}
    same_sign = np.sign(train_metrics["mean_ret"]) == np.sign(test_metrics["mean_ret"])
    holds = abs(test_metrics["mean_ret"]) >= 0.5 * abs(train_metrics["mean_ret"])
    big_enough = abs(test_metrics["mean_ret"]) >= 0.03
    t = test_metrics.get("t_stat", np.nan)
    significant = bool(not np.isnan(t) and abs(t) >= 1.5)
    return {
        "confirmed": bool(same_sign and holds and big_enough and significant),
        "is_mean": train_metrics["mean_ret"],
        "oos_mean": test_metrics["mean_ret"],
        "oos_t": round(float(t), 2) if not np.isnan(t) else np.nan,
        "oos_win_rate": test_metrics["win_rate"],
        "oos_n": test_metrics["n"],
        "oos_n_eff": test_metrics.get("n_eff", np.nan),
    }
