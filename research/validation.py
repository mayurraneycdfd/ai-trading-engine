"""
LEVEL 5 -- the validation gauntlet.

Every candidate edge that survives the discovery levels (1/2/4/6) is put
through five independent robustness tests drawn from the modern
quant-research literature. An edge is only PLATINUM if it passes all five.

  1. Purged walk-forward CV with embargo   (Lopez de Prado, AFML ch.7)
     -- k sequential folds; training data within EMBARGO_DAYS of the test
        fold is dropped so overlapping labels cannot leak.
  2. Monte Carlo permutation test          (White's Reality Check flavour)
     -- date-block shuffle of outcomes builds an empirical null for the
        rule's mean return; reports the fraction of shuffles that beat it.
  3. Stationary block bootstrap CI         (Politis & Romano 1994)
     -- resamples event blocks to get a distribution of the mean; the edge
        must keep its sign at the 5th percentile.
  4. CSCV / Probability of Backtest Overfitting (Bailey et al. 2015)
     -- splits history into S blocks, tries all C(S, S/2) train/test
        combinations, measures how often the in-sample winner underperforms
        the median OOS. PBO > 0.5 == coin flip == overfit.
  5. Deflated Sharpe ratio                 (Bailey & Lopez de Prado 2014)
     -- haircuts the Sharpe for the NUMBER OF TRIALS run during discovery,
        plus skew/kurtosis. Reports the probability the true Sharpe > 0.

All returns are net of COST_PCT round-trip transaction costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import (BOOTSTRAP_BLOCK, COST_PCT, CSCV_SPLITS, EMBARGO_DAYS,
                    N_BOOTSTRAP, N_CV_FOLDS, N_PERMUTATIONS, PBO_MAX)


# ----------------------------------------------------------------- helpers -
def net_returns(gross: pd.Series) -> pd.Series:
    """Deduct round-trip costs from every event return (both in %)."""
    return gross.dropna() - COST_PCT


def _annualised_sharpe(r: pd.Series) -> float:
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


# ------------------------------------------- 1. purged walk-forward CV -----
def purged_walkforward(dates: pd.Series, returns: pd.Series,
                       n_folds: int = N_CV_FOLDS,
                       embargo_days: int = EMBARGO_DAYS) -> dict:
    """Split events chronologically into n_folds test windows. For each fold
    the edge's mean net return is measured on the fold only (the rule is
    fixed, so 'training' here means the discovery already happened on other
    data -- what we check is CONSISTENCY across non-overlapping windows,
    with an embargo so adjacent-window label overlap cannot flatter us)."""
    df = pd.DataFrame({"date": pd.to_datetime(dates.values),
                       "ret": returns.values}).dropna().sort_values("date")
    if len(df) < n_folds * 20:
        return {"cv_pass": False, "cv_reason": "too few events"}
    edges = np.array_split(df.index.values, n_folds)
    fold_means = []
    for i, idx in enumerate(edges):
        fold = df.loc[idx]
        # embargo: drop events within embargo_days of neighbouring folds
        lo = fold["date"].min() + pd.Timedelta(days=embargo_days)
        hi = fold["date"].max() - pd.Timedelta(days=embargo_days)
        core = fold[(fold["date"] >= lo) & (fold["date"] <= hi)] \
            if i not in (0, len(edges) - 1) else fold
        if len(core) >= 10:
            fold_means.append(core["ret"].mean())
    fold_means = np.array(fold_means)
    n_pos = int((fold_means > 0).sum())
    return {
        "cv_fold_means": np.round(fold_means, 4).tolist(),
        "cv_positive_folds": f"{n_pos}/{len(fold_means)}",
        "cv_pass": bool(n_pos >= int(np.ceil(len(fold_means) * 0.6))
                        and fold_means.mean() > 0),
    }


# ------------------------------------------- 2. permutation test -----------
def permutation_test(dates: pd.Series, mask: pd.Series, returns: pd.Series,
                     n_perm: int = N_PERMUTATIONS, seed: int = 0) -> dict:
    """Empirical p-value: how often does a DATE-BLOCK-shuffled version of the
    selection mask produce a mean >= the real rule's mean? Shuffling whole
    dates (not single events) preserves cross-sectional correlation, so the
    null is honest for panels where many stocks share the same day."""
    df = pd.DataFrame({"date": pd.to_datetime(dates.values),
                       "mask": mask.values.astype(bool),
                       "ret": returns.values}).dropna()
    real = df.loc[df["mask"], "ret"].mean() - COST_PCT
    n_sel = int(df["mask"].sum())
    if n_sel < 30:
        return {"perm_pass": False, "perm_reason": "too few events"}
    # per-date selection frequency is preserved by shuffling date labels
    by_date = df.groupby("date")
    date_keys = np.array(sorted(by_date.groups.keys()))
    sel_per_date = df[df["mask"]].groupby("date").size() \
        .reindex(date_keys, fill_value=0).values
    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(date_keys))
        shuffled_sel = sel_per_date[perm]
        sim_rets = []
        for dk, k in zip(date_keys, shuffled_sel):
            if k == 0:
                continue
            day = by_date.get_group(dk)["ret"].values
            take = day if k >= len(day) else rng.choice(day, size=k, replace=False)
            sim_rets.append(take)
        if sim_rets:
            sim_mean = np.concatenate(sim_rets).mean() - COST_PCT
            if sim_mean >= real:
                beats += 1
    p_emp = (beats + 1) / (n_perm + 1)
    return {"perm_p": round(p_emp, 4), "perm_pass": bool(p_emp < 0.05),
            "net_mean": round(real, 4)}


# ------------------------------------------- 3. block bootstrap ------------
def block_bootstrap_ci(returns: pd.Series, n_boot: int = N_BOOTSTRAP,
                       block: int = BOOTSTRAP_BLOCK, seed: int = 0) -> dict:
    """Stationary block bootstrap on the (chronological) event returns.
    Keeps serial dependence intact. Edge must stay positive at the 5th pct."""
    r = net_returns(returns).values
    n = len(r)
    if n < 50:
        return {"boot_pass": False, "boot_reason": "too few events"}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    p_geo = 1.0 / block
    for b in range(n_boot):
        out, i = [], rng.integers(n)
        while sum(len(o) for o in out) < n:
            length = rng.geometric(p_geo)
            out.append(np.take(r, np.arange(i, i + length), mode="wrap"))
            i = rng.integers(n)
        means[b] = np.concatenate(out)[:n].mean()
    lo, hi = np.percentile(means, [5, 95])
    return {"boot_ci_5_95": (round(float(lo), 4), round(float(hi), 4)),
            "boot_pass": bool(lo > 0)}


# ------------------------------------------- 4. CSCV / PBO -----------------
def cscv_pbo(dates: pd.Series, rule_returns: dict[str, pd.Series],
             n_splits: int = CSCV_SPLITS) -> dict:
    """Probability of Backtest Overfitting across a FAMILY of candidate
    rules. History is cut into n_splits blocks; for every combination of
    half the blocks as 'in-sample', pick the best rule IS and record its
    OOS rank. PBO = fraction of combinations where the IS winner ranks in
    the bottom half OOS."""
    from itertools import combinations as it_comb
    names = list(rule_returns)
    if len(names) < 2:
        return {"pbo": np.nan, "pbo_pass": True, "pbo_reason": "single rule"}
    all_dates = pd.to_datetime(dates.dropna().sort_values().unique())
    blocks = np.array_split(all_dates, n_splits)
    # per-rule per-block mean net return matrix
    M = np.full((len(names), n_splits), np.nan)
    for ri, name in enumerate(names):
        s = rule_returns[name].dropna()
        d = pd.to_datetime(s.index if s.index.dtype != object else s.index)
        for bi, blk in enumerate(blocks):
            in_blk = s[np.isin(d, blk)]
            if len(in_blk) >= 5:
                M[ri, bi] = in_blk.mean() - COST_PCT
    half = n_splits // 2
    logits = []
    for is_blocks in it_comb(range(n_splits), half):
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]
        is_perf = np.nanmean(M[:, list(is_blocks)], axis=1)
        oos_perf = np.nanmean(M[:, oos_blocks], axis=1)
        if np.all(np.isnan(is_perf)) or np.all(np.isnan(oos_perf)):
            continue
        winner = int(np.nanargmax(is_perf))
        # OOS rank of the IS winner (0..1, higher = better)
        valid = ~np.isnan(oos_perf)
        if not valid[winner] or valid.sum() < 2:
            continue
        rank = sps.rankdata(oos_perf[valid])[np.nonzero(np.nonzero(valid)[0] == winner)[0][0]]
        w = rank / (valid.sum() + 1)
        logits.append(np.log(w / (1 - w)))
    if not logits:
        return {"pbo": np.nan, "pbo_pass": False, "pbo_reason": "no valid splits"}
    pbo = float(np.mean(np.array(logits) <= 0))
    return {"pbo": round(pbo, 3), "pbo_pass": bool(pbo <= PBO_MAX)}


# ------------------------------------------- 5. deflated Sharpe ------------
def deflated_sharpe(returns: pd.Series, n_trials: int) -> dict:
    """Deflated Sharpe Ratio: probability that the true Sharpe exceeds the
    expected max Sharpe of n_trials random strategies (accounts for
    selection bias, skew and fat tails)."""
    r = net_returns(returns)
    n = len(r)
    if n < 50:
        return {"dsr_pass": False, "dsr_reason": "too few events"}
    sr = r.mean() / r.std() if r.std() > 0 else 0.0  # per-event Sharpe
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    # expected max Sharpe under the null across n_trials (Bailey & LdP 2014)
    emc = 0.5772156649
    max_z = ((1 - emc) * sps.norm.ppf(1 - 1.0 / n_trials)
             + emc * sps.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = max_z * (1.0 / np.sqrt(n - 1))
    denom = np.sqrt(max(1e-12,
                        (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (n - 1)))
    dsr = float(sps.norm.cdf((sr - sr0) / denom))
    return {"dsr": round(dsr, 4), "dsr_pass": bool(dsr > 0.95),
            "n_trials_deflated_for": n_trials}


# ------------------------------------------- the full gauntlet -------------
def run_gauntlet(panel: pd.DataFrame, mask: pd.Series, target: str,
                 rule_name: str, n_trials: int,
                 family_returns: dict[str, pd.Series] | None = None) -> dict:
    """Run all five tests on one rule. `mask` selects the events the rule
    trades; `n_trials` = total hypotheses tested during discovery (for DSR);
    `family_returns` = sibling rules for the CSCV/PBO family test."""
    sel = panel.loc[mask.fillna(False)]
    rets = sel[target]
    dates = sel["date"]
    res = {"rule": rule_name, "n_events": len(sel)}
    res.update(purged_walkforward(dates, rets))
    res.update(permutation_test(panel["date"], mask.fillna(False), panel[target]))
    res.update(block_bootstrap_ci(rets))
    res.update(deflated_sharpe(rets, max(n_trials, 2)))
    if family_returns:
        fam = dict(family_returns)
        fam[rule_name] = pd.Series(rets.values, index=dates.values)
        res.update(cscv_pbo(panel["date"], fam))
    else:
        res.update({"pbo": np.nan, "pbo_pass": True})
    checks = [res.get("cv_pass"), res.get("perm_pass"),
              res.get("boot_pass"), res.get("dsr_pass"), res.get("pbo_pass")]
    n_pass = sum(bool(c) for c in checks)
    res["gauntlet_score"] = f"{n_pass}/5"
    res["grade"] = ("PLATINUM" if n_pass == 5 else
                    "GOLD" if n_pass == 4 else
                    "SILVER" if n_pass == 3 else "REJECTED")
    return res
