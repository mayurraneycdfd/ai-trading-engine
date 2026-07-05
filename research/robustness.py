"""
Boundary robustness + GP interpretability.

1. multi_boundary(): re-measures every confirmed rule's OOS mean at SEVERAL
   train/test boundary dates (BOUNDARY_DATES). A real edge should be
   OOS-positive after most cuts; an edge whose sign depends on where you
   cut the sample is noise dressed up as signal.

2. gp_interpretability(): an evolved formula does not trade until a human
   can articulate WHY it should work. This produces the review pack per
   alpha: which raw factors it uses, its correlation to each (is it just a
   re-discovery of a known factor?), turnover proxy, and a plain-English
   template the researcher must complete and sign off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BOUNDARY_DATES, BOUNDARY_MIN_PASS, COST_PCT, OUT_DIR


# ------------------------------------------- 1. multi-boundary check -------
def multi_boundary(panel: pd.DataFrame, masks: dict[str, pd.Series],
                   target: str) -> pd.DataFrame:
    """OOS net mean per rule for each alternative boundary. A rule passes if
    it is OOS-positive after >= BOUNDARY_MIN_PASS of the cuts."""
    rows = []
    dates = pd.to_datetime(panel["date"])
    costs = panel["cost_pct"] if "cost_pct" in panel.columns else COST_PCT
    net = panel[target] - costs
    for name, m in masks.items():
        row = {"rule": name}
        n_pos = 0
        for b in BOUNDARY_DATES:
            oos = m.fillna(False) & (dates > pd.Timestamp(b))
            r = net[oos].dropna()
            mu = float(r.mean()) if len(r) >= 20 else np.nan
            row[f"oos_after_{b}"] = round(mu, 4) if not np.isnan(mu) else np.nan
            row[f"n_after_{b}"] = int(len(r))
            if not np.isnan(mu) and mu > 0:
                n_pos += 1
        row["boundaries_positive"] = f"{n_pos}/{len(BOUNDARY_DATES)}"
        row["boundary_pass"] = bool(n_pos >= BOUNDARY_MIN_PASS)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------- 2. GP interpretability --------
def _leaf_factors(formula: str, factor_cols: list[str]) -> list[str]:
    return sorted({c for c in factor_cols if c in formula})


def gp_interpretability(l6: pd.DataFrame, panel: pd.DataFrame,
                        factor_cols: list[str], name: str) -> pd.DataFrame:
    """Review pack for every confirmed evolved alpha. The `approved` column
    ships as False -- a human must flip it after writing the rationale.
    Alphas with approved=False are EXCLUDED from the tradeable edge list."""
    rows = []
    for _, r in l6[l6.get("confirmed", False) == True].iterrows():  # noqa: E712
        leaves = _leaf_factors(r["formula"], factor_cols)
        # correlation of the alpha signal to each raw leaf factor:
        # near +/-1 means the GP just re-discovered a known factor
        sig = None
        if "tree" in r and r["tree"] is not None:
            try:
                sig = r["tree"].evaluate(panel)
            except Exception:
                sig = None
        redund = {}
        if sig is not None:
            for leaf in leaves[:6]:
                c = pd.Series(sig).corr(panel[leaf])
                if pd.notna(c):
                    redund[leaf] = round(float(c), 3)
        max_redund = max((abs(v) for v in redund.values()), default=np.nan)
        # turnover proxy: how often does the top-decile membership change?
        churn = np.nan
        if sig is not None:
            s = pd.Series(sig, index=panel.index)
            top = s > s.quantile(0.9)
            by_date = pd.DataFrame({"d": pd.to_datetime(panel["date"]).dt.normalize(),
                                    "t": top}).groupby("d")["t"].mean()
            churn = round(float(by_date.diff().abs().mean()), 4)
        rows.append({
            "formula": r["formula"],
            "oos_ic": r.get("oos_ic", np.nan),
            "raw_factors_used": ", ".join(leaves),
            "factor_correlations": str(redund),
            "max_factor_redundancy": max_redund,
            "likely_rediscovery": bool(max_redund > 0.9) if pd.notna(max_redund) else False,
            "top_decile_daily_churn": churn,
            "economic_rationale": "REQUIRED -- explain WHY this should predict returns",
            "approved": False,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / f"{name}_gp_review_pack.csv", index=False)
        n_re = int(df["likely_rediscovery"].sum())
        print(f"  gp review: {len(df)} evolved alphas need human sign-off "
              f"({n_re} flagged as likely factor re-discoveries)")
    return df
