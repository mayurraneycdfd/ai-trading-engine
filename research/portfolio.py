"""
LEVEL 7 -- portfolio-level validation of the CONFIRMED edge set.

Individual edges can each be real yet still add up to one trade: ten
"independent" rules that all fire on the same high-VIX gap days are a single
edge with hidden leverage. This layer measures the combined book:

  1. Signal-overlap matrix   -- Jaccard overlap of trade days between edges
  2. Daily-PnL correlation   -- flag pairs above PORT_CORR_MAX
  3. Combined-book equity    -- daily PnL of all edges with position caps
                                (max PORT_MAX_CONCURRENT simultaneous trades,
                                equal notional), max drawdown, Calmar
  4. Concurrency profile     -- distribution of simultaneous-signal counts
  5. Capacity per edge       -- max deployable notional so the trade stays
                                under PORT_CAPACITY_PCT of entry-window value

Output: portfolio_report.txt + portfolio_daily_pnl.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (COST_PCT, OUT_DIR, PORT_CAPACITY_PCT, PORT_CORR_MAX,
                    PORT_MAX_CONCURRENT, TRADE_VALUE_INR)


def _edge_daily_pnl(panel: pd.DataFrame, mask: pd.Series, target: str) -> pd.Series:
    sel = panel.loc[mask.fillna(False)]
    costs = sel["cost_pct"] if "cost_pct" in sel.columns else COST_PCT
    net = sel[target] - costs
    return net.groupby(pd.to_datetime(sel["date"]).dt.normalize()).mean()


def signal_overlap(masks: dict[str, pd.Series], panel: pd.DataFrame) -> pd.DataFrame:
    """Jaccard overlap of TRADE DATES between every pair of edges."""
    dates = {n: set(pd.to_datetime(panel.loc[m.fillna(False), "date"]).dt.normalize())
             for n, m in masks.items()}
    names = list(dates)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j and (dates[a] | dates[b]):
                jac = len(dates[a] & dates[b]) / len(dates[a] | dates[b])
                M.iloc[i, j] = M.iloc[j, i] = round(jac, 3)
    return M


def pnl_correlation(daily_pnls: dict[str, pd.Series]) -> tuple[pd.DataFrame, list]:
    df = pd.DataFrame(daily_pnls)
    corr = df.corr()
    flagged = [(a, b, round(float(corr.loc[a, b]), 3))
               for i, a in enumerate(corr.index)
               for b in corr.columns[i + 1:]
               if abs(corr.loc[a, b]) > PORT_CORR_MAX and df[[a, b]].dropna().shape[0] >= 20]
    return corr, flagged


def combined_book(panel: pd.DataFrame, masks: dict[str, pd.Series],
                  target: str) -> dict:
    """Equal-notional combined book with a concurrency cap. Returns equity
    stats + the daily PnL series."""
    trades = []
    for name, m in masks.items():
        sel = panel.loc[m.fillna(False)]
        costs = sel["cost_pct"] if "cost_pct" in sel.columns \
            else pd.Series(COST_PCT, index=sel.index)
        for idx, row in sel.iterrows():
            trades.append({"date": pd.Timestamp(row["date"]).normalize(),
                           "edge": name,
                           "net_ret": row[target] - costs.loc[idx]})
    if not trades:
        return {"error": "no trades"}
    tdf = pd.DataFrame(trades).dropna().sort_values("date")
    # concurrency cap: keep at most PORT_MAX_CONCURRENT trades per day
    # (first-come basis; real system would rank by signal strength)
    tdf["rank_in_day"] = tdf.groupby("date").cumcount()
    kept = tdf[tdf["rank_in_day"] < PORT_MAX_CONCURRENT]
    daily = kept.groupby("date")["net_ret"].mean()  # equal-weight book % PnL
    conc = tdf.groupby("date").size()

    equity = (1 + daily / 100).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    ann_ret = float(daily.mean() * 252)
    ann_vol = float(daily.std() * np.sqrt(252))
    max_dd = float(dd.min())
    return {
        "daily_pnl": daily,
        "n_trade_days": int(len(daily)),
        "n_trades_total": int(len(tdf)),
        "n_trades_capped_out": int(len(tdf) - len(kept)),
        "ann_return_pct": round(ann_ret, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(ann_ret / ann_vol, 2) if ann_vol > 0 else np.nan,
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(ann_ret / abs(max_dd), 2) if max_dd < 0 else np.nan,
        "concurrency_p50": int(conc.median()),
        "concurrency_p95": int(conc.quantile(0.95)),
        "concurrency_max": int(conc.max()),
    }


def capacity(panel: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Max deployable notional per edge: PORT_CAPACITY_PCT of the median
    entry-window traded value across the edge's events."""
    rows = []
    val = panel.get("open30_value")
    for name, m in masks.items():
        sel = panel.loc[m.fillna(False)]
        if val is not None and val.reindex(sel.index).notna().any():
            med_val = float(val.reindex(sel.index).median())
        else:
            med_val = 5e8  # liquid F&O default
        cap = med_val * PORT_CAPACITY_PCT / 100
        rows.append({"edge": name, "n_events": len(sel),
                     "median_entry_value_inr": round(med_val),
                     "max_notional_inr": round(cap),
                     "vs_assumed_trade_size": round(cap / TRADE_VALUE_INR, 1)})
    return pd.DataFrame(rows)


def run_portfolio(panel: pd.DataFrame, masks: dict[str, pd.Series],
                  target: str, name: str) -> dict:
    """Full portfolio analysis over the confirmed edge set."""
    if len(masks) == 0:
        return {"error": "no confirmed edges to combine"}
    daily_pnls = {n: _edge_daily_pnl(panel, m, target) for n, m in masks.items()}
    overlap = signal_overlap(masks, panel)
    corr, flagged = pnl_correlation(daily_pnls)
    book = combined_book(panel, masks, target)
    cap = capacity(panel, masks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Portfolio report -- {name}", "",
             f"Edges combined: {len(masks)}", ""]
    if "daily_pnl" in book:
        book["daily_pnl"].to_csv(OUT_DIR / f"{name}_portfolio_daily_pnl.csv")
        lines += ["## Combined book (equal weight, "
                  f"max {PORT_MAX_CONCURRENT} concurrent)", ""]
        for k in ("n_trade_days", "n_trades_total", "n_trades_capped_out",
                  "ann_return_pct", "ann_vol_pct", "sharpe",
                  "max_drawdown_pct", "calmar",
                  "concurrency_p50", "concurrency_p95", "concurrency_max"):
            lines.append(f"- {k}: {book[k]}")
    lines += ["", "## Signal-date overlap (Jaccard)", "",
              overlap.to_string(), "",
              "## Daily-PnL correlation flags "
              f"(|corr| > {PORT_CORR_MAX})", ""]
    if flagged:
        for a, b, c in flagged:
            lines.append(f"- REDUNDANT PAIR: {a} vs {b}: corr={c} "
                         "-- treat as ONE edge when sizing")
    else:
        lines.append("- none: edges are reasonably independent")
    lines += ["", "## Capacity per edge", "", cap.to_string(index=False)]
    (OUT_DIR / f"{name}_portfolio_report.txt").write_text("\n".join(lines))
    print(f"  level 7 portfolio: {len(masks)} edges, "
          f"sharpe={book.get('sharpe', 'n/a')}, "
          f"maxDD={book.get('max_drawdown_pct', 'n/a')}%, "
          f"{len(flagged)} redundant pairs")
    return {"overlap": overlap, "corr": corr, "flagged": flagged,
            "book": book, "capacity": cap}
