"""
Main entry point.

    python run_discovery.py                 # all symbols, both strategies
    python run_discovery.py --symbols RELIANCE TCS
    python run_discovery.py --strategy gap  # or: mr

Pipeline per symbol:
  1-min bars -> features (all factor categories) + labels (gap&go / meanrev)
Then all symbols are stacked into one event panel and edge discovery runs
at three levels (single / combos / cumulative model).

Outputs (research/output/):
  gap_level1_single_factors.csv     mr_level1_single_factors.csv
  gap_level2_combinations.csv       mr_level2_combinations.csv
  gap_level3_model_report.txt       mr_level3_model_report.txt
  confirmed_edges.md                <- human-readable summary of survivors
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import alpha_search as als
import data_audit as aud
import data_loader as dl
import edge_discovery as ed
import execution as exe
import features as feat
import hidden_edges as he
import labels as lab
import portfolio as port
import robustness as rob
import validation as val
from config import GAP_GO_HORIZONS, MR_HORIZONS, OUT_DIR


def build_panel(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (gap_panel, mr_panel): one row per event, features + labels."""
    sector_map = dl.load_sector_map()

    # pass 1: gaps for every stock (needed for cross-sectional factors)
    print("pass 1/2: computing gap panel across all stocks...")
    gap_cols = {}
    minutes = {}
    for sym in symbols:
        m = dl.load_minute_bars(sym)
        if m is None or len(m) < 500:
            continue
        minutes[sym] = m
        d = dl.to_daily(m)
        gap_cols[sym] = (d["open"] - d["close"].shift(1)) / d["close"].shift(1) * 100
    panel_gaps = pd.DataFrame(gap_cols)
    print(f"  {len(minutes)} symbols loaded")

    # pass 2: features + labels per stock
    print("pass 2/2: features and labels per stock...")
    gap_rows, mr_rows = [], []
    for i, (sym, m) in enumerate(minutes.items(), 1):
        print(f"  [{i}/{len(minutes)}] {sym}")
        f = feat.build_features(sym, m, panel_gaps, sector_map)
        g_lab, m_lab = lab.build_labels(m, panel_gaps[sym])

        if not g_lab.empty:
            merged = f.join(g_lab.drop(columns=["gap_pct"], errors="ignore"), how="inner")
            merged["date"] = merged.index
            gap_rows.append(merged)
        if not m_lab.empty:
            merged = f.join(m_lab, how="inner")
            merged["date"] = merged.index
            mr_rows.append(merged)

    gap_panel = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame()
    mr_panel = pd.concat(mr_rows, ignore_index=True) if mr_rows else pd.DataFrame()
    return gap_panel, mr_panel


def run_strategy(panel: pd.DataFrame, name: str, target: str, all_targets: list[str]):
    if panel.empty:
        print(f"  [{name}] no events found -- check data paths in config.py")
        return None, None, None, None, None, None, None, None
    print(f"\n=== {name.upper()}  target={target}  events={len(panel)} ===")

    l1 = ed.level1_single_factors(panel, target, all_targets)
    l1.to_csv(OUT_DIR / f"{name}_level1_single_factors.csv")
    conf1 = l1[l1["fdr_pass"] & l1["confirmed"]] if not l1.empty else l1
    print(f"  level 1: {len(l1)} factors tested, {len(conf1)} confirmed OOS + FDR")

    l2 = ed.level2_combinations(panel, target, l1)
    if not l2.empty:
        l2.to_csv(OUT_DIR / f"{name}_level2_combinations.csv", index=False)
        conf2 = l2[l2["fdr_pass"] & l2.get("oos_confirmed", False)]
        print(f"  level 2: {len(l2)} combos tested, {len(conf2)} confirmed")

    l3 = ed.level3_cumulative(panel, target, all_targets)
    with open(OUT_DIR / f"{name}_level3_model_report.txt", "w") as fh:
        for k, v in l3.items():
            fh.write(f"\n--- {k} ---\n{v}\n")
    if "error" not in l3:
        print(f"  level 3: OOS rank IC = {l3['oos_rank_ic']:.3f}, "
              f"long-short spread = {l3['oos_long_short_spread']:.3f}%")

    # LEVEL 4: hidden-edge detectors (thresholds, regimes, decay, lead-lag,
    # sequences, anomaly precursors, clustering, calendar)
    l4 = he.run_all(panel, target, all_targets, level1=l1)
    for det_name, df in l4.items():
        if df is not None and not df.empty:
            df.to_csv(OUT_DIR / f"{name}_level4_{det_name}.csv")
            n_conf = int(df.get("confirmed", pd.Series(dtype=bool)).sum()) \
                if "confirmed" in df.columns else 0
            print(f"  level 4 [{det_name}]: {len(df)} candidates, "
                  f"{n_conf} confirmed OOS")

    # LEVEL 6: genetic-programming alpha miner (evolves NEW formulas)
    factor_cols = ed._factor_cols(panel, target, all_targets)
    l6 = als.evolve_alphas(panel.dropna(subset=[target]), target, factor_cols)
    if not l6.empty:
        l6.drop(columns=["tree"]).to_csv(
            OUT_DIR / f"{name}_level6_evolved_alphas.csv", index=False)
        print(f"  level 6: {len(l6)} evolved alphas, "
              f"{int(l6['confirmed'].sum())} confirmed OOS")

    # LEVEL 5: validation gauntlet on every surviving candidate.
    # n_trials = total hypotheses tested across all levels (for DSR haircut)
    n_trials = len(l1) + (len(l2) if l2 is not None else 0) + \
        sum(len(df) for df in l4.values() if df is not None) + \
        (len(l6) * 50 if not l6.empty else 0)  # GP searched a big space
    gauntlet_rows = []
    family: dict[str, pd.Series] = {}

    # candidates from level 1
    conf1_idx = l1[l1["fdr_pass"] & l1["confirmed"]].head(10).index \
        if not l1.empty else []
    for f in conf1_idx:
        mask = ed._bucket_mask(panel, f, str(l1.loc[f, "best_bucket"]))
        sel = panel.loc[mask.fillna(False)]
        family[f"L1:{f}"] = pd.Series(sel[target].values, index=sel["date"].values)
    # candidates from level 6
    l6_masks = {}
    if not l6.empty:
        for _, r in l6[l6["confirmed"]].iterrows():
            m = als.alpha_top_decile_mask(r["tree"], panel, np.sign(r["oos_ic"]))
            l6_masks[f"L6:{r['formula'][:60]}"] = m
            sel = panel.loc[m.fillna(False)]
            family[f"L6:{r['formula'][:60]}"] = \
                pd.Series(sel[target].values, index=sel["date"].values)

    for f in conf1_idx:
        mask = ed._bucket_mask(panel, f, str(l1.loc[f, "best_bucket"]))
        fam = {k: v for k, v in family.items() if k != f"L1:{f}"}
        gauntlet_rows.append(val.run_gauntlet(
            panel, mask, target, f"L1:{f}", n_trials, fam))
    for rname, mask in l6_masks.items():
        fam = {k: v for k, v in family.items() if k != rname}
        gauntlet_rows.append(val.run_gauntlet(
            panel, mask, target, rname, n_trials, fam))

    l5 = pd.DataFrame(gauntlet_rows)
    if not l5.empty:
        l5.to_csv(OUT_DIR / f"{name}_level5_gauntlet.csv", index=False)
        n_plat = int((l5["grade"] == "PLATINUM").sum())
        n_gold = int((l5["grade"] == "GOLD").sum())
        print(f"  level 5 gauntlet: {len(l5)} rules tested -> "
              f"{n_plat} PLATINUM, {n_gold} GOLD "
              f"(net of costs, deflated for {n_trials} trials)")

    # collect the trade masks of every gauntlet-surviving rule (>= GOLD)
    all_masks: dict[str, pd.Series] = {}
    if not l5.empty:
        keep = set(l5.loc[l5["grade"].isin(["PLATINUM", "GOLD"]), "rule"])
        for f in conf1_idx:
            if f"L1:{f}" in keep:
                all_masks[f"L1:{f}"] = ed._bucket_mask(
                    panel, f, str(l1.loc[f, "best_bucket"]))
        for rname, m in l6_masks.items():
            if rname in keep:
                all_masks[rname] = m

    # STAGE 7: multi-boundary robustness -- edges must survive alternative
    # train/test cuts, not just the single TRAIN_END choice
    if all_masks:
        l7 = rob.multi_boundary(panel, all_masks, target)
        l7.to_csv(OUT_DIR / f"{name}_level7_boundary_robustness.csv", index=False)
        n_rob = int(l7["boundary_robust"].sum())
        print(f"  level 7 boundaries: {len(l7)} rules -> {n_rob} robust "
              f"across cuts")
        # drop non-robust edges from the tradeable set
        robust = set(l7.loc[l7["boundary_robust"], "rule"])
        all_masks = {k: v for k, v in all_masks.items() if k in robust}
    else:
        l7 = pd.DataFrame()

    # GP interpretability review pack (evolved alphas need human approval)
    if not l6.empty:
        factor_cols2 = ed._factor_cols(panel, target, all_targets)
        gp_review = rob.gp_interpretability(l6, panel, factor_cols2, name)
        if not gp_review.empty:
            gp_review.to_csv(OUT_DIR / f"{name}_gp_review_pack.csv", index=False)
            print(f"  GP review pack: {len(gp_review)} evolved alphas await "
                  "human approval (approved=False by default)")

    # STAGE 8: portfolio-level analysis of the final tradeable edge set
    if all_masks:
        l8 = port.run_portfolio(panel, all_masks, target, name)
        if "error" not in l8:
            l8["overlap"].to_csv(OUT_DIR / f"{name}_level8_signal_overlap.csv")
            if l8["pnl_corr"] is not None:
                l8["pnl_corr"].to_csv(OUT_DIR / f"{name}_level8_pnl_correlation.csv")
            l8["capacity"].to_csv(OUT_DIR / f"{name}_level8_capacity.csv", index=False)
            bk = l8["book"]
            print(f"  level 8 portfolio: combined book Sharpe {bk['sharpe']}, "
                  f"maxDD {bk['max_drawdown_pct']}%, "
                  f"{len(l8['corr_flags'])} correlated edge pairs flagged")
    else:
        l8 = {"error": "no edges survived to portfolio stage"}
    return l1, l2, l3, l4, l5, l6, l7, l8


def write_summary(results: dict):
    lines = ["# Confirmed Edges (out-of-sample + FDR survivors)\n",
             "\nGrades: PLATINUM = passed all 5 gauntlet tests "
             "(purged CV, permutation, bootstrap, deflated Sharpe, PBO), "
             "net of transaction costs.\n"]
    for name, (l1, l2, l3, l4, l5, l6, l7, l8) in results.items():
        lines.append(f"\n## {name}\n")
        if l1 is not None and not l1.empty:
            conf = l1[l1["fdr_pass"] & l1["confirmed"]]
            lines.append("### Single factors\n")
            for f, r in conf.head(20).iterrows():
                lines.append(f"- **{f}** in {r['best_bucket']}: "
                             f"OOS mean {r['oos_mean']:.3f}%, "
                             f"win {r['oos_win_rate']:.1%}, n={int(r['oos_n'])}")
        if l2 is not None and not l2.empty:
            conf = l2[l2["fdr_pass"] & l2.get("oos_confirmed", False)]
            lines.append("\n### Factor combinations\n")
            for _, r in conf.head(20).iterrows():
                lines.append(f"- {r['combo']}: OOS mean {r.get('oos_oos_mean', float('nan')):.3f}%")
        if l3 and "error" not in l3:
            lines.append("\n### Cumulative model (all factors)\n")
            lines.append(f"- OOS rank IC: {l3['oos_rank_ic']:.3f}")
            lines.append(f"- Top-decile OOS mean: {l3['oos_top_decile'].get('mean_ret', float('nan')):.3f}%")
            lines.append("- Top drivers: " + ", ".join(l3["feature_importance"].head(8).index))
        if l4:
            lines.append("\n### Hidden-edge detectors (level 4)\n")
            for det, df in l4.items():
                if df is None or df.empty:
                    continue
                if det == "C_edge_decay":
                    for _, r in df.iterrows():
                        lines.append(f"- [decay] {r['rule']}: {r['status']} "
                                     f"(early {r['early_mean']:.3f}% -> "
                                     f"recent {r['recent_mean']:.3f}%)")
                    continue
                if "confirmed" not in df.columns:
                    continue
                conf = df[df["confirmed"] == True]  # noqa: E712
                if "fdr_pass" in df.columns:
                    conf = conf[conf["fdr_pass"]]
                for idx, r in conf.head(10).iterrows():
                    label = (r.get("rule") or r.get("pattern") or r.get("condition")
                             or r.get("anomaly")
                             or (f"{r.get('regime', '')} {r.get('factor', '')}".strip())
                             or str(idx))
                    oos = r.get("oos_mean", r.get("oos_rank_ic", float("nan")))
                    lines.append(f"- [{det}] {label}: OOS {oos:.3f}, "
                                 f"n={int(r.get('oos_n', r.get('n', 0)))}")
        if l6 is not None and not l6.empty:
            conf = l6[l6["confirmed"]]
            if not conf.empty:
                lines.append("\n### Evolved alphas (level 6, GP-mined)\n")
                for _, r in conf.iterrows():
                    lines.append(f"- `{r['formula']}`: train IC {r['train_ic']:.3f}, "
                                 f"OOS IC {r['oos_ic']:.3f} (n={int(r['oos_n'])})")
        if l5 is not None and not l5.empty:
            lines.append("\n### Validation gauntlet (level 5) -- FINAL GRADES\n")
            order = {"PLATINUM": 0, "GOLD": 1, "SILVER": 2, "REJECTED": 3}
            graded = l5.iloc[l5["grade"].map(order).argsort()]
            for _, r in graded.iterrows():
                lines.append(
                    f"- **{r['grade']}** {r['rule']}: net mean "
                    f"{r.get('net_mean', float('nan')):.3f}%/trade, "
                    f"folds {r.get('cv_positive_folds', '?')}, "
                    f"perm p={r.get('perm_p', float('nan'))}, "
                    f"PBO={r.get('pbo', float('nan'))}, "
                    f"DSR={r.get('dsr', float('nan'))} "
                    f"[{r['gauntlet_score']}]")
        if l7 is not None and isinstance(l7, pd.DataFrame) and not l7.empty:
            lines.append("\n### Boundary robustness (level 7)\n")
            for _, r in l7.iterrows():
                status = "ROBUST" if r["boundary_robust"] else "FRAGILE"
                lines.append(f"- **{status}** {r['rule']}: positive after "
                             f"{int(r['n_positive'])}/{int(r['n_boundaries'])} cuts")
        if isinstance(l8, dict) and "error" not in l8:
            bk = l8["book"]
            lines.append("\n### Combined portfolio book (level 8)\n")
            lines.append(f"- Edges in book: {len(l8['capacity'])}")
            lines.append(f"- Annualised Sharpe (net): {bk['sharpe']}")
            lines.append(f"- Max drawdown: {bk['max_drawdown_pct']}%")
            lines.append(f"- Mean daily net PnL: {bk['daily_mean_pct']}%")
            lines.append(f"- Trades skipped by concurrency cap: {bk['n_skipped_cap']}")
            if l8["corr_flags"]:
                lines.append("- WARNING correlated edge pairs: " +
                             "; ".join(f"{a} ~ {b} (r={c})"
                                       for a, b, c in l8["corr_flags"]))
            for _, r in l8["capacity"].iterrows():
                lines.append(f"- capacity {r['rule']}: "
                             f"Rs {r['max_notional_inr']:,.0f}/trade")
    (OUT_DIR / "confirmed_edges.md").write_text("\n".join(lines))
    print(f"\nSummary written to {OUT_DIR / 'confirmed_edges.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--strategy", choices=["gap", "mr", "both"], default="both")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or dl.discover_symbols()
    if not symbols:
        print("No symbols found. Point config.PATHS['minute_bars'] at your parquet folder.")
        return

    # ---- STAGE 0: data integrity audit (corporate actions, survivorship,
    # suspect jumps, stale prices). Quarantined events never enter discovery.
    print("stage 0: data integrity audit...")
    audit = aud.run_audit(symbols)
    if not audit["report"].empty:
        audit["report"].to_csv(OUT_DIR / "data_audit_report.csv", index=False)
    print(f"  {len(audit['report'])} issues found, "
          f"{len(audit['quarantine'])} (symbol, date) events quarantined")
    if audit["survivorship_warning"]:
        print("  WARNING: no point-in-time F&O universe file found "
              f"-- backtest results may carry SURVIVORSHIP BIAS. "
              "Provide fno_universe.parquet [symbol, from_date, to_date] to fix.")

    gap_panel, mr_panel = build_panel(symbols)
    gap_panel = aud.apply_quarantine(gap_panel, audit)
    mr_panel = aud.apply_quarantine(mr_panel, audit)

    # ---- per-event execution costs (spread + impact + STT + auction slippage)
    if not gap_panel.empty:
        gap_panel = exe.add_event_costs(gap_panel, "gap")
        print(f"  gap cost model: {exe.summarize_costs(gap_panel)}")
    if not mr_panel.empty:
        mr_panel = exe.add_event_costs(mr_panel, "mr")
        print(f"  mr  cost model: {exe.summarize_costs(mr_panel)}")
    gap_targets = [f"cont_{h}m" for h in GAP_GO_HORIZONS] + \
                  [f"go_{h}m" for h in GAP_GO_HORIZONS] + \
                  ["cont_close", "filled_gap", "tb_ret", "tb_hit"]
    mr_targets = [f"revert_{h}m" for h in MR_HORIZONS] + \
                 [f"rev_{h}m" for h in MR_HORIZONS] + ["mr_z", "tb_ret", "tb_hit"]

    results = {}
    if args.strategy in ("gap", "both"):
        results["gap_and_go"] = run_strategy(gap_panel, "gap", "cont_30m", gap_targets)
    if args.strategy in ("mr", "both"):
        results["mean_reversion"] = run_strategy(mr_panel, "mr", "revert_30m", mr_targets)

    write_summary(results)


if __name__ == "__main__":
    main()
