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

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import data_loader as dl
import edge_discovery as ed
import features as feat
import labels as lab
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
        return None, None, None
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
    return l1, l2, l3


def write_summary(results: dict):
    lines = ["# Confirmed Edges (out-of-sample + FDR survivors)\n"]
    for name, (l1, l2, l3) in results.items():
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

    gap_panel, mr_panel = build_panel(symbols)
    gap_targets = [f"cont_{h}m" for h in GAP_GO_HORIZONS] + \
                  [f"go_{h}m" for h in GAP_GO_HORIZONS] + ["cont_close", "filled_gap"]
    mr_targets = [f"revert_{h}m" for h in MR_HORIZONS] + \
                 [f"rev_{h}m" for h in MR_HORIZONS] + ["mr_z"]

    results = {}
    if args.strategy in ("gap", "both"):
        results["gap_and_go"] = run_strategy(gap_panel, "gap", "cont_30m", gap_targets)
    if args.strategy in ("mr", "both"):
        results["mean_reversion"] = run_strategy(mr_panel, "mr", "revert_30m", mr_targets)

    write_summary(results)


if __name__ == "__main__":
    main()
