"""
Per-trade execution cost model. Replaces the flat COST_PCT assumption with a
cost that DEPENDS on the trade's own characteristics -- critical because
gap-and-go selects exactly the events (big gaps, opening auction) where
slippage is worst, so a flat cost systematically flatters the strategy.

  cost% = statutory (STT + stamp + brokerage)
        + spread (baseline, x OPEN_AUCTION_MULT at the open)
        + impact (IMPACT_COEF * sqrt(participation), x auction mult at open)
        + gap slippage (GAP_SLIPPAGE_COEF per 1% |gap| for gap entries)

Participation = TRADE_VALUE_INR / rupee value traded in the entry window.
All components are % of notional, round trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (GAP_SLIPPAGE_COEF, IMPACT_COEF, OPEN_AUCTION_MULT,
                    SPREAD_BASE_PCT, STAMP_BROKER_PCT, STT_PCT,
                    TRADE_VALUE_INR)

STATUTORY_PCT = STT_PCT + STAMP_BROKER_PCT


def trade_cost_pct(entry_value_traded: float | np.ndarray,
                   at_open: bool | np.ndarray = False,
                   gap_abs_pct: float | np.ndarray = 0.0) -> np.ndarray:
    """Round-trip cost in % of notional for one trade (vectorised).

    entry_value_traded : rupee value traded in the entry window (close*volume
                         summed over the first bars for open trades, or the
                         signal bar for intraday entries)
    at_open            : True for trades filled in/near the opening auction
    gap_abs_pct        : |gap| % for gap-entry trades (0 otherwise)
    """
    v = np.maximum(np.asarray(entry_value_traded, dtype=float), 1.0)
    participation = np.minimum(TRADE_VALUE_INR / v, 1.0)
    impact = IMPACT_COEF * np.sqrt(participation) * 100  # to % of notional
    spread = np.full_like(impact, SPREAD_BASE_PCT)
    mult = np.where(np.asarray(at_open, dtype=bool), OPEN_AUCTION_MULT, 1.0)
    gap_slip = GAP_SLIPPAGE_COEF * np.abs(np.asarray(gap_abs_pct, dtype=float))
    return STATUTORY_PCT + (spread + impact) * mult + gap_slip


def add_event_costs(panel: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Attach a per-event `cost_pct` column to the events panel.

    Uses columns already built by the feature layer:
      - open30_value  (rupee value traded in first 30 min) if present,
        else falls back to rvol proxy via `volume_open30` * price, else a
        conservative liquid-stock default.
      - gap (for gap strategy) drives auction slippage.
    """
    panel = panel.copy()
    # entry-window traded value, best available proxy
    if "open30_value" in panel.columns:
        val = panel["open30_value"].astype(float)
    elif {"volume_open30", "open_price"}.issubset(panel.columns):
        val = panel["volume_open30"].astype(float) * panel["open_price"].astype(float)
    else:
        val = pd.Series(5e8, index=panel.index)  # Rs 50 cr default (liquid F&O)
    val = val.fillna(val.median() if val.notna().any() else 5e8)

    if strategy == "gap":
        gap_abs = panel.get("gap", pd.Series(0.0, index=panel.index)).abs().fillna(0)
        panel["cost_pct"] = trade_cost_pct(val.values, at_open=True,
                                           gap_abs_pct=gap_abs.values)
    else:  # mean reversion: intraday entry, not at auction
        panel["cost_pct"] = trade_cost_pct(val.values, at_open=False)
    return panel


def summarize_costs(panel: pd.DataFrame) -> dict:
    c = panel["cost_pct"].dropna()
    return {"cost_mean": round(float(c.mean()), 4),
            "cost_p90": round(float(c.quantile(0.9)), 4),
            "cost_max": round(float(c.max()), 4)}
