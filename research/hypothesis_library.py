"""
Pre-registered hypothesis library: the 15 best institutional edge hypotheses
for Indian intraday/F&O trading, encoded as EXPLICIT, testable rules.

Why this exists: blind factor search pays a heavy multiple-testing penalty
(thousands of implicit tests). A PRE-REGISTERED hypothesis -- stated before
looking at results, with an economic WHY -- needs far less statistical
correction and is far more likely to survive live trading. These 15 are the
best-documented effects from institutional practice and academic literature,
adapted to the Indian market's specific structure (delivery data, F&O bans,
participant OI, GIFT Nifty, expiry mechanics).

Each hypothesis defines: a mask (which events), an expected direction
(+1 = expect positive target, -1 = expect negative), and the rationale.
They are evaluated with the SAME walk-forward + Kish-t machinery as
discovered edges, then reported side by side.

All referenced columns degrade gracefully -- a hypothesis whose data is
missing reports "no data" instead of crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRAIN_END, MIN_EVENTS
from stats import edge_metrics
from edge_discovery import train_test_split_by_date


def _col(panel: pd.DataFrame, name: str) -> pd.Series | None:
    return panel[name] if name in panel.columns else None


# Each entry: (name, rationale, mask_fn, direction)
# mask_fn(panel) -> boolean Series | None (None = required data missing)

def _h01_delivery_gap_continuation(p):
    """Gap up + yesterday's delivery z-score high + sector momentum positive."""
    g, dz, srm = _col(p, "gap_pct"), _col(p, "delivery_z20"), _col(p, "sector_rs_momentum")
    if g is None or dz is None:
        return None
    m = (g > 0.5) & (dz > 1.0)
    if srm is not None:
        m &= srm.fillna(0) > 0
    return m


def _h02_fii_positioning_fade(p):
    """Gap AGAINST heavy FII futures positioning fades (smart money wins)."""
    g, ls = _col(p, "gap_pct"), _col(p, "fii_fut_long_short")
    if g is None or ls is None:
        return None
    # fixed structural threshold (FII longs >= 1.5x shorts) -- avoids any
    # rolling-quantile lookahead across the mixed-symbol panel
    return (g < -0.5) & (ls > 1.5)


def _h03_earnings_gap_drift(p):
    """Post-earnings announcement drift: earnings-day gaps continue intraday."""
    g, e = _col(p, "gap_pct"), _col(p, "earnings_today")
    if g is None or e is None:
        return None
    return (e == 1) & (g.abs() > 1.0)


def _h04_ban_exit_squeeze(p):
    """F&O ban exit: pent-up positioning demand returns -> upward pressure."""
    ex = _col(p, "fno_ban_exit")
    if ex is None:
        return None
    return ex == 1


def _h05_short_squeeze(p):
    """High short interest percentile + gap up = squeeze fuel -> continuation."""
    g, si = _col(p, "gap_pct"), _col(p, "si_pctile_90d")
    if g is None or si is None:
        return None
    return (g > 0.5) & (si > 0.8)


def _h06_bulk_deal_breakout(p):
    """Recent bulk-deal accumulation + open above yesterday's high."""
    bd, dh = _col(p, "bulk_deal_net5d"), _col(p, "dist_prev_high")
    if bd is None or dh is None:
        return None
    return (bd > 0) & (dh > 0)


def _h07_gap_overshoot_fade(p):
    """Stock gap far beyond what GIFT Nifty implies -> overshoot fades."""
    g, gp = _col(p, "gap_pct"), _col(p, "gift_premium_pct")
    if g is None or gp is None:
        return None
    excess = g - gp.fillna(0)
    return excess.abs() > 1.5


def _h08_high_vix_reversion(p):
    """Mean-reversion signals work HARDER in high-VIX regimes."""
    z, vp = _col(p, "zscore_20d"), _col(p, "vix_pctile_1y")
    if z is None or vp is None:
        return None
    return (z < -2.0) & (vp > 0.7)


def _h09_index_inclusion_drift(p):
    """Index inclusion window: forced passive buying -> upward drift."""
    w = _col(p, "index_inclusion_window")
    if w is None:
        return None
    return w == 1


def _h10_post_holiday_gap_fade(p):
    """Gaps after holidays overreact to accumulated news -> fade."""
    g, ph = _col(p, "gap_pct"), _col(p, "post_holiday")
    if g is None or ph is None:
        return None
    return (ph == 1) & (g.abs() > 0.75)


def _h11_long_buildup_continuation(p):
    """Futures long buildup (OI up + basis widening) + gap up -> continue."""
    g, lb = _col(p, "gap_pct"), _col(p, "fut_long_buildup")
    if g is None or lb is None:
        return None
    return (g > 0.5) & (lb == 1)


def _h12_idiosyncratic_gap_fade(p):
    """Idiosyncratic gaps (stock >> sector) fade; common gaps continue."""
    gi = _col(p, "gap_idiosyncratic")
    if gi is None:
        return None
    return gi.abs() > 1.0


def _h13_delivery_spike_reversal(p):
    """Delivery z-spike after a down day = institutional accumulation -> up."""
    dz, r1 = _col(p, "delivery_z20"), _col(p, "ret_1d")
    if dz is None or r1 is None:
        return None
    return (dz > 2.0) & (r1 < -1.0)


def _h14_pledge_distress_momentum(p):
    """Rising promoter pledge + gap down = forced-selling spiral -> continue."""
    g, pc = _col(p, "gap_pct"), _col(p, "pledge_chg_90d")
    if g is None or pc is None:
        return None
    return (g < -0.5) & (pc > 5.0)


def _h15_global_riskoff_fade(p):
    """US VIX spike overnight + individual stock gap down: Indian market
    overreacts to US fear at the open, then recovers intraday."""
    g, uv = _col(p, "gap_pct"), _col(p, "usvix_ret_1d")
    if g is None or uv is None:
        return None
    return (g < -0.75) & (uv > 10.0)


HYPOTHESES = [
    ("H01_delivery_gap_continuation", +1, _h01_delivery_gap_continuation,
     "High-delivery gap-ups with sector tailwind continue (institutional conviction)"),
    ("H02_fii_positioning_fade", +1, _h02_fii_positioning_fade,
     "Gap-downs against heavy FII long positioning recover (smart money wins)"),
    ("H03_earnings_gap_drift", 0, _h03_earnings_gap_drift,
     "Earnings gaps drift in the gap direction (intraday PEAD)"),
    ("H04_ban_exit_squeeze", +1, _h04_ban_exit_squeeze,
     "F&O ban exits release pent-up positioning demand"),
    ("H05_short_squeeze", +1, _h05_short_squeeze,
     "High short interest + gap up = squeeze continuation"),
    ("H06_bulk_deal_breakout", +1, _h06_bulk_deal_breakout,
     "Bulk-deal accumulation + breakout open continues"),
    ("H07_gap_overshoot_fade", -1, _h07_gap_overshoot_fade,
     "Gaps beyond GIFT-implied fair value revert"),
    ("H08_high_vix_reversion", +1, _h08_high_vix_reversion,
     "Oversold z-scores snap back harder in high-VIX regimes"),
    ("H09_index_inclusion_drift", +1, _h09_index_inclusion_drift,
     "Index inclusion windows drift up on passive flows"),
    ("H10_post_holiday_gap_fade", -1, _h10_post_holiday_gap_fade,
     "Post-holiday gaps overreact and fade"),
    ("H11_long_buildup_continuation", +1, _h11_long_buildup_continuation,
     "Futures long buildup + gap up continues"),
    ("H12_idiosyncratic_gap_fade", -1, _h12_idiosyncratic_gap_fade,
     "Idiosyncratic gaps fade; common-factor gaps persist"),
    ("H13_delivery_spike_reversal", +1, _h13_delivery_spike_reversal,
     "Delivery spikes after down days mark institutional accumulation"),
    ("H14_pledge_distress_momentum", -1, _h14_pledge_distress_momentum,
     "Pledge-distress gap-downs keep falling (forced selling)"),
    ("H15_global_riskoff_fade", +1, _h15_global_riskoff_fade,
     "US-fear-driven gap-downs recover intraday"),
]


def run_hypotheses(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    """Evaluate all pre-registered hypotheses on train + OOS windows.
    direction: +1 tests mean>0, -1 tests mean<0, 0 tests |gap-direction|
    continuation (signed target vs gap direction)."""
    train, test = train_test_split_by_date(panel, TRAIN_END)
    rows = []
    for name, direction, fn, why in HYPOTHESES:
        try:
            mask = fn(panel)
        except Exception:
            mask = None
        if mask is None:
            rows.append({"hypothesis": name, "why": why, "status": "no data"})
            continue
        mask = mask.fillna(False)
        y = panel[target]
        if direction == 0 and "gap_dir" in panel.columns:
            y = y * panel["gap_dir"]           # continuation = gap-signed
            direction = +1
        elif direction == -1:
            y = -y                              # fade: flip sign, test mean>0
        m_tr = mask.loc[train.index]
        m_te = mask.loc[test.index]
        if m_tr.sum() < MIN_EVENTS or m_te.sum() < 30:
            rows.append({"hypothesis": name, "why": why, "status": "too few events",
                         "n_train": int(m_tr.sum()), "n_test": int(m_te.sum())})
            continue
        tr = edge_metrics(y.loc[train.index][m_tr], train.loc[m_tr, "date"])
        te = edge_metrics(y.loc[test.index][m_te], test.loc[m_te, "date"])
        t_oos = te.get("t_stat", np.nan)
        confirmed = (tr.get("mean_ret", 0) > 0 and te.get("mean_ret", 0) > 0
                     and not np.isnan(t_oos) and t_oos >= 1.5)
        rows.append({
            "hypothesis": name, "why": why, "status": "tested",
            "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
            "train_mean": round(tr.get("mean_ret", np.nan), 3),
            "oos_mean": round(te.get("mean_ret", np.nan), 3),
            "train_t": round(tr.get("t_stat", np.nan), 2),
            "oos_t": round(t_oos, 2) if not np.isnan(t_oos) else np.nan,
            "oos_hit": round(te.get("win_rate", np.nan), 3),
            "confirmed": bool(confirmed),
        })
    return pd.DataFrame(rows)
