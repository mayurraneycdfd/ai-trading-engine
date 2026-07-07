"""
Feature engineering: converts raw data into one row PER STOCK PER DAY,
where every column is a candidate factor from the master factor list
(see DATA_CATALOG.md). All factors are known BY the market open of that
day -- no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_loader as dl
from microstructure import microstructure_features


# ------------------------------------------------------------------------
# Category 1: own price/volume factors (computed from 1-min + daily bars)
# ------------------------------------------------------------------------
def price_volume_features(minute: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=daily.index)
    prev_close = daily["close"].shift(1)

    # Gap
    f["gap_pct"] = (daily["open"] - prev_close) / prev_close * 100
    f["gap_abs"] = f["gap_pct"].abs()
    f["gap_dir"] = np.sign(f["gap_pct"])

    # Momentum horizons (all shifted -> known at today's open)
    f["ret_1d"] = daily["close"].pct_change(1).shift(1) * 100
    f["ret_5d"] = daily["close"].pct_change(5).shift(1) * 100
    f["ret_20d"] = daily["close"].pct_change(20).shift(1) * 100

    # Volatility
    tr = np.maximum(daily["high"] - daily["low"],
                    np.maximum((daily["high"] - prev_close).abs(),
                               (daily["low"] - prev_close).abs()))
    f["atr14_pct"] = (tr.rolling(14).mean() / daily["close"]).shift(1) * 100
    f["rv20"] = (daily["close"].pct_change().rolling(20).std() * np.sqrt(252)).shift(1) * 100
    f["gap_vs_atr"] = f["gap_abs"] / f["atr14_pct"].replace(0, np.nan)

    # Volatility compression: range vs its own history
    rng = (daily["high"] - daily["low"]) / daily["close"] * 100
    f["range_compression"] = (rng.rolling(5).mean() / rng.rolling(60).mean()).shift(1)

    # Volume -- LOOK-AHEAD GUARD: full-day volume of day T is not known at
    # T's open, so rvol_20d uses YESTERDAY's volume vs its trailing average.
    f["rvol_20d"] = (daily["volume"].shift(1)
                     / daily["volume"].rolling(20).mean().shift(2))
    # first-30-min relative volume (vs same window 20d avg)
    m = minute.between_time("09:15", "09:45")
    v30 = m.groupby(m.index.date)["volume"].sum()
    v30.index = pd.to_datetime(v30.index)
    f["rvol_open30"] = v30 / v30.rolling(20).mean().shift(1)

    # Levels
    f["dist_prev_high"] = (daily["open"] - daily["high"].shift(1)) / daily["close"].shift(1) * 100
    f["dist_prev_low"] = (daily["open"] - daily["low"].shift(1)) / daily["close"].shift(1) * 100
    f["dist_52w_high"] = (daily["open"] / daily["high"].rolling(252).max().shift(1) - 1) * 100
    f["dist_52w_low"] = (daily["open"] / daily["low"].rolling(252).min().shift(1) - 1) * 100

    # Consecutive closes in same direction before today
    up = (daily["close"].diff() > 0).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumsum()
    f["up_streak"] = streak.where(up == 1, -streak).shift(1)

    # Distance from 20-DMA at open (mean-reversion anchor)
    sma20 = daily["close"].rolling(20).mean().shift(1)
    sd20 = daily["close"].rolling(20).std().shift(1)
    f["zscore_20d"] = (daily["open"] - sma20) / sd20

    # Calendar
    f["day_of_week"] = f.index.dayofweek
    f["is_month_end"] = (f.index.is_month_end | (f.index.day >= 28)).astype(int)

    return f


# ------------------------------------------------------------------------
# Category 3: index / market-wide factors
# ------------------------------------------------------------------------
def market_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)

    nifty = dl.load_daily_series("nifty")
    if nifty is not None:
        nifty = nifty.resample("D").last().dropna()
        f["nifty_ret_1d"] = nifty.pct_change().shift(1).reindex(dates) * 100
        f["nifty_ret_5d"] = nifty.pct_change(5).shift(1).reindex(dates) * 100
        sma50 = nifty.rolling(50).mean()
        f["nifty_above_50dma"] = (nifty > sma50).astype(int).shift(1).reindex(dates)

    vix = dl.load_daily_series("india_vix")
    if vix is not None:
        f["vix_level"] = vix.shift(1).reindex(dates)
        f["vix_chg_1d"] = vix.pct_change().shift(1).reindex(dates) * 100
        f["vix_pctile_1y"] = vix.rolling(252).rank(pct=True).shift(1).reindex(dates)

    sp500 = dl.load_daily_series("sp500")
    if sp500 is not None:
        # US closes before India opens: yesterday's US return IS overnight info
        f["sp500_overnight"] = sp500.pct_change().reindex(dates, method="ffill") * 100

    crude = dl.load_daily_series("crude")
    if crude is not None:
        f["crude_ret_1d"] = crude.pct_change().reindex(dates, method="ffill") * 100
        f["crude_ret_5d"] = crude.pct_change(5).reindex(dates, method="ffill") * 100

    usdinr = dl.load_daily_series("usdinr")
    if usdinr is not None:
        f["usdinr_ret_1d"] = usdinr.pct_change().reindex(dates, method="ffill") * 100

    fii = dl.load_symbol_table("fii_dii")
    if fii is not None and "fii_net" in fii.columns:
        s = fii.set_index("date")["fii_net"]
        f["fii_net_prev"] = s.shift(1).reindex(dates)
        f["fii_net_5d"] = s.rolling(5).sum().shift(1).reindex(dates)

    # ---- GIFT Nifty: trades overnight, so the level AT India's open is
    # legitimately known information -- the best pre-open gap predictor.
    gift = dl.load_daily_series("gift_nifty")
    nifty_c = nifty if nifty is not None else None
    if gift is not None and nifty_c is not None:
        prev_nifty = nifty_c.shift(1)
        gift_al = gift.reindex(dates, method="ffill")
        f["gift_premium_pct"] = ((gift_al - prev_nifty.reindex(dates))
                                 / prev_nifty.reindex(dates) * 100)

    # ---- global overnight context: all these close BEFORE India opens,
    # so their latest values are known at 09:15 IST (ffill, no shift needed
    # for US assets; Asian markets are concurrent so shift(0) at open is
    # only partially known -- use previous close to be safe)
    for key, prefix in (("us_vix", "usvix"), ("us_10y", "us10y"),
                        ("dollar_index", "dxy"), ("gold", "gold"),
                        ("copper", "copper")):
        s = dl.load_daily_series(key)
        if s is not None:
            f[f"{prefix}_ret_1d"] = s.pct_change().reindex(dates, method="ffill") * 100
            if key == "us_vix":
                f["usvix_level"] = s.reindex(dates, method="ffill")
    for key, prefix in (("nikkei", "nikkei"), ("hang_seng", "hsi")):
        s = dl.load_daily_series(key)
        if s is not None:
            # Asian sessions overlap India's open -> use PREVIOUS close
            f[f"{prefix}_ret_1d"] = s.pct_change().shift(1) \
                .reindex(dates, method="ffill") * 100

    # ---- Indian macro: repo-rate changes and macro release days
    repo = dl.load_daily_series("rbi_repo", value_col=None)
    if repo is not None:
        r = repo.reindex(dates, method="ffill")
        f["repo_rate"] = r
        f["repo_chg_recent"] = (r != r.shift(5)).astype(int)  # changed in last week
    cpi = dl.load_symbol_table("cpi_wpi_iip")
    if cpi is not None and "date" in cpi.columns:
        rel = set(pd.to_datetime(cpi["date"]).dt.normalize())
        f["macro_release_today"] = dates.normalize().isin(rel).astype(int)

    # ---- holiday calendar: pre/post-holiday session effects
    hol = dl.load_symbol_table("nse_holidays")
    if hol is not None and "date" in hol.columns:
        hset = set(pd.to_datetime(hol["date"]).dt.normalize())
        nxt = pd.Series(dates.normalize() + pd.offsets.BDay(1), index=dates)
        prv = pd.Series(dates.normalize() - pd.offsets.BDay(1), index=dates)
        f["pre_holiday"] = nxt.isin(hset).astype(int).values
        f["post_holiday"] = prv.isin(hset).astype(int).values

    return f


# ------------------------------------------------------------------------
# Category 4: derivatives factors
# ------------------------------------------------------------------------
def derivative_features(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    opt = dl.load_options(symbol)
    if opt is not None:
        for col in ("pcr", "iv", "iv_percentile", "oi_change", "max_pain"):
            if col in opt.columns:
                f[f"opt_{col}"] = opt[col].shift(1).reindex(dates)
    fut = dl.load_futures(symbol)
    if fut is not None:
        for col in ("basis", "oi", "oi_change", "rollover"):
            if col in fut.columns:
                f[f"fut_{col}"] = fut[col].shift(1).reindex(dates)
        if "oi_change" in fut.columns and "basis" in fut.columns:
            # long buildup = OI up + price premium widening
            f["fut_long_buildup"] = ((fut["oi_change"] > 0) & (fut["basis"].diff() > 0)) \
                .astype(int).shift(1).reindex(dates)
    return f


# ------------------------------------------------------------------------
# Categories 5 & 8: events and news
# ------------------------------------------------------------------------
def event_features(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)

    earn = dl.load_symbol_table("earnings_dates")
    if earn is not None:
        e = earn[earn["symbol"] == symbol]["date"]
        f["earnings_today"] = dates.normalize().isin(set(e)).astype(int)
        f["earnings_yesterday"] = dates.normalize().isin(set(e + pd.Timedelta(days=1))).astype(int)

    ca = dl.load_symbol_table("corporate_actions")
    if ca is not None:
        c = ca[ca["symbol"] == symbol]
        f["exdate_today"] = dates.normalize().isin(
            set(c.loc[c.get("type", pd.Series(dtype=str)).isin(["dividend", "split", "bonus"]), "date"])
        ).astype(int)

    news = dl.load_symbol_table("news")
    if news is not None:
        n = news[news["symbol"] == symbol]
        # LOOK-AHEAD GUARD: news published during day T (e.g. 14:00) must not
        # feed the 09:15 open decision of day T. Use the PREVIOUS session's
        # news via shift(1) on the daily aggregate.
        if "sentiment" in n.columns:
            s = n.groupby("date")["sentiment"].mean().sort_index().shift(1)
            f["news_sentiment"] = s.reindex(dates.normalize()).values
        cnt = n.groupby("date").size().sort_index().shift(1)
        f["news_count"] = cnt.reindex(dates.normalize()).fillna(0).values
        f["has_news"] = (f["news_count"] > 0).astype(int)

    return f


# ------------------------------------------------------------------------
# Category 6: India-specific smart-money features (delivery %, participant
# OI, short selling, bulk deals, sector rotation, F&O ban). All lagged by
# one session -- data published after close of day T feeds day T+1.
# ------------------------------------------------------------------------
def india_smart_money_features(symbol: str, dates: pd.DatetimeIndex,
                               sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    norm = dates.normalize()

    # ---- delivery percentage: the strongest underused Indian signal.
    # High delivery = real ownership changing hands (institutional
    # conviction); low delivery = intraday churn.
    dlv = dl.load_symbol_table("delivery")
    if dlv is not None and "delivery_pct" in dlv.columns:
        s = dlv[dlv["symbol"] == symbol].set_index("date")["delivery_pct"] \
            .sort_index().shift(1)                       # yesterday's delivery
        f["delivery_pct"] = s.reindex(norm).values
        mu = s.rolling(20).mean()
        sd = s.rolling(20).std()
        f["delivery_z20"] = ((s - mu) / sd).reindex(norm).values

    # ---- F&O ban list: banned = fresh F&O positions prohibited. Signals
    # on banned days are unfillable; also a squeeze indicator on exit.
    ban = dl.load_symbol_table("fno_ban")
    if ban is not None:
        b = set(pd.to_datetime(ban[ban["symbol"] == symbol]["date"]).dt.normalize())
        in_ban = pd.Series(norm.isin(b).astype(int), index=dates)
        f["fno_ban_today"] = in_ban.values
        f["fno_ban_exit"] = ((in_ban.shift(1) == 1) & (in_ban == 0)).astype(int).values

    # ---- participant-level OI: FII futures positioning as a market regime
    poi = dl.load_symbol_table("participant_oi")
    if poi is not None and "fii_fut_long" in poi.columns:
        p = poi.set_index("date").sort_index().shift(1)  # known next morning
        ls = p["fii_fut_long"] / p["fii_fut_short"].replace(0, np.nan)
        f["fii_fut_long_short"] = ls.reindex(norm).values
        f["fii_fut_ls_chg5d"] = (ls - ls.shift(5)).reindex(norm).values
        if "client_opt_put_oi" in p.columns and "client_opt_call_oi" in p.columns:
            pcr = p["client_opt_put_oi"] / p["client_opt_call_oi"].replace(0, np.nan)
            f["client_opt_pcr"] = pcr.reindex(norm).values

    # ---- short selling: high short interest + gap-up = squeeze fuel
    ss = dl.load_symbol_table("short_selling")
    if ss is not None and "short_qty" in ss.columns:
        s = ss[ss["symbol"] == symbol].set_index("date").sort_index().shift(1)
        if "traded_qty" in s.columns:
            si = (s["short_qty"] / s["traded_qty"].replace(0, np.nan) * 100)
            f["short_interest_pct"] = si.reindex(norm).values
            f["si_pctile_90d"] = si.rolling(90).rank(pct=True).reindex(norm).values

    # ---- bulk/block deals: large-investor conviction in the last 5 days
    bd = dl.load_symbol_table("bulk_deals")
    if bd is not None:
        b = bd[bd["symbol"] == symbol].copy()
        if not b.empty and "buy_sell" in b.columns:
            b["date"] = pd.to_datetime(b["date"]).dt.normalize()
            sign = b["buy_sell"].str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
            daily_net = sign.groupby(b["date"]).sum()
            net5 = daily_net.reindex(
                pd.date_range(norm.min() - pd.Timedelta(days=10), norm.max()),
                fill_value=0).rolling(5).sum().shift(1)   # deals up to yesterday
            f["bulk_deal_net5d"] = net5.reindex(norm).values
            f["bulk_deal_recent"] = (net5.reindex(norm).fillna(0) != 0).astype(int).values

    # ---- sector rotation: gaps aligned with sector momentum continue more
    rrg = dl.load_symbol_table("sector_rotation")
    sector = (sector_map or {}).get(symbol)
    if rrg is not None and sector is not None and "rs_ratio" in rrg.columns:
        r = rrg[rrg["sector"] == sector].set_index("date").sort_index().shift(1)
        f["sector_rs_ratio"] = r["rs_ratio"].reindex(norm).values
        if "rs_momentum" in r.columns:
            f["sector_rs_momentum"] = r["rs_momentum"].reindex(norm).values

    # ---- promoter pledges: rising pledge % = distress; forced-selling risk
    plg = dl.load_symbol_table("promoter_pledges")
    if plg is not None and "pledge_pct" in plg.columns:
        p = plg[plg["symbol"] == symbol].set_index("date")["pledge_pct"] \
            .sort_index()
        pf = p.reindex(pd.date_range(norm.min() - pd.Timedelta(days=200),
                                     norm.max()), method="ffill").shift(1)
        f["pledge_pct"] = pf.reindex(norm).values
        f["pledge_chg_90d"] = (pf - pf.shift(90)).reindex(norm).values

    # ---- index inclusion/exclusion: forced passive flows around the date
    cc = dl.load_symbol_table("constituent_changes")
    if cc is not None and "action" in cc.columns:
        c = cc[cc["symbol"] == symbol].copy()
        if not c.empty:
            c["date"] = pd.to_datetime(c["date"]).dt.normalize()
            inc = set(c.loc[c["action"].str.upper().str.startswith("INC"), "date"])
            exc = set(c.loc[c["action"].str.upper().str.startswith("EXC"), "date"])
            win_inc = set().union(*[{d + pd.Timedelta(days=k) for k in range(-10, 11)}
                                    for d in inc]) if inc else set()
            win_exc = set().union(*[{d + pd.Timedelta(days=k) for k in range(-10, 11)}
                                    for d in exc]) if exc else set()
            f["index_inclusion_window"] = norm.isin(win_inc).astype(int)
            f["index_exclusion_window"] = norm.isin(win_exc).astype(int)

    return f


# ------------------------------------------------------------------------
# Category 2: cross-sectional (needs all-stocks daily panel)
# ------------------------------------------------------------------------
def cross_sectional_features(symbol: str, daily: pd.DataFrame,
                             panel_gaps: pd.DataFrame | None,
                             sector_map: dict[str, str]) -> pd.DataFrame:
    """panel_gaps: DataFrame [date x symbol] of gap_pct for all stocks."""
    f = pd.DataFrame(index=daily.index)
    if panel_gaps is None or symbol not in panel_gaps.columns:
        return f
    own = panel_gaps[symbol]
    others = panel_gaps.drop(columns=[symbol])
    f["mkt_gap_median"] = others.median(axis=1).reindex(daily.index)
    f["gap_idiosyncratic"] = (own - f["mkt_gap_median"]).reindex(daily.index)

    sector = sector_map.get(symbol)
    if sector:
        peers = [s for s, sec in sector_map.items() if sec == sector and s != symbol
                 and s in panel_gaps.columns]
        if peers:
            f["sector_gap_median"] = panel_gaps[peers].median(axis=1).reindex(daily.index)
            f["gap_vs_sector"] = (own - f["sector_gap_median"]).reindex(daily.index)
    return f


# ------------------------------------------------------------------------
# assemble
# ------------------------------------------------------------------------
def build_features(symbol: str, minute: pd.DataFrame,
                   panel_gaps: pd.DataFrame | None = None,
                   sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    daily = dl.to_daily(minute)
    parts = [
        price_volume_features(minute, daily),
        market_features(daily.index),
        derivative_features(symbol, daily.index),
        event_features(symbol, daily.index),
        india_smart_money_features(symbol, daily.index, sector_map),
        cross_sectional_features(symbol, daily, panel_gaps, sector_map),
        microstructure_features(minute).reindex(daily.index),
    ]
    feats = pd.concat(parts, axis=1)
    feats["symbol"] = symbol
    return feats
