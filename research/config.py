"""
Central configuration for the edge-discovery engine.
Edit the paths to point at your parquet data folders.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
# Auto-detect the data root: prefers the user's arranged_analysis_data folder
_CANDIDATE_ROOTS = [Path("arranged_analysis_data"),
                    Path("../arranged_analysis_data"),
                    Path("data")]
DATA_ROOT = next((p for p in _CANDIDATE_ROOTS if p.exists()),
                 _CANDIDATE_ROOTS[0])

PATHS = {
    # 1-min OHLCV parquet per stock: columns [timestamp, open, high, low, close, volume]
    "minute_bars": DATA_ROOT / "minute_bars",        # minute_bars/{SYMBOL}.parquet
    "nifty": DATA_ROOT / "nifty.parquet",            # 1-min or daily nifty
    "sp500": DATA_ROOT / "sp500.parquet",            # daily S&P 500
    "india_vix": DATA_ROOT / "india_vix.parquet",    # daily India VIX
    "crude": DATA_ROOT / "crude.parquet",            # daily crude oil
    "usdinr": DATA_ROOT / "usdinr.parquet",          # daily USD/INR
    "options": DATA_ROOT / "options",                # options/{SYMBOL}.parquet (OI, IV, PCR)
    "futures": DATA_ROOT / "futures",                # futures/{SYMBOL}.parquet (basis, OI)
    "corporate_actions": DATA_ROOT / "corporate_actions.parquet",  # symbol, date, type
    "earnings_dates": DATA_ROOT / "earnings_dates.parquet",        # symbol, date
    "news": DATA_ROOT / "news.parquet",              # symbol, date, sentiment, category
    "correlations": DATA_ROOT / "correlations.parquet",  # precomputed pair correlations
    "fii_dii": DATA_ROOT / "fii_dii.parquet",        # date, fii_net, dii_net
    "sector_map": DATA_ROOT / "sector_map.parquet",  # symbol, sector
    # -------- India-specific high-value datasets (all OPTIONAL: the engine
    # degrades gracefully when a file is absent) ----------------------------
    "delivery": DATA_ROOT / "stock_delivery_daily.parquet",
    #   columns: symbol, date, delivery_pct  (0-100)
    "fno_ban": DATA_ROOT / "fno_ban_list.parquet",
    #   columns: symbol, date  (stock was in the F&O ban list on that date)
    "participant_oi": DATA_ROOT / "participant_oi_daily.parquet",
    #   columns: date, fii_fut_long, fii_fut_short, dii_fut_long,
    #            dii_fut_short, client_opt_put_oi, client_opt_call_oi
    "sector_rotation": DATA_ROOT / "sector_rotation_rrg.parquet",
    #   columns: sector, date, rs_ratio, rs_momentum
    "short_selling": DATA_ROOT / "short_selling.parquet",
    #   columns: symbol, date, short_qty, traded_qty
    "bulk_deals": DATA_ROOT / "bulk_deals.parquet",
    #   columns: symbol, date, buy_sell (BUY/SELL), value_inr
    # -------- global overnight context (daily close series) -----------------
    "gift_nifty": DATA_ROOT / "gift_nifty_overnight.parquet",
    "us_vix": DATA_ROOT / "us_vix.parquet",
    "us_10y": DATA_ROOT / "us_10y_yield.parquet",
    "dollar_index": DATA_ROOT / "dollar_index.parquet",
    "gold": DATA_ROOT / "gold.parquet",
    "copper": DATA_ROOT / "copper.parquet",
    "nikkei": DATA_ROOT / "nikkei.parquet",
    "hang_seng": DATA_ROOT / "hang_seng.parquet",
    # -------- Indian macro & event calendars --------------------------------
    "rbi_repo": DATA_ROOT / "rbi_repo_rate.parquet",
    #   columns: date, repo_rate (and optionally: policy_meeting flag)
    "cpi_wpi_iip": DATA_ROOT / "cpi_wpi_iip.parquet",
    #   columns: date, indicator (CPI/WPI/IIP), value  (date = release date)
    "nse_holidays": DATA_ROOT / "nse_holidays.parquet",
    #   columns: date
    "constituent_changes": DATA_ROOT / "constituent_changes.parquet",
    #   columns: symbol, date, action (INCLUDE/EXCLUDE), index_name
    "promoter_pledges": DATA_ROOT / "promoter_pledges.parquet",
    #   columns: symbol, date, pledge_pct
    "nifty_sector_indices": DATA_ROOT / "nifty_sector_indices.parquet",
    #   columns: date, sector (or index_name), close
}

# ---------------------------------------------------------- market hours ---
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
TZ = "Asia/Kolkata"

# ------------------------------------------------------------- universe ----
# Leave empty to auto-discover from minute_bars folder
SYMBOLS: list[str] = []

# ------------------------------------------------------ label definitions --
GAP_MIN_PCT = 0.5          # minimum |gap| % to count as a gap event
GAP_GO_HORIZONS = [15, 30, 60, 240]   # minutes after open to measure continuation
MR_ZSCORE_ENTRY = 2.0      # z-score for mean-reversion event trigger
MR_HORIZONS = [15, 30, 60, 120]       # minutes to measure reversion

# ------------------------------------------------- edge validation params --
TRAIN_END = "2021-12-31"   # walk-forward: fit/discover on <= this date
MIN_EVENTS = 100           # minimum events for a factor bucket to be considered
FDR_ALPHA = 0.05           # Benjamini-Hochberg false-discovery-rate threshold
N_BUCKETS = 5              # quantile buckets for single-factor analysis
COMBO_MAX_FACTORS = 3      # max factors in combination search
COMBO_TOP_SINGLE = 15      # only combine top-N single factors (combinatorial control)

# --------------------------------------- level 5: validation gauntlet ------
COST_PCT = 0.06            # round-trip cost+slippage per trade (% of notional)
                           # ~= STT + brokerage + impact for liquid F&O stocks
N_CV_FOLDS = 5             # purged walk-forward folds
EMBARGO_DAYS = 5           # embargo gap after each test fold (label overlap guard)
N_PERMUTATIONS = 500       # Monte Carlo permutations for empirical p-values
N_BOOTSTRAP = 1000         # stationary block bootstrap resamples
BOOTSTRAP_BLOCK = 20       # mean block length (events) for bootstrap
CSCV_SPLITS = 8            # CSCV partitions for PBO (must be even)
PBO_MAX = 0.5              # reject edges with overfit probability above this

# --------------------------------------- level 6: GP alpha miner -----------
GP_POPULATION = 200        # formulas per generation
GP_GENERATIONS = 12        # evolution rounds
GP_MAX_DEPTH = 4           # max expression-tree depth
GP_TOP_KEEP = 10           # alphas reported after validation
GP_SEED = 42

# --------------------------------------- triple-barrier labels -------------
TB_UP_MULT = 2.0           # profit barrier = TB_UP_MULT * event-day ATR%
TB_DN_MULT = 1.0           # stop barrier   = TB_DN_MULT * event-day ATR%
TB_MAX_MIN = 120           # vertical barrier: minutes after entry

# --------------------------------------- data integrity audit --------------
AUDIT_JUMP_PCT = 25.0      # overnight |gap| above this is a suspect corporate
                           # action unless explained by the CA file
AUDIT_STALE_DAYS = 5       # max consecutive identical closes before flagging
AUDIT_MIN_HISTORY_DAYS = 250   # symbols with less history are flagged
FNO_UNIVERSE_PATH = DATA_ROOT / "fno_universe.parquet"
                           # OPTIONAL point-in-time universe: [symbol, from_date, to_date]
                           # if absent the engine warns about survivorship risk

# --------------------------------------- execution cost model --------------
# Per-trade cost = statutory + brokerage + spread + impact (all % of notional)
STT_PCT = 0.025            # securities transaction tax (intraday sell side)
STAMP_BROKER_PCT = 0.007   # stamp duty + exchange charges + brokerage (round trip)
SPREAD_BASE_PCT = 0.03     # baseline half-spread for a liquid F&O stock
IMPACT_COEF = 0.10         # impact = IMPACT_COEF * sqrt(participation)
                           # participation = trade value / bar traded value
TRADE_VALUE_INR = 1_000_000    # notional per trade for impact estimate (Rs 10 lakh)
OPEN_AUCTION_MULT = 2.0    # spread+impact multiplier for trades at the open
                           # (gap trades execute in the auction / first minutes)
GAP_SLIPPAGE_COEF = 0.02   # extra slippage per 1% of |gap| for gap-entry trades

# --------------------------------------- portfolio-level validation --------
PORT_MAX_CONCURRENT = 10   # max simultaneous positions in the combined book
PORT_CORR_MAX = 0.7        # flag edge pairs with daily-PnL correlation above this
PORT_CAPACITY_PCT = 5.0    # max % of a stock's open-30min traded value deployable

# --------------------------------------- multi-boundary robustness ---------
BOUNDARY_DATES = ["2019-12-31", "2021-12-31", "2023-06-30"]
                           # alternative train/test cuts; an edge must be
                           # OOS-positive after a MAJORITY of the cuts
BOUNDARY_MIN_PASS = 2      # of len(BOUNDARY_DATES)

# ------------------------------------------------------------- outputs -----
OUT_DIR = Path(__file__).parent / "output"
