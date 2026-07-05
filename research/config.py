"""
Central configuration for the edge-discovery engine.
Edit the paths to point at your parquet data folders.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
DATA_ROOT = Path("data")  # change to your data root

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

# ------------------------------------------------------------- outputs -----
OUT_DIR = Path(__file__).parent / "output"
