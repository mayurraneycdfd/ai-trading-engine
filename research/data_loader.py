"""
Data loading layer. Every loader degrades gracefully: if a dataset is missing
the engine still runs with the factors it CAN compute.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from config import PATHS, SYMBOLS


def discover_symbols() -> list[str]:
    if SYMBOLS:
        return SYMBOLS
    folder = PATHS["minute_bars"]
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.parquet"))


def load_minute_bars(symbol: str) -> pd.DataFrame | None:
    """1-min OHLCV for one stock, indexed by timestamp."""
    path = PATHS["minute_bars"] / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def to_daily(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars to daily OHLCV plus useful intraday anchors."""
    g = minute.groupby(minute.index.date)
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    daily.index = pd.to_datetime(daily.index)
    return daily


def load_daily_series(key: str, value_col: str | None = None) -> pd.Series | None:
    """Generic daily series loader (nifty, sp500, vix, crude, usdinr...)."""
    path = PATHS[key]
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date")), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    df = df.sort_index()
    if value_col is None:
        value_col = next((c for c in df.columns if c.lower() == "close"), df.columns[-1])
    return df[value_col].astype(float)


def load_symbol_table(key: str) -> pd.DataFrame | None:
    """Tables keyed by (symbol, date): corporate actions, earnings, news."""
    path = PATHS[key]
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_options(symbol: str) -> pd.DataFrame | None:
    """Daily option summary per stock: expects columns like
    [date, pcr, iv, iv_percentile, oi_change, max_pain, put_wall, call_wall]."""
    folder = PATHS["options"]
    path = Path(folder) / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
    return df


def load_futures(symbol: str) -> pd.DataFrame | None:
    folder = PATHS["futures"]
    path = Path(folder) / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
    return df


def load_sector_map() -> dict[str, str]:
    df = load_symbol_table("sector_map")
    if df is None:
        return {}
    return dict(zip(df["symbol"], df["sector"]))
