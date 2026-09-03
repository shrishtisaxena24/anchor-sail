"""
Price data layer (yfinance) with the safeguards from the brief:
  * chunked downloads, per-symbol retry of anything that came back empty/short
  * reconciliation against the previous run's cached store (data/cache/prices.parquet, not
    committed): a symbol whose fresh history is shorter than what we already had falls back
    to the cached series and is flagged, so a silent partial download cannot move a signal
  * monthly aggregation from daily bars (open=first, high=max, low=min, close=last)
  * benchmark series with explicit source labels — no silent index substitution
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"

HISTORY_START = "2010-07-01"       # 65+ months of warm-up before the 2016-01 backtest start
NIFTY_CASH = "^NSEI"

# Benchmarks: ordered list of (yahoo symbol, label, kind). kind = "index" | "etf-proxy"
BENCHMARKS = {
    "CORE": {"name": "Nifty 100",
             "sources": [("^CNX100", "Nifty 100 (Yahoo ^CNX100)", "index"),
                         ("NIFTY_100.NS", "Nifty 100 (Yahoo NIFTY_100.NS)", "index")]},
    "PRECISION": {"name": "Nifty Midcap 150",
                  "sources": [("NIFTYMIDCAP150.NS", "Nifty Midcap 150 (Yahoo NIFTYMIDCAP150.NS)", "index"),
                              ("MID150BEES.NS", "ETF PROXY: Nippon Nifty Midcap 150 ETF (MID150BEES)", "etf-proxy")]},
    "FRONTIER": {"name": "Nifty Smallcap 250",
                 "sources": [("NIFTYSMLCAP250.NS", "Nifty Smallcap 250 (Yahoo NIFTYSMLCAP250.NS)", "index"),
                             ("MOSMALL250.NS", "ETF PROXY: Motilal Oswal Nifty Smallcap 250 ETF (MOSMALL250)", "etf-proxy")]},
    "SPECTRUM": {"name": "Nifty 500",
                 "sources": [("^CRSLDX", "Nifty 500 (Yahoo ^CRSLDX)", "index"),
                             ("NIFTY500.NS", "Nifty 500 (Yahoo NIFTY500.NS)", "index")]},
}
# A CSV downloaded from niftyindices.com (Reports -> Historical Data) placed at
# data/benchmarks/<PORTFOLIO>.csv with columns Date, Close overrides every source above.


def to_yahoo(sym: str) -> str:
    return f"{sym}.NS"


def from_yahoo(ysym: str) -> str:
    return ysym[:-3] if ysym.endswith(".NS") else ysym


# ----------------------------------------------------------------------------- download
def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    cols = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in cols):
        return None
    out = df[cols + (["Volume"] if "Volume" in df.columns else [])].copy()
    out = out.dropna(subset=["Close"])
    if out.empty:
        return None
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _download_batch(ysyms: list[str], start: str, auto_adjust: bool = True) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    if not ysyms:
        return out
    try:
        raw = yf.download(ysyms, start=start, interval="1d", auto_adjust=auto_adjust, group_by="ticker",
                          threads=True, progress=False, timeout=60)
    except Exception:
        return out
    if raw is None or raw.empty:
        return out
    for y in ysyms:
        try:
            df = _clean(_extract(raw, y, single=len(ysyms) == 1))
        except Exception:
            df = None
        if df is not None:
            out[y] = df
    return out


def _extract(raw: pd.DataFrame, y: str, single: bool) -> pd.DataFrame | None:
    """Pull one ticker's OHLC out of whatever column layout yfinance returned."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw if single else None
    lv0, lv1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
    if y in lv0:
        return raw[y]
    if y in lv1:
        return raw.xs(y, axis=1, level=1)
    if single:
        return raw.droplevel(1, axis=1) if "Close" in lv0 else raw.droplevel(0, axis=1)
    return None


def fetch_daily(symbols: list[str], start: str = HISTORY_START, chunk: int = 60, retries: int = 3,
                log=print) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """symbols: Yahoo symbols. Returns ({yahoo_symbol: daily OHLC}, warnings)."""
    got: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    todo = list(dict.fromkeys(symbols))
    for k in range(retries):
        if not todo:
            break
        if k:
            log(f"  retry {k}: {len(todo)} symbols")
            time.sleep(3 * k)
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            res = _download_batch(part, start)
            got.update(res)
            log(f"  batch {i // chunk + 1}: {len(res)}/{len(part)} ok")
            time.sleep(1.0)                                   # be polite to Yahoo
        todo = [s for s in todo if s not in got]
    # last resort: one by one
    for s in list(todo):
        res = _download_batch([s], start)
        if s in res:
            got[s] = res[s]; todo.remove(s)
    if todo:
        warnings.append(f"No price data for {len(todo)} symbol(s): {', '.join(from_yahoo(s) for s in todo)}")
    return got, warnings


def fetch_splits(symbols: list[str], since: str) -> dict[str, list[tuple[str, float]]]:
    """Stock splits (date, ratio) after `since` for the given Yahoo symbols — used to keep the
    ledger's quantity/stop/target consistent when a held stock splits."""
    import yfinance as yf
    out: dict[str, list[tuple[str, float]]] = {}
    for y in symbols:
        try:
            s = yf.Ticker(y).get_splits()
            if s is None or len(s) == 0:
                continue
            s.index = pd.to_datetime(s.index)
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
            ev = [(str(d.date()), float(r)) for d, r in s.items() if str(d.date()) > since and r and r > 0 and r != 1.0]
            if ev:
                out[y] = ev
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------------- reconciliation cache
def reconcile_with_cache(fresh: dict[str, pd.DataFrame], cache_name: str = "prices") -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Compare against last run's cache. If a fresh series is materially shorter than the cached one
    (partial download), use the cached series extended with any newer fresh rows, and flag it."""
    warnings: list[str] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{cache_name}.parquet"
    cached: dict[str, pd.DataFrame] = {}
    if p.exists():
        try:
            big = pd.read_parquet(p)
            for sym, g in big.groupby("symbol"):
                g = g.drop(columns=["symbol"]).set_index("Date").sort_index()
                cached[sym] = g
        except Exception as e:
            warnings.append(f"Price cache unreadable ({e}); rebuilt.")
    merged: dict[str, pd.DataFrame] = {}
    partial, dropped_latest = [], {}
    for sym in set(fresh) | set(cached):
        f, c = fresh.get(sym), cached.get(sym)
        if f is None:
            if c is not None:
                merged[sym] = c
                partial.append(sym)
            continue
        if c is None:
            merged[sym] = f
            continue
        # Fresh values win wherever Yahoo returned a row; dates Yahoo dropped (it removes the latest
        # session overnight before re-publishing the official bar, and sometimes returns partial
        # history) are filled from the cache so a bar never silently disappears between runs.
        m = f.combine_first(c).sort_index()
        merged[sym] = m
        if len(f) < 0.9 * len(c):
            partial.append(sym)
        if f.index.max() < c.index.max():
            dropped_latest.setdefault(str(c.index.max().date()), []).append(sym)
    if partial:
        warnings.append(f"Partial download for {len(partial)} symbol(s) — history completed from last good cache: "
                        f"{', '.join(sorted(from_yahoo(s) for s in partial))}")
    for dt, syms in dropped_latest.items():
        warnings.append(f"Yahoo has not (re)published the {dt} session for {len(syms)} symbol(s) — that bar is "
                        f"carried from the previous run's cache" + (f" ({', '.join(sorted(from_yahoo(s) for s in syms[:8]))}…)" if len(syms) <= 8 else "."))
    # save cache
    try:
        frames = []
        for sym, df in merged.items():
            g = df.copy(); g["symbol"] = sym; g.index.name = "Date"
            frames.append(g.reset_index())
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(p, index=False)
    except Exception as e:
        warnings.append(f"Could not write price cache: {e}")
    return merged, warnings


# ----------------------------------------------------------------------------- aggregation
def _month_rule() -> str:
    try:
        pd.Series(dtype=float, index=pd.DatetimeIndex([])).resample("ME")
        return "ME"
    except Exception:
        return "M"


MONTH_RULE = _month_rule()


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly bars from daily bars. The index is the LAST TRADING DAY of each month (not the
    calendar month-end), so it lines up with the daily series."""
    agg = daily.resample(MONTH_RULE).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    last_day = daily["Close"].resample(MONTH_RULE).apply(lambda s: s.index.max() if len(s) else pd.NaT)
    agg = agg.dropna(subset=["Close"])
    agg.index = pd.DatetimeIndex([last_day.loc[i] for i in agg.index])
    agg["month"] = [f"{d.year:04d}-{d.month:02d}" for d in agg.index]
    return agg


# ----------------------------------------------------------------------------- benchmarks
def load_benchmark(portfolio: str, start: str, log=print) -> tuple[pd.Series | None, str, list[str]]:
    """Returns (close series, source label, warnings)."""
    warnings = []
    override = ROOT / "data" / "benchmarks" / f"{portfolio}.csv"
    if override.exists():
        try:
            df = pd.read_csv(override, float_precision="round_trip")
            dcol = [c for c in df.columns if c.strip().lower() in ("date", "index date")][0]
            ccol = [c for c in df.columns if c.strip().lower() in ("close", "closing index value", "close price")][0]
            s = pd.Series(df[ccol].astype(float).values, index=pd.to_datetime(df[dcol], dayfirst=True)).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            return s, f"{BENCHMARKS[portfolio]['name']} (official CSV from niftyindices.com)", warnings
        except Exception as e:
            warnings.append(f"{portfolio}: benchmark override CSV unreadable ({e}); trying Yahoo.")
    for ysym, label, kind in BENCHMARKS[portfolio]["sources"]:
        got, _ = fetch_daily([ysym], start=start, retries=2, log=lambda *_: None)
        if ysym in got and len(got[ysym]) > 5:
            if kind == "etf-proxy":
                warnings.append(f"{portfolio}: official index unavailable on Yahoo — benchmark shown is an ETF proxy "
                                f"({ysym}); tracking error applies.")
            return got[ysym]["Close"], label, warnings
    warnings.append(f"{portfolio}: no benchmark data available ({BENCHMARKS[portfolio]['name']}).")
    return None, f"{BENCHMARKS[portfolio]['name']} — UNAVAILABLE", warnings
