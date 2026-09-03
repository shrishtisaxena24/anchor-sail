"""
Universe lists: NIFTY 50, NIFTY NEXT 50, NIFTY MIDCAP 150, NIFTY SMALLCAP 250.

Primary source: NSE's official constituent CSVs (nsearchives.nseindia.com).
Fallback: the last successfully fetched snapshot committed in data/universe/*.txt
(the fallback is *never* a hand-typed list, so it cannot drift from NSE; the dashboard
shows a visible warning with the snapshot date whenever a fallback is used).

NSE indices are mutually exclusive — any overlap between lists is reported as a warning.
"""
from __future__ import annotations

import io
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UNI_DIR = ROOT / "data" / "universe"

NSE_CSV = {
    "NIFTY50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTYNEXT50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTYMIDCAP150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}
EXPECTED = {"NIFTY50": 50, "NIFTYNEXT50": 50, "NIFTYMIDCAP150": 150, "NIFTYSMALLCAP250": 250}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _fetch_csv(url: str, retries: int = 3) -> pd.DataFrame | None:
    for k in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and "Symbol" in r.text[:2000]:
                return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    return None


def _read_snapshot(name: str) -> tuple[list[str], str | None]:
    p = UNI_DIR / f"{name}.txt"
    meta = UNI_DIR / "snapshot_meta.json"
    date = None
    if meta.exists():
        try:
            date = json.loads(meta.read_text()).get(name)
        except Exception:
            date = None
    if not p.exists():
        return [], date
    syms = [s.strip() for s in p.read_text().strip().replace("\n", ",").split(",") if s.strip()]
    return syms, date


def _write_snapshot(name: str, symbols: list[str], names: dict[str, str]):
    UNI_DIR.mkdir(parents=True, exist_ok=True)
    (UNI_DIR / f"{name}.txt").write_text(",".join(symbols) + "\n")
    meta_p = UNI_DIR / "snapshot_meta.json"
    meta = {}
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            meta = {}
    meta[name] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta_p.write_text(json.dumps(meta, indent=1))
    names_p = UNI_DIR / "company_names.json"
    allnames = {}
    if names_p.exists():
        try:
            allnames = json.loads(names_p.read_text())
        except Exception:
            allnames = {}
    allnames.update(names)
    names_p.write_text(json.dumps(allnames, indent=0, ensure_ascii=False))


def load_company_names() -> dict[str, str]:
    p = UNI_DIR / "company_names.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def load_universes(offline: bool = False) -> tuple[dict[str, list[str]], list[str], dict]:
    """Returns (lists, warnings, meta). lists[name] is the ordered symbol list (NSE CSV order)."""
    lists: dict[str, list[str]] = {}
    warnings: list[str] = []
    meta: dict = {"source": {}}
    for name, url in NSE_CSV.items():
        df = None if offline else _fetch_csv(url)
        if df is not None and "Symbol" in df.columns and len(df) == EXPECTED[name]:
            syms = [str(s).strip() for s in df["Symbol"].tolist()]
            names = {}
            if "Company Name" in df.columns:
                names = {str(s).strip(): str(n).strip() for s, n in zip(df["Symbol"], df["Company Name"])}
            _write_snapshot(name, syms, names)
            lists[name] = syms
            meta["source"][name] = "NSE live"
        else:
            syms, date = _read_snapshot(name)
            lists[name] = syms
            meta["source"][name] = f"snapshot {date or 'unknown date'}"
            if df is not None and len(df) != EXPECTED[name]:
                warnings.append(f"{name}: NSE returned {len(df)} rows (expected {EXPECTED[name]}); "
                                f"using committed snapshot from {date}.")
            else:
                warnings.append(f"{name}: NSE constituent list not reachable; using committed snapshot "
                                f"from {date} ({len(syms)} symbols).")
            if not syms:
                warnings.append(f"{name}: NO universe available (snapshot missing).")
    # mutual exclusivity check
    for a, b in itertools.combinations(lists, 2):
        ov = sorted(set(lists[a]) & set(lists[b]))
        if ov:
            warnings.append(f"Universe overlap {a} ∩ {b}: {', '.join(ov)} — one list is stale.")
    return lists, warnings, meta


def portfolio_universes(lists: dict[str, list[str]]) -> dict[str, list[str]]:
    core = lists["NIFTY50"] + [s for s in lists["NIFTYNEXT50"] if s not in lists["NIFTY50"]]
    precision = list(lists["NIFTYMIDCAP150"])
    frontier = list(lists["NIFTYSMALLCAP250"])
    seen, spectrum = set(), []
    for s in core + precision + frontier:
        if s not in seen:
            seen.add(s); spectrum.append(s)
    return {"CORE": core, "PRECISION": precision, "FRONTIER": frontier, "SPECTRUM": spectrum}
