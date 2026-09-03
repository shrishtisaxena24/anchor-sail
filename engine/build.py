"""
Anchor & Sail — orchestrator. Downloads prices, runs the strategy for the four portfolios,
updates the ledgers in data/state/*.json and writes docs/data.json for the dashboard.

    python engine/build.py            # real run (yfinance)
    python engine/build.py --mock     # synthetic prices, for testing the pipeline offline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy as S                                   # noqa: E402
from universes import load_universes, portfolio_universes, load_company_names   # noqa: E402
from data import (fetch_daily, reconcile_with_cache, to_monthly, load_benchmark, to_yahoo, from_yahoo,
                  fetch_splits, HISTORY_START, NIFTY_CASH, BENCHMARKS)        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "state"
DOCS = ROOT / "docs"
IST = ZoneInfo("Asia/Kolkata")

# ----------------------------------------------------------------------------- configuration
import os
INCEPTION_MONTH = os.environ.get("AS_INCEPTION_MONTH", "2026-08")   # capital committed at this month's last close
WARMUP_BARS = 65                   # monthly bars before signals are trusted (brief: 65-month warm-up)
CLOSE_HHMM = (15, 35)              # after this IST time the day's bar is treated as final

PORTFOLIOS = {
    "CORE":      {"label": "Core",      "universe": "NIFTY 50 + NIFTY NEXT 50", "ranking": False},
    "PRECISION": {"label": "Precision", "universe": "NIFTY MIDCAP 150",         "ranking": True},
    "FRONTIER":  {"label": "Frontier",  "universe": "NIFTY SMALLCAP 250",       "ranking": False},
    "SPECTRUM":  {"label": "Spectrum",  "universe": "All three pooled (~500)",  "ranking": True},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fnum(x, nd=2):
    if x is None:
        return None
    try:
        if isinstance(x, (float, np.floating)) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), nd)
    except Exception:
        return None


# ----------------------------------------------------------------------------- mock data
def mock_prices(symbols: list[str], start: str, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(start, end)
    out = {}
    for i, s in enumerate(symbols):
        n = len(idx)
        drift = rng.normal(0.0004, 0.0003)
        vol = rng.uniform(0.012, 0.03)
        r = rng.normal(drift, vol, n)
        # inject a few regimes so that %R arms and recovers
        for k in range(0, n, 260):
            seg = slice(k, min(n, k + 60))
            r[seg] += rng.choice([-0.004, 0.004, 0.0])
        close = 100 * np.exp(np.cumsum(r)) * rng.uniform(0.5, 30)
        o = close * (1 + rng.normal(0, 0.004, n))
        h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        out[s] = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": 1e5}, index=idx)
    return out


# ----------------------------------------------------------------------------- helpers
def month_key(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.year:04d}-{ts.month:02d}"


def last_completed_month(now_ist: datetime) -> str:
    """The most recent month whose bar is complete at `now_ist`."""
    y, m = now_ist.year, now_ist.month
    last_cal_day = (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).day
    after_close = (now_ist.hour, now_ist.minute) >= CLOSE_HHMM
    if now_ist.day == last_cal_day and after_close:
        return f"{y:04d}-{m:02d}"
    prev = pd.Timestamp(year=y, month=m, day=1) - pd.Timedelta(days=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def load_state(name: str) -> dict | None:
    p = STATE_DIR / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_state(name: str, state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{name}.json").write_text(json.dumps(state, indent=1, default=str))


def max_drawdown(navs: list[float]) -> float:
    peak, mdd = -1e18, 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)
    return mdd


# ----------------------------------------------------------------------------- main
def main(mock: bool = False, offline: bool = False):
    t0 = time.time()
    now_ist = datetime.now(IST)
    today = pd.Timestamp(now_ist.date())
    after_close = (now_ist.hour, now_ist.minute) >= CLOSE_HHMM
    warnings: list[str] = []

    # ---- universes
    lists, uwarn, umeta = load_universes(offline=offline or mock)
    warnings += uwarn
    unis = portfolio_universes(lists)
    all_syms = unis["SPECTRUM"]
    names = load_company_names()
    log(f"universe sizes: " + ", ".join(f"{k}={len(v)}" for k, v in unis.items()))

    # ---- prices
    ysyms = [to_yahoo(s) for s in all_syms] + [NIFTY_CASH]
    if mock:
        fresh = mock_prices(ysyms, HISTORY_START, today)
        warnings.append("MOCK DATA — synthetic prices, for pipeline testing only.")
    else:
        log(f"downloading {len(ysyms)} symbols from {HISTORY_START} …")
        fresh, w = fetch_daily(ysyms, HISTORY_START, log=log)
        warnings += w
        fresh, w = reconcile_with_cache(fresh)
        warnings += w
    if NIFTY_CASH not in fresh:
        raise SystemExit("^NSEI (cash leg / calendar) unavailable — aborting run, previous data.json kept.")
    nsei = fresh[NIFTY_CASH]
    data_date = nsei.index.max()
    prices = {from_yahoo(k): v for k, v in fresh.items() if k != NIFTY_CASH}

    # ---- calendar
    completed_key = last_completed_month(now_ist)
    running_key = month_key(today)
    nsei_m = to_monthly(nsei)
    month_last_day = {k: d for d, k in zip(nsei_m.index, nsei_m["month"])}
    if completed_key not in month_last_day:
        # e.g. month just completed but yfinance hasn't published the last day yet
        completed_key = max(k for k in month_last_day if k < running_key)
    log(f"data date {data_date.date()}, completed month {completed_key}, running month {running_key}")

    # ---- per-symbol indicators & ticker-level state
    snap: dict[str, dict] = {}
    entries_by_month: dict[str, dict[str, dict]] = {}     # month -> symbol -> candidate record
    prev_month_low: dict[str, dict[str, float]] = {}     # month -> symbol -> low of the previous month
    for sym in all_syms:
        d = prices.get(sym)
        if d is None or len(d) < 30:
            snap[sym] = {"symbol": sym, "name": names.get(sym, ""), "nodata": True}
            continue
        m = to_monthly(d)
        ind = S.calc_ind(m)
        comp = ind[ind["month"] <= completed_key]
        run = ind[ind["month"] == running_key]
        if comp.empty:
            snap[sym] = {"symbol": sym, "name": names.get(sym, ""), "nodata": True}
            continue
        positions, states = S.gen_positions(comp, min_bars=WARMUP_BARS)
        # entries per month (ticker level)
        for ts, row in states[states["entry"]].iterrows():
            mk = comp.loc[ts, "month"]
            entries_by_month.setdefault(mk, {})[sym] = {
                "symbol": sym, "name": names.get(sym, ""), "close": float(comp.loc[ts, "Close"]),
                "score": float(comp.loc[ts, "score"]), "wr": float(comp.loc[ts, "wr"]),
                "ema5": float(comp.loc[ts, "ema5"]), "ema15": float(comp.loc[ts, "ema15"]),
                "ema50": float(comp.loc[ts, "ema50"]), "date": str(ts.date())}
        # previous-month lows keyed by the month that FOLLOWS them (for trailing)
        mrows = list(zip(m["month"], m["Low"]))
        for (mk_prev, low_prev), (mk_next, _) in zip(mrows[:-1], mrows[1:]):
            prev_month_low.setdefault(mk_next, {})[sym] = float(low_prev)
        last = comp.iloc[-1]
        st = states.iloc[-1]
        tl_open = positions[-1] if positions and positions[-1].exit_idx is None else None
        # ticker-level position: is it open after the completed month?
        rec = {
            "symbol": sym, "name": names.get(sym, ""), "nodata": False,
            "wr": fnum(last["wr"]), "ema5": fnum(last["ema5"]), "ema15": fnum(last["ema15"]),
            "ema50": fnum(last["ema50"]), "trend": bool(last["trend"]), "score": fnum(last["score"]),
            "armed": bool(st["armed"]), "tl_inpos": bool(st["in_pos"]), "entry_month": bool(st["entry"]),
            "month_close": fnum(last["Close"]), "month": str(last["month"]),
            "tl_entry": fnum(tl_open.entry) if tl_open else None,
            "tl_entry_date": str(tl_open.entry_date.date()) if tl_open else None,
            "tl_stop": fnum(tl_open.stop) if tl_open else None,
        }
        # provisional (running month) readings — what the month looks like if it closed today
        if not run.empty:
            r = run.iloc[-1]
            rec.update({"wr_p": fnum(r["wr"]), "trend_p": bool(r["trend"]), "score_p": fnum(r["score"]),
                        "ema5_p": fnum(r["ema5"]), "ema15_p": fnum(r["ema15"]), "ema50_p": fnum(r["ema50"])})
            armed_p = rec["armed"] or (r["wr"] is not None and not math.isnan(r["wr"]) and r["wr"] < S.ARM_LEVEL)
            rec["would_trigger"] = bool(armed_p and (not rec["tl_inpos"]) and r["wr"] >= S.TRIGGER_LEVEL and bool(r["trend"]))
        else:
            rec.update({"wr_p": None, "trend_p": None, "score_p": None, "would_trigger": False})
        # latest daily OHLC
        ld = d.iloc[-1]
        pc = float(d["Close"].iloc[-2]) if len(d) > 1 else float(ld["Close"])
        rec.update({"date": str(d.index[-1].date()), "open": fnum(ld["Open"]), "high": fnum(ld["High"]),
                    "low": fnum(ld["Low"]), "close": fnum(ld["Close"]), "prev_close": fnum(pc),
                    "chg_pct": fnum((float(ld["Close"]) / pc - 1) * 100, 2) if pc else None,
                    "stale": bool(d.index[-1] < data_date - pd.Timedelta(days=3))})
        snap[sym] = rec

    # ---- trading calendar from ^NSEI
    cal = list(nsei.index)
    if INCEPTION_MONTH not in month_last_day:
        raise SystemExit(f"Inception month {INCEPTION_MONTH} not yet complete in the data.")
    inception_day = month_last_day[INCEPTION_MONTH]
    final_cutoff = today if after_close else today - pd.Timedelta(days=1)   # last day whose bar is final

    out_portfolios = {}
    for pname, cfg in PORTFOLIOS.items():
        uni = unis[pname]
        uni_set = set(uni)
        state = load_state(pname)
        if state is None:
            state = {"portfolio": pname, "inception": str(inception_day.date()), "capital": S.CAPITAL,
                     "cash_units": S.CAPITAL / float(nsei.loc[inception_day, "Close"]),
                     "ranking": cfg["ranking"], "created": now_ist.isoformat()}
        book = S.Book(state)
        bench, bench_label, bw = (None, "", []) if mock else load_benchmark(pname, start=HISTORY_START, log=log)
        if mock:
            bench = nsei["Close"] * 1.0; bench_label = "MOCK benchmark"
        warnings += bw
        lp = pd.Timestamp(state["last_processed_date"]) if state.get("last_processed_date") else None

        # ---- corporate actions: keep qty / entry / stop / target consistent through splits
        if not mock and book.open_positions():
            splits = fetch_splits([to_yahoo(s) for s in sorted(book.held_symbols())], since=state["inception"])
            for p in book.open_positions():
                for sd, ratio in splits.get(to_yahoo(p["symbol"]), []):
                    key = f"{sd}:{ratio}"
                    if sd <= p["entry_date"] or key in p.setdefault("splits_applied", []):
                        continue
                    p["qty"] = int(round(p["qty"] * ratio)); p["qty_open"] = int(round(p["qty_open"] * ratio))
                    for f in ("entry", "stop", "target", "scale_price"):
                        if p.get(f) is not None:
                            p[f] = float(p[f]) / ratio
                    p["splits_applied"].append(key)
                    book._event(sd, "SPLIT", p["symbol"], p["entry"], p["qty_open"], f"split {ratio:g}:1 applied to ledger")
                    warnings.append(f"{pname}: {p['symbol']} split {ratio:g}:1 on {sd} — quantity, entry, stop and target rescaled.")

        # ---- walk the calendar from inception
        for di, d in enumerate(cal):
            if d < inception_day:
                continue
            dk = str(d.date())
            nsei_c = float(nsei.loc[d, "Close"])
            # exits: only on days not yet finalised (protects the ledger from later data revisions)
            if lp is None or d > lp:
                bars = {}
                for p in book.open_positions():
                    s = p["symbol"]
                    if s in prices and d in prices[s].index:
                        r = prices[s].loc[d]
                        bars[s] = (float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]))
                mk = month_key(d)
                first_of_month = (di > 0 and month_key(cal[di - 1]) != mk)
                pml = prev_month_low.get(mk) if first_of_month else None
                book.check_exits(d, bars, nsei_c, pml)
            # month-end entries on a completed month
            mk = month_key(d)
            if mk <= completed_key and d == month_last_day.get(mk) and mk >= INCEPTION_MONTH \
                    and state.get("last_signal_month") != mk:
                cands = [entries_by_month.get(mk, {})[s] for s in uni if s in entries_by_month.get(mk, {})]
                closes_all = {s: float(prices[s].loc[d, "Close"]) for s in uni_set
                              if s in prices and d in prices[s].index}
                book.take_entries(d, mk, cands, closes_all, nsei_c, cfg["ranking"])
                log(f"{pname}: {mk} entries -> {sum(1 for r in book.s['signals_by_month'][mk] if r['status']=='Taken')} taken / {len(cands)} candidates")
            # valuation
            closes = {s: float(prices[s].loc[d, "Close"]) for s in book.held_symbols()
                      if s in prices and d in prices[s].index}
            nav, cash, inv = book.nav(closes, nsei_c)
            bval = None
            if bench is not None:
                bb = bench[bench.index <= d]
                if len(bb):
                    bval = float(bb.iloc[-1])
            state["history"][dk] = {"nav": round(nav, 2), "bench": bval, "cash": round(cash, 2), "inv": round(inv, 2)}
        state["last_processed_date"] = str(min(data_date, final_cutoff).date()) if data_date >= inception_day else None
        state["last_run"] = now_ist.isoformat()
        save_state(pname, state)

        # ---- build dashboard payload for this portfolio
        hist = sorted(state["history"].items())
        h0 = hist[0][1] if hist else None
        navs = [v["nav"] for _, v in hist]
        nav_now = navs[-1] if navs else S.CAPITAL
        bench_now = hist[-1][1]["bench"] if hist else None
        bench_0 = h0["bench"] if h0 else None
        ret = nav_now / S.CAPITAL - 1
        bret = (bench_now / bench_0 - 1) if (bench_now and bench_0) else None
        days = (pd.Timestamp(hist[-1][0]) - inception_day).days if hist else 0
        cagr = ((nav_now / S.CAPITAL) ** (365.25 / days) - 1) if days >= 365 else None
        bcagr = ((bench_now / bench_0) ** (365.25 / days) - 1) if (bret is not None and days >= 365) else None

        open_pos = []
        for p in book.open_positions():
            sp = snap.get(p["symbol"], {})
            ltp = sp.get("close")
            open_pos.append({**p,
                             "ltp": ltp, "day_open": sp.get("open"), "day_high": sp.get("high"), "day_low": sp.get("low"),
                             "pnl_pct": fnum((ltp / p["entry"] - 1) * 100) if ltp else None,
                             "unrealised": fnum((ltp - p["entry"]) * p["qty_open"]) if ltp else None,
                             "value": fnum(ltp * p["qty_open"]) if ltp else None,
                             "to_stop_pct": fnum((ltp / p["stop"] - 1) * 100) if ltp else None,
                             "to_target_pct": fnum((p["target"] / ltp - 1) * 100) if (ltp and not p["scaled"]) else None,
                             "wr_p": sp.get("wr_p"), "trend_p": sp.get("trend_p")})
        open_pos.sort(key=lambda r: (r["to_stop_pct"] if r["to_stop_pct"] is not None else 1e9))

        month_start = f"{running_key}-01"
        exits = []
        for p in book.positions:
            evs = [e for e in state["events"] if e["symbol"] == p["symbol"] and e["type"] in ("SELL", "SCALE")
                   and e["date"] >= min(month_start, str((data_date - pd.Timedelta(days=7)).date()))]
            if p["status"] == "closed" and p["exit_date"] and p["exit_date"] >= min(month_start, str((data_date - pd.Timedelta(days=7)).date())):
                exits.append({"type": "EXIT", "symbol": p["symbol"], "name": names.get(p["symbol"], ""),
                              "date": p["exit_date"], "reason": p["reason"], "entry": p["entry"],
                              "entry_date": p["entry_date"], "qty": p["qty"], "price": p["exit_price"],
                              "stop": p["stop"], "realised": fnum(p["realised"]),
                              "pnl_pct": fnum((p["exit_price"] / p["entry"] - 1) * 100),
                              "today": p["exit_date"] == str(data_date.date())})
            if p.get("scale_date") and p["scale_date"] >= min(month_start, str((data_date - pd.Timedelta(days=7)).date())):
                exits.append({"type": "SCALE-OUT", "symbol": p["symbol"], "name": names.get(p["symbol"], ""),
                              "date": p["scale_date"], "reason": "Target hit — sold 50%, stop to breakeven",
                              "entry": p["entry"], "entry_date": p["entry_date"], "qty": int(round(p["qty"] * S.SCALE_FRACTION)),
                              "price": p["scale_price"], "stop": p["stop"], "realised": None,
                              "pnl_pct": fnum((p["scale_price"] / p["entry"] - 1) * 100),
                              "today": p["scale_date"] == str(data_date.date())})
        exits.sort(key=lambda r: (not r["today"], r["date"]), reverse=False)
        exits.sort(key=lambda r: r["date"], reverse=True)

        sig_month = max(state["signals_by_month"]) if state["signals_by_month"] else None
        entries = []
        for r in (state["signals_by_month"].get(sig_month, []) if sig_month else []):
            sp = snap.get(r["symbol"], {})
            ltp = sp.get("close")
            entries.append({**r, "name": names.get(r["symbol"], ""), "ltp": ltp,
                            "from_entry_pct": fnum((ltp / r["close"] - 1) * 100) if ltp else None,
                            "still_open": r["symbol"] in book.held_symbols()})

        # watchlist — entry side
        held = book.held_symbols()
        wl_entry = []
        for s in uni:
            sp = snap.get(s)
            if not sp or sp.get("nodata") or s in held or sp["tl_inpos"] or not sp["armed"]:
                continue
            wr_ok = sp["wr"] is not None and sp["wr"] >= S.TRIGGER_LEVEL
            tr_ok = sp["trend"]
            if wr_ok and tr_ok:
                continue                     # would already have fired
            if not (wr_ok or tr_ok) and not sp.get("would_trigger"):
                continue                     # armed only — too far away
            waiting = []
            if not wr_ok:
                waiting.append(f"%R ≥ {int(S.TRIGGER_LEVEL)}")
            if not tr_ok:
                waiting.append("EMA5 > EMA15 > EMA50")
            wl_entry.append({**{k: sp.get(k) for k in ("symbol", "name", "close", "chg_pct", "wr", "wr_p", "trend", "trend_p",
                                                        "ema5", "ema15", "ema50", "score", "score_p", "would_trigger")},
                             "waiting_for": " & ".join(waiting)})
        wl_entry.sort(key=lambda r: (not r["would_trigger"], -(r["wr_p"] if r["wr_p"] is not None else -999)))

        wl_exit = [r for r in open_pos if (r["to_stop_pct"] is not None and r["to_stop_pct"] <= 5.0)
                   or (r["to_target_pct"] is not None and r["to_target_pct"] <= 5.0)]

        # full universe table
        uni_rows = []
        for s in uni:
            sp = dict(snap.get(s, {"symbol": s, "nodata": True}))
            if s in held:
                sp["state"] = "HELD"
            elif sp.get("nodata"):
                sp["state"] = "NO DATA"
            elif sp.get("entry_month") and sp.get("month") == completed_key:
                sp["state"] = "ENTRY SIGNAL"
            elif sp.get("tl_inpos"):
                sp["state"] = "IN CYCLE"          # ticker-level position open (strategy already 'in' this name)
            elif sp.get("armed") and sp.get("trend"):
                sp["state"] = "ARMED + TREND"
            elif sp.get("armed"):
                sp["state"] = "ARMED"
            else:
                sp["state"] = "—"
            uni_rows.append(sp)

        realised_total = sum(p["realised"] for p in book.positions if p["status"] == "closed")
        realised_total += sum(p["realised"] for p in book.positions if p["status"] == "open")   # scale-out cash
        unreal = sum((r["unrealised"] or 0) for r in open_pos)
        out_portfolios[pname] = {
            "name": pname, "label": cfg["label"], "universe_label": cfg["universe"], "ranking": cfg["ranking"],
            "universe_size": len(uni), "benchmark": BENCHMARKS[pname]["name"], "benchmark_source": bench_label,
            "inception": state["inception"], "capital": S.CAPITAL,
            "performance": {
                "nav": fnum(nav_now), "return_pct": fnum(ret * 100), "bench_return_pct": fnum(bret * 100) if bret is not None else None,
                "alpha_pct": fnum((ret - bret) * 100) if bret is not None else None,
                "cagr_pct": fnum(cagr * 100) if cagr is not None else None,
                "bench_cagr_pct": fnum(bcagr * 100) if bcagr is not None else None,
                "cash": fnum(hist[-1][1]["cash"]) if hist else S.CAPITAL, "invested": fnum(hist[-1][1]["inv"]) if hist else 0,
                "open_positions": len(open_pos), "slots": S.SLOTS, "days": days,
                "realised": fnum(realised_total), "unrealised": fnum(unreal),
                "max_dd_pct": fnum(max_drawdown(navs) * 100), "closed_trades": sum(1 for p in book.positions if p["status"] == "closed"),
                "wins": sum(1 for p in book.positions if p["status"] == "closed" and p["realised"] > 0),
                "bench_level": bench_now, "bench_base": bench_0,
            },
            "history": [{"date": k, "nav": v["nav"], "bench": v["bench"]} for k, v in hist],
            "exits": exits, "entries": entries, "signal_month": sig_month,
            "open_positions": open_pos, "watch_entry": wl_entry, "watch_exit": wl_exit,
            "universe": uni_rows,
            "events": state["events"][-200:],
        }

    # ---- market status
    if data_date.date() == today.date():
        status = "Closed — today's final bar" if after_close else "LIVE — intraday bar (Yahoo delay ≈ 15 min)"
    else:
        status = f"As of previous close {data_date.date()}"

    payload = {
        "generated_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "data_date": str(data_date.date()), "market_status": status,
        "completed_month": completed_key, "running_month": running_key, "inception_month": INCEPTION_MONTH,
        "nifty": {"close": fnum(float(nsei["Close"].iloc[-1])),
                  "chg_pct": fnum((float(nsei["Close"].iloc[-1]) / float(nsei["Close"].iloc[-2]) - 1) * 100) if len(nsei) > 1 else None},
        "universe_sources": umeta.get("source", {}),
        "warnings": warnings,
        "strategy": {"wr_len": S.WR_LEN, "arm": S.ARM_LEVEL, "trigger": S.TRIGGER_LEVEL, "emas": [5, 15, 50],
                     "stop_pct": -15, "target_pct": 30, "scale": 50, "slots": S.SLOTS, "capital": S.CAPITAL},
        "portfolios": out_portfolios,
        "runtime_sec": round(time.time() - t0, 1),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(payload, default=str, separators=(",", ":")))
    log(f"wrote docs/data.json ({(DOCS / 'data.json').stat().st_size // 1024} KB) in {time.time() - t0:.0f}s; "
        f"{len(warnings)} warning(s)")
    for w in warnings:
        log("  WARN", w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="synthetic prices (offline pipeline test)")
    ap.add_argument("--offline", action="store_true", help="do not fetch universe lists from NSE")
    a = ap.parse_args()
    main(mock=a.mock, offline=a.offline)
