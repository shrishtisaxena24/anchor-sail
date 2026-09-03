"""
Anchor & Sail — strategy engine.

Implements, exactly as specified in the brief:
  * Williams %R(14) on monthly bars, EMA 5/15/50 (pandas ewm, adjust=False)
  * Armed / Trigger / Trend entry on the completed monthly close
  * Exits in the fixed order: gap-stop at open -> stop at stop -> scale-out 50% at target,
    then breakeven stop and trailing to the previous bar's low
  * Position sizing corpus/25 (compounding), max 25 slots, ranking score (EMA5-EMA50)/EMA50
  * Indian delivery cost model applied inside the simulation

Two layers:
  1. Ticker level (gen_positions): a per-stock monthly state machine, independent of the
     portfolio — this is what produces "an entry fired this month" and the armed flag.
  2. Book level (Book): the live portfolio ledger — takes the ticker-level entries at each
     completed month-end subject to slots/ranking, and monitors the stop/target of every
     open position on DAILY bars (the brief: "the stop must be monitored daily").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- parameters
WR_LEN = 14
EMA_FAST, EMA_MID, EMA_SLOW = 5, 15, 50
ARM_LEVEL = -40.0
TRIGGER_LEVEL = -20.0
STOP_MULT = 0.85
TARGET_MULT = 1.30
SCALE_FRACTION = 0.50
SLOTS = 25
CAPITAL = 26_00_000.0

# costs (fractions of traded value unless noted)
TXN = 0.00325 / 100
STT = 0.10 / 100
STAMP = 0.015 / 100          # buy only
SEBI = 10 / 1e7              # Rs 10 per crore
GST = 0.18                   # on (TXN + SEBI)
DP = 23.60                   # Rs per sell
BROKERAGE = 0.0


def buy_cost(v: float) -> float:
    return (TXN + STT + STAMP + SEBI) * v + GST * (TXN + SEBI) * v + BROKERAGE


def sell_cost(v: float) -> float:
    return (TXN + STT + SEBI) * v + GST * (TXN + SEBI) * v + DP + BROKERAGE


# ----------------------------------------------------------------------------- indicators
def calc_ind(m: pd.DataFrame) -> pd.DataFrame:
    """m: monthly bars with columns Open, High, Low, Close (DatetimeIndex, one row per month).
    Returns a copy with wr, ema5, ema15, ema50, trend, score."""
    out = m.copy()
    hh = out["High"].rolling(WR_LEN).max()
    ll = out["Low"].rolling(WR_LEN).min()
    rng = (hh - ll).replace(0, np.nan)
    out["wr"] = 100.0 * (out["Close"] - hh) / rng
    out["ema5"] = out["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["ema15"] = out["Close"].ewm(span=EMA_MID, adjust=False).mean()
    out["ema50"] = out["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    out["trend"] = (out["ema5"] > out["ema15"]) & (out["ema15"] > out["ema50"])
    out["score"] = (out["ema5"] - out["ema50"]) / out["ema50"] * 100.0
    return out


# ----------------------------------------------------------------------------- ticker level
@dataclass
class TickerPosition:
    entry_idx: int
    entry_date: pd.Timestamp
    entry: float
    stop: float
    target: float
    scaled: bool = False
    weight: float = 1.0
    exit_idx: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: Optional[str] = None
    score: float = float("nan")


def gen_positions(ind: pd.DataFrame, min_bars: int = EMA_SLOW):
    """Ticker-level monthly simulation.

    ind: output of calc_ind (monthly). Only bars where the indicators are valid are used;
    the first `min_bars` bars are treated as warm-up (no signals).

    Returns (positions, states):
      positions: list[TickerPosition] (open one last, exit_idx None)
      states: DataFrame indexed like ind with columns armed (state AFTER the bar),
              entry (entry fired on this bar), in_pos (position open after this bar)
    """
    o, h, l, c = (ind[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    wr, trend, score = ind["wr"].to_numpy(), ind["trend"].to_numpy(), ind["score"].to_numpy()
    n = len(ind)
    armed = False
    pos: Optional[TickerPosition] = None
    positions: list[TickerPosition] = []
    st_armed = np.zeros(n, bool)
    st_entry = np.zeros(n, bool)
    st_inpos = np.zeros(n, bool)

    for i in range(n):
        # ---- exits, evaluated on each bar after entry, in the fixed order
        if pos is not None and i > pos.entry_idx:
            if pos.scaled:
                pos.stop = max(pos.stop, l[i - 1])            # trail to previous bar's low
            if o[i] <= pos.stop:
                pos.exit_idx, pos.exit_date, pos.exit_price = i, ind.index[i], float(o[i])
                pos.reason = "Stop (gap)"
                positions.append(pos); pos = None
            elif l[i] <= pos.stop:
                pos.exit_idx, pos.exit_date, pos.exit_price = i, ind.index[i], float(pos.stop)
                pos.reason = "SL Hit" if (not pos.scaled and pos.stop <= pos.entry) else "Trail (prev-low)"
                positions.append(pos); pos = None
            elif (not pos.scaled) and h[i] >= pos.target:
                pos.scaled = True
                pos.weight = SCALE_FRACTION
                pos.stop = max(pos.stop, pos.entry)           # breakeven

        # ---- entry logic on this bar's close
        valid = i >= min_bars and not (math.isnan(wr[i]) or math.isnan(score[i]))
        if valid and wr[i] < ARM_LEVEL:
            armed = True
        if valid and pos is None and armed and wr[i] >= TRIGGER_LEVEL and bool(trend[i]):
            pos = TickerPosition(entry_idx=i, entry_date=ind.index[i], entry=float(c[i]),
                                 stop=float(c[i]) * STOP_MULT, target=float(c[i]) * TARGET_MULT,
                                 score=float(score[i]))
            armed = False
            st_entry[i] = True
        st_armed[i] = armed
        st_inpos[i] = pos is not None

    if pos is not None:
        positions.append(pos)
    states = pd.DataFrame({"armed": st_armed, "entry": st_entry, "in_pos": st_inpos}, index=ind.index)
    return positions, states


# ----------------------------------------------------------------------------- book level
@dataclass
class Position:
    id: str
    symbol: str
    entry_date: str
    entry: float
    qty: int                     # original quantity
    qty_open: int                # currently held
    stop: float
    target: float
    scaled: bool = False
    score: float = 0.0
    rank: Optional[int] = None
    status: str = "open"         # open | closed
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    reason: Optional[str] = None
    realised: float = 0.0        # realised P&L net of costs (incl. scale-out)
    scale_date: Optional[str] = None
    scale_price: Optional[float] = None
    buy_value: float = 0.0
    buy_costs: float = 0.0
    sell_costs: float = 0.0


class Book:
    """Live portfolio ledger. All state lives in a plain dict so it can be saved as JSON."""

    def __init__(self, state: dict):
        self.s = state
        self.s.setdefault("positions", [])
        self.s.setdefault("events", [])
        self.s.setdefault("history", {})
        self.s.setdefault("signals_by_month", {})
        self.s.setdefault("last_processed_date", None)
        self.s.setdefault("last_signal_month", None)
        self.s.setdefault("seq", 0)

    # ---------------- helpers
    @property
    def positions(self) -> list[dict]:
        return self.s["positions"]

    def open_positions(self) -> list[dict]:
        return [p for p in self.positions if p["status"] == "open"]

    def held_symbols(self) -> set[str]:
        return {p["symbol"] for p in self.open_positions()}

    def cash_value(self, nsei_close: float) -> float:
        return self.s["cash_units"] * nsei_close

    def _event(self, date, typ, symbol, price, qty, note=""):
        self.s["events"].append({"date": str(date)[:10], "type": typ, "symbol": symbol,
                                 "price": round(float(price), 2), "qty": int(qty), "note": note})

    def _next_id(self, symbol):
        self.s["seq"] += 1
        return f"{symbol}-{self.s['seq']:04d}"

    # ---------------- transactions
    def buy(self, date, symbol, price, qty, nsei_close, score, rank):
        v = qty * price
        cost = buy_cost(v)
        self.s["cash_units"] -= (v + cost) / nsei_close
        p = Position(id=self._next_id(symbol), symbol=symbol, entry_date=str(date)[:10], entry=float(price),
                     qty=int(qty), qty_open=int(qty), stop=float(price) * STOP_MULT,
                     target=float(price) * TARGET_MULT, score=float(score), rank=rank,
                     buy_value=v, buy_costs=cost)
        self.positions.append(asdict(p))
        self._event(date, "BUY", symbol, price, qty, f"stop {p.stop:.2f} target {p.target:.2f}")
        return p

    def sell(self, p: dict, date, price, qty, nsei_close, reason):
        v = qty * price
        cost = sell_cost(v)
        self.s["cash_units"] += (v - cost) / nsei_close
        p["realised"] += (price - p["entry"]) * qty - cost
        p["sell_costs"] += cost
        p["qty_open"] -= qty
        if p["qty_open"] <= 0:
            p["status"] = "closed"
            p["exit_date"] = str(date)[:10]
            p["exit_price"] = float(price)
            p["reason"] = reason
            # buy costs are attributed at close
            p["realised"] -= p["buy_costs"]
        self._event(date, "SELL", p["symbol"], price, qty, reason)

    # ---------------- daily exit monitoring
    def check_exits(self, date, bars: dict, nsei_close: float, new_month_prev_low: dict | None):
        """bars: {symbol: (open, high, low, close)} for this day.
        new_month_prev_low: {symbol: previous month's low} when `date` is the first trading
        day of a new month (trail update), else None."""
        fired = []
        for p in self.open_positions():
            sym = p["symbol"]
            if new_month_prev_low is not None and p["scaled"] and sym in new_month_prev_low:
                pl = new_month_prev_low[sym]
                if pl is not None and not math.isnan(pl) and pl > p["stop"]:
                    old = p["stop"]
                    p["stop"] = float(pl)
                    self._event(date, "TRAIL", sym, pl, p["qty_open"], f"stop {old:.2f} -> {pl:.2f}")
            if sym not in bars:
                continue
            o, h, l, c = bars[sym]
            if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), (o, h, l, c))):
                continue
            if p["entry_date"] >= str(date)[:10]:
                continue                                        # exits start the bar after entry
            if o <= p["stop"]:
                self.sell(p, date, o, p["qty_open"], nsei_close, "Stop (gap)")
                fired.append((p, "Stop (gap)"))
            elif l <= p["stop"]:
                reason = "SL Hit" if (not p["scaled"] and p["stop"] <= p["entry"]) else "Trail (prev-low)"
                self.sell(p, date, p["stop"], p["qty_open"], nsei_close, reason)
                fired.append((p, reason))
            elif (not p["scaled"]) and h >= p["target"]:
                q = int(round(p["qty_open"] * SCALE_FRACTION))
                if q > 0:
                    self.sell(p, date, p["target"], q, nsei_close, "Target 50% scale-out")
                p["scaled"] = True
                p["scale_date"] = str(date)[:10]
                p["scale_price"] = float(p["target"])
                old = p["stop"]
                p["stop"] = max(p["stop"], p["entry"])
                self._event(date, "SCALE", sym, p["target"], q, f"stop {old:.2f} -> {p['stop']:.2f} (breakeven)")
                fired.append((p, "Target 50% scale-out"))
        return fired

    # ---------------- month-end entries
    def take_entries(self, date, month_key: str, candidates: list[dict], closes: dict, nsei_close: float,
                     ranking: bool):
        """candidates: [{symbol, close, score, wr, ema5, ema15, ema50}] whose ticker-level entry fired
        on this completed month. closes: {symbol: close} for marking the corpus."""
        if self.s.get("last_signal_month") == month_key:
            return self.s["signals_by_month"].get(month_key, [])
        held = self.held_symbols()
        cands = [c for c in candidates if c["symbol"] not in held]
        if ranking:
            cands.sort(key=lambda c: -c["score"])
        else:
            pass                                                # first-come = universe order
        # corpus = cash + marked open positions (compounding)
        corpus = self.cash_value(nsei_close) + sum(
            p["qty_open"] * closes.get(p["symbol"], p["entry"]) for p in self.open_positions())
        size = corpus / SLOTS
        free = SLOTS - len(self.open_positions())
        records = []
        for i, c in enumerate(cands):
            rec = dict(c)
            rec.update({"rank": i + 1, "size": round(size, 2), "stop": round(c["close"] * STOP_MULT, 2),
                        "target": round(c["close"] * TARGET_MULT, 2)})
            if free <= 0:
                rec["status"] = "No slot"; rec["qty"] = 0
                records.append(rec); continue
            price = c["close"]
            qty = int(math.floor(size / price))
            cash_avail = self.cash_value(nsei_close)
            while qty > 0 and qty * price + buy_cost(qty * price) > cash_avail:
                qty -= 1
            if qty <= 0:
                rec["status"] = "Insufficient cash"; rec["qty"] = 0
                records.append(rec); continue
            p = self.buy(date, c["symbol"], price, qty, nsei_close, c["score"], i + 1)
            rec.update({"status": "Taken", "qty": qty, "position_id": p.id})
            records.append(rec)
            free -= 1
        self.s["signals_by_month"][month_key] = records
        self.s["last_signal_month"] = month_key
        return records

    # ---------------- valuation
    def nav(self, closes: dict, nsei_close: float) -> tuple[float, float, float]:
        cash = self.cash_value(nsei_close)
        inv = 0.0
        for p in self.open_positions():
            px = closes.get(p["symbol"])
            if px is None or (isinstance(px, float) and math.isnan(px)):
                px = p["entry"]
            inv += p["qty_open"] * px
        return cash + inv, cash, inv
