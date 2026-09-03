"""Unit checks for the strategy engine against hand-built monthly series. Run: python -m pytest -q tests"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
import strategy as S  # noqa: E402


def bars(closes, highs=None, lows=None, opens=None):
    closes = np.asarray(closes, float)
    highs = closes * 1.01 if highs is None else np.asarray(highs, float)
    lows = closes * 0.99 if lows is None else np.asarray(lows, float)
    opens = closes if opens is None else np.asarray(opens, float)
    idx = pd.date_range("2015-01-31", periods=len(closes), freq="ME")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def test_indicators_match_formulas():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 80)))
    m = bars(c)
    ind = S.calc_ind(m)
    i = 60
    hh = m["High"].iloc[i - 13:i + 1].max(); ll = m["Low"].iloc[i - 13:i + 1].min()
    assert math.isclose(ind["wr"].iloc[i], 100 * (c[i] - hh) / (hh - ll), rel_tol=1e-12)
    # EMA with adjust=False: e_t = a*c_t + (1-a)*e_{t-1}, e_0 = c_0
    a = 2 / 51; e = c[0]
    for k in range(1, i + 1):
        e = a * c[k] + (1 - a) * e
    assert math.isclose(ind["ema50"].iloc[i], e, rel_tol=1e-12)
    assert math.isclose(ind["score"].iloc[i], (ind["ema5"].iloc[i] - ind["ema50"].iloc[i]) / ind["ema50"].iloc[i] * 100)


def test_entry_requires_arm_then_trigger_and_trend():
    # 20 flat bars, then a dip (arms), then a recovery to new highs (trend + %R>=-20)
    c = [100] * 20 + [95, 90, 80, 70] + [80, 92, 101, 105, 110, 115]
    m = bars(c)
    ind = S.calc_ind(m)
    pos, st = S.gen_positions(ind, min_bars=14)
    # armed at first bar where %R < -40
    first_armed = st.index[st["armed"]].min()
    assert ind.loc[first_armed, "wr"] < -40
    assert len(pos) == 1
    e = pos[0]
    assert ind.loc[e.entry_date, "wr"] >= -20 and ind.loc[e.entry_date, "trend"]
    assert math.isclose(e.stop, e.entry * 0.85) and math.isclose(e.target, e.entry * 1.30)
    # armed reset after the entry
    assert not st.loc[e.entry_date, "armed"]


def test_no_entry_without_prior_arm():
    c = list(np.linspace(100, 200, 40))       # steady uptrend, %R never below -40
    ind = S.calc_ind(bars(c))
    pos, st = S.gen_positions(ind, min_bars=14)
    assert pos == [] and not st["armed"].any()


def _entered_series():
    """Series that produces one entry at bar 29 (close 110). Following bars are appended by tests."""
    return [100] * 20 + [95, 90, 80, 70] + [80, 92, 101, 105, 110, 115]


def test_gap_stop_exits_at_open():
    c = _entered_series()
    n0 = len(c)
    c2 = c + [90]
    o = list(c2); o[n0] = 80          # next bar opens below the stop (115*0.85 = 97.75)
    m = bars(c2, opens=o, highs=np.array(c2) * 1.01, lows=np.array(c2) * 0.99)
    pos, _ = S.gen_positions(S.calc_ind(m), min_bars=14)
    p = pos[0]
    assert p.reason == "Stop (gap)" and p.exit_price == 80


def test_stop_hit_intra_bar_exits_at_stop():
    c = _entered_series()
    n0 = len(c)
    c2 = c + [100]
    lows = np.array(c2) * 0.99; lows[n0] = 96        # low pierces 97.75, open above
    m = bars(c2, lows=lows)
    pos, _ = S.gen_positions(S.calc_ind(m), min_bars=14)
    p = pos[0]
    assert p.reason == "SL Hit" and math.isclose(p.exit_price, 115 * 0.85)


def test_target_scale_then_breakeven_and_trail():
    c = _entered_series()                    # entry 115 at last bar; target 149.5
    n0 = len(c)
    c2 = c + [140, 150, 145, 100]
    highs = np.array(c2) * 1.01; highs[n0 + 1] = 151      # target hit on bar n0+1
    lows = np.array(c2) * 0.99
    lows[n0 + 1] = 142                                    # low of the scaling bar -> trail level next bar
    lows[n0 + 2] = 143                                    # above 142, no exit
    lows[n0 + 3] = 90                                     # trail (max(142, prev low 143) = 143) hit
    opens = np.array(c2, float); opens[n0 + 3] = 144      # open above the trailed stop -> exit at stop
    m = bars(c2, highs=highs, lows=lows, opens=opens)
    pos, _ = S.gen_positions(S.calc_ind(m), min_bars=14)
    p = pos[0]
    assert p.scaled and p.weight == 0.5
    assert p.reason == "Trail (prev-low)" and math.isclose(p.exit_price, 143)


def test_book_sizing_costs_and_slots():
    state = {"portfolio": "T", "inception": "2026-08-31", "capital": S.CAPITAL, "cash_units": S.CAPITAL / 25000.0}
    b = S.Book(state)
    cands = [{"symbol": f"S{i}", "close": 1000.0, "score": float(i), "wr": -10, "ema5": 1, "ema15": 1, "ema50": 1}
             for i in range(30)]
    recs = b.take_entries(pd.Timestamp("2026-08-31"), "2026-08", cands, {}, 25000.0, ranking=True)
    taken = [r for r in recs if r["status"] == "Taken"]
    assert len(taken) == 25 and len(b.open_positions()) == 25
    assert [r["symbol"] for r in recs[:3]] == ["S29", "S28", "S27"]          # ranking ON: highest score first
    assert taken[0]["qty"] == math.floor(S.CAPITAL / 25 / 1000.0)             # corpus/25 sizing
    v = taken[0]["qty"] * 1000.0
    spent = sum(p["buy_value"] + p["buy_costs"] for p in b.positions)
    assert math.isclose(S.CAPITAL - b.cash_value(25000.0), spent, rel_tol=1e-9)
    assert math.isclose(b.positions[0]["buy_costs"], S.buy_cost(v))
    # first-come ordering when ranking is OFF
    state2 = {"portfolio": "T", "inception": "2026-08-31", "capital": S.CAPITAL, "cash_units": S.CAPITAL / 25000.0}
    b2 = S.Book(state2)
    recs2 = b2.take_entries(pd.Timestamp("2026-08-31"), "2026-08", cands, {}, 25000.0, ranking=False)
    assert [r["symbol"] for r in recs2[:3]] == ["S0", "S1", "S2"]
    # no double-processing of the same month
    assert b.take_entries(pd.Timestamp("2026-08-31"), "2026-08", cands, {}, 25000.0, ranking=True) is recs or \
        len(b.open_positions()) == 25


def test_book_daily_exit_and_scale():
    state = {"portfolio": "T", "inception": "2026-08-31", "capital": S.CAPITAL, "cash_units": S.CAPITAL / 25000.0}
    b = S.Book(state)
    b.buy(pd.Timestamp("2026-08-31"), "AAA", 100.0, 1000, 25000.0, 1.0, 1)
    b.buy(pd.Timestamp("2026-08-31"), "BBB", 100.0, 1000, 25000.0, 1.0, 2)
    # day 1: AAA gaps below stop (85) -> exit at open; BBB hits target 130 -> scale 50%, stop -> 100
    fired = b.check_exits(pd.Timestamp("2026-09-01"), {"AAA": (80, 90, 79, 88), "BBB": (120, 131, 118, 128)}, 25000.0, None)
    reasons = {p["symbol"]: r for p, r in fired}
    assert reasons["AAA"] == "Stop (gap)" and reasons["BBB"] == "Target 50% scale-out"
    aaa = next(p for p in b.positions if p["symbol"] == "AAA")
    assert aaa["status"] == "closed" and aaa["exit_price"] == 80
    bbb = next(p for p in b.positions if p["symbol"] == "BBB")
    assert bbb["scaled"] and bbb["qty_open"] == 500 and bbb["stop"] == 100.0
    # new month: trail to previous month's low (110) then breach -> "Trail (prev-low)" at 110
    b.check_exits(pd.Timestamp("2026-10-01"), {"BBB": (115, 116, 105, 106)}, 25000.0, {"BBB": 110.0})
    assert bbb["status"] == "closed" and bbb["reason"] == "Trail (prev-low)" and bbb["exit_price"] == 110.0
    # realised P&L: AAA loses 20/share + costs; BBB gains 30*500 + 10*500 - costs
    assert aaa["realised"] < -20000 and bbb["realised"] > 19000


def test_cost_model_round_trip_about_25bp():
    v = 1_00_000.0
    rt = (S.buy_cost(v) + S.sell_cost(v)) / v * 100
    assert 0.22 < rt < 0.28
