# Anchor & Sail — live signal dashboard

Four portfolios, one strategy (monthly Williams %R(14) + EMA 5/15/50, long only), rebuilt
automatically by GitHub Actions and published on GitHub Pages. Setup: see `SETUP.md`.

| Portfolio | Universe | Ranking | Benchmark |
|---|---|---|---|
| CORE | NIFTY 50 + NIFTY NEXT 50 | OFF | Nifty 100 |
| PRECISION | NIFTY MIDCAP 150 | ON | Nifty Midcap 150 |
| FRONTIER | NIFTY SMALLCAP 250 | OFF | Nifty Smallcap 250 |
| SPECTRUM | all three pooled, deduplicated | ON | Nifty 500 |

Each book: ₹26,00,000 committed at the **August 2026 month-end close** (fresh start), size =
corpus / 25 (compounding), max 25 positions, idle cash earns the Nifty 50 (^NSEI) return.

## What the page shows, per portfolio

1. **Signals** — exits (stop / gap / target scale-out) hit this month with TODAY flagged;
   entry signals from the last completed month-end close with rank, qty, stop, target and
   whether the book took them (or "No slot"); the open book sorted by distance to stop.
2. **Performance vs benchmark** — NAV, return, benchmark return, excess, invested/cash,
   drawdown, growth-of-100 chart, realised/unrealised P&L.
3. **Watchlist** — entry side: armed names one condition away (with a "would trigger if the
   month closed today" flag); exit side: holdings within 5% of stop or target.
4. **Universe** — every stock with latest OHLC, day change, %R(14) for the completed month and
   the running month, EMA 5/15/50, armed/trend flags, score, and a state tag.

## How the engine follows the brief

* `engine/strategy.py` — indicators (`calc_ind`), per-stock monthly state machine
  (`gen_positions`) and the live ledger (`Book`). Parameters are at the top of the file.
* Entry needs, on the same **completed** monthly close: armed (%R < −40 earlier, flag persists
  until an entry fires) → %R ≥ −20 → EMA5 > EMA15 > EMA50. Entry price = that close.
* Exits are checked in the brief's order on every run using **daily** bars (the brief requires
  daily stop monitoring): open ≤ stop → out at open ("Stop (gap)"); low ≤ stop → out at stop
  ("SL Hit" / "Trail (prev-low)"); not scaled and high ≥ target → sell 50% at target, stop to
  breakeven. Once scaled, at the start of each new month the stop trails to the previous
  month's low.
* Costs (TXN, STT, stamp, SEBI, GST, DP) are charged inside the ledger; ≈0.25% round trip.
* The running month is **never** used for signal generation — entry signals change once a
  month; the open book and stops change daily.

## Interpretation choices (not fully pinned down by the brief)

* **Per-stock cycle.** `gen_positions` runs the strategy for each stock independently from
  2015; a stock whose per-stock cycle is already "in a position" (entered before this book
  started) does not produce a fresh entry until that cycle exits — shown as **IN CYCLE**. This
  mirrors how the reference backtest generates candidates.
* **First-come when ranking is OFF** = NSE constituent-list order.
* **Quantities** are whole shares (floor of size / price); a slot that cannot afford one share
  is reported as "Insufficient cash".
* **Cash leg** is held as Nifty 50 units (mark-to-market daily), which equals applying the
  ^NSEI return to cash.
* **Same-bar re-entry**: a stock may re-enter on the same monthly bar its previous per-stock
  position exited if all three conditions hold.

## Data notes (from the brief's findings)

* Prices: Yahoo Finance via yfinance, adjusted. Bulk downloads are chunked, retried
  per-symbol, and reconciled against the previous run's cache; anything served from cache or
  missing is listed in the warnings banner.
* Universes: NSE's official constituent CSVs, refreshed every run; if NSE is unreachable the
  last good snapshot in `data/universe/` is used **and a warning is shown**. Overlaps between
  lists are reported.
* Benchmarks: `^CNX100`, `NIFTYMIDCAP150.NS`, `NIFTYSMLCAP250.NS`, `^CRSLDX`; labelled ETF
  proxies only as a visible fallback; an official CSV in `data/benchmarks/` overrides both.
* Survivorship bias: the universe is today's members applied to the warm-up history.
* Yahoo's NSE feed is delayed about 15 minutes; intraday runs are for stop monitoring, the
  18:00 IST run is the day's final word.

## Files

```
engine/strategy.py   strategy + ledger            docs/index.html   the dashboard
engine/build.py      orchestrator (writes docs/data.json)   docs/data.json   generated each run
engine/data.py       yfinance download / reconcile / benchmarks
engine/universes.py  NSE constituent lists + snapshot fallback
data/state/*.json    the four ledgers (committed by the workflow)
data/universe/       last good constituent lists
tests/test_engine.py unit checks of every rule
.github/workflows/daily_dashboard.yml   the schedule
```
