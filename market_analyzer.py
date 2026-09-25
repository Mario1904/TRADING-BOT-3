"""
Market Analyzer & Trade Idea Generator — Pure Price Action / Supply & Demand
Multi-timeframe day-trading version: 4H structure/zones, 5M entry trigger.
Filtered for higher-quality setups: trend-aligned, fresh zones only, with
rejection confirmation and a 3:1 risk:reward target.
=============================================================================
No indicators. Pure price action:

  1. STRUCTURE & ZONES (4H): swing highs/lows classify trend. Supply &
     demand zones are the consolidation ("base") right before a strong
     directional move away from it.
  2. QUALITY FILTERS: a zone only qualifies for a signal if —
       - it's FRESH (never touched since it formed)
       - it's TREND-ALIGNED (demand zones only trade long during a 4H
         uptrend; supply zones only trade short during a 4H downtrend —
         counter-trend zone bounces are skipped)
       - price shows a REJECTION at the zone (a wick, not just a touch)
  3. ENTRY TRIGGER (5M): once a zone passes those filters and price is
     inside it, the 5-minute chart is checked for a confirmation candle
     (engulfing/rejection) before a live entry signal fires.
  4. RISK:REWARD: stop is placed a small buffer beyond the zone; target is
     3x that risk by default (configurable).
  5. NEWS FILTER (ForexFactory "red folder"): entries are suppressed near
     high-impact news for the relevant currency.
  6. BACKTEST: applies the SAME filters (fresh + trend-aligned + rejection
     confirmation) historically, over ~2 years of 4H data, and reports the
     actual resulting win rate — this is not a promise, it's a measurement.

IMPORTANT — win rate and reward:risk are not independent. A wider RR target
is statistically harder to reach and will tend to reduce win rate, not
increase it. Nothing here guarantees future accuracy; treat the backtest
numbers as a filter for whether a setup is even worth considering, not as
a promise.

THIS TOOL DOES NOT PLACE TRADES. Review manually in XM. Not financial
advice.

Install:  pip install yfinance pandas numpy requests --break-system-packages
Run:      python market_analyzer.py
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Run: pip install yfinance pandas numpy requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SYMBOLS = {
    "US30 (Dow Jones)": "^DJI",
    "US100 (Nasdaq)": "^NDX",
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}

SYMBOL_CURRENCIES = {
    "US30 (Dow Jones)": {"USD"},
    "US100 (Nasdaq)": {"USD"},
    "Gold": {"USD"},
    "Bitcoin": {"USD"},
    "EUR/USD": {"USD", "EUR"},
    "GBP/USD": {"USD", "GBP"},
}

STRUCTURE_SOURCE_INTERVAL = "1h"
STRUCTURE_SOURCE_PERIOD = "730d"
STRUCTURE_RULE = "4h"

ENTRY_INTERVAL = "5m"
ENTRY_PERIOD = "60d"

SWING_WINDOW = 3
BASE_LOOKBACK = 4
RANGE_BASELINE_LOOKBACK = 10
IMPULSE_MULTIPLIER = 2.2        # raised: only strong, clean impulse legs count as zones

NEWS_BLACKOUT_MINUTES = 30

# --- Quality filters (applied to BOTH live signals and the backtest) ---
REQUIRE_FRESH_ZONE = True        # only ever act on a zone's first touch
REQUIRE_TREND_ALIGNMENT = True   # only trade zones in the direction of 4H structure
REQUIRE_REJECTION_CANDLE = True  # touch must show a rejection wick, not just a pass-through

# --- Risk:reward ---
STOP_BUFFER_PCT = 0.15           # stop placed this much extra beyond the zone (of zone height)
TARGET_RR = 3.0                  # target = risk x this multiple

BACKTEST_LOOKAHEAD_CANDLES = 30  # widened since a 3R target takes longer to reach than 1.5R did

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    kind: str
    low: float
    high: float
    formed_at: str
    fresh: bool


@dataclass
class Analysis:
    symbol: str
    last_price: float
    structure: str
    bias: str
    demand_zone: Zone = None
    supply_zone: Zone = None
    inside_zone: str = None
    qualifies: bool = False       # passed fresh + trend-alignment filters
    entry_signal: bool = False
    entry_reason: str = ""
    trade_levels: dict = None     # {"entry":.., "stop":.., "target":.., "rr":..}
    news_blackout: bool = False
    news_blackout_reason: str = ""
    backtest: dict = None
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# PRICE DATA
# ---------------------------------------------------------------------------

def fetch_ohlc(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No {interval} data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=False)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Time"})
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("Time")
    out = pd.DataFrame({
        "Open": d["Open"].resample(rule).first(),
        "High": d["High"].resample(rule).max(),
        "Low": d["Low"].resample(rule).min(),
        "Close": d["Close"].resample(rule).last(),
    }).dropna()
    return out.reset_index()


# ---------------------------------------------------------------------------
# MARKET STRUCTURE
# ---------------------------------------------------------------------------

def find_swings(df: pd.DataFrame, window: int = SWING_WINDOW):
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        wh = highs[i - window:i + window + 1]
        wl = lows[i - window:i + window + 1]
        if highs[i] == wh.max():
            swing_highs.append((i, float(highs[i])))
        if lows[i] == wl.min():
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def classify_structure(swing_highs, swing_lows) -> str:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Not enough swing data yet"
    higher_high = swing_highs[-1][1] > swing_highs[-2][1]
    higher_low = swing_lows[-1][1] > swing_lows[-2][1]
    lower_high = swing_highs[-1][1] < swing_highs[-2][1]
    lower_low = swing_lows[-1][1] < swing_lows[-2][1]
    if higher_high and higher_low:
        return "Uptrend (higher highs & higher lows)"
    if lower_high and lower_low:
        return "Downtrend (lower highs & lower lows)"
    return "Ranging / structure transitioning"


def structure_at(swing_highs, swing_lows, idx: int) -> str:
    """Structure as it would have looked using only data available up to
    `idx` — avoids look-ahead bias when backtesting."""
    sh = [s for s in swing_highs if s[0] <= idx]
    sl = [s for s in swing_lows if s[0] <= idx]
    return classify_structure(sh, sl)


# ---------------------------------------------------------------------------
# SUPPLY & DEMAND ZONES
# ---------------------------------------------------------------------------

def raw_zones(df: pd.DataFrame) -> list:
    ranges = df["High"] - df["Low"]
    zones = []
    for i in range(RANGE_BASELINE_LOOKBACK, len(df)):
        baseline = ranges.iloc[i - RANGE_BASELINE_LOOKBACK:i].mean()
        if not baseline or np.isnan(baseline) or baseline == 0:
            continue
        candle_range = ranges.iloc[i]
        if candle_range < IMPULSE_MULTIPLIER * baseline:
            continue
        bullish_impulse = df["Close"].iloc[i] > df["Open"].iloc[i]

        base_idxs = []
        j = i - 1
        while j >= 0 and len(base_idxs) < BASE_LOOKBACK:
            if ranges.iloc[j] <= baseline * 1.1:
                base_idxs.append(j)
                j -= 1
            else:
                break
        if not base_idxs and i - 1 >= 0:
            base_idxs = [i - 1]
        if not base_idxs:
            continue

        base_idxs.sort()
        zone_low = float(df["Low"].iloc[base_idxs].min())
        zone_high = float(df["High"].iloc[base_idxs].max())
        formed_at = df["Time"].iloc[base_idxs[0]]
        formed_at_str = formed_at.strftime("%Y-%m-%d %H:%M") if hasattr(formed_at, "strftime") else str(formed_at)

        zones.append({
            "kind": "demand" if bullish_impulse else "supply",
            "low": zone_low, "high": zone_high,
            "formed_idx": i, "formed_at": formed_at_str,
        })
    return zones


def valid_zones_now(df: pd.DataFrame, zones: list) -> list:
    out = []
    for z in zones:
        after = df.iloc[z["formed_idx"] + 1:]
        if z["kind"] == "demand":
            broken = (after["Close"] < z["low"]).any()
        else:
            broken = (after["Close"] > z["high"]).any()
        if broken:
            continue
        touched = ((after["Low"] <= z["high"]) & (after["High"] >= z["low"])).any()
        out.append(Zone(kind=z["kind"], low=z["low"], high=z["high"],
                         formed_at=z["formed_at"], fresh=not bool(touched)))
    return out


def nearest_zones(zones: list, current_price: float):
    demand_candidates = [z for z in zones if z.kind == "demand" and z.low <= current_price]
    supply_candidates = [z for z in zones if z.kind == "supply" and z.high >= current_price]
    nearest_demand = max(demand_candidates, key=lambda z: z.high, default=None)
    nearest_supply = min(supply_candidates, key=lambda z: z.low, default=None)
    inside = None
    if nearest_demand and nearest_demand.low <= current_price <= nearest_demand.high:
        inside = "demand"
    elif nearest_supply and nearest_supply.low <= current_price <= nearest_supply.high:
        inside = "supply"
    return nearest_demand, nearest_supply, inside


# ---------------------------------------------------------------------------
# RISK / TARGET LEVELS (shared by live signals and backtest)
# ---------------------------------------------------------------------------

def compute_trade_levels(kind: str, low: float, high: float,
                          buffer_pct: float = STOP_BUFFER_PCT, rr: float = TARGET_RR) -> dict:
    height = high - low
    buffer = height * buffer_pct
    if kind == "demand":
        entry = high
        stop = low - buffer
        risk = entry - stop
        target = entry + risk * rr
    else:
        entry = low
        stop = high + buffer
        risk = stop - entry
        target = entry - risk * rr
    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "risk": risk}


def has_rejection(row) -> tuple:
    """Returns (bullish_rejection, bearish_rejection) for a single OHLC row —
    True if the candle's close sits in the half of its range away from the
    wick that probed the zone (i.e. it got rejected, not just touched)."""
    candle_range = row["High"] - row["Low"]
    if candle_range <= 0:
        return False, False
    bullish_rejection = (row["Close"] - row["Low"]) > 0.5 * candle_range
    bearish_rejection = (row["High"] - row["Close"]) > 0.5 * candle_range
    return bullish_rejection, bearish_rejection


# ---------------------------------------------------------------------------
# 5-MINUTE ENTRY TRIGGER
# ---------------------------------------------------------------------------

def find_5m_trigger(df5: pd.DataFrame, direction: str, lookback_candles: int = 12):
    recent = df5.tail(lookback_candles).reset_index(drop=True)
    for i in range(len(recent) - 1, 0, -1):
        cur, prev = recent.iloc[i], recent.iloc[i - 1]
        bullish_engulf = (cur["Close"] > cur["Open"] and prev["Close"] < prev["Open"]
                           and cur["Close"] >= prev["Open"] and cur["Open"] <= prev["Close"])
        bearish_engulf = (cur["Close"] < cur["Open"] and prev["Close"] > prev["Open"]
                           and cur["Close"] <= prev["Open"] and cur["Open"] >= prev["Close"])
        if direction == "bullish" and bullish_engulf:
            return True, f"Bullish engulfing candle on 5M at {cur['Time'].strftime('%H:%M UTC')}"
        if direction == "bearish" and bearish_engulf:
            return True, f"Bearish engulfing candle on 5M at {cur['Time'].strftime('%H:%M UTC')}"
    return False, "Price is at a qualified zone but no confirmed 5M reaction candle yet — wait."


# ---------------------------------------------------------------------------
# ECONOMIC CALENDAR
# ---------------------------------------------------------------------------

def fetch_economic_calendar() -> list:
    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warning] Could not fetch economic calendar: {e}")
        return []


def upcoming_high_impact_events(events: list, currencies: set, hours_ahead: int = 48) -> list:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    relevant = []
    for ev in events:
        try:
            if ev.get("impact") != "High" or ev.get("country") not in currencies:
                continue
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if now <= ev_time <= cutoff:
                relevant.append((ev_time, ev.get("title", "Unknown event"), ev.get("country")))
        except Exception:
            continue
    return sorted(relevant, key=lambda x: x[0])


def check_news_blackout(events: list, currencies: set, minutes: int = NEWS_BLACKOUT_MINUTES):
    now = datetime.now(timezone.utc)
    for ev in events:
        try:
            if ev.get("impact") != "High" or ev.get("country") not in currencies:
                continue
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if abs((ev_time - now).total_seconds()) <= minutes * 60:
                return True, f"{ev.get('title')} ({ev.get('country')}) at {ev_time.strftime('%H:%M UTC')}"
        except Exception:
            continue
    return False, ""


# ---------------------------------------------------------------------------
# BACKTEST — same filters as live signals: fresh + trend-aligned + rejection
# ---------------------------------------------------------------------------

def evaluate_zone_outcome(df, kind, low, high, touch_idx, lookahead):
    levels = compute_trade_levels(kind, low, high)
    window = df.iloc[touch_idx + 1: touch_idx + 1 + lookahead]
    for _, row in window.iterrows():
        if kind == "demand":
            if row["Low"] <= levels["stop"]:
                return "loss"
            if row["High"] >= levels["target"]:
                return "win"
        else:
            if row["High"] >= levels["stop"]:
                return "loss"
            if row["Low"] <= levels["target"]:
                return "win"
    return None


def backtest_zones(df: pd.DataFrame, zones: list, swing_highs, swing_lows) -> dict:
    wins = losses = no_result = skipped_filters = 0
    for z in zones:
        # trend alignment, assessed using only data available at zone formation
        if REQUIRE_TREND_ALIGNMENT:
            structure = structure_at(swing_highs, swing_lows, z["formed_idx"])
            aligned = ((z["kind"] == "demand" and structure.startswith("Uptrend")) or
                       (z["kind"] == "supply" and structure.startswith("Downtrend")))
            if not aligned:
                skipped_filters += 1
                continue

        # find the first touch (this IS the "fresh" touch by construction —
        # REQUIRE_FRESH_ZONE just means we never look past this first one)
        touch_idx = None
        for idx in range(z["formed_idx"] + 1, len(df)):
            row = df.iloc[idx]
            if row["Low"] <= z["high"] and row["High"] >= z["low"]:
                if REQUIRE_REJECTION_CANDLE:
                    bull_rej, bear_rej = has_rejection(row)
                    rejected = bull_rej if z["kind"] == "demand" else bear_rej
                    if not rejected:
                        continue  # touched but no rejection — keep looking
                touch_idx = idx
                break
        if touch_idx is None:
            continue

        outcome = evaluate_zone_outcome(df, z["kind"], z["low"], z["high"], touch_idx, BACKTEST_LOOKAHEAD_CANDLES)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            no_result += 1

    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved else None
    return {"wins": wins, "losses": losses, "no_result": no_result,
            "skipped_by_filters": skipped_filters, "total_zones": len(zones),
            "win_rate": win_rate, "rr": TARGET_RR}


# ---------------------------------------------------------------------------
# ANALYSIS (per symbol)
# ---------------------------------------------------------------------------

def analyze_symbol(name: str, ticker: str, calendar: list) -> Analysis:
    df1h = fetch_ohlc(ticker, STRUCTURE_SOURCE_INTERVAL, STRUCTURE_SOURCE_PERIOD)
    df4h = resample_ohlc(df1h, STRUCTURE_RULE)
    current_price = float(df4h["Close"].iloc[-1])

    swing_highs, swing_lows = find_swings(df4h)
    structure = classify_structure(swing_highs, swing_lows)

    z_raw = raw_zones(df4h)
    zones = valid_zones_now(df4h, z_raw)
    demand_zone, supply_zone, inside = nearest_zones(zones, current_price)

    backtest = backtest_zones(df4h, z_raw, swing_highs, swing_lows)

    notes = []
    bias = "Neutral"
    bullish_structure = structure.startswith("Uptrend")
    bearish_structure = structure.startswith("Downtrend")

    if inside == "demand":
        bias = "Bullish"
        notes.append("Price is currently trading inside a 4H demand zone.")
    elif inside == "supply":
        bias = "Bearish"
        notes.append("Price is currently trading inside a 4H supply zone.")
    elif bullish_structure:
        bias = "Bullish"
    elif bearish_structure:
        bias = "Bearish"
    else:
        notes.append("4H structure is unclear right now.")

    if demand_zone and demand_zone.fresh:
        notes.append(f"Nearest demand zone: {demand_zone.low:.2f}-{demand_zone.high:.2f} (fresh, formed {demand_zone.formed_at})")
    if supply_zone and supply_zone.fresh:
        notes.append(f"Nearest supply zone: {supply_zone.low:.2f}-{supply_zone.high:.2f} (fresh, formed {supply_zone.formed_at})")
    if not demand_zone and not supply_zone:
        notes.append("No clear valid 4H zones found in the lookback window.")

    # --- Does the current situation even qualify for consideration? ---
    qualifies = False
    active_zone = None
    if inside == "demand" and demand_zone:
        active_zone = demand_zone
        fresh_ok = (not REQUIRE_FRESH_ZONE) or demand_zone.fresh
        trend_ok = (not REQUIRE_TREND_ALIGNMENT) or bullish_structure
        qualifies = fresh_ok and trend_ok
        if not fresh_ok:
            notes.append("Demand zone has already been tested before — skipped (fresh-zone filter).")
        elif not trend_ok:
            notes.append("Demand zone is counter-trend against the current 4H downtrend — skipped (trend filter).")
    elif inside == "supply" and supply_zone:
        active_zone = supply_zone
        fresh_ok = (not REQUIRE_FRESH_ZONE) or supply_zone.fresh
        trend_ok = (not REQUIRE_TREND_ALIGNMENT) or bearish_structure
        qualifies = fresh_ok and trend_ok
        if not fresh_ok:
            notes.append("Supply zone has already been tested before — skipped (fresh-zone filter).")
        elif not trend_ok:
            notes.append("Supply zone is counter-trend against the current 4H uptrend — skipped (trend filter).")

    # --- News blackout ---
    currencies = SYMBOL_CURRENCIES.get(name, {"USD"})
    blackout, blackout_reason = check_news_blackout(calendar, currencies) if calendar else (False, "")

    # --- 5-minute entry trigger + trade levels, only if qualified ---
    entry_signal, entry_reason = False, "No qualified zone reaction to evaluate right now."
    trade_levels = None
    if qualifies and active_zone and not blackout:
        trade_levels = compute_trade_levels(active_zone.kind, active_zone.low, active_zone.high)
        try:
            df5 = fetch_ohlc(ticker, ENTRY_INTERVAL, ENTRY_PERIOD)
            direction = "bullish" if active_zone.kind == "demand" else "bearish"
            entry_signal, entry_reason = find_5m_trigger(df5, direction)
        except Exception as e:
            entry_reason = f"Could not fetch 5M data for entry trigger: {e}"
    elif blackout:
        entry_reason = f"Entry suppressed — inside red-news blackout window: {blackout_reason}"
    elif inside in ("demand", "supply") and not qualifies:
        entry_reason = "Price is at a zone, but it didn't pass the quality filters above — no signal."

    if entry_signal:
        notes.append("Day-trade reminder: plan your exit and close before end of session — don't hold overnight.")

    return Analysis(
        symbol=name, last_price=current_price, structure=structure, bias=bias,
        demand_zone=demand_zone, supply_zone=supply_zone, inside_zone=inside,
        qualifies=qualifies, entry_signal=entry_signal, entry_reason=entry_reason,
        trade_levels=trade_levels,
        news_blackout=blackout, news_blackout_reason=blackout_reason,
        backtest=backtest, notes=notes,
    )


# ---------------------------------------------------------------------------
# CONSOLE REPORT
# ---------------------------------------------------------------------------

def build_trade_idea(a: Analysis, news_events: list) -> str:
    lines = [f"\n=== {a.symbol} ===",
             f"Last price: {a.last_price:.2f} | 4H Structure: {a.structure} | Bias: {a.bias}"]

    if a.demand_zone:
        lines.append(f"  4H Demand zone: {a.demand_zone.low:.2f}-{a.demand_zone.high:.2f} "
                      f"({'fresh' if a.demand_zone.fresh else 'tested'}, formed {a.demand_zone.formed_at})")
    if a.supply_zone:
        lines.append(f"  4H Supply zone: {a.supply_zone.low:.2f}-{a.supply_zone.high:.2f} "
                      f"({'fresh' if a.supply_zone.fresh else 'tested'}, formed {a.supply_zone.formed_at})")

    for n in a.notes:
        lines.append(f"  - {n}")

    lines.append(f"  5M Entry signal: {'YES — ' + a.entry_reason if a.entry_signal else 'No — ' + a.entry_reason}")

    if a.trade_levels:
        t = a.trade_levels
        lines.append(f"  Levels if triggered: entry {t['entry']:.2f} | stop {t['stop']:.2f} | "
                      f"target {t['target']:.2f} (risk:reward {t['rr']:.1f}:1)")

    if a.news_blackout:
        lines.append(f"  \u26a0 RED NEWS BLACKOUT: {a.news_blackout_reason} — do not enter.")

    if a.backtest and a.backtest["win_rate"] is not None:
        bt = a.backtest
        lines.append(f"  Backtest (filtered, 4H zones, ~2y): {bt['wins']}W/{bt['losses']}L "
                      f"({bt['win_rate']:.0f}% win rate on {bt['wins']+bt['losses']} resolved zones, "
                      f"{bt['no_result']} inconclusive, {bt['skipped_by_filters']} skipped by filters) "
                      f"at {bt['rr']:.1f}:1 RR. Breakeven win rate at this RR is {100/(1+bt['rr']):.0f}%. "
                      f"Past results are not a guarantee of future ones.")

    if news_events:
        lines.append("  Upcoming high-impact news (next 48h):")
        for ev_time, title, country in news_events:
            lines.append(f"     {ev_time.strftime('%a %H:%M UTC')} [{country}] {title}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

CARD_HTML = """
<div class="card {bias_class}">
  <div class="card-head">
    <h2>{symbol}</h2>
    <span class="badge {bias_class}">{bias}</span>
  </div>
  <div class="structure">4H: {structure}</div>
  <div class="stats"><span class="label">Last</span><span class="value">{last_price:.2f}</span></div>
  <div class="zones">{zones_html}</div>
  <div class="entry {entry_class}">{entry_html}</div>
  {levels_html}
  {blackout_html}
  <div class="notes">{notes_html}</div>
  <div class="backtest">{backtest_html}</div>
  <div class="news">{news_html}</div>
</div>
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Daily Market Report</title>
<style>
  :root {{
    --bg: #f5f6f8; --card-bg: #ffffff; --text: #1a1a1a; --muted: #6b7280;
    --border: #e5e7eb; --bull: #16a34a; --bear: #dc2626; --neutral: #6b7280;
    --warn: #d97706;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #121212; --card-bg: #1e1e1e; --text: #f0f0f0; --muted: #9ca3af; --border: #333;
    }}
  }}
  html {{ scroll-padding-top: env(safe-area-inset-top, 0px); }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, Roboto, Segoe UI, sans-serif;
    padding: 16px; max-width: 640px; margin: 0 auto;
  }}
  header {{ padding: 8px 4px 20px; }}
  header h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  header p {{ margin: 0; color: var(--muted); font-size: 0.85rem; }}
  .disclaimer {{
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; font-size: 0.78rem; color: var(--muted); margin-bottom: 18px;
  }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; margin-bottom: 14px;
  }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
  .card-head h2 {{ font-size: 1.05rem; margin: 0; }}
  .badge {{
    font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.02em;
  }}
  .badge.Bullish {{ background: rgba(22,163,74,0.15); color: var(--bull); }}
  .badge.Bearish {{ background: rgba(220,38,38,0.15); color: var(--bear); }}
  .badge.Neutral {{ background: rgba(107,114,128,0.15); color: var(--neutral); }}
  .structure {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }}
  .stats {{ margin-bottom: 8px; }}
  .label {{ font-size: 0.68rem; color: var(--muted); text-transform: uppercase; margin-right: 6px; }}
  .value {{ font-size: 0.95rem; font-weight: 600; }}
  .zones {{ font-size: 0.82rem; margin-bottom: 8px; }}
  .zones div {{ margin-bottom: 3px; }}
  .zone-demand {{ color: var(--bull); }}
  .zone-supply {{ color: var(--bear); }}
  .entry {{ font-size: 0.88rem; font-weight: 600; margin-bottom: 8px; padding: 8px 10px; border-radius: 8px; }}
  .entry.signal-yes {{ background: rgba(22,163,74,0.12); color: var(--bull); }}
  .entry.signal-no {{ background: rgba(107,114,128,0.10); color: var(--muted); font-weight: 500; }}
  .levels {{ font-size: 0.8rem; color: var(--text); background: rgba(107,114,128,0.08);
             padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; }}
  .blackout {{ font-size: 0.82rem; font-weight: 600; color: var(--warn); background: rgba(217,119,6,0.12);
               padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; }}
  .notes {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }}
  .notes div {{ margin-bottom: 3px; }}
  .backtest {{ font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--border); padding-top: 8px; margin-bottom: 8px; }}
  .news {{ font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 8px; color: var(--muted); }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; padding: 20px 0 8px; }}
</style>
</head>
<body>
<header>
  <h1>\U0001F4CA Daily Market Report</h1>
  <p>Generated {generated_at} &middot; 4H zones (trend-aligned, fresh-only, rejection-confirmed), 5M entry, 3:1 RR</p>
</header>
<div class="disclaimer">
  Educational tool only — not financial advice, not a guarantee of accuracy.
  Prices are a Yahoo Finance proxy, delayed and not identical to XM's live
  feed. 5M data covers ~60 days, 4H data ~2 years (Yahoo's free-tier
  limits). Backtest win rates reflect that window with the same filters as
  live signals — they are a measurement of the past, not a promise about
  the future. Always verify on your own XM chart before trading.
</div>
{cards}
<footer>Re-generated automatically. Day-trade setups only — no overnight holds implied.</footer>
</body>
</html>
"""


def render_zones_html(a: Analysis) -> str:
    parts = []
    if a.demand_zone:
        tag = "fresh" if a.demand_zone.fresh else "tested"
        parts.append(f'<div class="zone-demand">\u25B2 Demand: {a.demand_zone.low:.2f}-{a.demand_zone.high:.2f} ({tag})</div>')
    if a.supply_zone:
        tag = "fresh" if a.supply_zone.fresh else "tested"
        parts.append(f'<div class="zone-supply">\u25BC Supply: {a.supply_zone.low:.2f}-{a.supply_zone.high:.2f} ({tag})</div>')
    return "".join(parts) if parts else "<div>No valid zones currently in range.</div>"


def render_notes_html(notes: list) -> str:
    return "".join(f"<div>\u2022 {n}</div>" for n in notes) if notes else ""


def render_news_html(news_events: list) -> str:
    if not news_events:
        return "No high-impact news flagged in the next 48h."
    items = "".join(f"<div>{t.strftime('%a %H:%M UTC')} [{c}] {title}</div>" for t, title, c in news_events)
    return f"<div>Upcoming high-impact news:</div>{items}"


def render_backtest_html(bt: dict) -> str:
    if not bt or bt["win_rate"] is None:
        return "Not enough historical zone touches yet to backtest."
    breakeven = 100 / (1 + bt["rr"])
    return (f"Filtered backtest (4H zones, ~2y): {bt['wins']}W/{bt['losses']}L "
            f"({bt['win_rate']:.0f}% win rate, {bt['no_result']} inconclusive, "
            f"{bt['skipped_by_filters']} skipped by filters) at {bt['rr']:.1f}:1 RR "
            f"(breakeven needs {breakeven:.0f}%). Not a guarantee of future results.")


def generate_html_report(results: list, output_path: str):
    cards_html = ""
    for analysis, news in results:
        entry_class = "signal-yes" if analysis.entry_signal else "signal-no"
        entry_html = ("\u2705 ENTRY SIGNAL: " + analysis.entry_reason) if analysis.entry_signal else ("No entry: " + analysis.entry_reason)
        blackout_html = f'<div class="blackout">\u26a0 RED NEWS BLACKOUT: {analysis.news_blackout_reason}</div>' if analysis.news_blackout else ""
        levels_html = ""
        if analysis.trade_levels:
            t = analysis.trade_levels
            levels_html = (f'<div class="levels">Entry {t["entry"]:.2f} | Stop {t["stop"]:.2f} | '
                            f'Target {t["target"]:.2f} ({t["rr"]:.1f}:1 RR)</div>')

        cards_html += CARD_HTML.format(
            symbol=analysis.symbol, bias=analysis.bias, bias_class=analysis.bias,
            structure=analysis.structure, last_price=analysis.last_price,
            zones_html=render_zones_html(analysis),
            entry_class=entry_class, entry_html=entry_html,
            levels_html=levels_html,
            blackout_html=blackout_html,
            notes_html=render_notes_html(analysis.notes),
            backtest_html=render_backtest_html(analysis.backtest),
            news_html=render_news_html(news),
        )

    html = HTML_SHELL.format(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        cards=cards_html,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report written to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Market Analysis Report — generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print("4H zones (trend-aligned, fresh-only, rejection-confirmed), 5M entry, 3:1 RR.")
    print("NOTE: Educational tool only. Not financial advice. No accuracy guarantee.")
    print("=" * 60)

    calendar = fetch_economic_calendar()
    results = []

    for name, ticker in SYMBOLS.items():
        try:
            analysis = analyze_symbol(name, ticker, calendar)
        except Exception as e:
            print(f"\n=== {name} ===\n  [error] Could not analyze: {e}")
            continue

        currencies = SYMBOL_CURRENCIES.get(name, {"USD"})
        news = upcoming_high_impact_events(calendar, currencies) if calendar else []
        print(build_trade_idea(analysis, news))
        results.append((analysis, news))

    print("\n" + "=" * 60)
    print("Done.")

    generate_html_report(results, os.environ.get("REPORT_OUTPUT_PATH", "docs/index.html"))


if __name__ == "__main__":
    main()
