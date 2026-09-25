"""
Market Analyzer & Trade Idea Generator — Pure Price Action / Supply & Demand
Multi-timeframe day-trading version: 4H structure/zones, 5M entry trigger.
=============================================================================
No indicators (no moving averages, RSI, MACD, etc). Pure price action:

  1. STRUCTURE & ZONES (4H): swing highs/lows classify trend (higher-highs/
     higher-lows = uptrend, lower-highs/lower-lows = downtrend). Supply &
     demand zones are the consolidation ("base") right before a strong
     directional move away from it.
  2. ENTRY TRIGGER (5M): once price is trading inside a valid 4H zone, the
     5-minute chart is checked for a confirmation candle (engulfing /
     rejection) before a live entry signal fires. This avoids entering the
     moment price touches a zone — it waits for the zone to actually react.
  3. NEWS FILTER (ForexFactory "red folder" high-impact events): entries
     are suppressed inside a blackout window around high-impact news for
     the relevant currency.
  4. BACKTEST (honesty check): reports a historical win rate for the zone
     strategy so you're trusting a tested process, not a blind promise.
     IMPORTANT DATA LIMITS (free Yahoo Finance data):
       - 5-minute candles: last ~60 days only.
       - Hourly candles (used to build 4H bars): last ~2 years only.
       - Daily candles: full history (several years+).
     True 4-year backtesting at 5-minute resolution is not possible with
     free data. The 4H zone backtest below covers ~2 years (the max
     available). For genuine multi-year, tick-accurate backtesting, use
     your XM MetaTrader Strategy Tester with the broker's own history.

THIS TOOL DOES NOT PLACE TRADES. Review manually in XM. Not financial
advice, and no win rate shown here is a promise about future performance.

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

# --- Structure/zone timeframe (4H, built by resampling 1H data) ---
STRUCTURE_SOURCE_INTERVAL = "1h"
STRUCTURE_SOURCE_PERIOD = "730d"    # Yahoo's max history for hourly data (~2y)
STRUCTURE_RULE = "4h"

# --- Entry trigger timeframe ---
ENTRY_INTERVAL = "5m"
ENTRY_PERIOD = "60d"                # Yahoo's max history for 5m data

SWING_WINDOW = 3
BASE_LOOKBACK = 4
RANGE_BASELINE_LOOKBACK = 10
IMPULSE_MULTIPLIER = 1.7

# How close to "now" a high-impact news event has to be (either side) to
# suppress a live entry signal.
NEWS_BLACKOUT_MINUTES = 30

# Backtest settings (run on the 4H structure timeframe, ~2 years of data)
BACKTEST_LOOKAHEAD_CANDLES = 20     # how many 4H candles to watch after a zone is touched
BACKTEST_RR = 1.5                   # target = risk x this multiple

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
    entry_signal: bool = False
    entry_reason: str = ""
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


# ---------------------------------------------------------------------------
# SUPPLY & DEMAND ZONES
# ---------------------------------------------------------------------------

def raw_zones(df: pd.DataFrame) -> list:
    """All zones ever formed in the dataset (used for backtesting), before
    filtering out the ones price has since broken through."""
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
    """Zones not yet broken by the end of the dataset, marked fresh/tested."""
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
# 5-MINUTE ENTRY TRIGGER
# ---------------------------------------------------------------------------

def find_5m_trigger(df5: pd.DataFrame, direction: str, lookback_candles: int = 12):
    """direction: 'bullish' (looking for demand-zone reaction) or 'bearish'."""
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
    return False, "Price is at the zone but no confirmed 5M reaction candle yet — wait."


# ---------------------------------------------------------------------------
# ECONOMIC CALENDAR (ForexFactory "red folder" = High impact)
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
            if ev.get("impact") != "High":
                continue
            if ev.get("country") not in currencies:
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
# BACKTEST (4H zone reliability, ~2 years — see data-limit note at top)
# ---------------------------------------------------------------------------

def evaluate_zone_outcome(df, z, touch_idx, lookahead, rr):
    risk = z["high"] - z["low"]
    if risk <= 0:
        return None
    if z["kind"] == "demand":
        stop, target = z["low"], z["high"] + risk * rr
    else:
        stop, target = z["high"], z["low"] - risk * rr
    window = df.iloc[touch_idx + 1: touch_idx + 1 + lookahead]
    for _, row in window.iterrows():
        if z["kind"] == "demand":
            if row["Low"] <= stop:
                return "loss"
            if row["High"] >= target:
                return "win"
        else:
            if row["High"] >= stop:
                return "loss"
            if row["Low"] <= target:
                return "win"
    return None


def backtest_zones(df: pd.DataFrame, zones: list) -> dict:
    wins = losses = no_result = 0
    for z in zones:
        touch_idx = None
        for idx in range(z["formed_idx"] + 1, len(df)):
            row = df.iloc[idx]
            if row["Low"] <= z["high"] and row["High"] >= z["low"]:
                touch_idx = idx
                break
        if touch_idx is None:
            continue
        outcome = evaluate_zone_outcome(df, z, touch_idx, BACKTEST_LOOKAHEAD_CANDLES, BACKTEST_RR)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            no_result += 1
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved else None
    return {"wins": wins, "losses": losses, "no_result": no_result,
            "total_zones": len(zones), "win_rate": win_rate}


# ---------------------------------------------------------------------------
# ANALYSIS (per symbol)
# ---------------------------------------------------------------------------

def analyze_symbol(name: str, ticker: str, calendar: list) -> Analysis:
    # 4H structure & zones, built from 1H data
    df1h = fetch_ohlc(ticker, STRUCTURE_SOURCE_INTERVAL, STRUCTURE_SOURCE_PERIOD)
    df4h = resample_ohlc(df1h, STRUCTURE_RULE)
    current_price = float(df4h["Close"].iloc[-1])

    swing_highs, swing_lows = find_swings(df4h)
    structure = classify_structure(swing_highs, swing_lows)

    z_raw = raw_zones(df4h)
    zones = valid_zones_now(df4h, z_raw)
    demand_zone, supply_zone, inside = nearest_zones(zones, current_price)

    backtest = backtest_zones(df4h, z_raw)

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

    # --- News blackout check (red folder) ---
    currencies = SYMBOL_CURRENCIES.get(name, {"USD"})
    blackout, blackout_reason = check_news_blackout(calendar, currencies) if calendar else (False, "")

    # --- 5-minute entry trigger, only checked when price is inside a zone ---
    entry_signal, entry_reason = False, "Price is not currently inside a valid zone — no entry to evaluate yet."
    if inside in ("demand", "supply") and not blackout:
        try:
            df5 = fetch_ohlc(ticker, ENTRY_INTERVAL, ENTRY_PERIOD)
            direction = "bullish" if inside == "demand" else "bearish"
            entry_signal, entry_reason = find_5m_trigger(df5, direction)
        except Exception as e:
            entry_reason = f"Could not fetch 5M data for entry trigger: {e}"
    elif blackout:
        entry_reason = f"Entry suppressed — inside red-news blackout window: {blackout_reason}"

    if entry_signal:
        notes.append("Day-trade reminder: this is an intraday setup — plan your exit and close before end of session, don't hold overnight.")

    return Analysis(
        symbol=name, last_price=current_price, structure=structure, bias=bias,
        demand_zone=demand_zone, supply_zone=supply_zone, inside_zone=inside,
        entry_signal=entry_signal, entry_reason=entry_reason,
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

    if a.news_blackout:
        lines.append(f"  \u26a0 RED NEWS BLACKOUT: {a.news_blackout_reason} — do not enter.")

    if a.backtest and a.backtest["win_rate"] is not None:
        bt = a.backtest
        lines.append(f"  Backtest (4H zones, ~last 2y): {bt['wins']}W/{bt['losses']}L "
                      f"({bt['win_rate']:.0f}% win rate on {bt['wins']+bt['losses']} resolved zones, "
                      f"{bt['no_result']} inconclusive). Past results are not a guarantee of future ones.")

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
  .blackout {{ font-size: 0.82rem; font-weight: 600; color: var(--warn); background: rgba(217,119,6,0.12);
               padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; }}
  .notes {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }}
  .notes div {{ margin-bottom: 3px; }}
  .backtest {{ font-size: 0.78rem; color: var(--muted); border-top: 1px solid var(--border); padding-top: 8px; margin-bottom: 8px; }}
  .news {{ font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 8px; color: var(--muted); }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; padding: 20px 0 8px; }}
</style>
</head>
<body>
<header>
  <h1>\U0001F4CA Daily Market Report</h1>
  <p>Generated {generated_at} &middot; 4H structure/zones, 5M entry, red-news filter</p>
</header>
<div class="disclaimer">
  Educational tool only — not financial advice, not a guarantee of accuracy.
  Prices are a Yahoo Finance proxy, delayed and not identical to XM's live
  feed. 5M data covers ~60 days, 4H data ~2 years (Yahoo's free-tier
  limits) — backtest win rates reflect that window, not 4 full years.
  Always verify on your own XM chart before trading.
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
    return (f"Backtest (4H zones, ~2y history): {bt['wins']}W/{bt['losses']}L "
            f"({bt['win_rate']:.0f}% win rate, {bt['no_result']} inconclusive). "
            f"Past results aren't a guarantee of future ones.")


def generate_html_report(results: list, output_path: str):
    cards_html = ""
    for analysis, news in results:
        entry_class = "signal-yes" if analysis.entry_signal else "signal-no"
        entry_html = ("\u2705 ENTRY SIGNAL: " + analysis.entry_reason) if analysis.entry_signal else ("No entry yet: " + analysis.entry_reason)
        blackout_html = f'<div class="blackout">\u26a0 RED NEWS BLACKOUT: {analysis.news_blackout_reason}</div>' if analysis.news_blackout else ""

        cards_html += CARD_HTML.format(
            symbol=analysis.symbol, bias=analysis.bias, bias_class=analysis.bias,
            structure=analysis.structure, last_price=analysis.last_price,
            zones_html=render_zones_html(analysis),
            entry_class=entry_class, entry_html=entry_html,
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
    print("4H structure/zones, 5M entry trigger, red-news filter. No indicators.")
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
