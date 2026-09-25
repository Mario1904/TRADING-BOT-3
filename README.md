# Market Analyzer & Trade Idea Generator (Day-Trading, Pure Price Action)

Generates day-trading setups for US30, US100, Gold, Bitcoin, and major forex
pairs using **pure price action — no indicators** (no moving averages, RSI,
MACD, etc.), across two timeframes:

- **4H — structure & zones**: swing highs/lows classify trend (higher-highs/
  higher-lows = uptrend, lower-highs/lower-lows = downtrend). Supply &
  demand zones are found using the "base before an impulse" method — a
  tight consolidation right before a strong directional move marks the
  zone left behind.
- **5M — entry trigger**: once price is trading inside a valid 4H zone, the
  5-minute chart is checked for a confirmation candle (an engulfing/
  rejection candle) before a live entry signal fires. This is meant to
  avoid jumping in the instant price touches a zone.
- **Red-folder news filter**: pulls ForexFactory's high-impact ("red")
  economic events and suppresses entry signals inside a blackout window
  around them (default ±30 min) for the relevant currency.
- **Backtest**: reports a historical win rate for the 4H zone strategy, so
  you can see a track record rather than take a blind promise.

It's built for **day trading** — every signal comes with a reminder to plan
your exit and close before end of session, not hold overnight.

It prints/publishes structured reports — it does **not** place trades for
you. You stay in control and execute manually in XM.

## Important: data limits (please read before trusting the numbers)

This uses free Yahoo Finance data, which has hard caps:

| Data | Max free history |
|---|---|
| 5-minute candles (entry trigger) | ~60 days |
| Hourly candles (built into 4H structure/zones) | ~2 years |
| Daily candles | several years+ |

**A true 4-year backtest at 5-minute entry resolution is not possible with
free data — no free source provides it.** The backtest in this tool covers
the 4H zone strategy over the ~2 years of hourly data Yahoo makes available;
it does not (and can't) validate the 5-minute entry trigger over 4 years,
because that data simply doesn't exist for free.

**If you want genuine multi-year, broker-accurate backtesting**, use your
XM MetaTrader platform's built-in Strategy Tester — it has full historical
tick data from XM itself, going back years, for free. This script is best
used for live daily/intraday decision support; MT4/MT5 is the right tool
for rigorous historical validation.

No win rate shown by this tool is a promise about future performance.

## Setup

```bash
pip install -r requirements.txt
```

(If you get an "externally managed environment" error on Linux, use:
`pip install -r requirements.txt --break-system-packages`)

## Run

```bash
python market_analyzer.py
```

## Getting it on your phone (no app needed)

The script writes an HTML report to `docs/index.html`. The included GitHub
Actions workflow runs it automatically **every 15 minutes** (so 5-minute
entry signals stay current) and publishes that report as a free webpage —
open the link in your Android browser whenever you want.

**One-time setup (~10 minutes):**

1. Create a free GitHub account if you don't have one.
2. Create a new **public** repository (e.g. `market-reports`) — must be
   public for free GitHub Pages.
3. Upload these files/folders, keeping the structure:
   - `market_analyzer.py`
   - `requirements.txt`
   - `.github/workflows/daily-analysis.yml`
4. Settings → Pages → Source: "Deploy from a branch", branch `main`,
   folder `/docs`. Save.
5. Actions tab → "Daily Market Analysis" → "Run workflow" (do this once
   manually to generate the first report).
6. Open `https://YOUR-USERNAME.github.io/market-reports/` on your phone,
   bookmark it or "Add to Home Screen" from Chrome.

It then refreshes itself every 15 minutes automatically. Edit the `cron`
line in `daily-analysis.yml` if you want a different frequency — public
repos get unlimited free Actions minutes, so this costs $0.

## Customizing

Open `market_analyzer.py` and edit near the top:

- `SYMBOLS` — add/remove instruments. Prices are a Yahoo Finance **proxy**
  for XM's CFD prices (close but not identical) — always cross-check
  against your XM chart before entering.
- `IMPULSE_MULTIPLIER` — how strong a move must be (relative to recent
  candle ranges) to count as breaking out of a base and leaving a zone.
  Raise for fewer, higher-conviction zones.
- `NEWS_BLACKOUT_MINUTES` — how wide a window around red-folder news
  suppresses entries.
- `BACKTEST_LOOKAHEAD_CANDLES` / `BACKTEST_RR` — how far forward and what
  risk:reward the backtest checks when grading a zone as a win or loss.

## Important

This is a decision-support tool, not financial advice, and not a signal
service with any guaranteed accuracy. The supply/demand zone logic and 5M
trigger are a simplified, rules-based reading of price action — treat every
output as a starting point for your own analysis, not a trade you take
automatically. Forex/CFD/crypto trading carries real risk of loss.

## Ideas for extending it

- Multi-timeframe confluence beyond 4H (e.g. daily structure agreement)
- Log signals to a CSV to build your own real track record over time
- Telegram/email push the instant a fresh entry signal fires
- Position-size calculator based on account balance and stop distance
