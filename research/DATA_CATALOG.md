# Data Catalog: Everything That Can Move a Stock Price

A comprehensive checklist of data categories for the Gap-and-Go + Mean Reversion AI engine.
Items marked [HAVE] are what you said you already have. Everything else is worth acquiring,
roughly ordered by expected edge-per-effort for Indian F&O intraday strategies.

---

## 1. Price & Volume (the core) [HAVE]
- 1-min OHLCV for all 210 F&O stocks (11 years)
- Derived timeframes: 5m, 15m, 30m, hourly, daily
- Overnight gap (close -> open), pre-open auction price/volume (NSE pre-open session 9:00-9:15)
- Intraday VWAP, rolling realized volatility, ATR
- Prior day/week/month high-low-close (support/resistance reference levels)
- 52-week high/low proximity, all-time high proximity
- Volume profile: relative volume (RVOL), volume at price, opening-range volume

## 2. Index & Market Breadth [HAVE partially]
- Nifty 50, Bank Nifty, Nifty Midcap, sector indices (IT, Pharma, Auto, Metal, Energy, FMCG, Realty)
- Index overnight gap and first-15-min direction
- Advance/decline ratio, % stocks above 20/50/200 DMA, new highs vs new lows
- GIFT Nifty (formerly SGX Nifty) overnight session — the single best predictor of Nifty's opening gap

## 3. Global Markets [HAVE partially]
- S&P 500, Nasdaq, Dow (previous close + overnight futures ES/NQ)
- Asian session: Nikkei, Hang Seng, Shanghai, Kospi (trade before/with India open)
- European open: DAX, FTSE (affects Indian afternoon session)
- MSCI Emerging Markets index, MSCI India rebalance dates

## 4. Volatility
- India VIX (level, change, term structure)
- Stock-level implied volatility (ATM IV from options chain)
- IV percentile / IV rank per stock
- Realized-vs-implied vol spread (variance risk premium)
- US VIX and VIX futures term structure

## 5. Options & Derivatives Data [HAVE]
- Open interest (OI) per strike, OI change, max pain
- Put-Call ratio (volume and OI based), per-stock and index
- IV skew (25-delta risk reversal proxy)
- Futures basis (futures - spot), rollover %, cost of carry
- Days to expiry (weekly/monthly), expiry-day flag
- F&O ban list (stocks in ban see distinct behavior)
- Gamma exposure estimates around key strikes (pinning effects)

## 6. Cross-Asset Macro [HAVE partially]
- Crude oil (Brent/WTI) — critical for OMCs, paints, aviation, tyres [HAVE]
- USD/INR — critical for IT, pharma exporters
- Gold/silver — jewellery, NBFC-gold-loan stocks
- US 10Y yield, India 10Y G-sec yield — banks, NBFCs, rate-sensitives
- Dollar index (DXY), Baltic Dry Index (shipping/metals)
- Metals: copper, aluminium, steel (LME) — metal stocks

## 7. Macro Events Calendar
- RBI policy dates (repo decisions), Fed FOMC dates
- India CPI, WPI, GDP, IIP, PMI releases
- US CPI, NFP, GDP releases (evening IST — affect next-day gaps)
- Union Budget day, monsoon forecasts (FMCG/agri/fertilizer)
- Election dates and exit polls (extreme gap days)

## 8. Corporate Actions & Events [HAVE]
- Earnings announcement dates + actual results vs estimates (surprise %)
- Dividends (ex-dates), splits, bonuses, rights issues, buybacks
- Mergers, demergers, delistings, open offers
- Board meeting dates, AGM dates
- Stock inclusion/exclusion from indices (Nifty rebalance) and F&O ban entry/exit
- Promoter pledge changes, insider trading disclosures (SAST filings)
- Bulk deals / block deals (NSE publishes daily)

## 9. Institutional Flows
- FII/DII daily cash-market net flows
- FII derivatives positioning (index futures long/short ratio)
- Mutual fund monthly flows (SIP data)
- Delivery percentage per stock per day (proxy for institutional accumulation)

## 10. News & Sentiment [HAVE]
- Headline news with timestamps (pre-market news drives gap quality)
- Analyst upgrades/downgrades, target price changes
- Rating agency actions (CRISIL, ICRA, Moody's on the company)
- Regulatory news: SEBI orders, court rulings, government policy for the sector
- Social sentiment (optional, noisy)
- **Key label to build: "gap WITH news" vs "gap WITHOUT news"** — the single most
  important classifier for gap-fade vs gap-go

## 11. Cross-Sectional / Relative Data [HAVE partially]
- Pairwise stock correlations (rolling 20/60/250-day) [HAVE]
- Sector membership and sector-relative return (stock return minus sector return)
- Beta to Nifty (rolling), residual return after removing market+sector
- Peer basket z-score (stock deviation from its correlated basket — core mean-reversion signal)
- Relative strength rank across the 210-stock universe

## 12. Calendar / Seasonality
- Day of week, day of month, month of year
- Monthly expiry day/week, weekly expiry day
- First/last trading day of month (flow effects), quarter-end (window dressing)
- Pre-holiday and post-holiday sessions
- Time-of-day buckets (9:15-9:30 open drive, 11:30-13:30 lunch lull, 14:30-15:30 close drive)

## 13. Regime / State Variables (derived)
- Market regime: trending vs choppy (e.g., ADX on Nifty, efficiency ratio)
- Volatility regime: India VIX percentile buckets
- Drawdown state of the index (bull/correction/bear)
- Correlation regime: average pairwise correlation (high = macro-driven market, mean reversion works differently)

## 14. Liquidity & Regulatory Microstructure
- Circuit limits and circuit hits (stocks locked at upper/lower circuit)
- ASM / GSM surveillance list membership
- Tick-level bid-ask spread and depth (if available — improves execution modeling)
- Lot size changes, tick size regime

---

## Priority for your two strategies

**Gap-and-Go depends most on:** #1 gaps, #2 index gap, #3 global overnight, #10 news-vs-no-news,
#5 OI change, #4 IV, #8 earnings dates, #12 expiry calendar.

**Mean Reversion depends most on:** #11 peer-basket residuals, #1 VWAP/z-scores, #4 vol regime,
#13 correlation regime, #10 absence of news (never fade a fundamental move), #9 delivery %.
