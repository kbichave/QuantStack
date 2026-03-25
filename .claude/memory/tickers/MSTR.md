# MSTR Research & Trading Memory

> Per-ticker memory file. Updated by research and trading loops.
> Read at START of any session involving this symbol.

## Fundamental Snapshot

_Last updated: 2026-03-25 (AV OVERVIEW + GLOBAL_QUOTE)_

| Metric | Value | Notes |
|--------|-------|-------|
| Revenue (TTM) | $477.2M | Software business (legacy); almost irrelevant to thesis |
| Revenue Growth YoY | +1.9% | Flat software revenue; company is now a Bitcoin treasury |
| Gross Margin | 68.7% | Software margins — irrelevant; BTC drives everything |
| Operating Margin | -4402% | Bitcoin unrealized losses dominate OpInc |
| P/E | N/A | Bitcoin accounting distorts P/E |
| P/S | 98.7x | Not meaningful — price follows BTC, not revenue |
| MCap | $47.1B | |
| EPS | -$15.23 | Accounting loss from Bitcoin |
| Beta | 3.633 | **Highest beta in watchlist** |
| % Institutions | 62.7% | |
| % Insiders | 0.18% | Saylor has pledged most shares |

## Price Action

_Last updated: 2026-03-25_

| Level | Value |
|-------|-------|
| Current Price | $136.25 (-1.41% today) |
| SMA 50 | $143.20 |
| SMA 200 | $265.69 |
| 52w High | $457.22 |
| 52w Low | $104.17 |
| % from 52wH | -70.2% |
| vs SMA 50 | -4.9% (just below) |
| vs SMA 200 | -48.7% (deeply below) |
| Analyst Target | $374.07 |
| Volume (today) | 19.1M |

**-70.2% from 52wH. Deeply below SMA 200. Mirrors Bitcoin drawdown. Near 52wL $104.**

## Investment Thesis

| Thesis Type | Fit | Evidence |
|-------------|-----|----------|
| Bitcoin Proxy | CORE | Holds ~570,000 BTC (~$48B); company IS a leveraged Bitcoin ETF |
| Leverage | KEY | Uses convertible notes to buy BTC; ~1.8x effective Bitcoin leverage |
| Options Activity | HIGHEST | Most active options name outside indexes (by dollar volume) |
| Valuation | N/A | Traditional metrics irrelevant; NAV premium/discount to BTC is what matters |

**Primary thesis**: The only publicly-traded leveraged Bitcoin treasury. 1.8x synthetic BTC exposure via convertible debt structure. When Bitcoin rallies, MSTR amplifies. When Bitcoin falls, MSTR amplifies further (leverage works both ways).
**Conviction**: 50% (depends entirely on Bitcoin thesis)
**Catalyst timeline**: Bitcoin price movements; new BTC purchase announcements; convertible note issuances

## Evidence Map

| Category | Key Findings | Tier | Direction |
|----------|-------------|------|-----------|
| Regime | -70.2% from high; near 52wL | tier_4 | bearish |
| Technicals | Below SMA50+200; -48.7% below SMA200 | tier_1 | bearish |
| BTC Correlation | Effectively 1.8x leveraged BTC ETF | — | follows BTC |
| Options | Deepest options market in individual names | tier_2 | high vol |
| Analyst | 2 Strong Buy, 1 Buy, 1 Hold, 0 Sell | — | bullish |

## Strategies

| ID | Name | Type | Backtest | WF OOS | ML AUC | Status |
|----|------|------|---------|--------|--------|--------|

## Research Log

### Iteration 1 (2026-03-25) — Initial AV data load
- BTC price as of 2026-03-25 ~$87K (infer from MCap $47.1B / 570K BTC ≈ $82K per BTC)
- MSTR NAV premium is key: if MSTR MCap >> BTC holdings value, it's expensive vs BTC

## Lessons (MSTR-specific)
1. **Not a software company**: Revenue and margins from software are irrelevant. The stock price is ~1.8x leveraged BTC.
2. **Options market depth**: MSTR has the most options activity of any individual stock outside ETFs. IV is usually elevated 80-120%.
3. **Convertible note mechanism**: Saylor issues convertible notes at low/zero coupon to buy more BTC. Dilutive over time but bullish BTC signal.
4. **NAV premium is the risk**: MSTR trades at a premium to its BTC holdings (leveraged premium). In bear markets, this premium compresses = amplified losses.
5. **52wH $457 reflects BTC peak euphoria**: BTC was ~$108K in Jan 2026. Current BTC ~$82-87K. MSTR at $136 = -70% from BTC peak, more than BTC's decline, due to premium compression.