# P08 Self-Interview: Options Market-Making

## Q1: How does the vol arb engine decide entry timing vs just signal generation?
**A:** The vol arb engine is a signal + strategy combination. The signal (IV vs realized vol divergence) identifies candidates. The strategy layer decides entry timing based on: (a) divergence magnitude exceeding threshold, (b) IV rank position (prefer selling high IV rank), (c) days to expiration window (30-60 DTE preferred for theta). Entry is via limit orders through the existing execution pipeline, not market orders.

## Q2: What happens when dispersion trading correlation spikes during a crisis?
**A:** This is the primary risk. Mitigation: (a) position size capped at 2% of portfolio per dispersion trade, (b) correlation monitor triggers exit when realized correlation exceeds implied by >0.15, (c) index hedge via long puts as tail protection. The risk gate enforces these limits. Discovery findings confirm the risk gate is inviolable.

## Q3: How does gamma scalping interact with the P06 hedging engine?
**A:** Gamma scalping uses the P06 HedgingEngine for its delta rebalancing. The difference is intent: P06 hedging is defensive (reduce risk), gamma scalping is offensive (profit from vol). The GammaScalpingStrategy extends HedgingEngine with: rehedge frequency (every 30min or 0.5% move), theta bleed tracking, and auto-exit when cumulative theta exceeds gamma profit.

## Q4: What's the options_market_maker agent's relationship to existing trading graph agents?
**A:** It's a new node in the trading graph, parallel to the existing entry_scanner. It runs every cycle during market hours. It has its own tool set (compute_greeks, get_iv_surface, score_trade_structure, simulate_trade_outcome — the P06 tools). Its proposals go through the same risk gate as all other trade proposals. It does not bypass any existing flow.

## Q5: How do you prevent the vol strategies from conflicting with directional equity strategies?
**A:** Portfolio-level delta is tracked. The risk gate has a max portfolio delta limit. If vol strategies generate net delta (from imperfect hedging), it counts against the same limit as equity positions. The options_market_maker agent sees the full portfolio Greeks before proposing new trades.

## Q6: What's the iron condor management protocol when a short strike is breached?
**A:** Roll the tested side: if the short put is breached, buy back the put spread and sell a new one at lower strikes. If the short call is breached, same logic upward. Close the entire position at 50% profit target or 200% loss limit. If underlying moves beyond the long strike, close immediately — the defined risk of the spread caps the loss.

## Q7: How does the system handle expiration risk / pin risk?
**A:** P06's pin risk monitoring detects when underlying is near a short strike within 2 DTE. For condors and verticals, the management rule is: close any position within 2 DTE of expiration if underlying is within 2% of either short strike. This avoids assignment risk and gamma explosion near expiry.

## Q8: What data sources are needed for IV rank and realized vol?
**A:** IV rank requires historical IV data — maintained by computing daily ATM IV from the IV surface (P06) and storing in the database. Realized vol is computed from OHLCV data (already available via Alpha Vantage / Alpaca IEX). Both are rolling calculations (IV rank = percentile over 252 trading days, realized vol = 21-day annualized std dev of log returns).
