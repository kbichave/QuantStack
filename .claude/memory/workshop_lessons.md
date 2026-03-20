# Workshop Lessons

> Accumulated R&D learnings — your research memory
> Read at START of every /workshop session
> Update after: /workshop, /reflect

## Backtesting Pitfalls

- SPY mean-reversion with extreme oversold filters (RSI<35, Stoch<15, CCI<-150) produces only 27-55 trades in 6 years depending on configuration
- Loosening oversold thresholds to 1-of-4 (87 trades) destroys edge: Sharpe -0.41, PF 0.76
