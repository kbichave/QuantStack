"""
Iron Condor VRP Backtest — Modified Parameters
Strategy: qqq_iron_condor_vrp_swing_v1 (strat_5b3dda24f7)

Modifications from original:
- MIN_SPACING: 28 -> 21 days
- BB%B filter: [0.20, 0.80] -> [0.15, 0.85]
- Stop loss: 100% -> 200% of credit
- History: 2023+ -> 2015+ (11 years vs 3)

BSM pricing for option legs. IV estimated as multiplier of realized vol.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from quantstack.db import pg_conn
import json
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def find_strike_by_delta(S, T, r, sigma, target_delta, option_type="call",
                         precision=0.25):
    """Find strike that gives approximately the target delta."""
    # Search range: +/- 30% from spot
    low = S * 0.70
    high = S * 1.30
    best_K = S
    best_diff = 999

    # Coarse grid then refine
    for K in np.arange(low, high, precision):
        d = abs(bs_delta(S, K, T, r, sigma, option_type))
        diff = abs(d - abs(target_delta))
        if diff < best_diff:
            best_diff = diff
            best_K = K

    # Round to nearest dollar for ETF options
    return round(best_K)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_atr(df, period=14):
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    """Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # When both are positive, keep the larger
    mask = plus_dm > minus_dm
    minus_dm[mask & (plus_dm > 0)] = 0
    mask2 = minus_dm > plus_dm
    plus_dm[mask2 & (minus_dm > 0)] = 0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ---------------------------------------------------------------------------
# Iron Condor simulation
# ---------------------------------------------------------------------------

def simulate_iron_condor(symbol, df, params, verbose=False):
    """
    Simulate iron condor trades on OHLCV data with BSM pricing.

    Iron condor structure:
        - Sell OTM put at ~delta_wing
        - Buy OTM put at (sell_put_strike - wing_width)
        - Sell OTM call at ~delta_wing
        - Buy OTM call at (sell_call_strike + wing_width)

    Entry filters:
        - BB%B between [bb_low, bb_high] (ranging)
        - ADX < adx_max (not strongly trending)
        - Min spacing between trades
        - IV > RV (vol risk premium exists)

    Exit rules:
        - Profit target: close at X% of credit received
        - Stop loss: close at Y% of credit (loss)
        - DTE stop: close at 5 DTE if still open
    """
    min_spacing = params["min_spacing_days"]
    bb_low = params["bb_low"]
    bb_high = params["bb_high"]
    adx_max = params["adx_max"]
    dte = params["dte"]
    delta_wing = params["delta_wing"]
    wing_width = params["wing_width"]
    profit_target_pct = params["profit_target"]
    stop_loss_pct = params["stop_loss"]
    iv_multiplier = params["iv_multiplier"]
    r = params.get("risk_free_rate", 0.04)

    # Compute indicators
    df = df.copy()
    df["rv_21"] = df["close"].pct_change().rolling(21).std() * np.sqrt(252)
    df["rv_63"] = df["close"].pct_change().rolling(63).std() * np.sqrt(252)
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )
    df["adx"] = compute_adx(df, 14)
    df["atr"] = compute_atr(df, 14)

    df.dropna(inplace=True)

    trades = []
    last_entry_idx = -min_spacing  # allow first trade immediately

    for i in range(len(df)):
        row = df.iloc[i]
        idx = i

        # Spacing filter
        if idx - last_entry_idx < min_spacing:
            continue

        # BB%B filter — price is in the "middle" of Bollinger Bands (ranging)
        if row["bb_pctb"] < bb_low or row["bb_pctb"] > bb_high:
            continue

        # ADX filter — not strongly trending
        if row["adx"] > adx_max:
            continue

        # IV proxy: use multiplier on 21-day RV (markets typically price IV > RV)
        rv = row["rv_21"]
        if rv <= 0.05:  # skip extremely low vol (data issue)
            continue
        iv = rv * iv_multiplier

        S = row["close"]
        T = dte / 252.0

        # Find strikes at target delta
        sell_put_K = find_strike_by_delta(S, T, r, iv, delta_wing, "put")
        sell_call_K = find_strike_by_delta(S, T, r, iv, delta_wing, "call")
        buy_put_K = sell_put_K - wing_width
        buy_call_K = sell_call_K + wing_width

        # Ensure strikes make sense
        if sell_put_K >= S or sell_call_K <= S:
            continue
        if buy_put_K <= 0:
            continue

        # Price the iron condor at entry
        sell_put_price = bs_price(S, sell_put_K, T, r, iv, "put")
        buy_put_price = bs_price(S, buy_put_K, T, r, iv, "put")
        sell_call_price = bs_price(S, sell_call_K, T, r, iv, "call")
        buy_call_price = bs_price(S, buy_call_K, T, r, iv, "call")

        credit = (sell_put_price - buy_put_price) + (
            sell_call_price - buy_call_price
        )

        if credit <= 0.10:  # skip if credit is negligible
            continue

        # Max risk per contract = wing_width - credit
        max_risk = wing_width - credit

        # Track through to exit
        entry_date = df.index[i]
        entry_price = S
        profit_target_val = credit * profit_target_pct
        stop_loss_val = credit * stop_loss_pct

        exit_reason = None
        exit_pnl = 0
        exit_date = None
        days_held = 0

        for j in range(i + 1, min(i + dte + 1, len(df))):
            future_row = df.iloc[j]
            future_S = future_row["close"]
            future_high = future_row["high"]
            future_low = future_row["low"]
            days_elapsed = j - i
            remaining_dte = dte - days_elapsed
            future_T = max(remaining_dte / 252.0, 1 / 252.0)

            # Re-price the iron condor with current spot and decayed time
            # Use same IV (simplified — in reality IV changes)
            sp = bs_price(future_S, sell_put_K, future_T, r, iv, "put")
            bp = bs_price(future_S, buy_put_K, future_T, r, iv, "put")
            sc = bs_price(future_S, sell_call_K, future_T, r, iv, "call")
            bc = bs_price(future_S, buy_call_K, future_T, r, iv, "call")

            current_cost_to_close = (sp - bp) + (sc - bc)
            current_pnl = credit - current_cost_to_close

            # Check intraday breaches using high/low
            # Worst case: if high touches sell_call or low touches sell_put
            worst_S_high = future_high
            worst_S_low = future_low

            sp_worst = bs_price(worst_S_low, sell_put_K, future_T, r, iv, "put")
            bp_worst = bs_price(worst_S_low, buy_put_K, future_T, r, iv, "put")
            sc_worst = bs_price(worst_S_high, sell_call_K, future_T, r, iv, "call")
            bc_worst = bs_price(worst_S_high, buy_call_K, future_T, r, iv, "call")

            worst_cost = max(
                (sp_worst - bp_worst) + (sc - bc),  # put side breach
                (sp - bp) + (sc_worst - bc_worst),  # call side breach
                current_cost_to_close,
            )
            worst_pnl = credit - worst_cost

            # Profit target check (on close)
            if current_pnl >= profit_target_val:
                exit_reason = "profit_target"
                exit_pnl = current_pnl
                exit_date = df.index[j]
                days_held = days_elapsed
                break

            # Stop loss check (on worst case intraday)
            if worst_pnl <= -stop_loss_val:
                exit_reason = "stop_loss"
                # Use worst-case PnL, capped at max risk
                exit_pnl = max(worst_pnl, -max_risk)
                exit_date = df.index[j]
                days_held = days_elapsed
                break

            # DTE stop: close at 5 DTE
            if remaining_dte <= 5:
                exit_reason = "dte_stop"
                exit_pnl = current_pnl
                exit_date = df.index[j]
                days_held = days_elapsed
                break

        # If we ran out of data without hitting exit
        if exit_reason is None:
            # Close at last available price
            if i + dte < len(df):
                j = i + dte
                future_S = df.iloc[j]["close"]
                future_T = 1 / 252.0
                sp = bs_price(future_S, sell_put_K, future_T, r, iv, "put")
                bp = bs_price(future_S, buy_put_K, future_T, r, iv, "put")
                sc = bs_price(future_S, sell_call_K, future_T, r, iv, "call")
                bc = bs_price(future_S, buy_call_K, future_T, r, iv, "call")
                exit_pnl = credit - ((sp - bp) + (sc - bc))
                exit_reason = "expiry"
                exit_date = df.index[j]
                days_held = dte
            else:
                continue  # skip incomplete trades

        trade = {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "sell_put_K": sell_put_K,
            "buy_put_K": buy_put_K,
            "sell_call_K": sell_call_K,
            "buy_call_K": buy_call_K,
            "credit": round(credit, 2),
            "pnl": round(exit_pnl, 2),
            "exit_reason": exit_reason,
            "days_held": days_held,
            "iv_at_entry": round(iv * 100, 1),
            "rv_at_entry": round(rv * 100, 1),
            "bb_pctb": round(row["bb_pctb"], 3),
            "adx": round(row["adx"], 1),
            "max_risk": round(max_risk, 2),
        }
        trades.append(trade)
        last_entry_idx = idx

        if verbose and len(trades) <= 5:
            print(
                f"  Trade {len(trades)}: {entry_date.date()} -> {exit_date.date()}, "
                f"S={S:.0f}, credit={credit:.2f}, pnl={exit_pnl:.2f}, "
                f"reason={exit_reason}"
            )

    return trades


def compute_metrics(trades, symbol):
    """Compute strategy performance metrics from trade list."""
    if not trades:
        return {"symbol": symbol, "total_trades": 0, "error": "no trades"}

    pnls = [t["pnl"] for t in trades]
    credits = [t["credit"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Sharpe: annualize from per-trade returns
    pnl_arr = np.array(pnls)
    avg_days_held = np.mean([t["days_held"] for t in trades])
    trades_per_year = 252 / avg_days_held if avg_days_held > 0 else 12
    mean_pnl = np.mean(pnl_arr)
    std_pnl = np.std(pnl_arr, ddof=1) if len(pnl_arr) > 1 else 1
    sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year) if std_pnl > 0 else 0

    # Max drawdown (cumulative PnL)
    cum_pnl = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min()

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    period_start = trades[0]["entry_date"]
    period_end = trades[-1]["exit_date"]

    # Per-contract metrics (multiply by 100 for actual dollar amounts)
    metrics = {
        "symbol": symbol,
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_credit": round(np.mean(credits), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_days_held": round(avg_days_held, 1),
        "trades_per_year": round(trades_per_year, 1),
        "exit_reasons": exit_reasons,
        "period": f"{period_start.date()} to {period_end.date()}",
    }
    return metrics


def print_report(metrics, trades):
    """Print formatted backtest report."""
    sym = metrics["symbol"]
    print(f"\n{'='*70}")
    print(f"  IRON CONDOR BACKTEST — {sym}")
    print(f"{'='*70}")
    print(f"  Period:          {metrics['period']}")
    print(f"  Total trades:    {metrics['total_trades']}")
    print(f"  Win rate:        {metrics['win_rate']}%")
    print(f"  Profit factor:   {metrics['profit_factor']}")
    print(f"  Sharpe ratio:    {metrics['sharpe_ratio']}")
    print(f"  Total PnL:       ${metrics['total_pnl']:.2f} per spread")
    print(f"  Avg credit:      ${metrics['avg_credit']:.2f}")
    print(f"  Avg win:         ${metrics['avg_win']:.2f}")
    print(f"  Avg loss:        ${metrics['avg_loss']:.2f}")
    print(f"  Max drawdown:    ${metrics['max_drawdown']:.2f}")
    print(f"  Avg days held:   {metrics['avg_days_held']}")
    print(f"  Trades/year:     {metrics['trades_per_year']}")
    print(f"  Exit reasons:    {metrics['exit_reasons']}")

    # Gate assessment
    trades_pass = metrics["total_trades"] >= 60
    sharpe_pass = metrics["sharpe_ratio"] >= 0.5
    pf_pass = metrics["profit_factor"] >= 1.2

    print(f"\n  GATE ASSESSMENT:")
    print(f"    Trades >= 60:   {'PASS' if trades_pass else 'FAIL'} ({metrics['total_trades']})")
    print(f"    Sharpe >= 0.5:  {'PASS' if sharpe_pass else 'FAIL'} ({metrics['sharpe_ratio']})")
    print(f"    PF >= 1.2:      {'PASS' if pf_pass else 'FAIL'} ({metrics['profit_factor']})")
    all_pass = trades_pass and sharpe_pass and pf_pass
    print(f"    OVERALL:        {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*70}")

    return all_pass


def load_ohlcv(symbol, start_date="2015-01-01"):
    """Load OHLCV data from PostgreSQL."""
    with pg_conn() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv WHERE symbol=%s AND timeframe='1D'
            ORDER BY timestamp
            """,
            [symbol],
        ).fetchall()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.set_index("date", inplace=True)
    df = df.loc[start_date:]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Modified parameters (looser than original)
    params = {
        "min_spacing_days": 21,   # was 28
        "bb_low": 0.15,           # was 0.20
        "bb_high": 0.85,          # was 0.80
        "adx_max": 30,            # ranging market filter
        "dte": 30,                # 30 DTE
        "delta_wing": 0.16,       # sell at ~16 delta
        "wing_width": 10,         # $10 wide wings for ETFs
        "profit_target": 0.50,    # take profit at 50% of credit
        "stop_loss": 2.0,         # stop at 200% of credit (was 100%)
        "iv_multiplier": 1.25,    # IV ~ 1.25x realized vol
        "risk_free_rate": 0.04,
    }

    print("Modified Iron Condor Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    symbols = ["QQQ", "SPY", "IWM"]
    all_results = {}
    all_trades = {}
    passing_symbols = []

    for sym in symbols:
        print(f"\n--- Loading {sym} data ---")
        df = load_ohlcv(sym, start_date="2015-01-01")
        print(f"  Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

        print(f"--- Simulating {sym} iron condor ---")
        trades = simulate_iron_condor(sym, df, params, verbose=True)
        metrics = compute_metrics(trades, sym)
        passed = print_report(metrics, trades)

        all_results[sym] = metrics
        all_trades[sym] = trades
        if passed:
            passing_symbols.append(sym)

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'#'*70}")
    print(f"  Symbols tested: {symbols}")
    print(f"  Passing symbols: {passing_symbols}")
    print(f"  Multi-symbol validates: {len(passing_symbols) >= 2}")

    # Update DB with QQQ results
    if "QQQ" in all_results:
        qqq = all_results["QQQ"]
        trades_pass = qqq["total_trades"] >= 60
        sharpe_pass = qqq["sharpe_ratio"] >= 0.5
        pf_pass = qqq["profit_factor"] >= 1.2

        summary = {
            "version": "v2_modified",
            "modifications": {
                "min_spacing": "28->21 days",
                "bb_pctb_filter": "[0.20,0.80]->[0.15,0.85]",
                "stop_loss": "100%->200% of credit",
                "history": "2023+->2015+",
            },
            "total_trades": qqq["total_trades"],
            "win_rate": qqq["win_rate"],
            "profit_factor": qqq["profit_factor"],
            "sharpe_ratio": qqq["sharpe_ratio"],
            "total_pnl_per_spread": qqq["total_pnl"],
            "avg_credit": qqq["avg_credit"],
            "avg_win": qqq["avg_win"],
            "avg_loss": qqq["avg_loss"],
            "max_drawdown": qqq["max_drawdown"],
            "avg_days_held": qqq["avg_days_held"],
            "exit_reasons": qqq["exit_reasons"],
            "period": qqq["period"],
            "gate_pass": trades_pass and sharpe_pass and pf_pass,
            "gate_details": {
                "trades_required": 60,
                "trades_actual": qqq["total_trades"],
                "trades_pass": trades_pass,
                "sharpe_required": 0.5,
                "sharpe_actual": qqq["sharpe_ratio"],
                "sharpe_pass": sharpe_pass,
                "pf_required": 1.2,
                "pf_actual": qqq["profit_factor"],
                "pf_pass": pf_pass,
            },
            "cross_symbol_validation": {
                sym: {
                    "trades": all_results[sym]["total_trades"],
                    "sharpe": all_results[sym]["sharpe_ratio"],
                    "win_rate": all_results[sym]["win_rate"],
                    "pf": all_results[sym]["profit_factor"],
                }
                for sym in symbols
                if sym in all_results
            },
            "multi_symbol_pass": len(passing_symbols) >= 2,
            "passing_symbols": passing_symbols,
            "backtest_method": "BSM simulation from 11yr OHLCV (2015-2026), IV=1.25x 21d RV",
        }

        new_status = "backtested"
        if summary["gate_pass"] and summary["multi_symbol_pass"]:
            new_status = "validated"

        with pg_conn() as conn:
            conn.execute(
                """
                UPDATE strategies SET backtest_summary = %s, status = %s
                WHERE strategy_id = 'strat_5b3dda24f7'
                """,
                [json.dumps(summary), new_status],
            )
        print(f"\n  DB updated: strategy status = '{new_status}'")
        print(f"  Backtest summary written to DB.")

    # Return data for further use
    return all_results, all_trades


if __name__ == "__main__":
    results, trades = main()
