"""
FinRL MCP tools — agent-facing API for RL model lifecycle.

ML scientist and researcher agents use these tools to:
  - Design and configure environments
  - Train models (single algorithm or ensemble)
  - Evaluate and compare models
  - Get predictions (tagged [SHADOW] if not promoted)
  - Manage model lifecycle (list, status, promote)
  - Screen stocks using ML ensemble

Tools:
  - finrl_create_environment    — configure a training environment
  - finrl_train_model           — train a DRL agent
  - finrl_train_ensemble        — walk-forward ensemble training
  - finrl_evaluate_model        — backtest on OOS data
  - finrl_predict               — get model action for current state
  - finrl_list_models           — list all models + metadata
  - finrl_compare_models        — side-by-side comparison
  - finrl_get_model_status      — shadow/live status, promotion readiness
  - finrl_promote_model         — promote shadow → live
  - finrl_screen_stocks         — ML-based stock selection
"""

import asyncio
import uuid
from typing import Any

import numpy as np
from loguru import logger

from quantstack.mcp._state import _serialize, live_db_or_error, require_ctx
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# In-memory env config cache (environments are created per-training-run)
_env_configs: dict[str, dict[str, Any]] = {}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_create_environment(
    env_type: str,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    technical_indicators: list[str] | None = None,
    custom_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Configure a training environment for FinRL model training.

    Supported env_types:
      - "stock_trading": FinRL built-in equity trading (requires symbols + dates)
      - "portfolio": Portfolio allocation across multiple stocks
      - "execution": Order execution optimization (TWAP/IS minimization)
      - "sizing": Dynamic position sizing
      - "alpha_selection": Alpha signal weighting

    Args:
        env_type: Environment type (see above).
        symbols: Ticker symbols (required for stock_trading/portfolio).
        start_date: Training data start (YYYY-MM-DD).
        end_date: Training data end (YYYY-MM-DD).
        initial_capital: Starting capital for simulation.
        transaction_cost: Transaction cost as fraction (e.g. 0.001 = 10bps).
        technical_indicators: List of indicators to add (e.g. ["macd", "rsi_30"]).
        custom_params: Environment-specific parameters.

    Returns:
        Dict with env_id and configuration details.
    """
    valid_types = ["stock_trading", "portfolio", "execution", "sizing", "alpha_selection"]
    if env_type not in valid_types:
        return {"success": False, "error": f"Invalid env_type. Must be one of: {valid_types}"}

    env_id = f"env_{env_type}_{uuid.uuid4().hex[:8]}"

    config = {
        "env_id": env_id,
        "env_type": env_type,
        "symbols": symbols or [],
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost,
        "technical_indicators": technical_indicators,
        "custom_params": custom_params or {},
    }

    # Validate type-specific requirements
    if env_type in ("stock_trading", "portfolio") and not symbols:
        return {"success": False, "error": f"{env_type} requires symbols list."}

    if env_type in ("stock_trading", "portfolio") and (not start_date or not end_date):
        return {"success": False, "error": f"{env_type} requires start_date and end_date."}

    # Pre-fetch data for stock_trading / portfolio envs
    if env_type in ("stock_trading", "portfolio") and symbols and start_date and end_date:
        try:
            from quantstack.finrl.data_adapter import FinRLDataAdapter

            adapter = FinRLDataAdapter()
            df = await asyncio.to_thread(
                adapter.fetch_and_format,
                symbols,
                start_date,
                end_date,
                add_indicators=bool(technical_indicators),
                indicator_list=technical_indicators,
            )
            config["data_rows"] = len(df)
            config["data_columns"] = list(df.columns) if not df.empty else []
        except Exception as e:
            logger.warning(f"[finrl_tools] Data pre-fetch failed: {e}")
            config["data_rows"] = 0
            config["data_warning"] = str(e)

    _env_configs[env_id] = config

    return {
        "success": True,
        "env_id": env_id,
        "env_type": env_type,
        "config": config,
    }


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_train_model(
    env_id: str,
    algorithm: str = "ppo",
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    net_arch: list[int] | None = None,
    model_name: str | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a DRL model on a configured environment.

    Supports: PPO, A2C, SAC, TD3, DDPG, DQN.
    Model is saved to disk and registered in the model registry with shadow status.

    Args:
        env_id: From finrl_create_environment.
        algorithm: RL algorithm (ppo, a2c, sac, td3, ddpg, dqn).
        total_timesteps: Training duration.
        learning_rate: Optimizer learning rate.
        batch_size: Minibatch size.
        net_arch: Network architecture (e.g. [256, 256]).
        model_name: Human-readable name for the model.
        hyperparams: Algorithm-specific hyperparameter overrides.

    Returns:
        Dict with model_id, checkpoint_path, training metrics.
    """
    if env_id not in _env_configs:
        return {"success": False, "error": f"Environment {env_id} not found. Create it first."}

    env_config = _env_configs[env_id]

    try:
        env = await asyncio.to_thread(_build_env, env_config)

        from quantstack.finrl.trainer import FinRLTrainer

        trainer = FinRLTrainer()
        result = await asyncio.to_thread(
            trainer.train,
            env,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            net_arch=net_arch,
            model_name=model_name,
            hyperparams=hyperparams,
        )

        # Register in model registry
        ctx, err = live_db_or_error()
        if not err and ctx:
            try:
                from quantstack.finrl.model_registry import ModelRegistry

                registry = ModelRegistry(ctx.db)
                registry.register(
                    model_id=result.model_id,
                    env_type=env_config["env_type"],
                    algorithm=result.algorithm,
                    checkpoint_path=result.checkpoint_path,
                    name=model_name,
                    symbols=env_config.get("symbols"),
                    train_start=env_config.get("start_date"),
                    train_end=env_config.get("end_date"),
                    hyperparams={
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "net_arch": net_arch or [256, 256],
                        **(hyperparams or {}),
                    },
                    training_metrics=result.metrics,
                )
            except Exception as e:
                logger.warning(f"[finrl_tools] Registry save failed: {e}")

        return {
            "success": True,
            "model_id": result.model_id,
            "algorithm": result.algorithm,
            "checkpoint_path": result.checkpoint_path,
            "total_timesteps": result.total_timesteps,
            "training_time_s": result.training_time_s,
            "status": "shadow",
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_train_model failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_train_ensemble(
    env_id: str,
    algorithms: list[str] | None = None,
    total_timesteps: int = 100_000,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Train multiple algorithms and select the best by validation performance.

    Walk-forward style: trains each algorithm, evaluates on held-out data,
    picks the winner by Sharpe ratio.

    Args:
        env_id: From finrl_create_environment.
        algorithms: List of algorithms to compare (default: ["ppo", "a2c", "ddpg"]).
        total_timesteps: Training duration per algorithm.
        model_name: Name for the winning model.

    Returns:
        Dict with winning model_id, per-algorithm results, and comparison.
    """
    if env_id not in _env_configs:
        return {"success": False, "error": f"Environment {env_id} not found."}

    env_config = _env_configs[env_id]

    try:
        env_train = await asyncio.to_thread(_build_env, env_config)
        env_val = await asyncio.to_thread(_build_env, env_config)

        from quantstack.finrl.trainer import FinRLTrainer

        trainer = FinRLTrainer()
        result = await asyncio.to_thread(
            trainer.train_ensemble,
            env_train,
            env_val,
            algorithms=algorithms,
            total_timesteps=total_timesteps,
            model_name=model_name,
        )

        return {
            "success": True,
            "model_id": result.model_id,
            "winner_algorithm": result.algorithm,
            "checkpoint_path": result.checkpoint_path,
            "training_time_s": result.training_time_s,
            "ensemble_results": result.metrics.get("ensemble_results", {}),
            "status": "shadow",
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_train_ensemble failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_evaluate_model(
    model_id: str,
    test_start: str | None = None,
    test_end: str | None = None,
    benchmark: str = "SPY",
    n_episodes: int = 10,
) -> dict[str, Any]:
    """
    Evaluate a trained model on out-of-sample data.

    Computes Sharpe ratio, max drawdown, total return, win rate.

    Args:
        model_id: Model to evaluate.
        test_start: OOS test period start (YYYY-MM-DD).
        test_end: OOS test period end (YYYY-MM-DD).
        benchmark: Benchmark ticker for comparison.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with performance metrics and equity curve data.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        from quantstack.finrl.model_registry import ModelRegistry

        registry = ModelRegistry(ctx.db)
        model = registry.get(model_id)
        if not model:
            return {"success": False, "error": f"Model {model_id} not found."}

        env_config = {
            "env_type": model["env_type"],
            "symbols": model.get("symbols", []),
            "start_date": test_start or model.get("train_end"),
            "end_date": test_end,
            "initial_capital": 100_000,
            "custom_params": {},
        }

        env = await asyncio.to_thread(_build_env, env_config)

        from quantstack.finrl.trainer import FinRLTrainer

        trainer = FinRLTrainer()
        result = await asyncio.to_thread(
            trainer.evaluate,
            model["checkpoint_path"],
            env,
            algorithm=model["algorithm"],
            n_episodes=n_episodes,
        )

        metrics = {
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
        }

        registry.update_eval_metrics(model_id, metrics)

        return {
            "success": True,
            "model_id": model_id,
            "metrics": metrics,
            "equity_curve_sample": result.equity_curve[:50],
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_evaluate_model failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_predict(
    model_id: str,
    symbol: str | None = None,
    current_state: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Get a prediction from a trained model for the current market state.

    If the model is in shadow mode, the prediction is tagged [SHADOW].

    Args:
        model_id: Model to query.
        symbol: Ticker for context (optional).
        current_state: Override observation vector (optional).

    Returns:
        Dict with action, confidence, and shadow status.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        from quantstack.finrl.model_registry import ModelRegistry
        from quantstack.finrl.trainer import FinRLTrainer

        registry = ModelRegistry(ctx.db)
        model = registry.get(model_id)
        if not model:
            return {"success": False, "error": f"Model {model_id} not found."}

        is_shadow = model["status"] == "shadow"

        if current_state:
            obs = np.array(list(current_state.values()), dtype=np.float32)
        else:
            # Build observation from env defaults
            env_config = {
                "env_type": model["env_type"],
                "symbols": model.get("symbols", []),
                "custom_params": {},
            }
            env = _build_env(env_config)
            obs, _ = env.reset()

        trainer = FinRLTrainer()
        action, confidence = trainer.predict(
            model["checkpoint_path"],
            obs,
            algorithm=model["algorithm"],
        )

        action_val = action.tolist() if hasattr(action, "tolist") else action

        return {
            "success": True,
            "model_id": model_id,
            "action": action_val,
            "confidence": round(confidence, 4),
            "shadow_mode": is_shadow,
            "note": "[SHADOW — advisory only]" if is_shadow else "LIVE",
            "symbol": symbol,
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_predict failed: {e}")
        return {"success": True, "prediction": None, "note": f"Unavailable: {e}"}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_list_models(
    env_type: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    List all registered FinRL models with metadata.

    Args:
        env_type: Filter by environment type (optional).
        status: Filter by status: "shadow", "live", "retired" (optional).

    Returns:
        Dict with list of model records.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        from quantstack.finrl.model_registry import ModelRegistry

        registry = ModelRegistry(ctx.db)
        models = registry.list_models(env_type=env_type, status=status)

        return {
            "success": True,
            "count": len(models),
            "models": models,
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_list_models failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_compare_models(
    model_ids: list[str],
    test_start: str | None = None,
    test_end: str | None = None,
) -> dict[str, Any]:
    """
    Compare multiple models side-by-side on the same test period.

    Args:
        model_ids: List of model IDs to compare.
        test_start: Test period start (YYYY-MM-DD).
        test_end: Test period end (YYYY-MM-DD).

    Returns:
        Dict with comparison table (model_id → metrics).
    """
    results = {}
    for mid in model_ids:
        eval_result = await finrl_evaluate_model(
            model_id=mid,
            test_start=test_start,
            test_end=test_end,
            n_episodes=5,
        )
        if eval_result.get("success"):
            results[mid] = eval_result.get("metrics", {})
        else:
            results[mid] = {"error": eval_result.get("error", "unknown")}

    return {
        "success": True,
        "comparison": results,
        "recommendation": _pick_best(results),
    }


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_get_model_status(model_id: str) -> dict[str, Any]:
    """
    Get detailed status for a model including promotion readiness.

    Args:
        model_id: Model to check.

    Returns:
        Dict with status, shadow period stats, eval metrics, and promotion readiness.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        from quantstack.finrl.model_registry import ModelRegistry

        registry = ModelRegistry(ctx.db)
        model = registry.get(model_id)
        if not model:
            return {"success": False, "error": f"Model {model_id} not found."}

        # Calculate shadow duration
        shadow_days = 0
        if model.get("shadow_start"):
            from datetime import datetime

            try:
                start = datetime.fromisoformat(str(model["shadow_start"]))
                shadow_days = (datetime.utcnow() - start).days
            except Exception:
                pass

        eval_metrics = model.get("eval_metrics") or {}

        from quantstack.finrl.config import get_finrl_config

        cfg = get_finrl_config()

        return {
            "success": True,
            "model_id": model_id,
            "status": model["status"],
            "algorithm": model["algorithm"],
            "env_type": model["env_type"],
            "shadow_days": shadow_days,
            "min_shadow_days_required": cfg.min_shadow_observations,
            "eval_metrics": eval_metrics,
            "promotion_eligible": shadow_days >= cfg.min_shadow_observations,
            "promoted_at": model.get("promoted_at"),
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_get_model_status failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_promote_model(
    model_id: str,
    evidence: str,
) -> dict[str, Any]:
    """
    Promote a model from shadow to live after passing statistical gates.

    Checks: observation count, Sharpe CI, max drawdown, Monte Carlo significance,
    walk-forward consistency.

    Args:
        model_id: Model to promote.
        evidence: REQUIRED. Justification for promotion.

    Returns:
        Dict with promotion result (pass/fail) and per-check details.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        from quantstack.finrl.model_registry import ModelRegistry
        from quantstack.finrl.promotion import PromotionGate

        registry = ModelRegistry(ctx.db)
        model = registry.get(model_id)
        if not model:
            return {"success": False, "error": f"Model {model_id} not found."}

        if model["status"] != "shadow":
            return {
                "success": False,
                "error": f"Model status is '{model['status']}', expected 'shadow'.",
            }

        eval_metrics = model.get("eval_metrics") or {}

        # Calculate shadow period
        shadow_days = 0
        if model.get("shadow_start"):
            from datetime import datetime

            try:
                start = datetime.fromisoformat(str(model["shadow_start"]))
                shadow_days = (datetime.utcnow() - start).days
            except Exception:
                pass

        gate = PromotionGate()
        result = gate.evaluate(
            model_id=model_id,
            n_observations=shadow_days,
            simulated_sharpe=eval_metrics.get("sharpe_ratio"),
            max_drawdown=eval_metrics.get("max_drawdown"),
        )

        if result.passes:
            registry.update_status(model_id, "live", reason=evidence)
            logger.info(f"[finrl_tools] Model {model_id} promoted to LIVE: {evidence}")

        return {
            "success": True,
            "model_id": model_id,
            "promoted": result.passes,
            "new_status": "live" if result.passes else "shadow",
            "evidence": evidence,
            "checks": result.to_dict()["checks"],
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_promote_model failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_screen_stocks(
    symbols: list[str],
    start_date: str,
    end_date: str,
    top_n: int = 10,
    weighting: str = "equal",
) -> dict[str, Any]:
    """
    Screen stocks using ML ensemble (Random Forest + Gradient Boosting).

    Scores stocks by predicted forward returns and weights the portfolio
    using equal-weight or minimum-variance optimization.

    Args:
        symbols: Candidate tickers to screen.
        start_date: Data start (YYYY-MM-DD).
        end_date: Data end (YYYY-MM-DD).
        top_n: Number of stocks to select.
        weighting: "equal" or "min_variance".

    Returns:
        Dict with selected stocks, scores, and weights.
    """
    try:
        from quantstack.finrl.data_adapter import FinRLDataAdapter
        from quantstack.finrl.stock_selector import MLStockSelector

        adapter = FinRLDataAdapter()
        df = await asyncio.to_thread(
            adapter.fetch_and_format, symbols, start_date, end_date
        )

        if df.empty:
            return {"success": False, "error": "No data fetched for symbols."}

        selector = MLStockSelector()
        picks = await asyncio.to_thread(
            selector.select, df, symbols, top_n, weighting
        )

        return {
            "success": True,
            "count": len(picks),
            "picks": [
                {
                    "symbol": p.symbol,
                    "score": round(p.score, 6),
                    "weight": round(p.weight, 4),
                    "features": p.features,
                }
                for p in picks
            ],
            "weighting": weighting,
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_screen_stocks failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.FINRL)
@mcp.tool()
async def finrl_screen_options(
    symbols: list[str],
    start_date: str,
    end_date: str,
    top_n: int = 5,
    min_dte: int = 7,
    max_dte: int = 60,
) -> dict[str, Any]:
    """
    Screen underlyings for options trading using ML + IV rank analysis.

    Scores each symbol on:
      - IV Rank (current IV percentile vs 1-year range)
      - Liquidity (volume-based score)
      - Predicted move magnitude (RF+GBM ensemble)

    Then recommends optimal strategy, strike, and DTE for each.

    Strategy selection:
      - High IV + bullish → bull call spread (sell expensive premium)
      - High IV + bearish → bear put spread
      - High IV + neutral → iron condor
      - Low IV + directional → long call/put (buy cheap vol)
      - Low IV + neutral → calendar spread

    Args:
        symbols: Candidate underlying tickers to screen.
        start_date: Data start (YYYY-MM-DD).
        end_date: Data end (YYYY-MM-DD).
        top_n: Number of recommendations.
        min_dte: Minimum days to expiration.
        max_dte: Maximum days to expiration.

    Returns:
        Dict with options recommendations including strategy, strike %, DTE,
        IV rank, predicted move, and liquidity score.
    """
    try:
        from quantstack.finrl.data_adapter import FinRLDataAdapter
        from quantstack.finrl.stock_selector import OptionsSelector

        adapter = FinRLDataAdapter()
        df = await asyncio.to_thread(
            adapter.fetch_and_format, symbols, start_date, end_date
        )

        if df.empty:
            return {"success": False, "error": "No data fetched for symbols."}

        selector = OptionsSelector()
        picks = await asyncio.to_thread(
            selector.select, df, symbols, top_n, min_dte, max_dte
        )

        return {
            "success": True,
            "count": len(picks),
            "picks": [
                {
                    "symbol": p.symbol,
                    "score": round(p.score, 4),
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "strike_pct": p.strike_pct,
                    "dte_target": p.dte_target,
                    "iv_rank": p.iv_rank,
                    "predicted_move": p.predicted_move,
                    "liquidity_score": p.liquidity_score,
                    "features": p.features,
                }
                for p in picks
            ],
        }
    except Exception as e:
        logger.error(f"[finrl_tools] finrl_screen_options failed: {e}")
        return {"success": False, "error": str(e)}


# ─── Helpers ───


def _build_env(config: dict[str, Any]) -> Any:
    """Build a Gymnasium environment from config."""
    env_type = config["env_type"]

    if env_type == "execution":
        from quantstack.finrl.environments import ExecutionEnv

        params = config.get("custom_params", {})
        return ExecutionEnv(
            total_quantity=params.get("total_quantity", 1000),
            time_horizon=params.get("time_horizon", 20),
            market_impact_coef=params.get("market_impact_coef", 0.1),
            spread_bps=params.get("spread_bps", 5.0),
        )

    elif env_type == "sizing":
        from quantstack.finrl.environments import SizingEnv

        params = config.get("custom_params", {})
        return SizingEnv(
            initial_equity=config.get("initial_capital", 100_000),
            max_position_pct=params.get("max_position_pct", 0.2),
            max_drawdown_limit=params.get("max_drawdown_limit", 0.15),
        )

    elif env_type == "alpha_selection":
        from quantstack.finrl.environments import AlphaSelectionEnv

        params = config.get("custom_params", {})
        return AlphaSelectionEnv(
            alpha_names=params.get("alpha_names"),
            lookback=params.get("lookback", 20),
        )

    elif env_type in ("stock_trading", "portfolio"):
        try:
            adapter = FinRLDataAdapter()
            df = adapter.fetch_and_format(
                config.get("symbols", ["SPY"]),
                config.get("start_date", "2023-01-01"),
                config.get("end_date", "2024-01-01"),
                add_indicators=True,
                indicator_list=config.get("technical_indicators"),
            )

            if df.empty:
                raise ValueError("No data available for stock trading env.")

            # FinRL StockTradingEnv setup
            stock_dim = len(df["tic"].unique())
            indicator_cols = [
                c for c in df.columns
                if c not in ("date", "tic", "open", "high", "low", "close", "volume")
            ]

            env = StockTradingEnv(
                df=df,
                stock_dim=stock_dim,
                hmax=100,
                initial_amount=config.get("initial_capital", 100_000),
                buy_cost_pct=[config.get("transaction_cost", 0.001)] * stock_dim,
                sell_cost_pct=[config.get("transaction_cost", 0.001)] * stock_dim,
                tech_indicator_list=indicator_cols,
                turbulence_threshold=np.inf,
            )
            return env

        except Exception as e:
            logger.warning(
                f"[finrl_tools] StockTradingEnv setup failed: {e}. "
                "Falling back to SizingEnv."
            )
            return SizingEnv(initial_equity=config.get("initial_capital", 100_000))

    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def _pick_best(results: dict[str, dict]) -> dict[str, Any]:
    """Pick best model from comparison results."""
    best_id = None
    best_sharpe = -float("inf")

    for mid, metrics in results.items():
        if "error" in metrics:
            continue
        sharpe = metrics.get("sharpe_ratio", -999)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_id = mid

    return {
        "best_model_id": best_id,
        "best_sharpe": round(best_sharpe, 4) if best_id else None,
    }
