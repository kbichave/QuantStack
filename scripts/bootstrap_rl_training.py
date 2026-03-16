#!/usr/bin/env python3
"""
Bootstrap RL Training Script.

Fetches 2 years of OHLCV data via AlphaVantage, trains all three RL agents
(SizingRLAgent, ExecutionRLAgent, AlphaSelectionAgent), and saves checkpoints.

Run once before deploying RL tools in production to ensure checkpoints exist.
Without checkpoints the tools degrade gracefully (return {} with shadow=True),
but shadow-period data collected before training is not useful for promotion.

Usage:
    python scripts/bootstrap_rl_training.py --symbols SPY QQQ IWM
    python scripts/bootstrap_rl_training.py --symbols SPY --skip-execution
    python scripts/bootstrap_rl_training.py --dry-run   # validate config only

Env vars honoured (same as RLProductionConfig):
    QUANTRL_SIZING_CHECKPOINT_PATH
    QUANTRL_EXECUTION_CHECKPOINT_PATH
    QUANTRL_META_CHECKPOINT_PATH
    ALPHAVANTAGE_API_KEY   (required unless --skip-data)
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap RL agent training")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        help="Symbols to fetch for execution environment training",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=504,  # 2 trading years
        help="Days of historical data to fetch",
    )
    p.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip AlphaVantage fetch (use existing KnowledgeStore data)",
    )
    p.add_argument(
        "--skip-sizing",
        action="store_true",
        help="Skip SizingRLAgent training",
    )
    p.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip ExecutionRLAgent training",
    )
    p.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip AlphaSelectionAgent training",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps per agent (reduce for quick smoke test)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and imports only — do not train or fetch data",
    )
    return p.parse_args()


def validate_imports() -> bool:
    """Check all required modules are importable."""
    ok = True
    for module in [
        "quantcore.rl.config",
        "quantcore.rl.features",
        "quantcore.rl.data_bridge",
        "quantcore.rl.sizing.agent",
        "quantcore.rl.sizing.environment",
        "quantcore.rl.execution.agent",
        "quantcore.rl.execution.environment",
        "quantcore.rl.meta.agent",
        "quantcore.rl.meta.environment",
        "quantcore.rl.training",
        "quant_pod.knowledge.store",
    ]:
        try:
            __import__(module)
            logger.info(f"  [OK] {module}")
        except ImportError as exc:
            logger.error(f"  [FAIL] {module}: {exc}")
            ok = False
    return ok


def bootstrap_data(symbols: list, lookback_days: int, api_key: str) -> None:
    """Fetch historical OHLCV data into KnowledgeStore via AlphaVantage."""
    from datetime import date, timedelta

    from quant_pod.knowledge.store import KnowledgeStore
    from quantcore.rl.data_bridge import KnowledgeStoreRLBridge

    store = KnowledgeStore()
    bridge = KnowledgeStoreRLBridge.from_knowledge_store(store)

    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    logger.info(f"Fetching {lookback_days} days of OHLCV for {symbols} via AlphaVantage...")
    logger.info("Rate limit: 5 calls/min — this will take several minutes for many symbols.")

    bridge.bootstrap_from_alphavantage(
        symbols=symbols,
        start_date=start_date,
        api_key=api_key,
    )
    logger.info("Data bootstrap complete.")


def train_sizing_agent(cfg, store, timesteps: int) -> None:
    """Train SizingRLAgent on real signal history, save checkpoint."""
    logger.info("=== Training SizingRLAgent ===")

    from quantcore.rl.sizing.environment import SizingEnvironment
    from quantcore.rl.sizing.agent import SizingRLAgent
    from quantcore.rl.training import Trainer, TrainingConfig

    env = SizingEnvironment.from_knowledge_store(store=store)

    if env.signals is None:
        logger.warning(
            "  [WARN] No real signals found in KnowledgeStore. "
            "SizingEnvironment will use synthetic signals. "
            "Run the full trading pipeline for a few days first to accumulate real signals."
        )

    agent = SizingRLAgent(state_dim=cfg.sizing_state_dim, action_dim=1)
    train_cfg = TrainingConfig(total_timesteps=timesteps, batch_size=64)

    trainer = Trainer(agent=agent, env=env, config=train_cfg)
    metrics = trainer.train()

    ckpt_path = cfg.sizing_checkpoint_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(ckpt_path))

    logger.info(
        f"  SizingRLAgent trained: "
        f"mean_reward={metrics.get('mean_episode_reward', 'N/A'):.2f}, "
        f"checkpoint={ckpt_path}"
    )


def train_execution_agent(cfg, store, symbols: list, timesteps: int) -> None:
    """Train ExecutionRLAgent on OHLCV data, save checkpoint."""
    logger.info("=== Training ExecutionRLAgent ===")

    from quantcore.rl.execution.environment import ExecutionEnvironment
    from quantcore.rl.execution.agent import ExecutionRLAgent
    from quantcore.rl.data_bridge import KnowledgeStoreRLBridge
    from quantcore.rl.training import Trainer, TrainingConfig

    # Use data for the first symbol with sufficient history
    bridge = KnowledgeStoreRLBridge.from_knowledge_store(store)
    ohlcv = None
    for sym in symbols:
        df = bridge.get_ohlcv_for_execution(sym, lookback_days=504)
        if not df.empty and len(df) >= 100:
            ohlcv = df
            logger.info(f"  Using {sym} ({len(df)} bars) for execution training.")
            break

    if ohlcv is None:
        logger.warning("  No OHLCV data found — training ExecutionRLAgent on synthetic data.")

    env = ExecutionEnvironment(data=ohlcv)
    agent = ExecutionRLAgent(state_dim=cfg.execution_state_dim, action_dim=5)
    train_cfg = TrainingConfig(total_timesteps=timesteps, batch_size=64)

    trainer = Trainer(agent=agent, env=env, config=train_cfg)
    metrics = trainer.train()

    ckpt_path = cfg.execution_checkpoint_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(ckpt_path))

    logger.info(
        f"  ExecutionRLAgent trained: "
        f"mean_reward={metrics.get('mean_episode_reward', 'N/A'):.2f}, "
        f"checkpoint={ckpt_path}"
    )


def train_meta_agent(cfg, store, timesteps: int) -> None:
    """Train AlphaSelectionAgent on real alpha return history, save checkpoint."""
    logger.info("=== Training AlphaSelectionAgent ===")

    from quantcore.rl.meta.environment import AlphaSelectionEnvironment
    from quantcore.rl.meta.agent import AlphaSelectionAgent
    from quantcore.rl.data_bridge import KnowledgeStoreRLBridge
    from quantcore.rl.training import Trainer, TrainingConfig

    bridge = KnowledgeStoreRLBridge.from_knowledge_store(store)
    alpha_names = [
        "WTI_BRENT_SPREAD", "CRACK_SPREAD", "EIA_INVENTORY",
        "MICROSTRUCTURE", "COMMODITY_REGIME", "CROSS_ASSET", "MACRO",
    ]

    if not bridge.has_sufficient_alpha_history(alpha_names, min_observations=20):
        logger.warning(
            "  [WARN] Insufficient alpha history for meta agent training. "
            "Need at least 20 closed trades with signal_type labels. "
            "Skipping AlphaSelectionAgent training — run the pipeline first."
        )
        return

    env = AlphaSelectionEnvironment.from_knowledge_store(
        store=store,
        alpha_names=alpha_names,
    )
    agent = AlphaSelectionAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
    )
    train_cfg = TrainingConfig(total_timesteps=timesteps, batch_size=64)

    trainer = Trainer(agent=agent, env=env, config=train_cfg)
    metrics = trainer.train()

    ckpt_path = cfg.meta_checkpoint_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(ckpt_path))

    logger.info(
        f"  AlphaSelectionAgent trained: "
        f"mean_reward={metrics.get('mean_episode_reward', 'N/A'):.2f}, "
        f"checkpoint={ckpt_path}"
    )


def main() -> int:
    args = parse_args()

    logger.info("QuantStack RL Bootstrap")
    logger.info("=" * 60)

    # Validate imports
    logger.info("Validating imports...")
    if not validate_imports():
        logger.error("Import validation failed. Fix missing dependencies before bootstrapping.")
        return 1

    if args.dry_run:
        logger.info("Dry run complete — all imports OK.")
        return 0

    # Load config
    from quantcore.rl.config import get_rl_config
    cfg = get_rl_config()
    logger.info(f"Config loaded: checkpoints at {cfg.sizing_checkpoint_path.parent}")

    # Resolve KnowledgeStore
    from quant_pod.knowledge.store import KnowledgeStore
    store = KnowledgeStore()

    # Fetch data
    if not args.skip_data:
        import os
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not api_key:
            logger.warning(
                "ALPHAVANTAGE_API_KEY not set. Skipping data fetch. "
                "Agents will train on existing KnowledgeStore data."
            )
        else:
            bootstrap_data(args.symbols, args.lookback_days, api_key)

    # Train agents
    errors = []
    if not args.skip_sizing:
        try:
            train_sizing_agent(cfg, store, args.timesteps)
        except Exception as exc:
            logger.error(f"SizingRLAgent training failed: {exc}")
            errors.append(f"sizing: {exc}")

    if not args.skip_execution:
        try:
            train_execution_agent(cfg, store, args.symbols, args.timesteps)
        except Exception as exc:
            logger.error(f"ExecutionRLAgent training failed: {exc}")
            errors.append(f"execution: {exc}")

    if not args.skip_meta:
        try:
            train_meta_agent(cfg, store, args.timesteps)
        except Exception as exc:
            logger.error(f"AlphaSelectionAgent training failed: {exc}")
            errors.append(f"meta: {exc}")

    if errors:
        logger.error(f"Bootstrap completed with {len(errors)} errors: {errors}")
        return 1

    logger.info("")
    logger.info("Bootstrap complete. RL agents are ready for shadow-mode deployment.")
    logger.info(
        "Next step: run the trading pipeline. RL tools will tag output "
        "[SHADOW – not yet validated] until PromotionGate thresholds are met "
        "(63 trading days minimum for sizing/execution, 126 for meta)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
