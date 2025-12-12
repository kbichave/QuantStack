"""
Configuration classes for QuantCore modules.

Provides typed, validated configuration for:
- Backtesting
- Feature engineering
- RL training
- Risk management

Supports loading from YAML files and environment variables.

Usage:
    from quantcore.core.config import BacktestConfig, load_config_from_yaml

    # Default config
    config = BacktestConfig()

    # From YAML
    config = load_config_from_yaml("configs/backtest/custom.yaml", BacktestConfig)

    # Validate
    config.validate()
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from quantcore.core.errors import ConfigurationError


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Attributes:
        initial_capital: Starting capital for backtest
        max_concurrent_trades: Maximum number of concurrent positions
        commission_per_trade: Fixed commission per trade
        slippage_pct: Slippage as percentage of price
        position_size_pct: Position size as percentage of capital
        stop_loss_atr_multiple: Stop loss as ATR multiple (None to disable)
        take_profit_atr_multiple: Take profit as ATR multiple (None to disable)
    """

    initial_capital: float = 100000.0
    max_concurrent_trades: int = 5
    commission_per_trade: float = 1.0
    slippage_pct: float = 0.001
    position_size_pct: float = 0.1
    stop_loss_atr_multiple: Optional[float] = 2.0
    take_profit_atr_multiple: Optional[float] = 3.0

    # Spread trading specific
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    stop_loss_zscore: Optional[float] = 5.0
    position_size: int = 1000
    spread_cost_per_barrel: float = 0.05

    def validate(self) -> None:
        """Validate configuration values."""
        if self.initial_capital <= 0:
            raise ConfigurationError(
                "initial_capital must be positive", value=self.initial_capital
            )
        if self.position_size_pct <= 0 or self.position_size_pct > 1:
            raise ConfigurationError(
                "position_size_pct must be in (0, 1]", value=self.position_size_pct
            )
        if self.exit_zscore >= self.entry_zscore:
            raise ConfigurationError(
                "exit_zscore must be less than entry_zscore",
                exit_zscore=self.exit_zscore,
                entry_zscore=self.entry_zscore,
            )


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    Attributes:
        include_waves: Compute wave pattern features
        include_rrg: Compute relative rotation graph features
        include_technical_indicators: Compute technical indicators
        enable_moving_averages: Include moving average features
        enable_oscillators: Include oscillator features
        enable_volatility_indicators: Include volatility features
        enable_volume_indicators: Include volume features
        include_trendlines: Compute trendline features
        include_candlestick_patterns: Compute candlestick patterns
        zscore_lookback: Lookback period for z-score calculation
        volatility_lookback: Lookback for volatility calculation
        correlation_lookback: Lookback for correlation calculation
    """

    include_waves: bool = True
    include_rrg: bool = True
    include_technical_indicators: bool = True
    enable_moving_averages: bool = True
    enable_oscillators: bool = True
    enable_volatility_indicators: bool = True
    enable_volume_indicators: bool = True
    include_trendlines: bool = True
    include_candlestick_patterns: bool = True
    include_quant_trend: bool = True
    include_quant_pattern: bool = True
    include_gann_features: bool = True
    include_mean_reversion: bool = True
    include_sentiment_features: bool = True

    # Lookback periods
    zscore_lookback: int = 60
    volatility_lookback: int = 20
    correlation_lookback: int = 60
    trendline_lookback: int = 50
    mr_lookback: int = 20
    mr_zscore_threshold: float = 2.0

    # Lag settings
    lag_features: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.zscore_lookback < 10:
            raise ConfigurationError(
                "zscore_lookback must be at least 10", value=self.zscore_lookback
            )
        if self.volatility_lookback < 5:
            raise ConfigurationError(
                "volatility_lookback must be at least 5", value=self.volatility_lookback
            )


@dataclass
class RLConfig:
    """
    Configuration for reinforcement learning.

    Attributes:
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        hidden_dim: Hidden layer size
        learning_rate: Optimizer learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration decay rate
        batch_size: Training batch size
        buffer_size: Replay buffer size
        target_update_freq: Target network update frequency
        training_steps: Total training steps
        checkpoint_freq: Checkpoint save frequency
    """

    # Network architecture
    state_dim: int = 12
    action_dim: int = 5
    hidden_dim: int = 128

    # Optimization
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft target update

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Training
    batch_size: int = 64
    buffer_size: int = 100000
    target_update_freq: int = 100
    training_steps: int = 10000
    checkpoint_freq: int = 2000

    # Environment
    max_episode_steps: int = 1000
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration values."""
        if self.learning_rate <= 0:
            raise ConfigurationError(
                "learning_rate must be positive", value=self.learning_rate
            )
        if not 0 < self.gamma <= 1:
            raise ConfigurationError("gamma must be in (0, 1]", value=self.gamma)
        if self.batch_size < 1:
            raise ConfigurationError(
                "batch_size must be at least 1", value=self.batch_size
            )


@dataclass
class RiskConfig:
    """
    Configuration for risk management.

    Attributes:
        max_risk_per_trade_bps: Maximum risk per trade in basis points
        max_daily_risk_pct: Maximum daily risk as percentage of portfolio
        max_drawdown_pct: Maximum allowed drawdown percentage
        soft_stop_drawdown_pct: Drawdown to trigger size reduction
        hard_stop_drawdown_pct: Drawdown to halt trading
        max_position_value: Maximum value of a single position
        max_portfolio_leverage: Maximum portfolio leverage
        var_confidence: VaR confidence level (e.g., 0.95)
        var_horizon_days: VaR horizon in days
    """

    max_risk_per_trade_bps: float = 25.0
    max_daily_risk_pct: float = 1.0
    max_drawdown_pct: float = 20.0
    soft_stop_drawdown_pct: float = 3.0
    hard_stop_drawdown_pct: float = 7.0
    max_position_value: float = 100000.0
    max_portfolio_leverage: float = 1.0
    var_confidence: float = 0.95
    var_horizon_days: int = 1

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_risk_per_trade_bps <= 0:
            raise ConfigurationError(
                "max_risk_per_trade_bps must be positive",
                value=self.max_risk_per_trade_bps,
            )
        if self.max_daily_risk_pct <= 0 or self.max_daily_risk_pct > 100:
            raise ConfigurationError(
                "max_daily_risk_pct must be in (0, 100]", value=self.max_daily_risk_pct
            )
        if not 0.5 <= self.var_confidence < 1:
            raise ConfigurationError(
                "var_confidence must be in [0.5, 1)", value=self.var_confidence
            )


@dataclass
class SpreadTradingConfig:
    """
    Configuration for WTI-Brent spread trading.

    Combines backtest, feature, and risk configs with spread-specific settings.
    """

    # Spread parameters
    zscore_lookback: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    stop_loss_zscore: Optional[float] = 5.0

    # Position sizing
    position_size_barrels: int = 1000
    max_position_value: float = 100000.0

    # Costs
    spread_cost_per_barrel: float = 0.05
    slippage_pct: float = 0.001

    # Timing
    max_holding_bars: int = 50

    # Train/Val/Test split dates
    train_end_date: str = "2018-01-01"
    val_end_date: str = "2021-01-01"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.exit_zscore >= self.entry_zscore:
            raise ConfigurationError(
                "exit_zscore must be less than entry_zscore",
                exit_zscore=self.exit_zscore,
                entry_zscore=self.entry_zscore,
            )
        if (
            self.stop_loss_zscore is not None
            and self.stop_loss_zscore <= self.entry_zscore
        ):
            raise ConfigurationError(
                "stop_loss_zscore must be greater than entry_zscore",
                stop_loss_zscore=self.stop_loss_zscore,
                entry_zscore=self.entry_zscore,
            )


# Type alias for all config types
ConfigType = Union[
    BacktestConfig, FeatureConfig, RLConfig, RiskConfig, SpreadTradingConfig
]


def load_config_from_yaml(
    path: Union[str, Path],
    config_class: type,
    validate: bool = True,
) -> ConfigType:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file
        config_class: Configuration class to instantiate
        validate: Whether to validate after loading

    Returns:
        Configuration instance

    Raises:
        ConfigurationError: If file not found or invalid
    """
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}", path=str(path)
        )

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {e}", path=str(path)
        )

    if data is None:
        data = {}

    # Filter to only fields that exist in the config class
    valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    config = config_class(**filtered_data)

    if validate and hasattr(config, "validate"):
        config.validate()

    return config


def save_config_to_yaml(
    config: ConfigType,
    path: Union[str, Path],
) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration instance
        path: Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: ConfigType) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
    return asdict(config)


def merge_configs(
    base: ConfigType,
    override: Dict[str, Any],
) -> ConfigType:
    """
    Merge override dictionary into base config.

    Args:
        base: Base configuration
        override: Dictionary of overrides

    Returns:
        New configuration with overrides applied
    """
    base_dict = asdict(base)
    base_dict.update(override)
    return type(base)(**base_dict)
