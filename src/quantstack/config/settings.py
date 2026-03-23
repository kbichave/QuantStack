"""
Application settings and configuration management.

Uses pydantic-settings for validation and environment variable loading.
"""

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Per-provider settings — separate classes so env prefixes don't collide.
# =============================================================================


class AlpacaSettings(BaseSettings):
    """Alpaca API credentials and mode flags.

    Environment variables (prefix ``ALPACA_``):
      ALPACA_API_KEY    — paper or live API key
      ALPACA_SECRET_KEY — matching secret
      ALPACA_PAPER      — "true" (default) for paper trading endpoint
    """

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", description="Alpaca API key")
    secret_key: str = Field(default="", description="Alpaca secret key")
    paper: bool = Field(default=True, description="Use paper trading endpoint")


class PolygonSettings(BaseSettings):
    """Polygon.io credentials.

    Environment variable: ``POLYGON_API_KEY``
    """

    model_config = SettingsConfigDict(
        env_prefix="POLYGON_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", description="Polygon.io API key")


class IBKRSettings(BaseSettings):
    """Interactive Brokers Gateway connection parameters.

    Environment variables (prefix ``IBKR_``):
      IBKR_HOST      — gateway hostname (default 127.0.0.1)
      IBKR_PORT      — 4001 for IB Gateway (lighter), 7497 for TWS
      IBKR_CLIENT_ID — unique integer per connection (0-999)
      IBKR_TIMEOUT   — connect timeout seconds
    """

    model_config = SettingsConfigDict(
        env_prefix="IBKR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="IB Gateway host")
    port: int = Field(
        default=4001, description="IB Gateway port (4001=gateway, 7497=TWS)"
    )
    client_id: int = Field(
        default=1, description="Client ID (must be unique per connection)"
    )
    timeout: int = Field(default=30, description="Connection timeout in seconds")


class FinancialDatasetsSettings(BaseSettings):
    """FinancialDatasets.ai API credentials and configuration.

    Environment variables (prefix ``FINANCIAL_DATASETS_``):
      FINANCIAL_DATASETS_API_KEY      — API key from financialdatasets.ai
      FINANCIAL_DATASETS_BASE_URL     — API base URL (default: production)
      FINANCIAL_DATASETS_RATE_LIMIT_RPM — requests per minute (Developer=1000)
    """

    model_config = SettingsConfigDict(
        env_prefix="FINANCIAL_DATASETS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", description="FinancialDatasets.ai API key")
    base_url: str = Field(
        default="https://api.financialdatasets.ai",
        description="API base URL",
    )
    rate_limit_rpm: int = Field(
        default=1000, description="Requests per minute (Developer=1000, Pro=unlimited)"
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars like openai_api_key
    )

    # ── Provider settings (nested) ────────────────────────────────────────────
    # Each provider reads its own env-prefix independently.  The nested model
    # approach avoids field name collisions (e.g. api_key is shared by all).
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    polygon: PolygonSettings = Field(default_factory=PolygonSettings)
    ibkr: IBKRSettings = Field(default_factory=IBKRSettings)
    financial_datasets: FinancialDatasetsSettings = Field(
        default_factory=FinancialDatasetsSettings
    )

    # Provider priority: comma-separated list tried left-to-right.
    # Registry skips providers whose credentials are missing.
    # Alpha Vantage is primary — single source for OHLCV, options,
    # fundamentals, macro.  Alpaca kept as fallback for OHLCV only.
    data_provider_priority: str = Field(
        default="alpha_vantage,alpaca",
        description="Comma-separated data provider priority (highest first)",
    )

    # ── Alpha Vantage API (legacy — keep for backward compat) ─────────────────
    alpha_vantage_api_key: str = Field(
        default="demo", description="Alpha Vantage API key"
    )
    alpha_vantage_base_url: str = Field(
        default="https://www.alphavantage.co/query",
        description="Alpha Vantage API base URL",
    )
    alpha_vantage_rate_limit: int = Field(
        default=5, description="API calls per minute (free tier = 5)"
    )

    # Database
    data_dir: Path = Field(
        default=Path("data"), description="Directory for data storage"
    )
    database_path: str = Field(
        default="data/trader.duckdb", description="Path to DuckDB database file"
    )

    # Symbol Universe
    symbols: list[str] = Field(
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
        description="Trading universe symbols",
    )
    benchmark_symbol: str = Field(
        default="SPY", description="Benchmark symbol for relative strength calculations"
    )

    # Data Settings
    data_start_date: str = Field(
        default="2019-01-01", description="Start date for historical data"
    )
    data_end_date: str = Field(
        default="",
        description="End date for historical data (defaults to today if empty)",
    )
    market_timezone: str = Field(
        default="America/New_York", description="Market timezone for data alignment"
    )

    # Training Settings
    train_start_date: str = Field(
        default="2019-01-01", description="Training period start"
    )
    train_end_date: str = Field(default="2022-12-31", description="Training period end")
    validation_start_date: str = Field(
        default="2023-01-01", description="Validation period start"
    )
    validation_end_date: str = Field(
        default="2023-06-30", description="Validation period end"
    )
    test_start_date: str = Field(default="2023-07-01", description="Test period start")
    test_end_date: str = Field(default="2024-12-31", description="Test period end")

    # Model Settings
    model_probability_threshold: float = Field(
        default=0.6, description="Minimum ML probability to take a trade"
    )

    # Risk Settings
    max_risk_per_trade_bps: float = Field(
        default=25.0, description="Maximum risk per trade in basis points"
    )
    max_daily_risk_pct: float = Field(
        default=1.0, description="Maximum daily risk as percentage of portfolio"
    )
    max_concurrent_trades: int = Field(
        default=5, description="Maximum concurrent open positions"
    )
    soft_stop_drawdown_pct: float = Field(
        default=3.0, description="Drawdown percentage to trigger size reduction"
    )
    hard_stop_drawdown_pct: float = Field(
        default=7.0, description="Drawdown percentage to halt trading"
    )

    # Transaction Costs (basis points)
    spread_cost_bps: float = Field(default=2.0, description="Bid-ask spread cost")
    slippage_cost_bps: float = Field(default=1.0, description="Execution slippage")
    fee_cost_bps: float = Field(default=2.0, description="Exchange and clearing fees")

    @model_validator(mode="after")
    def _apply_defaults_and_warn(self) -> "Settings":
        # Default data_end_date to today so data fetches are never silently capped in the past
        if not self.data_end_date:
            self.data_end_date = date.today().isoformat()

        # Warn early if the API key was not configured — avoids silent demo-mode data
        if self.alpha_vantage_api_key == "demo":
            logging.getLogger(__name__).warning(
                "ALPHA_VANTAGE_API_KEY is not set — using the demo key. "
                "Data fetches will be heavily rate-limited. "
                "Set ALPHA_VANTAGE_API_KEY in your .env file."
            )

        return self

    @property
    def total_transaction_cost_bps(self) -> float:
        """Total round-trip transaction cost in basis points."""
        return (self.spread_cost_bps + self.slippage_cost_bps + self.fee_cost_bps) * 2

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/trader.log", description="Log file path")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
