"""
Application settings and configuration management.

Uses pydantic-settings for validation and environment variable loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars like openai_api_key
    )

    # Alpha Vantage API
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
    symbols: List[str] = Field(
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
        default="2024-12-31", description="End date for historical data"
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

    @property
    def total_transaction_cost_bps(self) -> float:
        """Total round-trip transaction cost in basis points."""
        return (self.spread_cost_bps + self.slippage_cost_bps + self.fee_cost_bps) * 2

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/trader.log", description="Log file path")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
