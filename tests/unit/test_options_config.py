# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.config.options_config module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml

from quantcore.config.options_config import (
    VolatilityRegime,
    Sector,
    SymbolConfig,
    UniverseConfig,
    OptionsConfigLoader,
    get_options_config,
    get_symbol_config,
)


class TestVolatilityRegimeEnum:
    """Test VolatilityRegime enum."""

    def test_values(self):
        """Test enum values."""
        assert VolatilityRegime.LOW.value == "LOW"
        assert VolatilityRegime.MEDIUM.value == "MEDIUM"
        assert VolatilityRegime.HIGH.value == "HIGH"

    def test_count(self):
        """Test number of regimes."""
        assert len(VolatilityRegime) == 3


class TestSectorEnum:
    """Test Sector enum."""

    def test_major_sectors(self):
        """Test major sector values exist."""
        assert Sector.TECHNOLOGY.value == "TECHNOLOGY"
        assert Sector.HEALTHCARE.value == "HEALTHCARE"
        assert Sector.FINANCIALS.value == "FINANCIALS"
        assert Sector.ENERGY.value == "ENERGY"

    def test_special_sectors(self):
        """Test special sector values."""
        assert Sector.BENCHMARK.value == "BENCHMARK"
        assert Sector.ETF.value == "ETF"
        assert Sector.SMALL_CAP.value == "SMALL_CAP"


class TestSymbolConfig:
    """Test SymbolConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic symbol config."""
        config = SymbolConfig(
            symbol="AAPL",
            name="Apple Inc",
            sector=Sector.TECHNOLOGY,
        )

        assert config.symbol == "AAPL"
        assert config.name == "Apple Inc"
        assert config.sector == Sector.TECHNOLOGY
        assert config.is_etf is False

    def test_default_values(self):
        """Test default values are set correctly."""
        config = SymbolConfig(
            symbol="TEST",
            name="Test Stock",
            sector=Sector.TECHNOLOGY,
        )

        assert config.iv_rank_low == 30.0
        assert config.iv_rank_high == 70.0
        assert config.max_contracts == 10
        assert config.min_dte == 20
        assert config.max_dte == 45
        assert config.preferred_dte == 30
        assert config.earnings_buffer_days == 5
        assert config.volatility_regime == VolatilityRegime.MEDIUM
        assert config.min_option_volume == 100
        assert config.max_spread_pct == 0.10
        assert config.notes == ""

    def test_etf_config(self):
        """Test ETF configuration."""
        config = SymbolConfig(
            symbol="SPY",
            name="SPDR S&P 500",
            sector=Sector.BENCHMARK,
            is_etf=True,
        )

        assert config.is_etf is True

    def test_custom_parameters(self):
        """Test custom parameter values."""
        config = SymbolConfig(
            symbol="TSLA",
            name="Tesla Inc",
            sector=Sector.CONSUMER_DISCRETIONARY,
            iv_rank_low=40.0,
            iv_rank_high=80.0,
            max_contracts=5,
            volatility_regime=VolatilityRegime.HIGH,
        )

        assert config.iv_rank_low == 40.0
        assert config.iv_rank_high == 80.0
        assert config.max_contracts == 5
        assert config.volatility_regime == VolatilityRegime.HIGH


class TestUniverseConfig:
    """Test UniverseConfig dataclass."""

    @pytest.fixture
    def sample_config(self) -> UniverseConfig:
        """Create sample universe config."""
        return UniverseConfig(
            benchmarks=[
                SymbolConfig("SPY", "S&P 500 ETF", Sector.BENCHMARK, is_etf=True),
                SymbolConfig("QQQ", "Nasdaq 100 ETF", Sector.BENCHMARK, is_etf=True),
            ],
            sector_etfs=[
                SymbolConfig("XLK", "Tech Select SPDR", Sector.TECHNOLOGY, is_etf=True),
            ],
            other_etfs=[
                SymbolConfig("IWM", "Russell 2000 ETF", Sector.SMALL_CAP, is_etf=True),
            ],
            stocks=[
                SymbolConfig("AAPL", "Apple Inc", Sector.TECHNOLOGY),
                SymbolConfig("MSFT", "Microsoft Corp", Sector.TECHNOLOGY),
            ],
        )

    def test_all_symbols(self, sample_config):
        """Test all_symbols property."""
        all_syms = sample_config.all_symbols
        assert len(all_syms) == 6
        symbols = [s.symbol for s in all_syms]
        assert "SPY" in symbols
        assert "AAPL" in symbols

    def test_all_etfs(self, sample_config):
        """Test all_etfs property."""
        etfs = sample_config.all_etfs
        assert len(etfs) == 4  # 2 benchmarks + 1 sector + 1 other
        for etf in etfs:
            assert etf.is_etf is True

    def test_symbol_list(self, sample_config):
        """Test symbol_list property."""
        symbols = sample_config.symbol_list
        assert len(symbols) == 6
        assert "SPY" in symbols
        assert "AAPL" in symbols

    def test_tradable_symbols(self, sample_config):
        """Test tradable_symbols property."""
        tradable = sample_config.tradable_symbols
        assert len(tradable) == 6

    def test_get_symbol(self, sample_config):
        """Test get_symbol method."""
        aapl = sample_config.get_symbol("AAPL")
        assert aapl is not None
        assert aapl.symbol == "AAPL"
        assert aapl.sector == Sector.TECHNOLOGY

    def test_get_symbol_not_found(self, sample_config):
        """Test get_symbol returns None for unknown symbol."""
        result = sample_config.get_symbol("UNKNOWN")
        assert result is None

    def test_get_sector_etf(self, sample_config):
        """Test get_sector_etf method."""
        tech_etf = sample_config.get_sector_etf(Sector.TECHNOLOGY)
        assert tech_etf == "XLK"

    def test_get_sector_etf_not_found(self, sample_config):
        """Test get_sector_etf returns None for missing sector."""
        result = sample_config.get_sector_etf(Sector.HEALTHCARE)
        assert result is None

    def test_get_symbols_by_sector(self, sample_config):
        """Test get_symbols_by_sector method."""
        tech_symbols = sample_config.get_symbols_by_sector(Sector.TECHNOLOGY)
        assert "XLK" in tech_symbols
        assert "AAPL" in tech_symbols
        assert "MSFT" in tech_symbols


class TestOptionsConfigLoader:
    """Test OptionsConfigLoader class."""

    def test_default_config_when_no_file(self):
        """Test default config is created when YAML not found."""
        with TemporaryDirectory() as tmpdir:
            loader = OptionsConfigLoader(config_dir=Path(tmpdir))
            config = loader.load()

            assert len(config.benchmarks) == 2
            assert config.benchmarks[0].symbol == "SPY"
            assert len(config.stocks) == 2

    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create universe.yaml
            universe_data = {
                "defaults": {
                    "min_dte": 25,
                    "max_dte": 50,
                    "iv_rank_low": 25.0,
                    "iv_rank_high": 75.0,
                },
                "benchmarks": [
                    {
                        "symbol": "SPY",
                        "name": "S&P 500",
                        "sector": "BENCHMARK",
                        "is_etf": True,
                    },
                ],
                "stocks": [
                    {"symbol": "NVDA", "name": "NVIDIA", "sector": "TECHNOLOGY"},
                ],
            }

            with open(config_dir / "universe.yaml", "w") as f:
                yaml.dump(universe_data, f)

            loader = OptionsConfigLoader(config_dir=config_dir)
            config = loader.load()

            assert len(config.benchmarks) == 1
            assert config.benchmarks[0].symbol == "SPY"
            assert len(config.stocks) == 1
            assert config.stocks[0].symbol == "NVDA"

            # Check defaults were applied
            assert config.default_min_dte == 25
            assert config.default_max_dte == 50

    def test_ticker_params_override(self):
        """Test ticker-specific parameters override defaults."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create universe.yaml
            universe_data = {
                "defaults": {"iv_rank_low": 30.0, "iv_rank_high": 70.0},
                "stocks": [
                    {"symbol": "TSLA", "name": "Tesla", "sector": "TECHNOLOGY"},
                ],
            }

            # Create ticker_params.yaml
            ticker_data = {
                "TSLA": {
                    "iv_rank_low": 40.0,
                    "iv_rank_high": 80.0,
                    "volatility_regime": "HIGH",
                    "notes": "High volatility stock",
                },
            }

            with open(config_dir / "universe.yaml", "w") as f:
                yaml.dump(universe_data, f)

            with open(config_dir / "ticker_params.yaml", "w") as f:
                yaml.dump(ticker_data, f)

            loader = OptionsConfigLoader(config_dir=config_dir)
            config = loader.load()

            tsla = config.get_symbol("TSLA")
            assert tsla is not None
            assert tsla.iv_rank_low == 40.0
            assert tsla.iv_rank_high == 80.0
            assert tsla.volatility_regime == VolatilityRegime.HIGH
            assert tsla.notes == "High volatility stock"

    def test_caching(self):
        """Test config is cached after first load."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create config file so we don't get default each time
            universe_data = {
                "defaults": {"min_dte": 20},
                "benchmarks": [
                    {
                        "symbol": "SPY",
                        "name": "S&P 500",
                        "sector": "BENCHMARK",
                        "is_etf": True,
                    }
                ],
            }
            with open(config_dir / "universe.yaml", "w") as f:
                yaml.dump(universe_data, f)

            loader = OptionsConfigLoader(config_dir=config_dir)
            config1 = loader.load()
            config2 = loader.load()

            assert config1 is config2

    def test_get_symbol_params(self):
        """Test get_symbol_params method."""
        with TemporaryDirectory() as tmpdir:
            loader = OptionsConfigLoader(config_dir=Path(tmpdir))
            spy = loader.get_symbol_params("SPY")

            assert spy is not None
            assert spy.symbol == "SPY"

    def test_get_earnings_buffer(self):
        """Test get_earnings_buffer method."""
        with TemporaryDirectory() as tmpdir:
            loader = OptionsConfigLoader(config_dir=Path(tmpdir))

            # Default for known symbol
            buffer = loader.get_earnings_buffer("SPY")
            assert buffer == 5

            # Default for unknown symbol
            buffer = loader.get_earnings_buffer("UNKNOWN")
            assert buffer == 5

    def test_get_iv_thresholds(self):
        """Test get_iv_thresholds method."""
        with TemporaryDirectory() as tmpdir:
            loader = OptionsConfigLoader(config_dir=Path(tmpdir))

            low, high = loader.get_iv_thresholds("SPY")
            assert low == 30.0
            assert high == 70.0

    def test_get_max_contracts(self):
        """Test get_max_contracts method."""
        with TemporaryDirectory() as tmpdir:
            loader = OptionsConfigLoader(config_dir=Path(tmpdir))

            contracts = loader.get_max_contracts("SPY")
            assert contracts == 10


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_options_config(self):
        """Test get_options_config returns UniverseConfig."""
        # Reset singleton
        import quantcore.config.options_config as module

        module._config_loader = None

        config = get_options_config()
        assert isinstance(config, UniverseConfig)

    def test_get_symbol_config(self):
        """Test get_symbol_config function."""
        import quantcore.config.options_config as module

        module._config_loader = None

        # Default config has SPY
        spy = get_symbol_config("SPY")
        assert spy is not None
        assert spy.symbol == "SPY"

    def test_get_symbol_config_not_found(self):
        """Test get_symbol_config returns None for unknown."""
        import quantcore.config.options_config as module

        module._config_loader = None

        result = get_symbol_config("UNKNOWN_SYMBOL_XYZ")
        assert result is None
