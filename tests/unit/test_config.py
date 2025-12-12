"""
Tests for configuration classes.

Verifies:
1. Default configuration values
2. Validation logic
3. YAML loading/saving
4. Config merging
"""

import tempfile
from pathlib import Path

import pytest

from quantcore.core.config import (
    BacktestConfig,
    FeatureConfig,
    RLConfig,
    RiskConfig,
    SpreadTradingConfig,
    load_config_from_yaml,
    save_config_to_yaml,
    config_to_dict,
    merge_configs,
)
from quantcore.core.errors import ConfigurationError


class TestBacktestConfig:
    """Test BacktestConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.max_concurrent_trades == 5
        assert config.slippage_pct == 0.001

    def test_custom_values(self):
        """Can set custom values."""
        config = BacktestConfig(
            initial_capital=50000.0,
            position_size_pct=0.2,
        )

        assert config.initial_capital == 50000.0
        assert config.position_size_pct == 0.2

    def test_validation_passes_valid_config(self):
        """Valid config passes validation."""
        config = BacktestConfig()
        config.validate()  # Should not raise

    def test_validation_fails_negative_capital(self):
        """Validation fails for negative capital."""
        config = BacktestConfig(initial_capital=-1000)

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validation_fails_invalid_position_size(self):
        """Validation fails for invalid position size."""
        config = BacktestConfig(position_size_pct=1.5)

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validation_fails_exit_gte_entry(self):
        """Validation fails if exit_zscore >= entry_zscore."""
        config = BacktestConfig(entry_zscore=2.0, exit_zscore=2.5)

        with pytest.raises(ConfigurationError):
            config.validate()


class TestFeatureConfig:
    """Test FeatureConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = FeatureConfig()

        assert config.include_waves is True
        assert config.zscore_lookback == 60
        assert config.lag_features is True

    def test_disable_features(self):
        """Can disable individual feature groups."""
        config = FeatureConfig(
            include_waves=False,
            include_rrg=False,
        )

        assert config.include_waves is False
        assert config.include_rrg is False

    def test_validation_passes_valid_config(self):
        """Valid config passes validation."""
        config = FeatureConfig()
        config.validate()  # Should not raise

    def test_validation_fails_small_lookback(self):
        """Validation fails for too small lookback."""
        config = FeatureConfig(zscore_lookback=5)

        with pytest.raises(ConfigurationError):
            config.validate()


class TestRLConfig:
    """Test RLConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = RLConfig()

        assert config.state_dim == 12
        assert config.action_dim == 5
        assert config.hidden_dim == 128
        assert config.learning_rate == 1e-4

    def test_custom_architecture(self):
        """Can configure custom network architecture."""
        config = RLConfig(
            state_dim=20,
            action_dim=10,
            hidden_dim=256,
        )

        assert config.state_dim == 20
        assert config.action_dim == 10
        assert config.hidden_dim == 256

    def test_validation_passes_valid_config(self):
        """Valid config passes validation."""
        config = RLConfig()
        config.validate()  # Should not raise

    def test_validation_fails_negative_lr(self):
        """Validation fails for negative learning rate."""
        config = RLConfig(learning_rate=-0.01)

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validation_fails_invalid_gamma(self):
        """Validation fails for gamma > 1."""
        config = RLConfig(gamma=1.5)

        with pytest.raises(ConfigurationError):
            config.validate()


class TestRiskConfig:
    """Test RiskConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = RiskConfig()

        assert config.max_risk_per_trade_bps == 25.0
        assert config.var_confidence == 0.95

    def test_validation_passes_valid_config(self):
        """Valid config passes validation."""
        config = RiskConfig()
        config.validate()  # Should not raise

    def test_validation_fails_negative_risk(self):
        """Validation fails for negative risk."""
        config = RiskConfig(max_risk_per_trade_bps=-10)

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validation_fails_invalid_var_confidence(self):
        """Validation fails for invalid VaR confidence."""
        config = RiskConfig(var_confidence=1.5)

        with pytest.raises(ConfigurationError):
            config.validate()


class TestSpreadTradingConfig:
    """Test SpreadTradingConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = SpreadTradingConfig()

        assert config.entry_zscore == 2.0
        assert config.exit_zscore == 0.0
        assert config.position_size_barrels == 1000

    def test_validation_passes_valid_config(self):
        """Valid config passes validation."""
        config = SpreadTradingConfig()
        config.validate()  # Should not raise

    def test_validation_fails_exit_gte_entry(self):
        """Validation fails if exit_zscore >= entry_zscore."""
        config = SpreadTradingConfig(entry_zscore=1.5, exit_zscore=2.0)

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validation_fails_stop_less_than_entry(self):
        """Validation fails if stop_loss_zscore <= entry_zscore."""
        config = SpreadTradingConfig(entry_zscore=2.0, stop_loss_zscore=1.5)

        with pytest.raises(ConfigurationError):
            config.validate()


class TestYAMLOperations:
    """Test YAML loading and saving."""

    def test_save_and_load_backtest_config(self):
        """Can save and load BacktestConfig."""
        config = BacktestConfig(
            initial_capital=50000.0,
            position_size_pct=0.15,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backtest.yaml"
            save_config_to_yaml(config, path)

            loaded = load_config_from_yaml(path, BacktestConfig)

            assert loaded.initial_capital == config.initial_capital
            assert loaded.position_size_pct == config.position_size_pct

    def test_save_and_load_rl_config(self):
        """Can save and load RLConfig."""
        config = RLConfig(
            hidden_dim=256,
            learning_rate=5e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rl.yaml"
            save_config_to_yaml(config, path)

            loaded = load_config_from_yaml(path, RLConfig)

            assert loaded.hidden_dim == config.hidden_dim
            assert loaded.learning_rate == config.learning_rate

    def test_load_nonexistent_file(self):
        """Raises error for non-existent file."""
        with pytest.raises(ConfigurationError):
            load_config_from_yaml("/nonexistent/path.yaml", BacktestConfig)

    def test_load_ignores_extra_fields(self):
        """Loading ignores fields not in config class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"

            # Write YAML with extra field
            with open(path, "w") as f:
                f.write("initial_capital: 75000\n")
                f.write("unknown_field: 123\n")

            # Should load without error, ignoring unknown_field
            loaded = load_config_from_yaml(path, BacktestConfig)
            assert loaded.initial_capital == 75000.0


class TestConfigUtilities:
    """Test config utility functions."""

    def test_config_to_dict(self):
        """Can convert config to dictionary."""
        config = BacktestConfig(initial_capital=50000.0)
        result = config_to_dict(config)

        assert isinstance(result, dict)
        assert result["initial_capital"] == 50000.0

    def test_merge_configs(self):
        """Can merge config with overrides."""
        base = BacktestConfig(
            initial_capital=100000.0,
            position_size_pct=0.1,
        )

        override = {"initial_capital": 50000.0}
        merged = merge_configs(base, override)

        assert merged.initial_capital == 50000.0
        assert merged.position_size_pct == 0.1  # Unchanged


class TestConfigValidationWithYAML:
    """Test validation during YAML loading."""

    def test_validation_on_load(self):
        """Validation runs during load by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"

            # Write invalid config
            with open(path, "w") as f:
                f.write("initial_capital: -1000\n")

            with pytest.raises(ConfigurationError):
                load_config_from_yaml(path, BacktestConfig, validate=True)

    def test_skip_validation_on_load(self):
        """Can skip validation during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"

            # Write invalid config
            with open(path, "w") as f:
                f.write("initial_capital: -1000\n")

            # Should not raise with validate=False
            loaded = load_config_from_yaml(path, BacktestConfig, validate=False)
            assert loaded.initial_capital == -1000
