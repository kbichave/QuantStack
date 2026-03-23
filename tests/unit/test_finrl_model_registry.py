"""Tests for quantstack.finrl.model_registry."""

import duckdb
import pytest

from quantstack.finrl.model_registry import ModelRegistry


@pytest.fixture
def registry():
    """In-memory DuckDB registry."""
    conn = duckdb.connect(":memory:")
    reg = ModelRegistry(conn)
    yield reg
    conn.close()


class TestModelRegistry:
    def test_register_and_get(self, registry):
        registry.register(
            model_id="test_001",
            env_type="execution",
            algorithm="dqn",
            checkpoint_path="/tmp/model",
            name="Test DQN",
            symbols=["SPY"],
        )
        model = registry.get("test_001")
        assert model is not None
        assert model["model_id"] == "test_001"
        assert model["env_type"] == "execution"
        assert model["algorithm"] == "dqn"
        assert model["status"] == "shadow"

    def test_list_models_empty(self, registry):
        models = registry.list_models()
        assert models == []

    def test_list_models_with_filter(self, registry):
        registry.register("m1", "execution", "dqn", "/tmp/m1")
        registry.register("m2", "sizing", "ppo", "/tmp/m2")
        registry.register("m3", "execution", "ppo", "/tmp/m3")

        exec_models = registry.list_models(env_type="execution")
        assert len(exec_models) == 2

        ppo_models = registry.list_models(env_type="sizing")
        assert len(ppo_models) == 1

    def test_update_status(self, registry):
        registry.register("m1", "execution", "dqn", "/tmp/m1")
        registry.update_status("m1", "live", reason="passed gate")
        model = registry.get("m1")
        assert model["status"] == "live"
        assert model["promoted_at"] is not None

    def test_update_eval_metrics(self, registry):
        registry.register("m1", "execution", "dqn", "/tmp/m1")
        metrics = {"sharpe_ratio": 1.2, "max_drawdown": 0.08}
        registry.update_eval_metrics("m1", metrics)
        model = registry.get("m1")
        assert model["eval_metrics"]["sharpe_ratio"] == 1.2

    def test_delete(self, registry):
        registry.register("m1", "execution", "dqn", "/tmp/m1")
        registry.delete("m1")
        assert registry.get("m1") is None

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_symbols_stored_as_json(self, registry):
        registry.register("m1", "execution", "dqn", "/tmp/m1", symbols=["SPY", "QQQ"])
        model = registry.get("m1")
        assert model["symbols"] == ["SPY", "QQQ"]
