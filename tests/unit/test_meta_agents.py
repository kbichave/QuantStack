"""Tests for meta-agent guardrails, prompt optimizer, and conventions."""

from __future__ import annotations

from quantstack.meta.guardrails import (
    META_COMMIT_PREFIX,
    PROTECTED_FILES,
    check_sharpe_regression,
    is_protected,
    validate_meta_change,
)
from quantstack.meta.prompt_optimizer import (
    MAX_VARIANTS_PER_WEEK,
    apply_ab_split,
    evaluate_ab_results,
)


# --- Prompt optimizer ---


def test_max_variants_per_week_is_3():
    assert MAX_VARIANTS_PER_WEEK == 3


def test_ab_split_adds_variant_field():
    config = {"agent_id": "scanner", "model": "sonnet"}
    result = apply_ab_split(config, "variant_b_prompt")
    assert result["prompt_variant"] == "variant_b_prompt"
    # Original keys preserved.
    assert result["agent_id"] == "scanner"
    assert result["model"] == "sonnet"


def test_evaluate_ab_results_picks_better_sharpe():
    assert evaluate_ab_results([0.5, 0.6], [0.8, 0.9]) == "B"
    assert evaluate_ab_results([0.8, 0.9], [0.5, 0.6]) == "A"
    # Tie goes to A (incumbent).
    assert evaluate_ab_results([0.7], [0.7]) == "A"


# --- Guardrails ---


def test_sharpe_decline_triggers_revert():
    # 20% decline (> 10% threshold) => should revert.
    assert check_sharpe_regression(1.0, 0.8) is True
    # 5% decline (< 10% threshold) => no revert.
    assert check_sharpe_regression(1.0, 0.92) is False


def test_protected_file_blocks_risk_gate():
    assert is_protected("src/quantstack/execution/risk_gate.py") is True


def test_protected_file_blocks_kill_switch():
    assert is_protected("src/quantstack/execution/kill_switch.py") is True


def test_protected_file_blocks_db():
    assert is_protected("src/quantstack/db.py") is True


def test_protected_file_blocks_execution_dir():
    # Anything under execution/ is protected via the glob pattern.
    assert is_protected("src/quantstack/execution/some_new_module.py") is True


def test_non_protected_file_allowed():
    assert is_protected("src/quantstack/tools/langchain/signal_tools.py") is False
    assert is_protected("src/quantstack/meta/config.py") is False


def test_meta_change_blocked_on_test_failure():
    approved, reason = validate_meta_change(
        changed_files=["src/quantstack/meta/config.py"],
        test_result=False,
        sharpe_before=1.0,
        sharpe_after=1.0,
    )
    assert approved is False
    assert "Test suite" in reason


def test_all_meta_changes_use_meta_prefix():
    assert META_COMMIT_PREFIX == "meta:"
