# Section 01: Secrets and Env Var Hardening

## Goal

Create a startup-time environment validation layer that hard-crashes on invalid configuration. No silent defaults, no partial boots. If the system starts, it started with valid configuration. Additionally, harden `.env` file permissions, document every variable in `.env.example`, and add credential rotation procedures to the ops runbook.

## Dependencies

- **None.** This section is a foundation piece with no external dependencies.
- **Blocks:** section-08 (kill switch recovery) reads env-validated state.

---

## Tests First

All tests live in `tests/unit/test_env_validation.py`. The validation module under test is `src/quantstack/config/validation.py`.

**Fixtures needed:** an `env_override` context manager that temporarily sets/unsets environment variables and restores originals on exit. Place in `tests/conftest.py` if not already present.

```python
# tests/unit/test_env_validation.py

import pytest
from unittest.mock import patch

# Fixture: env_override — context manager that sets/unsets env vars for each test.
# Example usage: with env_override({"TRADER_PG_URL": "postgresql://localhost/test"}):

# --- Required vars ---

def test_validate_passes_with_all_required_vars_set():
    """All 4 required vars present and valid -> no exception."""

def test_validate_exits_when_TRADER_PG_URL_missing():
    """Missing TRADER_PG_URL -> SystemExit(1) with message naming the var."""

def test_validate_exits_when_ALPHA_VANTAGE_API_KEY_missing():
    """Missing ALPHA_VANTAGE_API_KEY -> SystemExit(1)."""

def test_validate_exits_when_ALPACA_API_KEY_missing():
    """Missing ALPACA_API_KEY -> SystemExit(1)."""

def test_validate_exits_when_ALPACA_SECRET_KEY_missing():
    """Missing ALPACA_SECRET_KEY -> SystemExit(1)."""

# --- Typed vars ---

def test_RISK_MAX_POSITION_PCT_valid_float():
    """RISK_MAX_POSITION_PCT="0.05" -> passes."""

def test_RISK_MAX_POSITION_PCT_non_numeric_exits():
    """RISK_MAX_POSITION_PCT="ten" -> SystemExit(1) with clear error."""

def test_RISK_MAX_POSITION_PCT_out_of_range_exits():
    """RISK_MAX_POSITION_PCT="1.5" -> SystemExit(1) (must be 0.0-1.0)."""

def test_AV_DAILY_CALL_LIMIT_valid_int():
    """AV_DAILY_CALL_LIMIT="25000" -> passes."""

def test_AV_DAILY_CALL_LIMIT_negative_exits():
    """AV_DAILY_CALL_LIMIT="-1" -> SystemExit(1)."""

def test_FORWARD_TESTING_SIZE_SCALAR_valid():
    """FORWARD_TESTING_SIZE_SCALAR="0.5" -> passes."""

def test_FORWARD_TESTING_SIZE_SCALAR_out_of_range():
    """FORWARD_TESTING_SIZE_SCALAR="2.0" -> SystemExit(1) (must be 0.0-1.0)."""

# --- Boolean vars ---

def test_USE_REAL_TRADING_true_passes():
    """USE_REAL_TRADING="true" -> passes."""

def test_USE_REAL_TRADING_True_case_insensitive():
    """USE_REAL_TRADING="True" -> passes (case insensitive)."""

def test_USE_REAL_TRADING_yes_exits():
    """USE_REAL_TRADING="yes" -> SystemExit(1) (only true/false accepted)."""

def test_ALPACA_PAPER_false_passes():
    """ALPACA_PAPER="false" -> passes."""

# --- Optional vars ---

def test_missing_GROQ_API_KEY_warns_but_does_not_exit(caplog):
    """Missing GROQ_API_KEY -> WARNING log but no SystemExit."""

def test_missing_DISCORD_WEBHOOK_URL_warns_but_does_not_exit(caplog):
    """Missing DISCORD_WEBHOOK_URL -> WARNING log but no SystemExit."""

# --- Redaction ---

def test_error_for_api_key_redacts_value():
    """Error message for *_API_KEY vars must NOT contain the raw value."""

def test_error_for_non_secret_var_shows_value():
    """Error message for non-secret vars (e.g. RISK_MAX_POSITION_PCT) shows the actual bad value."""
```

---

## Implementation Details

### 9.1 — `.env` permissions check in `start.sh`

**File:** `start.sh`

Insert a new section between the existing `.env` load (section 1, line ~24) and the prerequisites check (section 2, line ~33). This block must run after confirming `.env` exists but before sourcing it for Docker Compose.

Behavior:
- Read file permissions using `stat`. macOS uses `stat -f '%Lp'`, Linux uses `stat -c '%a'`. Detect the platform and use the right flag.
- If permissions are not `600`: print a warning to stderr explaining the risk, then offer to fix with `chmod 600 .env`. Since `start.sh` runs non-interactively in Docker/CI, the script should **warn and continue** (not block). The warning is for the human running it locally.
- If `.env` does not exist: the existing check on line 24-27 already handles this (errors and exits). No change needed.

The check goes in `start.sh` only — it is a shell-level concern, not a Python concern. The Python validation module (below) handles value correctness.

### 9.2 — Env var type validation module

**New file:** `src/quantstack/config/validation.py`

This module exports a single public function:

```python
def validate_environment() -> None:
    """Validate all critical environment variables at startup.

    Reads os.environ directly. On any validation failure, logs the
    exact variable name, expected type/range, and actual value (redacted
    if the var name contains 'KEY', 'SECRET', 'PASSWORD', or 'TOKEN'),
    then raises SystemExit(1).

    Called once at process startup in each runner, before graph init.
    """
```

**Validation tiers (complete list):**

| Tier | Variables | Rule |
|------|-----------|------|
| **Required** (crash if missing) | `TRADER_PG_URL` | Non-empty, starts with `postgresql://` |
| | `ALPHA_VANTAGE_API_KEY` | Non-empty string |
| | `ALPACA_API_KEY` | Non-empty string |
| | `ALPACA_SECRET_KEY` | Non-empty string |
| **Typed float 0-1** (crash if present but invalid) | `RISK_MAX_POSITION_PCT` | Parses as float, 0.0 <= x <= 1.0 |
| | `FORWARD_TESTING_SIZE_SCALAR` | Same |
| **Typed positive int** | `AV_DAILY_CALL_LIMIT` | Parses as int, > 0 |
| **Typed positive float** | `ROLLING_DRAWDOWN_MULTIPLIER` | Parses as float, > 0 |
| **Boolean** (crash if present but not true/false) | `ALPACA_PAPER` | Case-insensitive `true` or `false` |
| | `USE_REAL_TRADING` | Same |
| | `USE_FORWARD_TESTING_FOR_ENTRIES` | Same |
| | `LANGFUSE_RETENTION_ENABLED` | Same |
| **Optional** (warn if missing) | `GROQ_API_KEY` | Log WARNING, continue |
| | `DISCORD_WEBHOOK_URL` | Log WARNING, continue |
| | `RESEARCH_SYMBOL_OVERRIDE` | Log WARNING, continue |

Design notes:
- Use `os.environ.get()` for each variable. Typed/boolean/optional vars that are absent are simply skipped (they have defaults elsewhere in the codebase). Only required vars crash on absence.
- Redaction rule: if the variable name contains any of `KEY`, `SECRET`, `PASSWORD`, `TOKEN` (case-insensitive), replace the value with `***REDACTED***` in error messages. All other variables show their actual value to aid debugging.
- The function should collect ALL errors before exiting, not fail on the first one. This way the operator sees every misconfiguration in a single run, not one at a time.
- Use `logging.getLogger(__name__)` for output. Call `sys.exit(1)` after logging all errors if any were found.
- No external dependencies beyond stdlib (`os`, `sys`, `logging`, `re`).

### 9.2b — Wiring into runners

**Files to modify:**
- `src/quantstack/runners/trading_runner.py`
- `src/quantstack/runners/research_runner.py`
- `src/quantstack/runners/supervisor_runner.py`

Add at the top of each runner's entry point (before any graph initialization, DB connection, or LLM setup):

```python
from quantstack.config.validation import validate_environment
validate_environment()
```

This ensures every container validates its environment on startup. If validation fails, the container exits immediately with a clear error message, and Docker Compose marks it as failed.

### 9.2c — Export from config package

**File:** `src/quantstack/config/__init__.py`

Add `validate_environment` to the imports and `__all__` list so it is accessible as `quantstack.config.validate_environment`.

### 9.3 — `.env.example` documentation

**File:** `.env.example`

The current `.env.example` (360 lines) already has good section organization and comments. The gaps to fill:

1. Add explicit `# Required` / `# Optional` / `# Type: float, range 0.0-1.0` annotations to every variable that the validation module checks. Currently some variables have comments but not all specify type expectations or whether they are required.

2. Add `LANGFUSE_RETENTION_ENABLED` and `LANGFUSE_RETENTION_DAYS` variables (these come from section-09 but should be documented here for completeness):
   ```
   # Langfuse trace retention (config stub — cleanup not yet implemented)
   LANGFUSE_RETENTION_ENABLED=false    # Type: boolean (true/false)
   LANGFUSE_RETENTION_DAYS=30          # Type: positive integer
   ```

3. Ensure every variable in the validation module's tier table has a matching entry in `.env.example`. Cross-reference and fill any gaps.

### 9.4 — Credential rotation documentation

**File:** `docs/ops-runbook.md`

Append a new `## Credential Rotation` section at the end of the existing runbook. Content:

| Credential | Where to regenerate | What to update | Restart required |
|------------|-------------------|----------------|-----------------|
| Alpha Vantage API key | alphavantage.co account dashboard | `ALPHA_VANTAGE_API_KEY` in `.env` | Yes — all containers |
| Alpaca API key + secret | Alpaca dashboard → Paper/Live → API Keys | `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` in `.env` | Yes — all containers |
| PostgreSQL password | `ALTER USER ... PASSWORD '...'` in psql | `POSTGRES_PASSWORD` + `TRADER_PG_URL` in `.env` | Yes — all containers |
| Langfuse keys | Langfuse Settings → API Keys | `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` in `.env` | Yes — all containers |
| Discord webhook | Server Settings → Integrations → Webhooks | `DISCORD_WEBHOOK_URL` in `.env` | Yes — supervisor-graph |

For each credential, include a short procedure:
1. Generate the new credential in the provider's dashboard
2. Update the value in `.env`
3. Run `./stop.sh && ./start.sh` to restart with new credentials
4. Verify the service works (e.g., check Langfuse traces appear, check a test trade executes in paper mode)

Warn explicitly: **never rotate credentials while positions are open** — the trading graph may fail to manage exits during the restart window. Check `positions` table for `status='open'` before rotating.

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/config/validation.py` | **Create** | `validate_environment()` function with tiered validation |
| `src/quantstack/config/__init__.py` | Modify | Add `validate_environment` to exports |
| `src/quantstack/runners/trading_runner.py` | Modify | Add `validate_environment()` call at top |
| `src/quantstack/runners/research_runner.py` | Modify | Add `validate_environment()` call at top |
| `src/quantstack/runners/supervisor_runner.py` | Modify | Add `validate_environment()` call at top |
| `start.sh` | Modify | Add `.env` permissions check (warn if not 600) |
| `.env.example` | Modify | Add type/range/required annotations to all validated vars |
| `docs/ops-runbook.md` | Modify | Append credential rotation section |
| `tests/unit/test_env_validation.py` | **Create** | Unit tests for validation module |
| `tests/conftest.py` | Modify | Add `env_override` fixture if not present |
