# Section 03: CI/CD Pipeline Re-enablement

## Background

QuantStack has two GitHub Actions workflows that were disabled by appending `.disabled` to their filenames while active development continued without CI enforcement. The workflows are:

- `.github/workflows/ci.yml.disabled` -- runs on push/PR to `main`/`develop` with 5 stages: ruff lint + format check, mypy type check (execution path only), pytest matrix (3.11/3.12 x ubuntu/macos), bandit security scan (HIGH only), Trivy Docker image scan (CRITICAL/HIGH on main push only). An `all-checks` gate job aggregates lint + test + security as merge requirements. Integration tests run only on schedule/manual dispatch.
- `.github/workflows/release.yml.disabled` -- triggered by `v*.*.*` tags or manual dispatch. Validates version format, runs tests, builds package, publishes to PyPI (trusted publishing), creates GitHub Release, and deploys docs via mkdocs.

The CI workflow uses `uv` for dependency management, targets `packages/quantcore` and `packages/quant_pod` paths (which may need updating since the codebase has been restructured to `src/quantstack/`), and expects a 60% minimum coverage threshold.

This section blocks **section-10-sbom-scanning**, which adds `pip-audit` and `cyclonedx-py` steps to the CI pipeline after it is green.

## Tests

There are no new application test files to create for this section. The testing work IS the implementation: getting the existing test suite and linting tools to pass in CI. The verification criteria are:

1. `ruff check src/ tests/` passes (or all remaining violations are intentionally suppressed with inline comments)
2. `ruff format --check src/ tests/` passes
3. `mypy src/quantstack/execution/ --config-file pyproject.toml` passes (or remaining issues use `# type: ignore[code]` with justification)
4. `pytest tests/ -m "not slow and not integration and not requires_api and not requires_gpu"` passes with 60% coverage (or the threshold is temporarily lowered with a documented rationale)
5. `bandit -r src/ -lll` passes (or false positives are suppressed with `# nosec` and a comment)
6. Docker build succeeds and Trivy scan passes (or unfixable CVEs are listed in `.trivyignore`)
7. The `all-checks` gate job goes green
8. Post-deployment smoke test in `start.sh` verifies container health after `docker compose up`

## Implementation

### Step 1: Rename disabled workflows

Rename both files to remove the `.disabled` suffix:

- `.github/workflows/ci.yml.disabled` -> `.github/workflows/ci.yml`
- `.github/workflows/release.yml.disabled` -> `.github/workflows/release.yml`

This immediately activates CI on the next push. Expect failures -- that is the point. The next steps triage and fix them.

### Step 2: Update stale paths in ci.yml

The existing `ci.yml` references `packages/quantcore` and `packages/quant_pod` in multiple places (ruff targets, mypy targets, coverage targets). The codebase has been restructured to `src/quantstack/`. Update all path references:

- Ruff lint: `uv run ruff check src/ tests/`
- Ruff format: `uv run ruff format --check src/ tests/`
- MyPy: `uv run mypy src/quantstack/execution --config-file pyproject.toml`
- Pytest coverage: `--cov=src/quantstack`
- Bandit: `uvx bandit -r src/ -lll --exclude src/quantstack/rl/` (if RL dir exists at new path)

Similarly update `release.yml` if it references old paths (the `pip install -e ".[dev,ml]"` and bare `pytest tests/` should still work if `pyproject.toml` is correct).

### Step 3: Triage and fix CI failures stage by stage

Work through failures in this order (easiest to hardest, most value earliest):

**3a. Ruff (mechanical, fast)**

Run locally:
```bash
ruff check src/ tests/ --fix
ruff format src/ tests/
```

Commit all auto-fixes. Review any remaining violations that require manual intervention. For rules that conflict with the codebase's established style (unlikely but possible), add rule exclusions to `pyproject.toml` under `[tool.ruff]` with a comment explaining the exclusion.

**3b. Pytest (highest value)**

Run the same command CI will use:
```bash
pytest tests/ -v \
  --cov=src/quantstack \
  --cov-report=term-missing \
  --cov-fail-under=60 \
  -m "not slow and not integration and not requires_api and not requires_gpu"
```

Categorize each failure:

| Category | Action |
|----------|--------|
| Genuine bug (logic error, wrong assertion) | Fix the bug |
| Import error (module moved/renamed) | Fix the import path |
| Missing fixture | Add fixture to `tests/conftest.py` or the relevant conftest |
| Environment-dependent (needs DB, API key, Docker) | Add appropriate marker (`@pytest.mark.integration`, `@pytest.mark.requires_api`) so it is excluded from the default CI run |
| Flaky (passes sometimes, fails others) | Mark with `@pytest.mark.skip(reason="flaky: <description of flakiness>")` and open a tracking issue or add a TODO with context |

If coverage is below 60%, either write targeted tests for uncovered critical paths (execution, risk gate, kill switch) or temporarily lower the threshold in `pyproject.toml` with a comment: `# TODO(cicd): restore to 60 after test backfill -- currently at X% as of YYYY-MM-DD`.

**3c. MyPy (execution path only)**

Run:
```bash
mypy src/quantstack/execution --config-file pyproject.toml
```

Focus areas:
- Third-party libraries without type stubs: add `# type: ignore[import-untyped]` or install stubs (`types-requests`, `types-pytz`, etc.)
- Untyped function signatures in the execution path: add return type annotations at minimum
- Do NOT chase 100% type coverage across the full codebase -- that is a separate effort

**3d. Bandit (security)**

Run:
```bash
bandit -r src/ -lll
```

Review HIGH-severity findings. Common patterns:
- B608 (SQL injection): suppress with `# nosec B608 -- parameterized query, no user input` if the SQL is truly safe
- B105/B106 (hardcoded passwords): suppress if the "password" is a test fixture or default value clearly marked as such
- Any real issue: fix it

**3e. Trivy (Docker image scan)**

This only runs on pushes to `main`. Build locally first:
```bash
docker build -t quantstack:test .
```

If Trivy reports CRITICAL/HIGH CVEs:
- Update the base image version in `Dockerfile`
- Update pinned dependency versions in `pyproject.toml`
- For CVEs with no fix available: add to `.trivyignore` with a comment noting the CVE ID, affected package, and why it cannot be fixed yet

### Step 4: Branch protection

After CI is green, enable branch protection on `main` via GitHub settings (or `gh api`):

- Require status checks to pass before merging: enable, select `All Checks Pass` as required
- Do NOT require pull request reviews (solo developer workflow)
- Do NOT require linear history or signed commits (unnecessary overhead for a solo project)

This can be configured via:
```bash
gh api repos/{owner}/{repo}/branches/main/protection \
  -X PUT \
  -f required_status_checks='{"strict":true,"contexts":["All Checks Pass"]}' \
  -f enforce_admins=false \
  -f required_pull_request_reviews=null \
  -f restrictions=null
```

### Step 5: Post-deployment smoke test in start.sh

Add a smoke test block to `start.sh` after the graph services are started (after the existing step 14 health check loop, before the status summary). The existing step 14 already waits for graph health checks but only warns on timeout. The smoke test makes this a hard gate:

```bash
# ---------------------------------------------------------------------------
# 14b. Post-deployment smoke test (hard gate)
# ---------------------------------------------------------------------------
echo "[start.sh] Running post-deployment smoke test..."
SMOKE_OK=true
for svc in postgres trading-graph research-graph supervisor-graph; do
    HEALTH=$(docker compose ps --format json "$svc" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list): data = data[0]
    print(data.get('Health', data.get('health', 'unknown')))
except: print('unknown')
" 2>/dev/null || echo "unknown")
    if [[ "$HEALTH" != *"healthy"* ]]; then
        echo "  FAIL: $svc is not healthy (status: $HEALTH)"
        SMOKE_OK=false
    fi
done

if [[ "$SMOKE_OK" != "true" ]]; then
    echo "ERROR: Post-deployment smoke test failed. Diagnostic logs:" >&2
    docker compose logs --tail=50 >&2
    exit 1
fi
echo "[start.sh] Smoke test passed — all critical services healthy."
```

This catches configuration errors, missing environment variables, and container crashes immediately after deployment rather than discovering them when the first trade attempt fails.

## Key Files

| File | Action |
|------|--------|
| `.github/workflows/ci.yml.disabled` | Rename to `.github/workflows/ci.yml`, update paths |
| `.github/workflows/release.yml.disabled` | Rename to `.github/workflows/release.yml`, review paths |
| `pyproject.toml` | Possibly adjust ruff/mypy/pytest config sections |
| `start.sh` | Add post-deployment smoke test block |
| `.trivyignore` | Create if needed for unfixable CVEs |
| `tests/conftest.py` | May need new fixtures or marker registrations |
| Source files under `src/quantstack/` | Ruff auto-fixes, type annotations, bandit suppressions |

## Dependencies

- **No dependencies on other sections.** This section can be implemented immediately.
- **Blocks section-10-sbom-scanning**, which adds `pip-audit` and `cyclonedx-py` steps to the CI pipeline. Section 10 should only be started after the `all-checks` gate is green.

## Risks and Mitigations

**Risk: Cascade of failures makes CI feel intractable.** The existing workflows have been disabled while significant restructuring occurred (MCP removal, path changes from `packages/` to `src/`). Many failures will be path-related rather than logic bugs.

Mitigation: Fix one stage at a time in the order specified (ruff -> pytest -> mypy -> bandit -> trivy). Get the `all-checks` gate green even if individual stages have suppressed issues. Each suppression must be documented with a reason and tracked for future resolution.

**Risk: Coverage threshold too high for current test state.** The CI requires 60% coverage. If the codebase has grown significantly since CI was disabled, coverage may have dropped.

Mitigation: Temporarily lower the threshold with a documented rationale in `pyproject.toml`. Priority coverage targets are `src/quantstack/execution/` (risk gate, kill switch, order management) since these are the highest-blast-radius code paths.

**Risk: Docker build fails due to dependency changes.** The Dockerfile may reference old package names or have incompatible base images.

Mitigation: The Docker/Trivy stage only runs on `main` pushes, not PRs. Fix it after the core stages (lint, test, security) are green. This prevents Docker issues from blocking the entire CI pipeline.
