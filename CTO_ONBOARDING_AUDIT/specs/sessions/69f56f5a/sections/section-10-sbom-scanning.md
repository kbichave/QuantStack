# Section 10: SBOM Scanning

## Overview

This section adds Software Bill of Materials (SBOM) generation and dependency vulnerability scanning to the CI pipeline. It introduces two new CI steps: `pip-audit` for scanning installed Python packages against the Python Packaging Advisory Database, and `cyclonedx-py` for generating a machine-readable SBOM artifact on every build.

This is intentionally minimal. The goal is to establish the scanning baseline so vulnerabilities surface automatically in CI rather than being discovered in production.

## Dependency

**Depends on: Section 03 (CI/CD Pipeline Re-enablement).** The CI workflow must be renamed from `ci.yml.disabled` to `ci.yml`, triaged, and passing green before adding new steps. Adding SBOM scanning to a broken pipeline produces noise, not signal.

Do not start this section until the `all-checks` gate job in `.github/workflows/ci.yml` is passing on `main`.

## Tests

There is no application code in this section -- only CI workflow configuration and a dev dependency addition. Verification is that the CI pipeline passes with the new steps included.

**Manual verification checklist:**

- Push a branch with the changes below and confirm the GitHub Actions run completes.
- The `security` job now includes `pip-audit` and `cyclonedx` steps.
- `pip-audit` exits 0 (or exits non-zero only for genuine CRITICAL/HIGH CVEs).
- The `sbom.json` artifact appears in the GitHub Actions run artifacts.
- Known-unfixable CVEs listed in `.pip-audit-known-vulnerabilities` are suppressed and do not fail the build.

## Implementation

### 10.1 Add `cyclonedx-py` as a dev dependency

**File:** `pyproject.toml`

Add `cyclonedx-py` to the dev dependency group (same group that contains `pytest`, `ruff`, etc.). This tool generates CycloneDX-format SBOMs from the installed Python environment.

```toml
# In [project.optional-dependencies] or [dependency-groups] dev section:
"cyclonedx-py>=4.0,<5"
```

After adding, run `uv sync --all-packages --group dev` locally to verify it resolves without conflicts.

### 10.2 Add pip-audit step to CI workflow

**File:** `.github/workflows/ci.yml`

Add a `pip-audit` step to the existing `security` job, after the Bandit scan. The `security` job already has Python and `uv` set up, so the new step can reuse that environment.

The step should use the official GitHub Action `pypa/gh-action-pip-audit@v1.1.0`. Configuration:

- **Vulnerability database:** Python Packaging Advisory Database (default, no configuration needed).
- **Failure threshold:** Fail the build on CRITICAL and HIGH severity vulnerabilities. This matches the severity threshold already used by both Bandit (`-lll` = HIGH only) and Trivy (`severity: "CRITICAL,HIGH"`).
- **Known-vulnerability suppression:** Point to a `.pip-audit-known-vulnerabilities` file that lists CVE IDs for vulnerabilities that have no fix available upstream. This prevents the build from being permanently red due to unfixable transitive dependency issues.

The step belongs in the `security` job (not a new job) because it is a security scan and shares the same environment setup. Adding it as a separate job would duplicate the Python/uv installation steps.

Sketch of the workflow addition (inside the `security` job, after the Bandit step):

```yaml
      - name: Install dependencies for audit
        run: uv sync --all-packages

      - name: Run pip-audit
        uses: pypa/gh-action-pip-audit@v1.1.0
        with:
          # Fail on critical/high only — matches Bandit and Trivy thresholds
          vulnerability-service: pypi
          # Suppress known-unfixable CVEs
          ignore-vulns: .pip-audit-known-vulnerabilities
```

### 10.3 Create the known-vulnerabilities suppression file

**File:** `.pip-audit-known-vulnerabilities` (repository root)

Create this file empty initially. As pip-audit runs and surfaces CVEs that cannot be resolved (because the fix is not yet released upstream or the vulnerable package is a transitive dependency pinned by another library), add CVE IDs one per line with a comment explaining why the suppression exists and when to revisit.

Format:

```
# Each line: CVE-ID  # reason, revisit date
# Example:
# CVE-2024-XXXXX  # transitive via somelib, no fix available as of 2026-04-06
```

An empty file means no suppressions -- the build fails on any CRITICAL/HIGH finding, which is the correct starting posture.

### 10.4 Add CycloneDX SBOM generation step to CI workflow

**File:** `.github/workflows/ci.yml`

Add an SBOM generation step to the `security` job, after `pip-audit`. This step generates a CycloneDX JSON SBOM and uploads it as a build artifact.

Sketch of the workflow addition:

```yaml
      - name: Generate SBOM
        run: uv run cyclonedx-py environment --output sbom.json --output-format json

      - name: Upload SBOM artifact
        uses: actions/upload-artifact@v4
        with:
          name: sbom-${{ github.sha }}
          path: sbom.json
          retention-days: 90
```

The SBOM artifact serves as an audit trail. If a vulnerability is disclosed in a dependency after the build, the SBOM can be scanned retroactively to determine whether the affected version was in use at that point. The 90-day retention balances storage cost against audit utility.

### 10.5 Verify the all-checks gate still includes security

**File:** `.github/workflows/ci.yml`

The existing `all-checks` gate job depends on `[lint, test, security]`. Since `pip-audit` and `cyclonedx` are added as steps within the `security` job (not new jobs), no change is needed to the gate. If either new step fails, the `security` job fails, which fails the gate.

Confirm this by inspecting the `all-checks` job's `needs` array -- it should already include `security`.

## Key Files Summary

| File | Action |
|------|--------|
| `.github/workflows/ci.yml` | Add pip-audit step, SBOM generation step, and artifact upload to `security` job |
| `pyproject.toml` | Add `cyclonedx-py>=4.0,<5` to dev dependencies |
| `.pip-audit-known-vulnerabilities` | Create (initially empty) |

## Risks and Mitigations

**pip-audit may surface unfixable CVEs on first run.** Transitive dependencies frequently have known vulnerabilities with no available fix. If the first CI run fails due to these, triage each finding: if a fix exists, update the dependency; if not, add to `.pip-audit-known-vulnerabilities` with a comment and revisit date. Do not suppress without investigation.

**cyclonedx-py version compatibility.** The `cyclonedx-py` v4.x CLI changed its interface from v3.x. Pin to `>=4.0,<5` and use the v4 command syntax (`cyclonedx-py environment`). If the installed version does not match, the step will fail with a clear error.

**SBOM artifact size.** For a project with 100+ transitive dependencies, the CycloneDX JSON is typically 200-500KB. Well within GitHub Actions artifact limits (500MB). No mitigation needed.

**Build time impact.** pip-audit typically completes in 10-30 seconds (network call to PyPI advisory DB). cyclonedx-py is local-only and completes in under 5 seconds. Combined overhead is under 1 minute, negligible relative to the existing test and Docker build steps.
