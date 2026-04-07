# Section 01: Docker Security — Localhost Bindings & Password Hardening

## Background

QuantStack runs its full stack via Docker Compose: PostgreSQL (pgvector), Langfuse (observability), Ollama (embeddings/local LLM), a dashboard, and a FinRL worker. All services currently expose ports to all network interfaces (0.0.0.0), meaning any device on the same network can reach them. Three services also have hardcoded or default-fallback passwords, so if `.env` is missing or incomplete, they start with known credentials.

This section covers two independent but related fixes:

1. **Bind all published ports to 127.0.0.1** so services are only reachable from the host machine.
2. **Remove all default password fallbacks** and add startup validation so the system refuses to start with weak or missing credentials.

These are pure configuration changes with no application code impact. Inter-container communication uses Docker's internal bridge network and is unaffected by host port bindings.

---

## Part A: Bind All Docker Services to Localhost

### What to Change

**File: `docker-compose.yml`**

Every `ports:` mapping that uses the `"HOST:CONTAINER"` format must be prefixed with `127.0.0.1:`. There are five services with published ports:

| Service | Current | Target |
|---------|---------|--------|
| `postgres` (line 36) | `"5434:5432"` | `"127.0.0.1:5434:5432"` |
| `ollama` (line 89) | `"11434:11434"` | `"127.0.0.1:11434:11434"` |
| `langfuse` (line 114) | `"3100:3000"` | `"127.0.0.1:3100:3000"` |
| `dashboard` (line 282) | `"8421:8421"` | `"127.0.0.1:8421:8421"` |
| `finrl-worker` (line 320) | `"8090:8090"` | `"127.0.0.1:8090:8090"` |

Note: `langfuse-db` does not publish any ports (it is only reachable from within the Docker network). No change needed there.

### Why This Is Safe

Docker Compose services communicate via the `quantstack-net` bridge network using container names as hostnames (e.g., `postgres:5432`, `ollama:11434`). Published ports are only used for host-to-container access (e.g., your local Python process connecting to `localhost:5434`). Binding to `127.0.0.1` restricts host-level access to the loopback interface without affecting inter-container traffic at all.

### Remote Access

If you ever need to reach a service from another machine (e.g., a laptop connecting to a remote dev server), use an SSH tunnel:

```bash
ssh -L 5434:localhost:5434 your-server
```

Do not revert the binding to `0.0.0.0`.

### Manual Verification

After applying:

```bash
docker compose up -d postgres ollama langfuse dashboard finrl-worker
# From the host — should succeed:
curl -s http://127.0.0.1:3100/api/public/health
# From another machine on the same network — should be refused/timeout:
curl -s http://<host-ip>:3100/api/public/health
```

Also verify inter-container connectivity still works by checking that graph services start and pass their health checks (they connect to postgres, ollama, and langfuse via internal Docker DNS).

---

## Part B: Remove Default Password Fallbacks

### What to Change in docker-compose.yml

**File: `docker-compose.yml`**

Three password values currently have hardcoded defaults. Remove the `:-default` fallback syntax so Docker Compose requires the env var to be set:

| Location | Current | Target |
|----------|---------|--------|
| `postgres` → `POSTGRES_PASSWORD` (line 40) | `${POSTGRES_PASSWORD:-quantstack}` | `${POSTGRES_PASSWORD}` |
| `langfuse-db` → `POSTGRES_PASSWORD` (line 66) | `${LANGFUSE_DB_PASSWORD:-langfuse}` | `${LANGFUSE_DB_PASSWORD}` |
| `langfuse` → `DATABASE_URL` (line 119) | `...${LANGFUSE_DB_PASSWORD:-langfuse}@...` | `...${LANGFUSE_DB_PASSWORD}@...` |
| `langfuse` → `LANGFUSE_INIT_USER_PASSWORD` (line 131) | `quantstack123` (hardcoded) | `${LANGFUSE_INIT_USER_PASSWORD}` |

Without these env vars set, `docker compose up` will emit warnings about unset variables, and PostgreSQL will refuse to start (empty password is rejected).

**Out of scope for this section:** `NEXTAUTH_SECRET` and `SALT` also have default fallbacks (`change-me-random-secret` and `change-me-random-salt`). These are Langfuse internal auth tokens, not database credentials. Track them for a future hardening pass but do not change them in this PR.

### What to Change in start.sh

**File: `start.sh`**

Add a password validation block between the existing env var check (section 3, lines 47-58) and the infrastructure startup (section 4, line 63). This block runs after `.env` is sourced and after the existing required-var check.

The validation must check three variables:

- `POSTGRES_PASSWORD` — must be set, must not equal `"quantstack"`, must be 12+ characters
- `LANGFUSE_DB_PASSWORD` — must be set, must not equal `"langfuse"`, must be 12+ characters
- `LANGFUSE_INIT_USER_PASSWORD` — must be set, must not equal `"quantstack123"`, must be 12+ characters

If any check fails, print a specific error message naming the variable and the reason (missing, default value, too short), then `exit 1` before any Docker commands run.

The validation logic as a shell snippet (signature/structure only — adapt to match the existing script style):

```bash
# ---------------------------------------------------------------------------
# 3b. Validate passwords (no defaults, no weak values)
# ---------------------------------------------------------------------------
validate_password() {
    local var_name="$1"
    local default_value="$2"
    local value="${!var_name:-}"

    if [[ -z "$value" ]]; then
        echo "ERROR: ${var_name} is not set. Set a strong password (12+ characters) in .env" >&2
        return 1
    fi
    if [[ "$value" == "$default_value" ]]; then
        echo "ERROR: ${var_name} is using the insecure default value '${default_value}'. Change it in .env" >&2
        return 1
    fi
    if [[ ${#value} -lt 12 ]]; then
        echo "ERROR: ${var_name} is too short (${#value} chars). Use 12+ characters." >&2
        return 1
    fi
    return 0
}

PW_ERRORS=0
validate_password "POSTGRES_PASSWORD" "quantstack" || ((PW_ERRORS++))
validate_password "LANGFUSE_DB_PASSWORD" "langfuse" || ((PW_ERRORS++))
validate_password "LANGFUSE_INIT_USER_PASSWORD" "quantstack123" || ((PW_ERRORS++))

if [[ $PW_ERRORS -gt 0 ]]; then
    echo "ERROR: Fix the ${PW_ERRORS} password issue(s) above before starting." >&2
    exit 1
fi
```

### What to Change in .env.example

**File: `.env.example`**

The file already contains `POSTGRES_PASSWORD=quantstack` (line 161) and `LANGFUSE_DB_PASSWORD=langfuse` (line 158). Update these to placeholder values and add the new required variable:

- Change `POSTGRES_PASSWORD=quantstack` to `POSTGRES_PASSWORD=CHANGE_ME_MIN_12_CHARS`
- Change `LANGFUSE_DB_PASSWORD=langfuse` to `LANGFUSE_DB_PASSWORD=CHANGE_ME_MIN_12_CHARS`
- Add `LANGFUSE_INIT_USER_PASSWORD=CHANGE_ME_MIN_12_CHARS` near the other Langfuse settings

### Migration Impact

Anyone running QuantStack without these env vars explicitly set will be broken by this change. The clear error messages from `start.sh` are the migration path — they tell the operator exactly which variable to set and what the requirements are. This is intentional: starting with known credentials is worse than not starting at all.

---

## Tests

**Test file: `tests/unit/test_startup_validation.py`**

These tests validate the `start.sh` password checking logic. Since `start.sh` is a bash script, the tests invoke it as a subprocess and check exit codes and stderr output. The tests should source only the validation portion, not actually start Docker. One approach: extract the validation into a helper that can be tested independently, or run `start.sh` with Docker unavailable (it will fail at the docker check, so mock or skip that prereq).

Alternatively, test the validation logic by running the relevant portion of the script in isolation:

```python
"""Tests for start.sh password validation."""
import subprocess

# Test: start.sh rejects missing POSTGRES_PASSWORD
# Setup: Run start.sh with POSTGRES_PASSWORD unset (empty env)
# Assert: exit code 1, stderr contains "POSTGRES_PASSWORD"

# Test: start.sh rejects default password "quantstack"
# Setup: Run with POSTGRES_PASSWORD=quantstack (plus other valid vars)
# Assert: exit code 1, stderr mentions default value

# Test: start.sh rejects short passwords (< 12 chars)
# Setup: Run with POSTGRES_PASSWORD=short
# Assert: exit code 1, stderr mentions minimum length

# Test: start.sh rejects default LANGFUSE_DB_PASSWORD "langfuse"
# Setup: Run with LANGFUSE_DB_PASSWORD=langfuse
# Assert: exit code 1, stderr mentions LANGFUSE_DB_PASSWORD

# Test: start.sh rejects default LANGFUSE_INIT_USER_PASSWORD "quantstack123"
# Setup: Run with LANGFUSE_INIT_USER_PASSWORD=quantstack123
# Assert: exit code 1, stderr mentions LANGFUSE_INIT_USER_PASSWORD

# Test: start.sh accepts valid passwords (all 3 set, 12+ chars, non-default)
# Setup: Set all three to valid 12+ char non-default values
# Assert: password validation passes (script may fail later at docker check, but
#         stderr does NOT contain any password-related errors)
```

For the docker-compose.yml changes (localhost binding, removed defaults), manual verification is the appropriate approach — there is no meaningful way to unit test YAML configuration. The verification steps are documented in Part A above.

---

## Dependencies

This section has **no dependencies** on other sections. It can be implemented first and in parallel with sections 02, 04, and 06.

No other section depends on this one either — the security hardening is orthogonal to all application-level changes.

---

## Checklist

- [ ] `docker-compose.yml`: All 5 port mappings prefixed with `127.0.0.1:`
- [ ] `docker-compose.yml`: `POSTGRES_PASSWORD` fallback removed (line 40)
- [ ] `docker-compose.yml`: `LANGFUSE_DB_PASSWORD` fallback removed (lines 66, 119)
- [ ] `docker-compose.yml`: `LANGFUSE_INIT_USER_PASSWORD` uses env var (line 131)
- [ ] `start.sh`: Password validation block added after existing env var checks
- [ ] `start.sh`: Validation rejects missing, default, and short (<12 char) passwords
- [ ] `start.sh`: Clear error messages name the offending variable
- [ ] `.env.example`: Default passwords replaced with `CHANGE_ME_MIN_12_CHARS` placeholders
- [ ] `.env.example`: `LANGFUSE_INIT_USER_PASSWORD` added
- [ ] Manual: `docker compose up -d` succeeds with valid `.env`
- [ ] Manual: Services unreachable from non-localhost interfaces
- [ ] Manual: Inter-container communication (graph → postgres, graph → ollama) still works
- [ ] Tests: `test_startup_validation.py` passes for all 6 cases
