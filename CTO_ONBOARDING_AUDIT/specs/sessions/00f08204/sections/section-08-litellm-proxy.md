# Section 08: LiteLLM Proxy Deployment

## Overview

Deploy LiteLLM as a Docker service that sits between all graph services and LLM providers. All LLM calls route through it. LiteLLM handles provider fallback, circuit breaking, retries, and cost tracking natively. `get_chat_model()` in `src/quantstack/llm/provider.py` becomes a thin wrapper returning a LangChain `ChatOpenAI` pointed at the proxy.

**Why this matters:** Provider availability is checked at startup only today. A mid-execution 429 or API credit exhaustion (as happened 2026-04-05 with the Anthropic key) blocks the entire system with no automatic recovery. LiteLLM adds runtime resilience with zero agent-level code changes.

## Dependencies

- **section-01-consolidate-llm-configs** must be complete: the dual config system (`llm_config.py` vs `llm/provider.py`) must be consolidated into a single routing layer before LiteLLM can be wired as the single entry point.
- **section-07-hardcoded-strings** must be complete: all hardcoded model strings must route through `get_chat_model()` so that LiteLLM proxy routing covers the full codebase, not just agents.

## Tests (Write First)

All tests below should be written as stubs that fail before implementation begins.

### Unit Tests

File: `tests/unit/test_llm_provider.py` (extend existing)

```python
# Test: get_chat_model with LITELLM_PROXY_URL env var returns a ChatOpenAI instance
#   - Set LITELLM_PROXY_URL="http://litellm:4000" in env
#   - Call get_chat_model("heavy")
#   - Assert return type is ChatOpenAI
#   - Assert the instance's base_url points to "http://litellm:4000/v1"
#   - Assert the model name passed is "heavy" (tier name used directly, no translation)

# Test: get_chat_model without LITELLM_PROXY_URL falls back to direct provider routing
#   - Ensure LITELLM_PROXY_URL is NOT set
#   - Call get_chat_model("heavy")
#   - Assert return type is NOT ChatOpenAI (should be ChatBedrock, ChatAnthropic, etc.)
#   - This confirms backward compatibility for local dev without LiteLLM running

# Test: tier names map directly to LiteLLM model names (no translation layer)
#   - For each tier in ("heavy", "medium", "light", "bulk"):
#     - Set LITELLM_PROXY_URL
#     - Call get_chat_model(tier)
#     - Assert the model parameter on the returned ChatOpenAI is exactly the tier string
#   - This ensures the LiteLLM config.yaml model group names match the tier names

# Test: temperature parameter is forwarded through LiteLLM path
#   - Set LITELLM_PROXY_URL
#   - Call get_chat_model("heavy", temperature=0.7)
#   - Assert the returned ChatOpenAI has temperature=0.7

# Test: thinking parameter is forwarded as extra_body through LiteLLM path
#   - Set LITELLM_PROXY_URL
#   - Call get_chat_model("heavy", thinking={"type": "adaptive"})
#   - Assert the returned ChatOpenAI has model_kwargs or extra headers for thinking
```

### Integration Tests

File: `tests/integration/test_litellm.py` (new, `@pytest.mark.integration` marker)

These tests require a running LiteLLM Docker service and are skipped in CI unless `LITELLM_PROXY_URL` is set.

```python
import pytest

pytestmark = pytest.mark.integration

# Test: LiteLLM health endpoint responds
#   - GET http://localhost:4000/health
#   - Assert 200 status

# Test: request through LiteLLM returns valid LLM response
#   - Send a simple chat completion to http://localhost:4000/v1/chat/completions
#     with model="heavy"
#   - Assert response contains choices[0].message.content

# Test: simulated 429 triggers fallback to next provider
#   - This requires LiteLLM's test mode or a mock provider
#   - Verify that when the primary provider (Bedrock) returns 429,
#     LiteLLM routes to the next provider (Anthropic)
#   - Check LiteLLM logs or response headers for fallback indication

# Test: provider cooldown prevents requests to cooled-down provider
#   - After 3 consecutive failures on a provider, verify that
#     subsequent requests skip the cooled-down provider for cooldown_time seconds

# Test: Zscaler cert is mounted and provider connections succeed
#   - Verify the container has /etc/ssl/certs/zscaler.pem
#   - Verify SSL_CERT_FILE and REQUESTS_CA_BUNDLE are set in the container env
#   - Make a real LLM call and confirm no certificate errors
```

## Implementation

### Step 1: Create `litellm_config.yaml`

File: `litellm_config.yaml` (project root, next to `docker-compose.yml`)

This file defines model groups that LiteLLM uses for routing and fallback. The model group names must match the existing tier names (`heavy`, `medium`, `light`) so that `get_chat_model()` can pass the tier name directly as the model parameter with no translation layer.

**Model group definitions:**

- **`heavy`** (Sonnet-class, complex reasoning):
  - Priority 1: `bedrock/us.anthropic.claude-sonnet-4-6` (Bedrock Sonnet)
  - Priority 2: `anthropic/claude-sonnet-4-6` (Anthropic direct)
  - Priority 3: `groq/llama-3.3-70b-versatile` (degraded capability fallback)

- **`medium`** (Haiku-class, structured extraction):
  - Priority 1: `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`
  - Priority 2: `anthropic/claude-haiku-4-5`

- **`light`** (cheapest Haiku, simple coordination):
  - Priority 1: `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`
  - Priority 2: `anthropic/claude-haiku-4-5`

- **`embedding`** (local):
  - Priority 1: `ollama/mxbai-embed-large`

- **`bulk`** (OPRO/TextGrad loops, cost-sensitive):
  - Priority 1: `groq/llama-3.3-70b-versatile`
  - Priority 2: `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`

**Router settings to include:**

```yaml
router_settings:
  routing_strategy: "simple-shuffle"   # round-robin within same priority
  allowed_fails: 3                     # failures before cooldown
  cooldown_time: 60                    # seconds provider is unavailable
  num_retries: 2                       # retries per request before fallback
  retry_after: 5                       # minimum seconds between retries
  enable_pre_call_checks: true         # validate context window before sending
  retry_policy:
    RateLimitErrorRetries: 3
    TimeoutErrorRetries: 2
    AuthenticationErrorRetries: 0      # don't retry auth failures — they won't self-heal
```

Each model deployment in the YAML should include:
- `model_name`: the tier name (e.g., `heavy`)
- `litellm_params.model`: the provider-prefixed model string
- `litellm_params.api_key`: reference to the env var (e.g., `os.environ/ANTHROPIC_API_KEY`)
- `litellm_params.priority`: integer (lower = higher priority)

For Bedrock deployments, include `aws_access_key_id`, `aws_secret_access_key`, and `aws_region_name` in `litellm_params`.

### Step 2: Add LiteLLM Service to Docker Compose

File: `docker-compose.yml`

Add a `litellm` service in the Infrastructure section (after `langfuse`, before the Graph Services section). Key configuration:

- **Image:** `ghcr.io/berriai/litellm:main-latest` (pin to a specific SHA after initial validation)
- **Container name:** `quantstack-litellm`
- **Port:** `127.0.0.1:4000:4000` (exposed to host for debugging; internal Docker network for graph services)
- **Volumes:**
  - `./litellm_config.yaml:/app/config.yaml:ro` (the config file created in Step 1)
  - `~/.zscaler_certifi_bundle.pem:/etc/ssl/certs/zscaler.pem:ro` (Zscaler cert for outbound HTTPS)
- **Command:** `--config /app/config.yaml --port 4000 --detailed_debug` (remove `--detailed_debug` after initial validation)
- **Environment variables:**
  - `SSL_CERT_FILE=/etc/ssl/certs/zscaler.pem`
  - `REQUESTS_CA_BUNDLE=/etc/ssl/certs/zscaler.pem`
  - `ANTHROPIC_API_KEY` (from `.env`)
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (for Bedrock)
  - `GROQ_API_KEY`
  - `LITELLM_MASTER_KEY` (for admin API; generate and store in `.env`)
  - `OLLAMA_API_BASE=http://ollama:11434` (for embedding tier, internal Docker network)
- **Health check:** `curl -f http://localhost:4000/health || exit 1` with `interval: 15s`, `timeout: 10s`, `retries: 5`, `start_period: 30s`
- **Depends on:** nothing (standalone service; providers are external)
- **Networks:** `quantstack-net`
- **Memory limit:** `512m`
- **Restart:** `unless-stopped`

Update the three graph services (`trading-graph`, `research-graph`, `supervisor-graph`) to add a `depends_on` entry for litellm with `condition: service_healthy`. Add the `LITELLM_PROXY_URL` environment variable to each, defaulting to empty (opt-in during rollout):

```yaml
environment:
  LITELLM_PROXY_URL: ${LITELLM_PROXY_URL:-}  # Set to http://litellm:4000 to enable
```

### Step 3: Refactor `get_chat_model()` in `src/quantstack/llm/provider.py`

The current `get_chat_model(tier, thinking)` function instantiates provider-specific ChatModel classes directly via `_instantiate_chat_model()`. After this change, it checks for `LITELLM_PROXY_URL` first and, if set, returns a `ChatOpenAI` pointed at the proxy. The existing direct-provider path remains as fallback.

**Signature change** (must be backward compatible with existing callers):

```python
def get_chat_model(
    tier: str,
    thinking: dict | None = None,
    temperature: float | None = None,  # added by section-05
) -> BaseChatModel:
```

**Logic flow:**

1. Validate `tier` (same as current: reject `"embedding"`, reject unknown tiers)
2. Check `os.environ.get("LITELLM_PROXY_URL")`
3. **If set:** return `ChatOpenAI(model=tier, base_url=f"{proxy_url}/v1", temperature=temperature or 0.0, api_key="not-needed")`. The `api_key` is required by `ChatOpenAI` but LiteLLM manages actual keys. If `thinking` is provided, pass it via `model_kwargs` or `extra_body`.
4. **If not set:** fall through to existing `_instantiate_chat_model()` path (unchanged)

The `CHAT_TIERS` set in `src/quantstack/llm/config.py` must be updated to include `"bulk"` so that `get_chat_model("bulk")` works for OPRO/TextGrad loops (added in section-07).

**Key invariant:** The tier name passed to `get_chat_model()` is the exact model name LiteLLM expects. No mapping, no translation. This is why the `litellm_config.yaml` model groups must be named `heavy`, `medium`, `light`, `bulk`.

### Step 4: Configure LiteLLM Callbacks for Observability

LiteLLM supports callback integrations. Configure it to log provider switches and failures to Langfuse so the Supervisor graph's health monitor can detect routing anomalies.

In `litellm_config.yaml`, add a `litellm_settings` section:

```yaml
litellm_settings:
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]
  callbacks: ["langfuse"]
```

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST=http://langfuse:3000` in the litellm service environment so it can report to the internal Langfuse instance.

Events to monitor via Langfuse after deployment:
- **Fallback events:** which provider failed, which took over, the error message
- **Cooldown entry/exit:** which deployment and duration
- **Budget breach:** if per-key budgets are configured later, which agent/key exceeded

### Step 5: Gradual Rollout

This is the migration strategy. Do not flip all graph services at once.

1. **Phase A — Deploy LiteLLM (additive).** Add the service to `docker-compose.yml` and bring it up. Verify `curl http://localhost:4000/health` returns OK. Verify it can reach all providers by sending test requests for each model group. No existing services are changed.

2. **Phase B — Supervisor graph (lowest risk).** Set `LITELLM_PROXY_URL=http://litellm:4000` only for `supervisor-graph` in `docker-compose.yml`. Run for at least one full cycle. Verify in Langfuse that traces show LiteLLM routing (model name will be the tier name, not the full provider string). Check for any errors.

3. **Phase C — Trading graph.** Set `LITELLM_PROXY_URL` for `trading-graph`. Run in paper mode for one full trading session. Compare execution outcomes with pre-LiteLLM runs.

4. **Phase D — Research graph.** Set `LITELLM_PROXY_URL` for `research-graph`. This is the highest-volume service, so monitor cost and latency.

5. **Phase E — Cleanup.** After all graphs confirmed working through LiteLLM for at least one week, consider removing the direct provider initialization code (`_instantiate_chat_model()`, `FALLBACK_ORDER`, `_validate_provider()`) from `provider.py`. Keep it gated behind a feature flag or simply leave it as dead code with a dated comment, since it provides a known-good fallback path.

### Zscaler Considerations

LiteLLM makes outbound HTTPS calls to Anthropic (`api.anthropic.com`), AWS Bedrock (`bedrock-runtime.us-east-1.amazonaws.com`), and Groq (`api.groq.com`). On this network, Zscaler SSL inspection intercepts these connections. Without the Zscaler cert bundle mounted in the container, all outbound calls fail with `certificate verify failed`.

The Docker Compose volume mount (`~/.zscaler_certifi_bundle.pem:/etc/ssl/certs/zscaler.pem:ro`) and environment variables (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`) handle this. Verify connectivity to all three providers during Phase A of the rollout.

If LiteLLM uses `httpx` internally instead of `requests`, you may also need to set `HTTPX_CA_BUNDLE` or `CURL_CA_BUNDLE`. Check LiteLLM's HTTP client library and set the appropriate env var.

## Rollback

If LiteLLM causes issues at any point:

1. Unset `LITELLM_PROXY_URL` for the affected graph service(s) in `docker-compose.yml`
2. Restart the affected service(s): `docker compose up -d <service-name>`
3. The existing direct-provider path in `get_chat_model()` activates automatically when `LITELLM_PROXY_URL` is empty

The LiteLLM Docker service can remain running or be stopped independently. It has no side effects when no clients point to it.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `litellm_config.yaml` | **Create** | Model group definitions, router settings, callback config |
| `docker-compose.yml` | **Modify** | Add `litellm` service; add `LITELLM_PROXY_URL` env var to graph services |
| `src/quantstack/llm/provider.py` | **Modify** | Add LiteLLM proxy path in `get_chat_model()`; keep direct-provider as fallback |
| `src/quantstack/llm/config.py` | **Modify** | Add `"bulk"` to `CHAT_TIERS` and `VALID_TIERS`; add `bulk` tier to `ProviderConfig` and all provider entries |
| `tests/unit/test_llm_provider.py` | **Modify** | Add unit tests for LiteLLM proxy routing |
| `tests/integration/test_litellm.py` | **Create** | Integration tests for health, fallback, cooldown, Zscaler cert |
| `.env.example` | **Modify** | Add `LITELLM_PROXY_URL` and `LITELLM_MASTER_KEY` entries |

## Verification Checklist

After implementation, confirm each of these:

- [ ] `docker compose up litellm` starts and health check passes
- [ ] `curl http://localhost:4000/health` returns 200
- [ ] `curl http://localhost:4000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"heavy","messages":[{"role":"user","content":"hello"}]}'` returns a valid response
- [ ] Unit tests pass with `LITELLM_PROXY_URL` set and unset
- [ ] Langfuse shows LiteLLM-routed traces with correct model metadata
- [ ] All three graph services can be individually toggled to use LiteLLM via env var
- [ ] Removing `LITELLM_PROXY_URL` from a graph service reverts to direct provider routing with no errors
