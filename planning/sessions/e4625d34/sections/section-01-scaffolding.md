# Section 01: Project Scaffolding and Docker Compose Stack

## Goal

Create the project directory structure, Docker Compose stack (8 services), updated Dockerfile, and pyproject.toml additions required for the CrewAI migration. This section is the foundation that all other sections depend on. Nothing in this section involves business logic -- it is purely infrastructure and packaging.

## Existing State

The project already has:

- `src/quantstack/` — the full existing codebase (preserved, not modified)
- `src/quantstack/crews/` — contains `decoder_crew.py`, `registry.py`, `schemas.py`, `__init__.py` (existing crew code, preserved)
- `Dockerfile` — single-stage production image using `python:3.11-slim-bookworm` with uv for dependency management
- `start.sh` — current tmux-based launcher (will be updated in section-10, not here)
- `pyproject.toml` — existing dependencies including `langfuse>=2.0.0`, `boto3`, `anthropic`, `pyyaml`, etc.

The following do NOT yet exist and must be created:

- `src/quantstack/crews/trading/`, `research/`, `supervisor/` subdirectories with `config/` YAML dirs
- `src/quantstack/crewai_tools/` (22 tool wrapper modules -- content in section-03)
- `src/quantstack/llm/` (provider management -- content in section-02)
- `src/quantstack/rag/` (RAG pipeline -- content in section-06)
- `src/quantstack/runners/` (continuous loop runners -- content in section-09)
- `src/quantstack/health/` (heartbeat, watchdog, shutdown -- content in section-07)
- `docker-compose.yml` (does not exist yet)

---

## Tests (Write First)

File: `tests/unit/test_scaffolding.py`

```python
"""Tests for Section 01: Project Scaffolding.

Validates that the directory structure, Docker Compose config,
Dockerfile, and pyproject.toml are correctly set up for the
CrewAI migration.
"""

import pathlib
import tomllib

import pytest
import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"


class TestDirectoryStructure:
    """Verify all required directories and __init__.py files exist."""

    @pytest.mark.parametrize("subdir", [
        "crews/trading/config",
        "crews/research/config",
        "crews/supervisor/config",
        "crewai_tools",
        "llm",
        "rag",
        "runners",
        "health",
    ])
    def test_directory_exists(self, subdir: str) -> None:
        assert (SRC_ROOT / subdir).is_dir(), f"Missing directory: src/quantstack/{subdir}"

    @pytest.mark.parametrize("pkg", [
        "crews/trading",
        "crews/research",
        "crews/supervisor",
        "crewai_tools",
        "llm",
        "rag",
        "runners",
        "health",
    ])
    def test_init_py_exists(self, pkg: str) -> None:
        assert (SRC_ROOT / pkg / "__init__.py").is_file(), (
            f"Missing __init__.py in src/quantstack/{pkg}"
        )


class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    @pytest.fixture()
    def compose(self) -> dict:
        path = PROJECT_ROOT / "docker-compose.yml"
        assert path.is_file(), "docker-compose.yml does not exist"
        return yaml.safe_load(path.read_text())

    REQUIRED_SERVICES = [
        "postgres",
        "langfuse-db",
        "langfuse",
        "ollama",
        "chromadb",
        "trading-crew",
        "research-crew",
        "supervisor-crew",
    ]

    def test_all_services_defined(self, compose: dict) -> None:
        services = set(compose.get("services", {}).keys())
        for svc in self.REQUIRED_SERVICES:
            assert svc in services, f"Missing service: {svc}"

    def test_each_service_has_healthcheck(self, compose: dict) -> None:
        for svc_name in self.REQUIRED_SERVICES:
            svc = compose["services"][svc_name]
            assert "healthcheck" in svc, f"Service {svc_name} missing healthcheck"

    def test_crew_services_depend_on_infrastructure(self, compose: dict) -> None:
        infra = {"postgres", "ollama", "chromadb", "langfuse"}
        for crew in ("trading-crew", "research-crew", "supervisor-crew"):
            deps = compose["services"][crew].get("depends_on", {})
            dep_names = set(deps.keys()) if isinstance(deps, dict) else set(deps)
            assert infra.issubset(dep_names), (
                f"{crew} must depend on all infra services; missing {infra - dep_names}"
            )

    def test_named_volumes_defined(self, compose: dict) -> None:
        volumes = set(compose.get("volumes", {}).keys())
        for vol in ("postgres-data", "ollama-data", "chromadb-data", "langfuse-db-data"):
            assert vol in volumes, f"Missing named volume: {vol}"

    def test_shared_network(self, compose: dict) -> None:
        networks = compose.get("networks", {})
        assert len(networks) >= 1, "At least one Docker network must be defined"


class TestPyprojectToml:
    """Validate pyproject.toml has the crewai dependency group."""

    @pytest.fixture()
    def pyproject(self) -> dict:
        path = PROJECT_ROOT / "pyproject.toml"
        return tomllib.loads(path.read_text())

    def test_crewai_optional_group_exists(self, pyproject: dict) -> None:
        optional = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "crewai" in optional, "Missing [project.optional-dependencies] crewai group"

    def test_crewai_group_includes_core_packages(self, pyproject: dict) -> None:
        crewai_deps = pyproject["project"]["optional-dependencies"]["crewai"]
        dep_names = [d.split("[")[0].split(">")[0].split("=")[0].strip() for d in crewai_deps]
        for pkg in ("crewai", "chromadb", "ollama", "nest-asyncio"):
            assert pkg in dep_names, f"crewai group missing {pkg}"


class TestDockerfile:
    """Validate Dockerfile supports CrewAI optional deps."""

    def test_dockerfile_exists(self) -> None:
        assert (PROJECT_ROOT / "Dockerfile").is_file()

    def test_dockerfile_installs_crewai_extras(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "crewai" in content, "Dockerfile must install the crewai optional group"
```

---

## Implementation Details

### 1. New Directory Structure

Create the following directories and empty `__init__.py` files. Do NOT create any business logic files -- those belong to later sections. The only files created here are `__init__.py` stubs and YAML config placeholders.

```
src/quantstack/
  crews/
    trading/
      __init__.py           # empty
      config/
        agents.yaml         # placeholder: "# TradingCrew agents — see section-04"
        tasks.yaml          # placeholder: "# TradingCrew tasks — see section-05"
    research/
      __init__.py           # empty
      config/
        agents.yaml         # placeholder
        tasks.yaml          # placeholder
    supervisor/
      __init__.py           # empty
      config/
        agents.yaml         # placeholder
        tasks.yaml          # placeholder
  crewai_tools/
    __init__.py             # empty
  llm/
    __init__.py             # empty
  rag/
    __init__.py             # empty
  runners/
    __init__.py             # empty
  health/
    __init__.py             # empty
```

The `config/` directories do NOT need `__init__.py` (they contain YAML, not Python).

### 2. Docker Compose Stack (`docker-compose.yml`)

Create `docker-compose.yml` at the project root. Eight services:

**postgres** -- Existing quantstack database.
- Image: `postgres:16-alpine`
- Named volume: `postgres-data` mounted at `/var/lib/postgresql/data`
- Health check: `pg_isready -U quantstack`
- Port: 5432
- Environment: `POSTGRES_DB=quantstack`, `POSTGRES_USER=quantstack`, `POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-quantstack}`

**langfuse-db** -- Dedicated Postgres for Langfuse (separate from quantstack).
- Image: `postgres:16-alpine`
- Named volume: `langfuse-db-data`
- Health check: `pg_isready -U langfuse`
- Environment: `POSTGRES_DB=langfuse`, `POSTGRES_USER=langfuse`, `POSTGRES_PASSWORD=${LANGFUSE_DB_PASSWORD:-langfuse}`

**ollama** -- Local LLM and embedding server.
- Image: `ollama/ollama`
- Named volume: `ollama-data` mounted at `/root/.ollama`
- Health check: `curl -f http://localhost:11434/api/tags || exit 1`
- Port: 11434

**chromadb** -- Vector store for RAG.
- Image: `chromadb/chroma`
- Named volume: `chromadb-data` mounted at `/chroma/chroma`
- Health check: `curl -f http://localhost:8000/api/v1/heartbeat || exit 1`
- Port: 8000
- Environment: `IS_PERSISTENT=TRUE`, `ANONYMIZED_TELEMETRY=FALSE`

**langfuse** -- Self-hosted observability.
- Image: `langfuse/langfuse`
- Depends on: `langfuse-db` (healthy)
- Health check: `curl -f http://localhost:3000/api/public/health || exit 1`
- Port: 3000
- Environment (from `.env`): `DATABASE_URL=postgresql://langfuse:${LANGFUSE_DB_PASSWORD:-langfuse}@langfuse-db:5432/langfuse`, `NEXTAUTH_SECRET`, `SALT`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`

**trading-crew** -- TradingCrew continuous runner.
- Build context: `.` (current dir), Dockerfile
- Command: `python -m quantstack.runners.trading_runner`
- Depends on: postgres, ollama, chromadb, langfuse (all with `condition: service_healthy`)
- Health check: `python -c "from quantstack.health.heartbeat import check; check('trading')"`
- Restart: `unless-stopped`
- Environment: from `.env` file
- Stop grace period: 90s

**research-crew** -- ResearchCrew continuous runner.
- Same pattern as trading-crew, command: `python -m quantstack.runners.research_runner`
- Health check checks `research` heartbeat

**supervisor-crew** -- SupervisorCrew continuous runner.
- Same pattern, command: `python -m quantstack.runners.supervisor_runner`
- Health check checks `supervisor` heartbeat
- Lower resource requirements (lighter LLM tier)

All services share:
- A single Docker network (`quantstack-net`)
- Env file reference (`env_file: .env`)
- Log driver config: `json-file` with `max-size: 50m`, `max-file: "5"`

Named volumes: `postgres-data`, `langfuse-db-data`, `ollama-data`, `chromadb-data`

### 3. Dockerfile Update

The existing `Dockerfile` uses `python:3.11-slim-bookworm` with uv. Modify it to also install the `crewai` optional dependency group. The key change is in the pip install line -- it should become:

```dockerfile
uv pip install -e ".[crewai]"
```

Also ensure `curl` is available in the final image (it is already installed in the existing Dockerfile's apt-get step) since the health check scripts for crew containers use Python, but the ollama/chromadb health checks use curl.

The entrypoint is NOT set in the Dockerfile -- each service's `command` in docker-compose.yml specifies its own entry point.

### 4. pyproject.toml Addition

Add a new optional dependency group under `[project.optional-dependencies]`:

```toml
crewai = [
    "crewai[tools]>=0.100.0",
    "chromadb>=0.5.0",
    "ollama>=0.4.0",
    "nest-asyncio>=1.6.0",
    "openinference-instrumentation-crewai>=0.1.0",
]
```

Notes on the dependency list:
- `crewai[tools]` pulls in the core framework plus tool utilities. Provider-specific extras (anthropic, bedrock, openai, etc.) are handled by crewai's internal LiteLLM dependency, which is already in the project's main dependencies.
- `chromadb` -- vector store client for RAG
- `ollama` -- Python client for local Ollama server
- `nest-asyncio` -- required for calling async functions from within CrewAI's synchronous tool execution context
- `openinference-instrumentation-crewai` -- Langfuse/OpenTelemetry integration for tracing

### 5. `.env.example`

Create `.env.example` at the project root as a reference for required environment variables. This documents what goes in `.env` (which is gitignored):

```
# === LLM Provider ===
LLM_PROVIDER=bedrock
LLM_FALLBACK_ENABLED=true

# === AWS (for Bedrock) ===
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

# === Anthropic Direct (fallback) ===
ANTHROPIC_API_KEY=

# === OpenAI (fallback) ===
OPENAI_API_KEY=

# === Gemini (fallback) ===
GEMINI_API_KEY=

# === Ollama ===
OLLAMA_BASE_URL=http://ollama:11434

# === Existing QuantStack vars ===
TRADER_PG_URL=postgresql://quantstack:quantstack@postgres:5432/quantstack
ALPHA_VANTAGE_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true
USE_REAL_TRADING=false

# === Langfuse ===
LANGFUSE_SECRET_KEY=sk-lf-change-me
LANGFUSE_PUBLIC_KEY=pk-lf-change-me
NEXTAUTH_SECRET=change-me-random-secret
SALT=change-me-random-salt
LANGFUSE_DB_PASSWORD=langfuse

# === Postgres ===
POSTGRES_PASSWORD=quantstack

# === Execution ===
EXECUTION_ENABLED=false
```

---

## Dependencies on Other Sections

This section has **no dependencies** -- it is the first batch. All other sections depend on it:

- Section 02 (LLM Providers) populates `src/quantstack/llm/`
- Section 03 (Tool Wrappers) populates `src/quantstack/crewai_tools/`
- Section 04 (Agent Definitions) populates the `config/agents.yaml` files
- Section 05 (Crew Workflows) populates `config/tasks.yaml` and crew.py files
- Section 06 (RAG Pipeline) populates `src/quantstack/rag/`
- Section 07 (Self-Healing) populates `src/quantstack/health/`
- Section 08 (Observability) adds Langfuse instrumentation
- Section 09 (Runners) populates `src/quantstack/runners/`
- Section 10 (Scripts) updates `start.sh`, creates `stop.sh`, `status.sh`

---

## Acceptance Criteria

1. All tests in `tests/unit/test_scaffolding.py` pass.
2. `docker compose config` validates without errors.
3. `docker compose build` completes successfully (Dockerfile builds with crewai extras).
4. All 8 directories under `src/quantstack/` exist with `__init__.py`.
5. The 6 placeholder YAML files exist in crew config dirs.
6. `.env.example` exists with all documented variables.
7. No existing code is modified or broken -- only additive changes.
