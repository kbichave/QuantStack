# Section 01: Dependency & Scaffolding Changes

This section covers all foundational changes needed before any LangGraph code is written: dependency updates in `pyproject.toml`, directory structure creation, Dockerfile updates, and docker-compose skeleton changes. Every subsequent section depends on this one completing first.

---

## Tests (Write These First)

The test file already exists at `tests/unit/test_scaffolding.py`. It currently validates the **old** CrewAI scaffolding. Replace its contents entirely with tests for the **new** LangGraph scaffolding. The tests below should all fail before implementation and pass after.

```python
"""Tests for Section 01: Project Scaffolding (LangGraph migration).

Validates that dependency changes, directory structure, Docker Compose config,
and Dockerfile are correctly set up for the LangGraph migration.
"""

import pathlib
import tomllib

import pytest
import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"


class TestPyprojectToml:
    """Validate pyproject.toml dependency changes."""

    @pytest.fixture()
    def pyproject(self) -> dict:
        path = PROJECT_ROOT / "pyproject.toml"
        return tomllib.loads(path.read_text())

    def test_langgraph_optional_group_exists(self, pyproject: dict) -> None:
        optional = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "langgraph" in optional, "Missing [project.optional-dependencies] langgraph group"

    def test_langgraph_group_includes_core_packages(self, pyproject: dict) -> None:
        lg_deps = pyproject["project"]["optional-dependencies"]["langgraph"]
        dep_names = [d.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for d in lg_deps]
        for pkg in ("langgraph", "langchain-core", "langchain-anthropic",
                     "langgraph-checkpoint-postgres", "psycopg", "psycopg-pool",
                     "pgvector", "watchdog"):
            assert pkg in dep_names, f"langgraph group missing {pkg}"

    def test_crewai_optional_group_removed(self, pyproject: dict) -> None:
        optional = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "crewai" not in optional, "crewai optional group should be removed"

    def test_langgraph_version_pinned(self, pyproject: dict) -> None:
        lg_deps = pyproject["project"]["optional-dependencies"]["langgraph"]
        langgraph_dep = [d for d in lg_deps if d.startswith("langgraph") and "checkpoint" not in d][0]
        assert "<" in langgraph_dep, "langgraph must have an upper bound pin (pre-1.0, breaking changes)"

    def test_nest_asyncio_still_present(self, pyproject: dict) -> None:
        """nest-asyncio is still needed by mcp_bridge tools until Phase 4."""
        all_deps = []
        for group_deps in pyproject["project"]["optional-dependencies"].values():
            all_deps.extend(group_deps)
        all_deps.extend(pyproject.get("project", {}).get("dependencies", []))
        dep_names = [d.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for d in all_deps]
        assert "nest-asyncio" in dep_names, "nest-asyncio must remain until mcp_bridge migration is complete"

    def test_all_group_references_langgraph(self, pyproject: dict) -> None:
        all_group = pyproject["project"]["optional-dependencies"]["all"]
        all_str = " ".join(all_group)
        assert "langgraph" in all_str, "all group must include langgraph"
        assert "crewai" not in all_str, "all group must not reference crewai"


class TestDirectoryStructure:
    """Verify new graphs/ directory and expected structure."""

    @pytest.mark.parametrize("subdir", [
        "graphs",
        "graphs/research",
        "graphs/research/config",
        "graphs/trading",
        "graphs/trading/config",
        "graphs/supervisor",
        "graphs/supervisor/config",
        "tools/langchain",
        "tools/functions",
    ])
    def test_new_directory_exists(self, subdir: str) -> None:
        assert (SRC_ROOT / subdir).is_dir(), f"Missing directory: src/quantstack/{subdir}"

    @pytest.mark.parametrize("pkg", [
        "graphs",
        "graphs/research",
        "graphs/trading",
        "graphs/supervisor",
        "tools/langchain",
        "tools/functions",
    ])
    def test_init_py_exists(self, pkg: str) -> None:
        assert (SRC_ROOT / pkg / "__init__.py").is_file(), (
            f"Missing __init__.py in src/quantstack/{pkg}"
        )

    def test_graphs_state_module_exists(self) -> None:
        assert (SRC_ROOT / "graphs" / "state.py").is_file(), "Missing graphs/state.py"

    def test_graphs_config_module_exists(self) -> None:
        assert (SRC_ROOT / "graphs" / "config.py").is_file(), "Missing graphs/config.py"

    @pytest.mark.parametrize("subdir", [
        "crews/trading",
        "crews/research",
        "crews/supervisor",
        "crews",
        "crews/risk",
        "llm",
        "rag",
        "runners",
        "health",
    ])
    def test_kept_directories_still_exist(self, subdir: str) -> None:
        """Pure Python modules that are NOT deleted."""
        assert (SRC_ROOT / subdir).is_dir(), f"Directory should still exist: src/quantstack/{subdir}"


class TestCrewaiCleanup:
    """Verify CrewAI-specific files and directories are removed."""

    def test_crewai_tools_directory_removed(self) -> None:
        assert not (SRC_ROOT / "crewai_tools").is_dir(), "src/quantstack/crewai_tools/ should be deleted"

    def test_crewai_compat_removed(self) -> None:
        assert not (SRC_ROOT / "crewai_compat.py").is_file(), "src/quantstack/crewai_compat.py should be deleted"

    def test_crewai_docs_removed(self) -> None:
        assert not (PROJECT_ROOT / "docs" / "crewai_docs_md").is_dir(), "docs/crewai_docs_md/ should be deleted"


class TestDockerCompose:
    """Validate docker-compose.yml changes."""

    @pytest.fixture()
    def compose(self) -> dict:
        path = PROJECT_ROOT / "docker-compose.yml"
        assert path.is_file(), "docker-compose.yml does not exist"
        return yaml.safe_load(path.read_text())

    def test_chromadb_service_removed(self, compose: dict) -> None:
        services = set(compose.get("services", {}).keys())
        assert "chromadb" not in services, "chromadb service should be removed"

    def test_graph_services_exist(self, compose: dict) -> None:
        services = set(compose.get("services", {}).keys())
        for svc in ("trading-graph", "research-graph", "supervisor-graph"):
            assert svc in services, f"Missing service: {svc}"

    def test_crew_services_removed(self, compose: dict) -> None:
        services = set(compose.get("services", {}).keys())
        for svc in ("trading-crew", "research-crew", "supervisor-crew"):
            assert svc not in services, f"Old service still present: {svc}"

    def test_postgres_supports_pgvector(self, compose: dict) -> None:
        pg_image = compose["services"]["postgres"]["image"]
        assert "pgvector" in pg_image, f"Postgres image must include pgvector: got {pg_image}"

    def test_chromadb_volume_removed(self, compose: dict) -> None:
        volumes = set(compose.get("volumes", {}).keys())
        assert "chromadb-data" not in volumes, "chromadb-data volume should be removed"

    def test_graph_services_do_not_depend_on_chromadb(self, compose: dict) -> None:
        for svc_name in ("trading-graph", "research-graph", "supervisor-graph"):
            deps = compose["services"][svc_name].get("depends_on", {})
            dep_names = set(deps.keys()) if isinstance(deps, dict) else set(deps)
            assert "chromadb" not in dep_names, f"{svc_name} must not depend on chromadb"

    def test_infrastructure_services_still_present(self, compose: dict) -> None:
        services = set(compose.get("services", {}).keys())
        for svc in ("postgres", "langfuse-db", "langfuse", "ollama"):
            assert svc in services, f"Infrastructure service missing: {svc}"

    def test_shared_network(self, compose: dict) -> None:
        networks = compose.get("networks", {})
        assert len(networks) >= 1, "At least one Docker network must be defined"


class TestDockerfile:
    """Validate Dockerfile installs langgraph deps."""

    def test_dockerfile_exists(self) -> None:
        assert (PROJECT_ROOT / "Dockerfile").is_file()

    def test_dockerfile_installs_langgraph_extras(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "langgraph" in content, "Dockerfile must install the langgraph optional group"

    def test_dockerfile_does_not_reference_crewai(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "crewai" not in content.lower(), "Dockerfile must not reference crewai"


class TestEnvExample:
    """Validate .env.example cleanup."""

    def test_no_crewai_references_in_section_headers(self) -> None:
        content = (PROJECT_ROOT / ".env.example").read_text()
        # The section header "CrewAI / Docker Compose Stack" should be updated
        assert "CrewAI" not in content, ".env.example should not reference CrewAI"
```

---

## Implementation Details

### 1. Update `pyproject.toml`

**File**: `pyproject.toml` (project root)

**Remove** the entire `crewai` optional dependency group (lines 117-125 in current file):
```
crewai = [
    "crewai[tools]>=0.100.0",
    "chromadb>=0.5.0",
    "ollama>=0.4.0",
    "nest-asyncio>=1.6.0",
    "openinference-instrumentation-crewai>=0.1.0",
    "duckduckgo-search>=6.0.0",
    "httpx>=0.27.0",
]
```

**Add** a `langgraph` optional dependency group in its place:
```
langgraph = [
    "langgraph>=0.4.0,<0.5.0",
    "langchain-core>=0.3.0,<0.4.0",
    "langchain-anthropic>=0.3.0",
    "langgraph-checkpoint-postgres>=3.0.0",
    "psycopg[binary]>=3.1.0",
    "psycopg-pool>=3.1.0",
    "pgvector>=0.3.0",
    "watchdog>=4.0.0",
    "nest-asyncio>=1.6.0",
    "ollama>=0.4.0",
]
```

**Why `nest-asyncio` stays**: It is still needed by `mcp_bridge/` tools that use `run_async()`. It will be removed only after all mcp_bridge tools are migrated (end of Phase 4, covered in section-05-tool-layer).

**Why `ollama` stays**: Still used for local LLM fallback and embedding generation.

**Update** the `all` group to reference `langgraph` instead of `crewai`:
```
all = [
    "quantstack[dev,langgraph]",
]
```

**Pin rationale**: LangGraph is pre-1.0 and has had breaking changes between minor versions. The `>=0.4.0,<0.5.0` pin prevents silent breakage on `uv lock` / `pip install` updates. Same for `langchain-core`.

### 2. Create Directory Structure

Create the following new directories and files under `src/quantstack/`:

```
src/quantstack/graphs/
    __init__.py          # empty or re-exports
    state.py             # placeholder — TypedDict schemas (section-04)
    config.py            # placeholder — AgentConfig + loader (section-03)
    config_watcher.py    # placeholder — hot-reload (section-03)
    research/
        __init__.py
        graph.py         # placeholder — build_research_graph() (section-07)
        nodes.py         # placeholder — node functions (section-07)
        config/          # directory only, YAML files added in section-03
    trading/
        __init__.py
        graph.py         # placeholder
        nodes.py         # placeholder
        config/
    supervisor/
        __init__.py
        graph.py         # placeholder
        nodes.py         # placeholder
        config/
src/quantstack/tools/langchain/
    __init__.py          # placeholder — LLM-facing @tool wrappers (section-05)
src/quantstack/tools/functions/
    __init__.py          # placeholder — node-callable functions (section-05)
```

All `__init__.py` files should be empty. The `.py` placeholders (`state.py`, `config.py`, `config_watcher.py`, `graph.py`, `nodes.py`) should contain a module docstring only, describing what will go there. For example:

```python
"""TypedDict state schemas for all LangGraph graphs.

Defines ResearchState, TradingState, and SupervisorState.
Implemented in section-04-state-schemas.
"""
```

The `config/` directories under each graph are plain directories (no `__init__.py`) -- they hold YAML files, not Python packages.

### 3. Delete CrewAI-Specific Files

**Delete entirely:**

- `src/quantstack/crewai_tools/` -- 25 files, all CrewAI `BaseTool` subclasses. Replaced by `tools/langchain/` and `tools/functions/` in section-05.
- `src/quantstack/crewai_compat.py` -- compatibility shim for CrewAI's `BaseTool`. No longer needed.
- `docs/crewai_docs_md/` -- ~180 files of crawled CrewAI reference docs. Not project documentation.

**Do NOT delete:**

- `src/quantstack/crews/` -- this directory contains pure Python modules (schemas, registry, decoder_crew, safety_gate) that have no CrewAI framework dependency. They are kept and reused. The CrewAI-dependent `crew.py` files within each subdirectory (research, trading, supervisor) will be removed in later sections when their LangGraph replacements are built.
- `src/quantstack/tools/mcp_bridge/` -- these are migrated incrementally in section-05, not deleted here.

### 4. Update Dockerfile

**File**: `Dockerfile` (project root)

Change the install line (currently line 38):
```dockerfile
# OLD:
RUN uv pip install --system --no-cache -e ".[all]"

# NEW:
RUN uv pip install --system --no-cache -e ".[all]"
```

The install command itself does not change because it installs `.[all]`. What changes is that the `all` group in `pyproject.toml` now references `langgraph` instead of `crewai`. However, the comment on line 37 must be updated:

```dockerfile
# OLD:
# Install all dependencies (includes crewai optional group)

# NEW:
# Install all dependencies (includes langgraph optional group)
```

### 5. Update `docker-compose.yml`

**File**: `docker-compose.yml` (project root)

**5a. Switch PostgreSQL image to pgvector-enabled image:**

Change:
```yaml
postgres:
    image: postgres:16-alpine
```
To:
```yaml
postgres:
    image: pgvector/pgvector:pg16
```

This image includes the `vector` extension pre-installed. The pgvector extension itself is enabled in section-09-rag-migration via `CREATE EXTENSION IF NOT EXISTS vector;`.

**5b. Remove the `chromadb` service** (lines 94-119 in current file). Delete the entire service block.

**5c. Remove the `chromadb-data` volume** from the `volumes:` section at the bottom.

**5d. Rename crew services to graph services:**

- `trading-crew` becomes `trading-graph`
- `research-crew` becomes `research-graph`
- `supervisor-crew` becomes `supervisor-graph`

Update corresponding `container_name` values:
- `quantstack-trading-crew` becomes `quantstack-trading-graph`
- `quantstack-research-crew` becomes `quantstack-research-graph`
- `quantstack-supervisor-crew` becomes `quantstack-supervisor-graph`

**5e. Remove `chromadb` from `depends_on`** in all three graph services. The dependency list becomes:
```yaml
depends_on:
    postgres:
        condition: service_healthy
    ollama:
        condition: service_healthy
    langfuse:
        condition: service_healthy
```

**5f. Update health check commands** to match new service names:
```yaml
healthcheck:
    test: ["CMD", "python", "-c", "from quantstack.health.heartbeat import check_health; check_health('trading')"]
```
The health check function argument (`'trading'`, `'research'`, `'supervisor'`) stays the same -- only the Docker service name changes. No code change needed in the health module.

**5g. Keep all other services unchanged**: `langfuse-db`, `langfuse`, `ollama`, `api` (legacy profile), `alpaca-mcp` (mcp-servers profile).

### 6. Update `.env.example`

**File**: `.env.example` (project root)

Change the section header (currently line 149):
```
# CrewAI / Docker Compose Stack
```
To:
```
# LangGraph / Docker Compose Stack
```

Change the LLM Configuration section header comment (currently line 43):
```
# LLM Configuration (CrewAI agents)
```
To:
```
# LLM Configuration
```

No new environment variables are needed for LangGraph. The `LANGFUSE_*` variables are already present. The `TRADER_PG_URL` is already used for PostgreSQL. LangGraph checkpointing uses the same database connection.

---

## Dependencies on Other Sections

This section has no dependencies -- it is the first to execute.

All other sections depend on this one:
- **section-02 through section-05, section-09, section-10, section-12** can begin immediately after this section is complete (Batch 2, parallelizable).
- The `graphs/` directory placeholders created here are populated by sections 03, 04, 06, 07, and 08.
- The `tools/langchain/` and `tools/functions/` directories created here are populated by section-05.

---

## Verification Checklist

After implementation, run:

```bash
uv run pytest tests/unit/test_scaffolding.py -v
```

All tests in the rewritten `test_scaffolding.py` should pass. Additionally verify manually:

1. `uv lock` succeeds with the new dependencies (no resolution conflicts)
2. `uv run python -c "import langgraph; print(langgraph.__version__)"` succeeds
3. `docker compose config` parses without errors (validates YAML syntax)
4. The `graphs/` directory tree matches the structure specified above
5. `src/quantstack/crewai_tools/` and `src/quantstack/crewai_compat.py` no longer exist
