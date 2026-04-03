# Section 12: Docker & Infrastructure Cleanup

## Background

QuantStack runs as a Docker Compose stack with infrastructure services (postgres, ollama, chromadb, langfuse) and three "crew" application services (trading-crew, research-crew, supervisor-crew). The CrewAI-to-LangGraph migration requires several infrastructure changes:

1. **Remove ChromaDB** -- vector storage moves to pgvector inside the existing PostgreSQL instance (see section-09 for the RAG migration itself).
2. **Switch PostgreSQL image** to one that ships with pgvector.
3. **Rename crew services to graph services** -- cosmetic but clarifies the new architecture.
4. **Clean up environment variables** -- remove CrewAI-specific vars, update comments.
5. **Update Dockerfile** -- install the `langgraph` dependency group instead of `crewai`.
6. **Update start.sh** -- remove chromadb references, update service names, replace ChromaDB RAG check with pgvector equivalent.

**Depends on**: section-01-scaffolding (pyproject.toml must have the `langgraph` optional group defined before the Dockerfile can install it).

---

## Tests (Write First)

These tests validate the Docker and infrastructure artifacts. They are static file-content assertions -- no running containers required.

```python
# tests/unit/test_docker_cleanup.py

"""
Static assertions on Docker and infrastructure files.
Validates the CrewAI → LangGraph migration is reflected in
docker-compose.yml, Dockerfile, .env.example, and start.sh.
"""

import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestDockerCompose:
    """Validate docker-compose.yml after migration."""

    @pytest.fixture
    def compose_content(self) -> str:
        return (PROJECT_ROOT / "docker-compose.yml").read_text()

    def test_no_chromadb_service(self, compose_content: str):
        """ChromaDB service must be removed entirely."""
        assert "chromadb:" not in compose_content
        assert "chromadb/chroma" not in compose_content

    def test_no_chromadb_volume(self, compose_content: str):
        """ChromaDB volume must be removed."""
        assert "chromadb-data" not in compose_content

    def test_postgres_image_supports_pgvector(self, compose_content: str):
        """Postgres image must include pgvector extension."""
        assert "pgvector/pgvector:pg16" in compose_content

    def test_no_crew_service_names(self, compose_content: str):
        """Crew service names must be renamed to graph."""
        assert "trading-crew:" not in compose_content
        assert "research-crew:" not in compose_content
        assert "supervisor-crew:" not in compose_content

    def test_graph_service_names_exist(self, compose_content: str):
        """Graph service names must be present."""
        assert "trading-graph:" in compose_content
        assert "research-graph:" in compose_content
        assert "supervisor-graph:" in compose_content

    def test_graph_services_no_chromadb_dependency(self, compose_content: str):
        """Graph services must not depend on chromadb."""
        # No service should list chromadb in depends_on
        assert "chromadb" not in compose_content


class TestDockerfile:
    """Validate Dockerfile after migration."""

    @pytest.fixture
    def dockerfile_content(self) -> str:
        return (PROJECT_ROOT / "Dockerfile").read_text()

    def test_installs_langgraph_group(self, dockerfile_content: str):
        """Dockerfile must install the langgraph optional dependency group."""
        # Should reference the langgraph extra, not crewai/all
        assert "langgraph" in dockerfile_content

    def test_no_crewai_reference(self, dockerfile_content: str):
        """Dockerfile must not reference crewai."""
        assert "crewai" not in dockerfile_content.lower()


class TestEnvExample:
    """Validate .env.example after migration."""

    @pytest.fixture
    def env_content(self) -> str:
        return (PROJECT_ROOT / ".env.example").read_text()

    def test_no_crewai_prefixed_variables(self, env_content: str):
        """No CREWAI_ prefixed env vars should remain."""
        for line in env_content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                assert not stripped.startswith("CREWAI_"), (
                    f"Found CrewAI env var: {stripped}"
                )

    def test_no_crewai_section_header(self, env_content: str):
        """The 'CrewAI' section header should be updated."""
        assert "CrewAI" not in env_content


class TestStartScript:
    """Validate start.sh after migration."""

    @pytest.fixture
    def start_content(self) -> str:
        return (PROJECT_ROOT / "start.sh").read_text()

    def test_no_chromadb_references(self, start_content: str):
        """start.sh must not reference chromadb."""
        assert "chromadb" not in start_content.lower()

    def test_no_crew_service_references(self, start_content: str):
        """start.sh must use graph service names, not crew."""
        assert "trading-crew" not in start_content
        assert "research-crew" not in start_content
        assert "supervisor-crew" not in start_content

    def test_graph_service_references(self, start_content: str):
        """start.sh must reference the new graph service names."""
        assert "trading-graph" in start_content
        assert "research-graph" in start_content
        assert "supervisor-graph" in start_content
```

---

## Implementation Details

### 1. docker-compose.yml

**File**: `docker-compose.yml`

**1a. Remove the `chromadb` service block** (lines 94-119 in current file). Delete the entire service definition including image, container_name, ports, environment, volumes, healthcheck, mem_limit, restart, networks, and logging.

**1b. Switch postgres image** from `postgres:16-alpine` to `pgvector/pgvector:pg16`. This is a drop-in replacement that includes the `vector` extension. All other postgres configuration (ports, env, volumes, healthcheck) stays the same.

**1c. Rename crew services to graph services**:

| Before | After |
|--------|-------|
| `trading-crew` (service key) | `trading-graph` |
| `quantstack-trading-crew` (container_name) | `quantstack-trading-graph` |
| `research-crew` (service key) | `research-graph` |
| `quantstack-research-crew` (container_name) | `quantstack-research-graph` |
| `supervisor-crew` (service key) | `supervisor-graph` |
| `quantstack-supervisor-crew` (container_name) | `quantstack-supervisor-graph` |

The `command` fields stay the same (`python -m quantstack.runners.trading_runner`, etc.) -- the runner entry points are unchanged.

**1d. Remove chromadb from depends_on** in all three graph services. Each currently depends on `postgres`, `ollama`, `chromadb`, and `langfuse`. Remove `chromadb` from each.

**1e. Update the top-of-file comment** from "CrewAI Stack" to "LangGraph Stack". Update the `docker compose logs` example to reference `trading-graph` instead of `trading-crew`.

**1f. Remove `chromadb-data` from the `volumes:` section** at the bottom of the file.

**1g. Update section comment** from `# -- Crew Services --` to `# -- Graph Services --`.

### 2. Dockerfile

**File**: `Dockerfile`

**2a. Change the install command** from:
```
RUN uv pip install --system --no-cache -e ".[all]"
```
to:
```
RUN uv pip install --system --no-cache -e ".[langgraph]"
```

This installs the `langgraph` optional dependency group defined in pyproject.toml (section-01). The `[all]` group previously pulled in crewai and other dependencies that are no longer needed.

**2b. Update the comment** on the install line from "includes crewai optional group" to "includes langgraph optional group".

### 3. .env.example

**File**: `.env.example`

**3a. Update the section header** around line 149. Change:
```
# CrewAI / Docker Compose Stack
```
to:
```
# LangGraph / Docker Compose Stack
```

**3b. Remove any `CREWAI_` prefixed variables** if present. Currently there are none as active env vars, but the section header references CrewAI.

**3c. Update the LLM Configuration comment** around line 43. Change:
```
# LLM Configuration (CrewAI agents)
```
to:
```
# LLM Configuration
```

**3d. Add a comment about LangGraph checkpointing** in the Storage section:
```
# LangGraph checkpointing uses the same TRADER_PG_URL — no additional config needed.
```

### 4. start.sh

**File**: `start.sh`

**4a. Remove chromadb from infrastructure startup** (line 54). Change:
```bash
docker compose up -d postgres langfuse-db ollama chromadb langfuse
```
to:
```bash
docker compose up -d postgres langfuse-db ollama langfuse
```

**4b. Remove chromadb from infrastructure health wait** (line 60). Change:
```bash
INFRA_SERVICES="postgres ollama chromadb langfuse"
```
to:
```bash
INFRA_SERVICES="postgres ollama langfuse"
```

**4c. Replace the ChromaDB RAG check** (lines 173-188). The current block uses `chromadb.HttpClient` to check if collections are empty. Replace with a pgvector-based check:

```bash
# ---------------------------------------------------------------------------
# 11. RAG ingestion (first-run only)
# ---------------------------------------------------------------------------
echo "[start.sh] Checking RAG collections..."
RAG_EMPTY=$(docker compose run --rm trading-graph python -c "
try:
    from quantstack.db import open_db
    conn = open_db()
    row = conn.execute(\"SELECT COUNT(*) FROM embeddings\").fetchone()
    conn.close()
    print('yes' if not row or row[0] == 0 else 'no')
except:
    print('yes')
" 2>/dev/null || echo "yes")

if [[ "$RAG_EMPTY" == "yes" ]]; then
    echo "[start.sh] Embeddings table is empty — running memory ingestion..."
    docker compose run --rm trading-graph python -m quantstack.rag.ingest || true
fi
```

**4d. Rename all crew service references** throughout the file:
- `trading-crew` -> `trading-graph` (appears in steps 7, 8, 9, 10, 11, 12, 13, 14)
- `research-crew` -> `research-graph` (appears in step 13, 14)
- `supervisor-crew` -> `supervisor-graph` (appears in step 13, 14)

**4e. Update the script header comment** from "Starts infrastructure (postgres, ollama, chromadb, langfuse)" to "Starts infrastructure (postgres, ollama, langfuse)".

**4f. Update the final status banner** logs line from `trading-crew` to `trading-graph`.

**4g. Update step 13 comment** from "Start crew services" to "Start graph services".

**4h. Update step 14 comment** from "Wait for crew health checks" to "Wait for graph health checks". Update the loop variable and echo text accordingly.

### 5. Verify no other files reference chromadb or crew service names

After making the above changes, search the codebase for any remaining references to:
- `chromadb` in Docker/infrastructure context (the RAG migration in section-09 handles code references)
- `trading-crew`, `research-crew`, `supervisor-crew` as Docker service names
- `stop.sh` and `status.sh` if they exist -- apply the same renames

---

## Rollback Considerations

All changes in this section are additive or cosmetic renames. If the migration needs to be rolled back:

- The `pgvector/pgvector:pg16` image is a superset of `postgres:16-alpine` -- it runs standard PostgreSQL plus the vector extension. Reverting to the alpine image is safe; pgvector tables will simply be inaccessible until the extension is re-added.
- ChromaDB service can be re-added to docker-compose.yml from git history. The `chromadb-data` volume is not deleted by removing the service definition -- it persists until explicitly pruned with `docker volume rm`.
- Service renames are cosmetic. The underlying runner commands (`python -m quantstack.runners.*`) are unchanged.
