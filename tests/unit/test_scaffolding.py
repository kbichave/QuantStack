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

    def test_langgraph_dependency_exists(self, pyproject: dict) -> None:
        lg_deps = pyproject["project"]["optional-dependencies"]["langgraph"]
        langgraph_dep = [d for d in lg_deps if d.startswith("langgraph") and "checkpoint" not in d][0]
        assert ">=" in langgraph_dep, "langgraph must have a minimum version pin"

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
        assert "CrewAI" not in content, ".env.example should not reference CrewAI"
