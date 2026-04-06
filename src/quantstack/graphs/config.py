"""AgentConfig dataclass and YAML config loader."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_VALID_TIERS = frozenset({"heavy", "medium", "light"})
_ALWAYS_LOADED_WARN_THRESHOLD = 10


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent profile loaded from YAML."""

    name: str
    role: str
    goal: str
    backstory: str
    llm_tier: str
    max_iterations: int = 20
    timeout_seconds: int = 600
    tools: tuple[str, ...] = ()
    always_loaded_tools: tuple[str, ...] = ()
    thinking: dict | None = None

    def __post_init__(self) -> None:
        if self.llm_tier not in _VALID_TIERS:
            raise ValueError(
                f"Invalid llm_tier '{self.llm_tier}' for agent '{self.name}'. "
                f"Must be one of: {sorted(_VALID_TIERS)}"
            )
        if not self.role:
            raise ValueError(f"Agent '{self.name}' missing required field: role")
        if not self.goal:
            raise ValueError(f"Agent '{self.name}' missing required field: goal")

        # always_loaded_tools must be a subset of tools
        tools_set = set(self.tools)
        for tool_name in self.always_loaded_tools:
            if tool_name not in tools_set:
                raise ValueError(
                    f"Agent '{self.name}': always_loaded_tools contains '{tool_name}' "
                    f"which is not in the agent's tools list"
                )

        # risk_check must never be deferred — if it's in tools and deferred loading
        # is active (always_loaded_tools is non-empty), risk_check must be always-loaded
        if (
            self.always_loaded_tools
            and "risk_check" in tools_set
            and "risk_check" not in self.always_loaded_tools
        ):
            raise ValueError(
                f"Agent '{self.name}': risk_check is in tools but not in "
                f"always_loaded_tools. The risk gate must never be deferred."
            )

        if len(self.always_loaded_tools) > _ALWAYS_LOADED_WARN_THRESHOLD:
            warnings.warn(
                f"Agent '{self.name}' has {len(self.always_loaded_tools)} "
                f"always_loaded_tools; consider reducing to benefit from deferred loading",
                UserWarning,
                stacklevel=2,
            )


class _DuplicateKeyLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate keys."""
    pass


def _check_duplicate_keys(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in mapping:
            raise ValueError(
                f"Duplicate agent name '{key}' in YAML file "
                f"at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node)
    return mapping


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _check_duplicate_keys,
)


def load_agent_configs(yaml_path: Path) -> dict[str, AgentConfig]:
    """Load and validate agent configs from a YAML file.

    Returns a mapping of agent_name -> AgentConfig.
    Raises ValueError on validation failures.
    """
    text = yaml_path.read_text()
    raw = yaml.load(text, Loader=_DuplicateKeyLoader)
    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

    configs: dict[str, AgentConfig] = {}
    for name, fields in raw.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Agent '{name}' must be a mapping, got {type(fields).__name__}")

        tools_raw = fields.get("tools", [])
        tools = tuple(tools_raw) if isinstance(tools_raw, list) else ()

        always_loaded_raw = fields.get("always_loaded_tools", [])
        always_loaded = tuple(always_loaded_raw) if isinstance(always_loaded_raw, list) else ()

        configs[name] = AgentConfig(
            name=name,
            role=fields.get("role", ""),
            goal=fields.get("goal", ""),
            backstory=fields.get("backstory", ""),
            llm_tier=fields.get("llm_tier", ""),
            max_iterations=fields.get("max_iterations", 20),
            timeout_seconds=fields.get("timeout_seconds", 600),
            tools=tools,
            always_loaded_tools=always_loaded,
            thinking=fields.get("thinking"),
        )

    return configs


def validate_tool_references(
    configs: dict[str, AgentConfig],
    tool_registry: dict[str, Any],
) -> None:
    """Raise ValueError if any agent references a tool not in the registry."""
    for agent_name, cfg in configs.items():
        for tool_name in cfg.tools:
            if tool_name not in tool_registry:
                raise ValueError(
                    f"Agent '{agent_name}' references unknown tool '{tool_name}'. "
                    f"Available tools: {sorted(tool_registry.keys())}"
                )
