# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Prompt Loading Module for TradingCrew.

This module provides utilities to load agent and task configurations
from JSON files stored in the prompts/ directory structure.

Directory Structure:
    prompts/
    ├── ics/
    │   ├── data/
    │   │   └── data_ingestion_ic.json
    │   ├── market_monitor/
    │   │   ├── market_snapshot_ic.json
    │   │   └── regime_detector_ic.json
    │   ├── technicals/
    │   │   ├── trend_momentum_ic.json
    │   │   ├── volatility_ic.json
    │   │   └── structure_levels_ic.json
    │   ├── quant/
    │   │   ├── statarb_ic.json
    │   │   └── options_vol_ic.json
    │   └── risk/
    │       ├── risk_limits_ic.json
    │       └── calendar_events_ic.json
    ├── pod_managers/
    │   ├── data_pod_manager.json
    │   ├── market_monitor_pod_manager.json
    │   ├── technicals_pod_manager.json
    │   ├── quant_pod_manager.json
    │   └── risk_pod_manager.json
    ├── assistant/
    │   └── trading_assistant.json
    └── supertrader/
        └── super_trader.json

Each JSON file contains:
    - name: Agent identifier
    - role: Agent's role description
    - goal: Agent's objective
    - backstory: Agent's context/personality
    - settings: dict with llm, reasoning, verbose, allow_delegation, etc.
    - tools: list of tool names (optional)
    - pod: pod affiliation for ICs (optional)
    - managed_ics: list of ICs for pod managers (optional)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# =============================================================================
# PROMPT LOADER
# =============================================================================


class PromptLoader:
    """
    Loads agent configurations from JSON files in prompts/ directory.

    Usage:
        loader = PromptLoader()

        # Load a specific agent config
        config = loader.load_agent("data_ingestion_ic")

        # Load all ICs
        ics = loader.load_all_ics()

        # Load all pod managers
        managers = loader.load_all_pod_managers()
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Path to prompts directory. Defaults to module directory.
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.prompts_dir = prompts_dir

        # Cache loaded configs
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.debug(f"PromptLoader initialized with dir: {self.prompts_dir}")

    def _find_json_file(self, name: str) -> Optional[Path]:
        """Find JSON file by agent name."""
        # Search in all subdirectories
        for json_path in self.prompts_dir.rglob(f"{name}.json"):
            return json_path
        return None

    def load_agent(self, name: str) -> Dict[str, Any]:
        """
        Load agent configuration by name.

        Args:
            name: Agent name (e.g., "data_ingestion_ic", "trading_assistant")

        Returns:
            Dict with agent configuration

        Raises:
            FileNotFoundError: If agent config file not found
        """
        if name in self._cache:
            return self._cache[name]

        json_path = self._find_json_file(name)
        if json_path is None:
            raise FileNotFoundError(f"Agent config not found: {name}")

        with open(json_path) as f:
            config = json.load(f)

        self._cache[name] = config
        logger.debug(f"Loaded agent config: {name}")
        return config

    def load_all_ics(self) -> Dict[str, Dict[str, Any]]:
        """Load all IC agent configurations."""
        ics = {}
        ics_dir = self.prompts_dir / "ics"

        if not ics_dir.exists():
            logger.warning(f"ICs directory not found: {ics_dir}")
            return ics

        for json_path in ics_dir.rglob("*.json"):
            with open(json_path) as f:
                config = json.load(f)
            name = config.get("name", json_path.stem)
            ics[name] = config
            self._cache[name] = config

        logger.info(f"Loaded {len(ics)} IC configs")
        return ics

    def load_all_pod_managers(self) -> Dict[str, Dict[str, Any]]:
        """Load all pod manager configurations."""
        managers = {}
        managers_dir = self.prompts_dir / "pod_managers"

        if not managers_dir.exists():
            logger.warning(f"Pod managers directory not found: {managers_dir}")
            return managers

        for json_path in managers_dir.glob("*.json"):
            with open(json_path) as f:
                config = json.load(f)
            name = config.get("name", json_path.stem)
            managers[name] = config
            self._cache[name] = config

        logger.info(f"Loaded {len(managers)} pod manager configs")
        return managers

    def load_assistant(self) -> Dict[str, Any]:
        """Load trading assistant configuration."""
        return self.load_agent("trading_assistant")

    def load_supertrader(self) -> Dict[str, Any]:
        """Load super trader configuration."""
        return self.load_agent("super_trader")

    def get_agent_config(self, name: str) -> Dict[str, Any]:
        """
        Get CrewAI-compatible agent configuration dict.

        Transforms JSON config to format expected by CrewAI Agent constructor.

        Args:
            name: Agent name

        Returns:
            Dict with role, goal, backstory keys
        """
        config = self.load_agent(name)

        # Extract core agent fields
        return {
            "role": config.get("role", ""),
            "goal": config.get("goal", ""),
            "backstory": config.get("backstory", ""),
        }

    def get_agent_settings(self, name: str) -> Dict[str, Any]:
        """
        Get agent settings (llm, reasoning, verbose, etc.).

        Args:
            name: Agent name

        Returns:
            Dict with settings
        """
        config = self.load_agent(name)
        return config.get("settings", {})

    def get_agent_tools(self, name: str) -> List[str]:
        """
        Get list of tool names for an agent.

        Args:
            name: Agent name

        Returns:
            List of tool names
        """
        config = self.load_agent(name)
        return config.get("tools", [])

    def list_all_agents(self) -> Dict[str, List[str]]:
        """
        List all available agents by category.

        Returns:
            Dict with categories as keys and agent names as values
        """
        result = {
            "ics": [],
            "pod_managers": [],
            "assistant": [],
            "supertrader": [],
        }

        # ICs
        ics_dir = self.prompts_dir / "ics"
        if ics_dir.exists():
            for json_path in ics_dir.rglob("*.json"):
                result["ics"].append(json_path.stem)

        # Pod Managers
        managers_dir = self.prompts_dir / "pod_managers"
        if managers_dir.exists():
            for json_path in managers_dir.glob("*.json"):
                result["pod_managers"].append(json_path.stem)

        # Assistant
        assistant_dir = self.prompts_dir / "assistant"
        if assistant_dir.exists():
            for json_path in assistant_dir.glob("*.json"):
                result["assistant"].append(json_path.stem)

        # SuperTrader
        supertrader_dir = self.prompts_dir / "supertrader"
        if supertrader_dir.exists():
            for json_path in supertrader_dir.glob("*.json"):
                result["supertrader"].append(json_path.stem)

        return result


# =============================================================================
# SINGLETON LOADER
# =============================================================================

_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


def load_agent_config(name: str) -> Dict[str, Any]:
    """Convenience function to load agent config."""
    return get_prompt_loader().load_agent(name)


def load_all_ics() -> Dict[str, Dict[str, Any]]:
    """Convenience function to load all IC configs."""
    return get_prompt_loader().load_all_ics()


def load_all_pod_managers() -> Dict[str, Dict[str, Any]]:
    """Convenience function to load all pod manager configs."""
    return get_prompt_loader().load_all_pod_managers()


__all__ = [
    "PromptLoader",
    "get_prompt_loader",
    "load_agent_config",
    "load_all_ics",
    "load_all_pod_managers",
]
