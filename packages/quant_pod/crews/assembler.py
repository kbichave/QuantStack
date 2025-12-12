from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

from quant_pod.crews.registry import (
    IC_REGISTRY,
    POD_DEPENDENCIES,
    POD_MANAGER_REGISTRY,
    PROFILE_DEFAULTS,
    registry_snapshot,
)
from quant_pod.crews.schemas import TaskEnvelope


@dataclass
class PodSelection:
    """Pod + IC roster assembled for a task envelope."""

    asset_class: str
    ic_agents: List[str] = field(default_factory=list)
    pod_managers: List[str] = field(default_factory=list)
    profile_used: str = ""
    declined: List[str] = field(default_factory=list)

    def ensure_dependencies(self) -> None:
        """Ensure pod managers have their dependent ICs scheduled."""
        for manager in list(self.pod_managers):
            for ic in POD_DEPENDENCIES.get(manager, []):
                if ic not in self.ic_agents:
                    self.ic_agents.append(ic)

    def prune_by_asset(self, registry: Dict[str, Dict[str, Sequence[str]]]) -> None:
        """Drop units that do not support the requested asset class."""
        allowed_ics = []
        for ic in self.ic_agents:
            if self.asset_class in registry["ics"].get(ic, {}).get(
                "asset_classes", set()
            ):
                allowed_ics.append(ic)
            else:
                self.declined.append(ic)
        self.ic_agents = allowed_ics

        allowed_pods = []
        for pod in self.pod_managers:
            if self.asset_class in registry["pod_managers"].get(pod, {}).get(
                "asset_classes", set()
            ):
                allowed_pods.append(pod)
            else:
                self.declined.append(pod)
        self.pod_managers = allowed_pods

    def as_log_dict(self) -> Dict[str, object]:
        return {
            "asset_class": self.asset_class,
            "profile": self.profile_used,
            "ics": self.ic_agents,
            "pod_managers": self.pod_managers,
            "declined": self.declined,
        }


class CrewAssembler:
    """Hybrid LLM + profile-based crew assembler."""

    def __init__(self):
        self._registry = {
            "ics": IC_REGISTRY,
            "pod_managers": POD_MANAGER_REGISTRY,
        }

    def assemble(
        self,
        envelope: TaskEnvelope,
        llm_decider: Optional[Callable[[str], str]] = None,
    ) -> PodSelection:
        profile = PROFILE_DEFAULTS.get(
            envelope.asset_class, PROFILE_DEFAULTS["equities"]
        )

        selection = PodSelection(
            asset_class=envelope.asset_class,
            ic_agents=list(profile["ics"]),
            pod_managers=list(profile["pod_managers"]),
            profile_used=envelope.asset_class,
        )

        if llm_decider:
            override = self._llm_override(envelope, llm_decider)
            if override:
                selection.ic_agents = override.get("ics", selection.ic_agents)
                selection.pod_managers = override.get(
                    "pod_managers", selection.pod_managers
                )

        selection.ensure_dependencies()
        selection.prune_by_asset(self._registry)
        selection.ic_agents = _dedupe_in_order(selection.ic_agents)
        selection.pod_managers = _dedupe_in_order(selection.pod_managers)

        logger.info(
            "Crew assembled",
            extra={
                "roster": selection.as_log_dict(),
                "registry_profiles": list(PROFILE_DEFAULTS.keys()),
            },
        )
        return selection

    def _llm_override(
        self,
        envelope: TaskEnvelope,
        llm_decider: Callable[[str], str],
    ) -> Optional[Dict[str, List[str]]]:
        """Optional LLM override: expects JSON with {ics:[], pod_managers:[]}."""
        prompt = self._build_prompt(envelope)
        try:
            raw = llm_decider(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"LLM decider failed, using profile defaults: {exc}")
            return None

        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM decider returned non-JSON; ignoring override")
            return None

        if not isinstance(parsed, dict):
            return None

        ics = parsed.get("ics")
        pod_managers = parsed.get("pod_managers")
        result: Dict[str, List[str]] = {}
        if isinstance(ics, list):
            result["ics"] = [str(x) for x in ics]
        if isinstance(pod_managers, list):
            result["pod_managers"] = [str(x) for x in pod_managers]
        return result or None

    def _build_prompt(self, envelope: TaskEnvelope) -> str:
        registry_view = registry_snapshot()
        return (
            "Assemble a trading crew for the task below. "
            "Return JSON with two arrays: ics and pod_managers.\n\n"
            f"Task intent: {envelope.task_intent}\n"
            f"Asset class: {envelope.asset_class}\n"
            f"Instrument type: {envelope.instrument_type}\n"
            f"Priority: {envelope.priority}\n"
            f"Notes: {envelope.notes}\n\n"
            f"Registry: {json.dumps(registry_view)}\n"
        )


def _dedupe_in_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
