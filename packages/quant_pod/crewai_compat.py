"""Lightweight compatibility layer for optional CrewAI dependency.

The real project uses `crewai` for agents, tools, crews, and flows. The test
suite only needs basic class shells so imports work even when `crewai` is not
installed. When CrewAI is available we simply re-export the real symbols.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

try:  # pragma: no cover - prefer real CrewAI when present
    from crewai import Agent, Crew, Process, Task  # type: ignore
    from crewai.agents.agent_builder.base_agent import BaseAgent  # type: ignore
    from crewai.project import (  # type: ignore
        CrewBase,
        after_kickoff,
        agent,
        before_kickoff,
        crew,
        task,
    )
    from crewai.tools import BaseTool  # type: ignore
    from crewai.flow.flow import Flow, listen, router, start  # type: ignore
except ImportError:  # pragma: no cover - executed in test environments without CrewAI

    class _Stub:
        """Minimal stub object with no behavior."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class Agent(_Stub):
        pass

    class Task(_Stub):
        pass

    class BaseAgent(_Stub):
        pass

    class Process:
        """Enum-like container for process types."""

        sequential = "sequential"
        hierarchical = "hierarchical"

    class Crew(_Stub):
        """Very small Crew stub that stores agents and tasks."""

        def __init__(
            self,
            agents: Optional[List[Any]] = None,
            tasks: Optional[List[Any]] = None,
            process: Any = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(agents, tasks, process, **kwargs)
            self.agents = agents or []
            self.tasks = tasks or []
            self.process = process

    def agent(fn: Callable) -> Callable:
        return fn

    def task(fn: Callable) -> Callable:
        return fn

    def crew(*args: Any, **kwargs: Any) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def before_kickoff(fn: Callable) -> Callable:
        return fn

    def after_kickoff(fn: Callable) -> Callable:
        return fn

    def CrewBase(cls: Any) -> Any:
        return cls

    class BaseTool(_Stub):
        """Stub BaseTool with a simple call-through to `_run`."""

        name: str
        description: str
        args_schema: Any
        return_direct: bool

        def __init__(
            self,
            name: str | None = None,
            description: str | None = None,
            args_schema: Any = None,
            return_direct: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(name, description, args_schema, return_direct, **kwargs)
            self.name = name or self.__class__.__name__
            self.description = description or ""
            self.args_schema = args_schema
            self.return_direct = return_direct

        def _run(
            self, *args: Any, **kwargs: Any
        ) -> Any:  # pragma: no cover - override in subclasses
            raise NotImplementedError(
                "_run must be implemented by subclasses or monkeypatched in tests"
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._run(*args, **kwargs)

    class Flow(_Stub):
        """Minimal Flow stub that just holds state."""

        def __init__(self, state: Any = None, *args: Any, **kwargs: Any) -> None:
            self.state = (
                state
                or getattr(self, "state", None)
                or getattr(self, "state_class", lambda: None)()
            )
            super().__init__(self.state, *args, **kwargs)

    def start(event: str | None = None) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def listen(event: str | None = None) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def router(fn: Callable) -> Callable:
        return fn


__all__ = [
    "Agent",
    "Crew",
    "Process",
    "Task",
    "BaseAgent",
    "BaseTool",
    "CrewBase",
    "agent",
    "task",
    "crew",
    "before_kickoff",
    "after_kickoff",
    "Flow",
    "listen",
    "router",
    "start",
]
