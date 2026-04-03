# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Minimal BaseTool base class for QuantStack tool wrappers.

Provides a lightweight tool interface for classes that need name/description/
args_schema and a _run() method. Framework-agnostic — does not depend on
CrewAI or LangChain.
"""

from typing import Any


class BaseTool:
    """Minimal tool base class with _run contract."""

    name: str = ""
    description: str = ""
    args_schema: Any = None
    return_direct: bool = False

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement _run()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)
