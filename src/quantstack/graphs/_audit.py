"""Temporary runtime instrumentation for ghost field detection.

Wrap node return values to log any undeclared keys at runtime during paper
trading. Remove this module after the Pydantic migration (section-03) is
validated over 10+ paper trading cycles with zero ghost fields logged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def audit_node_return(node_name: str, state_class: type, returned: dict) -> dict:
    """Log any keys in *returned* that aren't declared in *state_class*.

    Returns *returned* unchanged so this can be inserted transparently:

        return audit_node_return("my_node", TradingState, result_dict)
    """
    if not isinstance(returned, dict):
        return returned
    declared = set(state_class.__annotations__.keys())
    actual = set(returned.keys())
    ghost = actual - declared
    if ghost:
        logger.warning(
            "GHOST FIELDS: node=%s returned undeclared keys: %s",
            node_name,
            sorted(ghost),
        )
    return returned
