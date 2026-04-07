"""Meta-agent subsystem for self-modification with guardrails.

Exports the most commonly needed symbols so callers can do::

    from quantstack.meta import get_threshold, PROTECTED_FILES
"""

from quantstack.meta.config import get_threshold
from quantstack.meta.guardrails import PROTECTED_FILES

__all__ = ["get_threshold", "PROTECTED_FILES"]
