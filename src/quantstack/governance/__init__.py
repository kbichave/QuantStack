# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Hierarchical governance — CIO mandate enforcement layer.

The governance package implements a three-tier control system:
  1. CIO Agent generates a DailyMandate once per day.
  2. mandate_check() enforces it as a hard gate before risk_gate.
  3. Conservative default kicks in if no mandate exists after 09:30 ET.
"""

from quantstack.governance.mandate import DailyMandate, get_active_mandate
from quantstack.governance.mandate_check import MandateVerdict, mandate_check

__all__ = [
    "DailyMandate",
    "MandateVerdict",
    "get_active_mandate",
    "mandate_check",
]
