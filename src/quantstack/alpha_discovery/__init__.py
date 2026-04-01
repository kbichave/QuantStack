# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AlphaDiscoveryEngine — overnight strategy discovery.

Discovers parameter combinations that pass IS/OOS statistical filters and
registers them as 'draft' strategies for human review in /workshop sessions.

Never auto-promotes to forward_testing. All discovered strategies are
status='draft', source='generated' until a human /workshop session reviews them.

Usage:
    from quantstack.alpha_discovery.engine import AlphaDiscoveryEngine

    result = AlphaDiscoveryEngine(dry_run=True).run(symbols=["XOM", "SPY"])
"""

from quantstack.alpha_discovery.engine import AlphaDiscoveryEngine
from quantstack.alpha_discovery.grammar_gp import GPConfig, GrammarGP

__all__ = ["AlphaDiscoveryEngine", "GrammarGP", "GPConfig"]
