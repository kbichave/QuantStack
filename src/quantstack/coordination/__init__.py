# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Coordination layer for autonomous Ralph loops.

Provides inter-loop communication (event bus), atomic strategy transitions
(status lock), autonomous promotion/demotion (auto-promoter), degradation
enforcement, loop health monitoring, portfolio-level entry gating, and
daily digest reporting.

All state is persisted in the shared PostgreSQL database.
"""
