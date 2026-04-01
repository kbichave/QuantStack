# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Waves and regime mixin — Wave scenario and regime state CRUD for KnowledgeStore."""

import uuid
from datetime import datetime

from quantstack.db import PgConnection
from quantstack.knowledge.models import (
    RegimeState,
    WaveScenario,
)


class WavesRegimeMixin:
    """Wave scenario and regime state operations."""

    conn: PgConnection

    # =========================================================================
    # WAVE SCENARIO OPERATIONS
    # =========================================================================

    def save_wave_scenario(self, scenario: WaveScenario) -> str:
        """Save a wave scenario."""
        if scenario.id is None:
            scenario.id = str(uuid.uuid4())[:8]

        data = scenario.model_dump()

        # Check if exists
        existing = self.conn.execute(
            "SELECT id FROM wave_scenarios WHERE id = ?", [scenario.id]
        ).fetchone()

        if existing:
            # Update
            cols = [k for k in data.keys() if k != "id"]
            set_clause = ", ".join([f"{k} = ?" for k in cols])
            self.conn.execute(
                f"UPDATE wave_scenarios SET {set_clause} WHERE id = ?",
                [data[k] for k in cols] + [scenario.id],
            )
        else:
            # Insert
            cols = list(data.keys())
            placeholders = ", ".join(["?" for _ in cols])
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO wave_scenarios ({col_names}) VALUES ({placeholders})",
                [data[k] for k in cols],
            )

        self.conn.commit()
        return scenario.id

    def get_active_wave_scenarios(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> list[WaveScenario]:
        """Get active wave scenarios."""
        query = "SELECT * FROM wave_scenarios WHERE is_active = TRUE"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY confidence DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [WaveScenario(**dict(zip(cols, row, strict=False))) for row in results]

    def invalidate_wave_scenario(self, scenario_id: str) -> None:
        """Mark a wave scenario as invalidated."""
        self.conn.execute(
            "UPDATE wave_scenarios SET is_active = FALSE, invalidated_at = ? WHERE id = ?",
            [datetime.now(), scenario_id],
        )
        self.conn.commit()

    # =========================================================================
    # REGIME STATE OPERATIONS
    # =========================================================================

    def save_regime_state(self, state: RegimeState) -> int:
        """Save a regime state."""
        data = state.model_dump()

        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO regime_states ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_current_regime(
        self,
        symbol: str,
        timeframe: str = "daily",
    ) -> RegimeState | None:
        """Get most recent regime state for symbol."""
        result = self.conn.execute(
            """SELECT * FROM regime_states
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC LIMIT 1""",
            [symbol, timeframe],
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        return RegimeState(**dict(zip(cols, result, strict=False)))
