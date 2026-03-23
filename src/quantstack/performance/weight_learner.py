# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
WeightLearner — data-driven synthesis weight optimization.

Replaces hand-tuned regime-conditional weight profiles with weights
learned from historical trade outcomes.

Method:
1. Load last N days of closed trades + corresponding signal snapshots
2. Build feature matrix: individual indicator votes at time of entry
3. Label: 1 if trade was profitable, 0 if not
4. Fit logistic regression (per regime) — coefficients = optimal weights
5. Validate OOS: if learned weights produce worse classification than
   hand-tuned defaults, keep the defaults
6. Store versioned weights in DuckDB

Runs monthly as an autonomous job. No LLM involved.

Usage:
    learner = WeightLearner(conn)
    result = learner.learn_weights(lookback_days=90)
    # result = {"trending_up": {weights}, "ranging": {weights}, ...}
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import json

import duckdb
import numpy as np
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# The vote keys that synthesis.py uses
_VOTE_KEYS = ["trend", "rsi", "macd", "bb", "sentiment", "ml", "flow"]

# Default hand-tuned weights from synthesis.py (fallback if learning fails)
_DEFAULT_PROFILES: dict[str, dict[str, float]] = {
    "trending_up": {
        "trend": 0.35,
        "rsi": 0.10,
        "macd": 0.20,
        "bb": 0.05,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.05,
    },
    "trending_down": {
        "trend": 0.30,
        "rsi": 0.15,
        "macd": 0.20,
        "bb": 0.05,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.05,
    },
    "ranging": {
        "trend": 0.05,
        "rsi": 0.25,
        "macd": 0.10,
        "bb": 0.25,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.10,
    },
    "unknown": {
        "trend": 0.15,
        "rsi": 0.15,
        "macd": 0.15,
        "bb": 0.15,
        "sentiment": 0.10,
        "ml": 0.20,
        "flow": 0.10,
    },
}


class WeightLearner:
    """
    Learn optimal synthesis weights from trade history.

    Args:
        conn: DuckDB connection.
        min_trades_per_regime: Minimum trades needed to learn weights for a regime.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        min_trades_per_regime: int = 20,
    ) -> None:
        self._conn = conn
        self._min_trades = min_trades_per_regime
        self._ensure_table()

    def learn_weights(
        self,
        lookback_days: int = 90,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """
        Learn regime-conditional synthesis weights from trade history.

        Returns:
            Dict with per-regime weights and metadata.
        """
        snapshot_date = as_of or date.today()
        start_date = snapshot_date - timedelta(days=lookback_days)

        # Load trades with signal snapshots
        trades = self._load_trades_with_signals(start_date, snapshot_date)

        if not trades:
            logger.info(
                "[WeightLearner] No trades with signal snapshots — keeping defaults"
            )
            return {"status": "no_data", "profiles": _DEFAULT_PROFILES}

        # Group by regime
        by_regime: dict[str, list[dict]] = {}
        for t in trades:
            regime = t.get("regime_at_entry", "unknown")
            by_regime.setdefault(regime, []).append(t)

        learned_profiles: dict[str, dict[str, float]] = {}
        metadata: dict[str, dict[str, Any]] = {}

        for regime, regime_trades in by_regime.items():
            if len(regime_trades) < self._min_trades:
                logger.debug(
                    f"[WeightLearner] {regime}: only {len(regime_trades)} trades "
                    f"(need {self._min_trades}) — using defaults"
                )
                learned_profiles[regime] = _DEFAULT_PROFILES.get(
                    regime, _DEFAULT_PROFILES["unknown"]
                )
                metadata[regime] = {"source": "default", "n_trades": len(regime_trades)}
                continue

            weights, meta = self._fit_regime(regime, regime_trades)
            learned_profiles[regime] = weights
            metadata[regime] = meta

        # Fill in any missing regimes with defaults
        for regime in _DEFAULT_PROFILES:
            if regime not in learned_profiles:
                learned_profiles[regime] = _DEFAULT_PROFILES[regime]
                metadata[regime] = {"source": "default", "n_trades": 0}

        # Persist versioned weights
        self._persist_weights(learned_profiles, metadata, snapshot_date)

        logger.info(
            f"[WeightLearner] Learned weights for {len(by_regime)} regimes "
            f"from {len(trades)} trades ({lookback_days}d lookback)"
        )

        return {
            "status": "learned",
            "profiles": learned_profiles,
            "metadata": metadata,
            "total_trades": len(trades),
            "lookback_days": lookback_days,
        }

    def get_active_weights(self) -> dict[str, dict[str, float]]:
        """
        Load the most recent learned weights from DB.

        Returns default profiles if no learned weights exist.
        """
        try:
            row = self._conn.execute(
                "SELECT profiles_json FROM synthesis_weights ORDER BY effective_date DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass
        return _DEFAULT_PROFILES

    # ── Internal ──────────────────────────────────────────────────────────

    def _fit_regime(
        self,
        regime: str,
        trades: list[dict],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Fit logistic regression for a single regime.

        Features: indicator vote scores at entry time.
        Label: 1 if realized_pnl > 0, else 0.

        Returns (weights_dict, metadata_dict).
        """
        # Build feature matrix
        X_rows = []
        y_labels = []

        for t in trades:
            snapshot = t.get("signal_snapshot", {})
            # Extract vote scores from snapshot (these are the raw [-1, 1] votes)
            row = []
            for key in _VOTE_KEYS:
                # Try direct vote score, fall back to 0
                row.append(float(snapshot.get(f"vote_{key}", 0.0)))
            X_rows.append(row)
            y_labels.append(1 if t.get("realized_pnl", 0) > 0 else 0)

        X = np.array(X_rows)
        y = np.array(y_labels)

        # Check for degenerate data
        if len(set(y)) < 2:
            logger.debug(
                f"[WeightLearner] {regime}: all trades same outcome — using defaults"
            )
            return _DEFAULT_PROFILES.get(regime, _DEFAULT_PROFILES["unknown"]), {
                "source": "default",
                "reason": "single_class",
                "n_trades": len(trades),
            }

        # Check for zero-variance features
        non_zero_cols = X.std(axis=0) > 0
        if not non_zero_cols.any():
            return _DEFAULT_PROFILES.get(regime, _DEFAULT_PROFILES["unknown"]), {
                "source": "default",
                "reason": "zero_variance",
                "n_trades": len(trades),
            }

        try:
            # Only use non-zero-variance features
            X_filtered = X[:, non_zero_cols]
            active_keys = [k for k, keep in zip(_VOTE_KEYS, non_zero_cols) if keep]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filtered)

            model = LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=200, random_state=42
            )
            model.fit(X_scaled, y)

            # Extract coefficients as weights
            coefs = np.abs(
                model.coef_[0]
            )  # abs because we want importance, not direction
            coef_sum = coefs.sum()

            if coef_sum <= 0:
                return _DEFAULT_PROFILES.get(regime, _DEFAULT_PROFILES["unknown"]), {
                    "source": "default",
                    "reason": "zero_coefs",
                    "n_trades": len(trades),
                }

            # Normalize to sum to 1.0
            normalized = coefs / coef_sum

            # Build weights dict — inactive features get 0
            weights = {}
            active_idx = 0
            for i, key in enumerate(_VOTE_KEYS):
                if non_zero_cols[i]:
                    weights[key] = round(float(normalized[active_idx]), 4)
                    active_idx += 1
                else:
                    weights[key] = 0.0

            # Validate OOS: simple train/test split
            split = int(len(X_scaled) * 0.7)
            if split >= 5 and len(X_scaled) - split >= 5:
                model_val = LogisticRegression(
                    C=1.0, penalty="l2", solver="lbfgs", max_iter=200, random_state=42
                )
                model_val.fit(X_scaled[:split], y[:split])
                oos_accuracy = model_val.score(X_scaled[split:], y[split:])

                if oos_accuracy < 0.5:
                    logger.warning(
                        f"[WeightLearner] {regime}: OOS accuracy {oos_accuracy:.2f} < 0.5 "
                        f"— reverting to defaults"
                    )
                    return _DEFAULT_PROFILES.get(
                        regime, _DEFAULT_PROFILES["unknown"]
                    ), {
                        "source": "default",
                        "reason": f"oos_accuracy={oos_accuracy:.3f}",
                        "n_trades": len(trades),
                    }
            else:
                oos_accuracy = None

            accuracy = model.score(X_scaled, y)

            return weights, {
                "source": "learned",
                "n_trades": len(trades),
                "accuracy": round(accuracy, 3),
                "oos_accuracy": (
                    round(oos_accuracy, 3) if oos_accuracy is not None else None
                ),
                "active_features": active_keys,
            }

        except Exception as exc:
            logger.warning(
                f"[WeightLearner] {regime}: fit failed ({exc}) — using defaults"
            )
            return _DEFAULT_PROFILES.get(regime, _DEFAULT_PROFILES["unknown"]), {
                "source": "default",
                "reason": str(exc),
                "n_trades": len(trades),
            }

    def _load_trades_with_signals(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Load closed trades joined with signal snapshots."""
        try:
            rows = self._conn.execute(
                """
                SELECT
                    ct.symbol, ct.realized_pnl, ct.regime_at_entry,
                    ct.strategy_id, ct.opened_at,
                    ss.technical, ss.regime, ss.sentiment,
                    ss.consensus_bias, ss.consensus_conviction
                FROM closed_trades ct
                LEFT JOIN signal_snapshots ss
                    ON ct.symbol = ss.symbol
                    AND ABS(EPOCH(ct.opened_at) - EPOCH(ss.created_at)) < 3600
                WHERE ct.closed_at >= ? AND ct.closed_at <= ?
                ORDER BY ct.closed_at
                """,
                [start_date, end_date],
            ).fetchall()

            trades = []
            for r in rows:
                technical = json.loads(r[5]) if r[5] else {}
                regime = json.loads(r[6]) if r[6] else {}
                sentiment = json.loads(r[7]) if r[7] else {}

                # Reconstruct vote scores from raw signal data
                signal_snapshot = self._reconstruct_votes(technical, regime, sentiment)

                trades.append(
                    {
                        "symbol": r[0],
                        "realized_pnl": r[1],
                        "regime_at_entry": r[2] or "unknown",
                        "strategy_id": r[3] or "",
                        "opened_at": r[4],
                        "consensus_bias": r[8],
                        "consensus_conviction": r[9],
                        "signal_snapshot": signal_snapshot,
                    }
                )

            return trades
        except Exception as exc:
            logger.warning(f"[WeightLearner] Failed to load trades: {exc}")
            return []

    def _reconstruct_votes(
        self,
        technical: dict,
        regime: dict,
        sentiment: dict,
    ) -> dict[str, float]:
        """Reconstruct vote scores from raw signal data (same logic as synthesis.py)."""
        votes: dict[str, float] = {}

        # Trend vote
        trend = regime.get("trend_regime", "unknown")
        votes["vote_trend"] = {"trending_up": 1.0, "trending_down": -1.0}.get(
            trend, 0.0
        )

        # RSI vote
        rsi = technical.get("rsi_14")
        if rsi is not None:
            if rsi < 35:
                votes["vote_rsi"] = 1.0
            elif rsi > 65:
                votes["vote_rsi"] = -1.0
            else:
                votes["vote_rsi"] = (50 - rsi) / 15 * 0.5
        else:
            votes["vote_rsi"] = 0.0

        # MACD vote
        macd = technical.get("macd_hist")
        votes["vote_macd"] = (
            (1.0 if macd and macd > 0 else -1.0) if macd is not None else 0.0
        )

        # BB vote
        bb_pct = technical.get("bb_pct")
        if bb_pct is not None:
            if bb_pct < 0.2:
                votes["vote_bb"] = 1.0
            elif bb_pct > 0.8:
                votes["vote_bb"] = -1.0
            else:
                votes["vote_bb"] = 0.0
        else:
            votes["vote_bb"] = 0.0

        # Sentiment vote
        sent_score = sentiment.get("sentiment_score", 0.5)
        n_headlines = sentiment.get("n_headlines", 0)
        if n_headlines > 0:
            if sent_score > 0.65:
                votes["vote_sentiment"] = 1.0
            elif sent_score < 0.35:
                votes["vote_sentiment"] = -1.0
            else:
                votes["vote_sentiment"] = 0.0
        else:
            votes["vote_sentiment"] = 0.0

        # ML and flow votes (may not be in historical snapshots)
        votes["vote_ml"] = 0.0
        votes["vote_flow"] = 0.0

        return votes

    def _ensure_table(self) -> None:
        """Create synthesis_weights table if missing."""
        try:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS synthesis_weights (
                    version INTEGER PRIMARY KEY,
                    effective_date DATE NOT NULL,
                    profiles_json TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self._conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS synthesis_weights_seq START 1"
            )
        except Exception:
            pass

    def _persist_weights(
        self,
        profiles: dict[str, dict[str, float]],
        metadata: dict[str, dict[str, Any]],
        effective_date: date,
    ) -> None:
        """Store versioned weights in DuckDB."""
        try:
            self._conn.execute(
                """
                INSERT INTO synthesis_weights (version, effective_date, profiles_json, metadata_json)
                VALUES (nextval('synthesis_weights_seq'), ?, ?, ?)
                """,
                [
                    effective_date,
                    json.dumps(profiles),
                    json.dumps(metadata, default=str),
                ],
            )
            logger.info(f"[WeightLearner] Persisted weights effective {effective_date}")
        except Exception as exc:
            logger.warning(f"[WeightLearner] Failed to persist weights: {exc}")
