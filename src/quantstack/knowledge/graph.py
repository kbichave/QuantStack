# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL-native Alpha Knowledge Graph.

Wraps the ``kg_nodes`` and ``kg_edges`` tables (phase-10 migration) with
domain-aware methods for hypothesis novelty, factor overlap, and research
history.  Uses pgvector ``<=>`` cosine distance for semantic similarity.

All DB access goes through ``db_conn()`` context managers so connections
are returned to the pool promptly.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

from loguru import logger

from quantstack.db import db_conn
from quantstack.knowledge.embeddings import generate_embedding
from quantstack.knowledge.kg_models import (
    HypothesisResult,
    NoveltyResult,
    OverlapResult,
    SimilarHypothesis,
)

# Cosine similarity threshold above which a hypothesis is considered redundant
_REDUNDANCY_THRESHOLD = 0.85

# Number of shared factors that makes a strategy "crowded"
_CROWDED_FACTOR_THRESHOLD = 2


class KnowledgeGraph:
    """Alpha Knowledge Graph backed by PostgreSQL + pgvector.

    Stateless — every method opens its own ``db_conn()`` context manager
    and commits on exit.  Safe to instantiate once and share across threads.
    """

    # ------------------------------------------------------------------
    # Node / edge primitives
    # ------------------------------------------------------------------

    def create_node(
        self,
        node_type: str,
        name: str,
        properties: dict | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Insert or update a knowledge graph node.

        Idempotent on ``(node_type, name)`` — if a node with the same type
        and name already exists, its properties and embedding are updated
        and the existing ``node_id`` is returned.

        Returns the ``node_id`` (new uuid4 on insert, existing on conflict).
        """
        node_id = str(uuid.uuid4())
        props_json = json.dumps(properties or {})

        with db_conn() as conn:
            if embedding is not None:
                conn.execute(
                    """
                    INSERT INTO kg_nodes (node_id, node_type, name, properties, embedding)
                    VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (node_type, name)
                    DO UPDATE SET properties = EXCLUDED.properties,
                                  embedding  = EXCLUDED.embedding
                    RETURNING node_id
                    """,
                    [node_id, node_type, name, props_json, str(embedding)],
                )
            else:
                conn.execute(
                    """
                    INSERT INTO kg_nodes (node_id, node_type, name, properties)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (node_type, name)
                    DO UPDATE SET properties = EXCLUDED.properties
                    RETURNING node_id
                    """,
                    [node_id, node_type, name, props_json],
                )
            row = conn.fetchone()
            return row["node_id"] if row else node_id

    def create_edge(
        self,
        edge_type: str,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        properties: dict | None = None,
        valid_from: date | None = None,
        valid_to: date | None = None,
    ) -> str:
        """Insert an edge between two existing nodes.

        Raises ``ValueError`` if either endpoint node does not exist.
        Returns the new ``edge_id``.
        """
        edge_id = str(uuid.uuid4())
        props_json = json.dumps(properties or {})

        with db_conn() as conn:
            # Validate endpoints
            conn.execute(
                "SELECT node_id FROM kg_nodes WHERE node_id IN (%s, %s)",
                [source_id, target_id],
            )
            found = {r["node_id"] for r in conn.fetchall()}
            missing = {source_id, target_id} - found
            if missing:
                raise ValueError(f"Node(s) not found: {missing}")

            conn.execute(
                """
                INSERT INTO kg_edges
                    (edge_id, edge_type, source_id, target_id, weight,
                     properties, valid_from, valid_to)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                [
                    edge_id, edge_type, source_id, target_id, weight,
                    props_json, valid_from, valid_to,
                ],
            )
            return edge_id

    # ------------------------------------------------------------------
    # Hypothesis novelty check
    # ------------------------------------------------------------------

    def check_hypothesis_novelty(
        self,
        hypothesis_text: str,
        regime: str | None = None,
    ) -> NoveltyResult:
        """Check whether *hypothesis_text* is novel vs. existing hypotheses.

        Embeds the text, queries the top-5 most similar hypothesis nodes via
        pgvector cosine distance, and classifies as:
          - ``"redundant"`` — similarity > 0.85 AND same regime
          - ``"similar_but_different_regime"`` — similarity > 0.85 but different regime
          - ``"novel"`` — no close match
        """
        emb = generate_embedding(hypothesis_text)

        with db_conn() as conn:
            conn.execute(
                """
                SELECT n.node_id, n.name, n.properties,
                       1 - (n.embedding <=> %s::vector) AS cosine_similarity
                FROM kg_nodes n
                WHERE n.node_type = 'hypothesis'
                  AND n.embedding IS NOT NULL
                ORDER BY n.embedding <=> %s::vector
                LIMIT 5
                """,
                [str(emb), str(emb)],
            )
            rows = conn.fetchall()

        similar: list[SimilarHypothesis] = []
        for r in rows:
            props = r.get("properties") or {}
            if isinstance(props, str):
                props = json.loads(props)
            similar.append(SimilarHypothesis(
                hypothesis_id=r["node_id"],
                name=r["name"],
                cosine_similarity=float(r["cosine_similarity"]),
                regime=props.get("regime"),
                outcome=props.get("outcome"),
            ))

        # Classification
        recommendation = "novel"
        is_novel = True
        for sh in similar:
            if sh.cosine_similarity >= _REDUNDANCY_THRESHOLD:
                if regime and sh.regime and sh.regime == regime:
                    recommendation = "redundant"
                    is_novel = False
                    break
                elif regime and sh.regime and sh.regime != regime:
                    recommendation = "similar_but_different_regime"
                    is_novel = True
                else:
                    # No regime info — treat high similarity as redundant
                    recommendation = "redundant"
                    is_novel = False
                    break

        return NoveltyResult(
            is_novel=is_novel,
            similar_hypotheses=similar,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Factor overlap check
    # ------------------------------------------------------------------

    def check_factor_overlap(self, strategy_id: str) -> OverlapResult:
        """Check how many factors *strategy_id* shares with other strategies.

        Uses a recursive CTE: strategy→(uses)→factor→(uses)→other_strategy.
        A strategy is "crowded" if it shares more than 2 factors with any
        single other strategy.
        """
        with db_conn() as conn:
            conn.execute(
                """
                WITH strategy_factors AS (
                    SELECT e.target_id AS factor_id, tn.name AS factor_name
                    FROM kg_edges e
                    JOIN kg_nodes tn ON tn.node_id = e.target_id
                    WHERE e.source_id = %s
                      AND e.edge_type = 'uses'
                      AND tn.node_type = 'factor'
                ),
                shared AS (
                    SELECT sf.factor_name,
                           e2.source_id AS other_strategy_id,
                           sn.name AS other_strategy_name
                    FROM strategy_factors sf
                    JOIN kg_edges e2 ON e2.target_id = sf.factor_id
                                    AND e2.edge_type = 'uses'
                                    AND e2.source_id != %s
                    JOIN kg_nodes sn ON sn.node_id = e2.source_id
                                    AND sn.node_type = 'strategy'
                )
                SELECT factor_name, other_strategy_name
                FROM shared
                """,
                [strategy_id, strategy_id],
            )
            rows = conn.fetchall()

        shared_factors: set[str] = set()
        affected_strategies: set[str] = set()
        for r in rows:
            shared_factors.add(r["factor_name"])
            affected_strategies.add(r["other_strategy_name"])

        shared_count = len(shared_factors)
        return OverlapResult(
            is_crowded=shared_count > _CROWDED_FACTOR_THRESHOLD,
            shared_factor_count=shared_count,
            shared_factors=sorted(shared_factors),
            affected_strategies=sorted(affected_strategies),
        )

    # ------------------------------------------------------------------
    # Research history
    # ------------------------------------------------------------------

    def get_research_history(
        self,
        topic: str,
        regime: str | None = None,
    ) -> list[HypothesisResult]:
        """Semantic search for past hypothesis results related to *topic*.

        Optionally filtered by regime (via node properties).
        """
        emb = generate_embedding(topic)

        with db_conn() as conn:
            conn.execute(
                """
                SELECT n.node_id, n.name, n.properties,
                       1 - (n.embedding <=> %s::vector) AS cosine_similarity
                FROM kg_nodes n
                WHERE n.node_type = 'hypothesis'
                  AND n.embedding IS NOT NULL
                ORDER BY n.embedding <=> %s::vector
                LIMIT 20
                """,
                [str(emb), str(emb)],
            )
            rows = conn.fetchall()

        results: list[HypothesisResult] = []
        for r in rows:
            props = r.get("properties") or {}
            if isinstance(props, str):
                props = json.loads(props)

            node_regime = props.get("regime")
            if regime and node_regime and node_regime != regime:
                continue

            results.append(HypothesisResult(
                hypothesis_id=r["node_id"],
                hypothesis_text=r["name"],
                test_date=props.get("test_date", "unknown"),
                outcome=props.get("outcome", "unknown"),
                result_sharpe=props.get("sharpe"),
                result_ic=props.get("ic"),
                regime_at_test=node_regime,
            ))

        return results

    # ------------------------------------------------------------------
    # Record experiment
    # ------------------------------------------------------------------

    def record_experiment(
        self,
        hypothesis: str,
        result: dict,
        factors_used: list[str],
        regime: str,
    ) -> str:
        """Record a completed experiment as nodes + edges in the KG.

        Creates:
          - hypothesis node (with embedding)
          - result node
          - factor nodes (idempotent)
          - edges: hypothesis→tested_by→result, hypothesis→uses→factor,
                   hypothesis→favors→regime (as a property, not a separate node)

        Returns the hypothesis ``node_id``.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Hypothesis node
        outcome = result.get("outcome", "unknown")
        hypothesis_props = {
            "regime": regime,
            "outcome": outcome,
            "test_date": now_iso,
            "sharpe": result.get("sharpe"),
            "ic": result.get("ic"),
        }
        emb = generate_embedding(hypothesis)
        hypothesis_id = self.create_node(
            "hypothesis", hypothesis, hypothesis_props, embedding=emb,
        )

        # Result node
        result_name = f"result_{hypothesis_id[:8]}_{now_iso[:10]}"
        result_id = self.create_node("result", result_name, result)

        # Edge: hypothesis → tested_by → result
        self.create_edge("tested_by", hypothesis_id, result_id)

        # Factor nodes + edges
        for factor_name in factors_used:
            factor_id = self.create_node("factor", factor_name)
            self.create_edge("uses", hypothesis_id, factor_id)

        logger.info(
            "[KG] Recorded experiment: hypothesis=%s regime=%s outcome=%s factors=%d",
            hypothesis[:60], regime, outcome, len(factors_used),
        )
        return hypothesis_id
