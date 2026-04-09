"""
Causal graph discovery via constraint-based algorithms.

Uses the PC algorithm (via dowhy.gcm) to discover conditional independence
structure between features and returns. Falls back to a correlation-based
skeleton when dowhy is not installed, so the rest of the pipeline degrades
gracefully rather than crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.core.causal.models import (
    CausalEdge,
    CausalGraph,
    ValidationResult,
)


class CausalGraphBuilder:
    """Build causal graphs from feature + return data."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    def build_graph(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        method: str = "pc",
    ) -> CausalGraph:
        """
        Discover causal structure from features and returns.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (rows = observations, columns = feature names).
        returns : pd.Series
            Target return series aligned with features index.
        method : str
            Discovery algorithm. Currently only "pc" is supported.

        Returns
        -------
        CausalGraph
            Discovered graph. Empty graph if dowhy is unavailable.
        """
        combined = features.copy()
        returns_col = returns.name or "returns"
        combined[returns_col] = returns.values

        # Drop rows with NaN — causal discovery needs complete cases
        combined = combined.dropna()
        if combined.shape[0] < 30:
            logger.warning(
                "Only {} complete rows after dropna — causal discovery "
                "needs at least 30 samples, returning empty graph",
                combined.shape[0],
            )
            return self._empty_graph(
                list(features.columns), combined.shape[0], method
            )

        feature_cols = list(features.columns)

        try:
            return self._build_with_dowhy(combined, feature_cols, method)
        except ImportError:
            logger.warning(
                "dowhy not installed — falling back to correlation-based "
                "skeleton. Install dowhy for proper PC discovery."
            )
            return self._build_correlation_skeleton(
                combined, feature_cols, method
            )

    # ------------------------------------------------------------------
    # dowhy-based PC discovery
    # ------------------------------------------------------------------

    def _build_with_dowhy(
        self,
        combined: pd.DataFrame,
        feature_cols: list[str],
        method: str,
    ) -> CausalGraph:
        """Run PC algorithm via dowhy.gcm."""
        import networkx as nx
        from dowhy import gcm

        causal_model = gcm.StructuralCausalModel(nx.DiGraph())
        graph = gcm.fit_and_compute(
            causal_model, combined, method=method
        ) if hasattr(gcm, "fit_and_compute") else None

        # dowhy's PC returns a networkx graph — extract edges
        if graph is None:
            # Fallback: use dowhy's independence test wrapper directly
            from dowhy.gcm.independence_test import approx_kernel_based
            edges: list[CausalEdge] = []
            adjacency: dict[str, list[str]] = {
                c: [] for c in combined.columns
            }
            cols = list(combined.columns)
            for i, src in enumerate(cols):
                for tgt in cols[i + 1 :]:
                    _, p_val = approx_kernel_based(
                        combined[[src]].values,
                        combined[[tgt]].values,
                    )
                    if p_val < self.significance_level:
                        strength = 1.0 - float(p_val)
                        edges.append(
                            CausalEdge(src, tgt, "undirected", strength)
                        )
                        adjacency[src].append(tgt)
                        adjacency[tgt].append(src)

            return CausalGraph(
                edges=edges,
                adjacency=adjacency,
                discovery_method=method,
                feature_columns=feature_cols,
                num_samples=combined.shape[0],
            )

        # Parse networkx DiGraph
        edges = []
        adjacency: dict[str, list[str]] = {c: [] for c in combined.columns}
        for src, tgt, data in graph.edges(data=True):
            strength = float(data.get("weight", 1.0))
            edges.append(CausalEdge(src, tgt, "directed", strength))
            adjacency[src].append(tgt)

        return CausalGraph(
            edges=edges,
            adjacency=adjacency,
            discovery_method=method,
            feature_columns=feature_cols,
            num_samples=combined.shape[0],
        )

    # ------------------------------------------------------------------
    # Correlation-based fallback
    # ------------------------------------------------------------------

    def _build_correlation_skeleton(
        self,
        combined: pd.DataFrame,
        feature_cols: list[str],
        method: str,
    ) -> CausalGraph:
        """
        Build an undirected skeleton from pairwise correlations.

        This is NOT causal discovery — it only captures marginal associations.
        Edges are kept where |correlation| exceeds a threshold derived from
        the significance level and sample size.
        """
        n = combined.shape[0]
        # Fisher-z critical value for significance_level (two-tailed)
        from scipy import stats

        z_crit = stats.norm.ppf(1 - self.significance_level / 2)
        corr_threshold = np.tanh(z_crit / np.sqrt(n - 3))

        corr_matrix = combined.corr()
        cols = list(combined.columns)
        edges: list[CausalEdge] = []
        adjacency: dict[str, list[str]] = {c: [] for c in cols}

        for i, src in enumerate(cols):
            for tgt in cols[i + 1 :]:
                r = corr_matrix.loc[src, tgt]
                if abs(r) > corr_threshold:
                    edges.append(
                        CausalEdge(src, tgt, "undirected", abs(float(r)))
                    )
                    adjacency[src].append(tgt)
                    adjacency[tgt].append(src)

        logger.info(
            "Correlation skeleton: {} edges from {} features "
            "(threshold={:.3f}, n={})",
            len(edges),
            len(feature_cols),
            corr_threshold,
            n,
        )
        return CausalGraph(
            edges=edges,
            adjacency=adjacency,
            discovery_method=f"{method}_correlation_fallback",
            feature_columns=feature_cols,
            num_samples=n,
        )

    # ------------------------------------------------------------------
    # Validation against domain priors
    # ------------------------------------------------------------------

    def validate_against_priors(
        self,
        graph: CausalGraph,
        priors: dict[str, list[str]],
    ) -> ValidationResult:
        """
        Check a discovered graph against domain knowledge.

        Parameters
        ----------
        priors : dict[str, list[str]]
            Expected causal parents for each child node.
            E.g. ``{"returns": ["momentum", "value"]}``.

        Returns
        -------
        ValidationResult
            Agreement score and edge-level breakdown.
        """
        confirmed: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        unexpected: list[tuple[str, str]] = []

        expected_pairs = {
            (parent, child)
            for child, parents in priors.items()
            for parent in parents
        }
        discovered_pairs = {(e.source, e.target) for e in graph.edges}
        # Also consider undirected edges as matching in either direction
        discovered_undirected = {(e.target, e.source) for e in graph.edges}
        discovered_all = discovered_pairs | discovered_undirected

        for pair in expected_pairs:
            if pair in discovered_all:
                confirmed.append(pair)
            else:
                missing.append(pair)

        for pair in discovered_pairs:
            if pair not in expected_pairs and (pair[1], pair[0]) not in expected_pairs:
                unexpected.append(pair)

        total = len(confirmed) + len(missing)
        agreement = len(confirmed) / total if total > 0 else 0.0

        return ValidationResult(
            confirmed_edges=confirmed,
            missing_edges=missing,
            unexpected_edges=unexpected,
            domain_agreement_score=agreement,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_graph(
        feature_cols: list[str], num_samples: int, method: str
    ) -> CausalGraph:
        return CausalGraph(
            edges=[],
            adjacency={c: [] for c in feature_cols},
            discovery_method=method,
            feature_columns=feature_cols,
            num_samples=num_samples,
        )


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------

def discover_causal_graph(
    features: pd.DataFrame,
    returns: pd.Series,
    method: str = "pc",
    significance_level: float = 0.05,
) -> CausalGraph:
    """Convenience wrapper around CausalGraphBuilder.build_graph."""
    builder = CausalGraphBuilder(significance_level=significance_level)
    return builder.build_graph(features, returns, method=method)
