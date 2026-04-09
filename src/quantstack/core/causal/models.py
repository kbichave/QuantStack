"""
Causal inference data models.

Dataclasses for causal graph structure, treatment effects, and validation
results used throughout the causal alpha discovery pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CausalEdge:
    """A single edge in a causal graph."""

    source: str
    target: str
    edge_type: str  # "directed" or "undirected"
    strength: float


@dataclass
class CausalGraph:
    """
    Discovered causal graph with edges, adjacency structure, and metadata.

    The graph is the output of a causal discovery algorithm (e.g. PC) and
    represents conditional independence relationships between features and
    returns.
    """

    edges: list[CausalEdge]
    adjacency: dict[str, list[str]]
    discovery_method: str
    feature_columns: list[str]
    num_samples: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def directed_edges(self) -> list[CausalEdge]:
        """Return only directed edges (causal claims)."""
        return [e for e in self.edges if e.edge_type == "directed"]

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        return {
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "strength": e.strength,
                }
                for e in self.edges
            ],
            "adjacency": self.adjacency,
            "discovery_method": self.discovery_method,
            "feature_columns": self.feature_columns,
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CausalGraph:
        """Deserialize from dictionary."""
        edges = [
            CausalEdge(
                source=e["source"],
                target=e["target"],
                edge_type=e["edge_type"],
                strength=e["strength"],
            )
            for e in data["edges"]
        ]
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            edges=edges,
            adjacency=data["adjacency"],
            discovery_method=data["discovery_method"],
            feature_columns=data["feature_columns"],
            num_samples=data["num_samples"],
            created_at=created_at,
        )


@dataclass
class ValidationResult:
    """Result of validating a discovered graph against domain priors."""

    confirmed_edges: list[tuple[str, str]]
    missing_edges: list[tuple[str, str]]
    unexpected_edges: list[tuple[str, str]]
    domain_agreement_score: float


@dataclass
class CausalHypothesis:
    """A testable causal hypothesis: treatment -> outcome with expected direction."""

    treatment: str
    outcome: str
    expected_direction: str  # "positive" or "negative"
    description: str


@dataclass
class TreatmentEffect:
    """Estimated treatment effect with confidence interval and refutation status."""

    hypothesis: CausalHypothesis
    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    refutation_passed: bool
    regime_stability: float
