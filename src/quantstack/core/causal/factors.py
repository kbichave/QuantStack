"""
Causal factor library — pre-defined hypotheses and signal weighting.

Houses the priority hypotheses that the research pipeline tests first, plus
utilities to batch-estimate treatment effects and convert significant effects
into signal weights for the signal engine.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from quantstack.core.causal.models import CausalHypothesis, TreatmentEffect
from quantstack.core.causal.treatment_effects import TreatmentEstimator

# ------------------------------------------------------------------
# Pre-defined priority hypotheses
# ------------------------------------------------------------------

PRIORITY_HYPOTHESES: list[CausalHypothesis] = [
    CausalHypothesis(
        treatment="earnings_revision",
        outcome="fwd_30d_return",
        expected_direction="positive",
        description=(
            "Upward earnings revisions cause positive 30-day forward "
            "returns via delayed analyst information incorporation."
        ),
    ),
    CausalHypothesis(
        treatment="insider_buy",
        outcome="fwd_60d_return",
        expected_direction="positive",
        description=(
            "Insider purchases signal private positive information, "
            "driving 60-day forward returns."
        ),
    ),
    CausalHypothesis(
        treatment="short_interest_change",
        outcome="fwd_20d_return",
        expected_direction="negative",
        description=(
            "Rising short interest reflects informed bearish conviction, "
            "leading to negative 20-day forward returns."
        ),
    ),
    CausalHypothesis(
        treatment="analyst_upgrade",
        outcome="fwd_10d_return",
        expected_direction="positive",
        description=(
            "Analyst upgrades cause short-term price appreciation as "
            "the market re-prices the stock to the new consensus."
        ),
    ),
    CausalHypothesis(
        treatment="volume_surge",
        outcome="fwd_5d_return",
        expected_direction="positive",
        description=(
            "Unusual volume surges on up-days indicate institutional "
            "accumulation, predicting 5-day momentum continuation."
        ),
    ),
]


# ------------------------------------------------------------------
# Batch estimation
# ------------------------------------------------------------------

def build_causal_factor_library(
    features_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> list[TreatmentEffect]:
    """
    Estimate treatment effects for all priority hypotheses.

    Skips hypotheses whose treatment or outcome column is missing from the
    provided data rather than failing the entire batch.

    Parameters
    ----------
    features_df : pd.DataFrame
        Wide feature matrix. Must contain treatment columns.
    returns_df : pd.DataFrame
        DataFrame with forward-return columns (e.g. fwd_5d_return).

    Returns
    -------
    list[TreatmentEffect]
        One entry per hypothesis that could be evaluated.
    """
    combined = features_df.join(returns_df, how="inner")
    estimator = TreatmentEstimator(method="dml")
    results: list[TreatmentEffect] = []

    for hyp in PRIORITY_HYPOTHESES:
        if hyp.treatment not in combined.columns:
            logger.info(
                "Skipping hypothesis '{}' — treatment column '{}' not in data",
                hyp.description[:60],
                hyp.treatment,
            )
            continue
        if hyp.outcome not in combined.columns:
            logger.info(
                "Skipping hypothesis '{}' — outcome column '{}' not in data",
                hyp.description[:60],
                hyp.outcome,
            )
            continue

        # Use all other feature columns as confounders (exclude treatment
        # and outcome to avoid conditioning on colliders)
        confounders = [
            c
            for c in features_df.columns
            if c != hyp.treatment and c != hyp.outcome
        ]

        logger.info(
            "Estimating: {} -> {} ({} confounders)",
            hyp.treatment,
            hyp.outcome,
            len(confounders),
        )
        effect = estimator.estimate(
            combined,
            treatment_col=hyp.treatment,
            outcome_col=hyp.outcome,
            confounders=confounders,
            hypothesis=hyp,
        )
        results.append(effect)

    logger.info(
        "Causal factor library: {}/{} hypotheses evaluated",
        len(results),
        len(PRIORITY_HYPOTHESES),
    )
    return results


# ------------------------------------------------------------------
# Signal weight conversion
# ------------------------------------------------------------------

def get_causal_signal_weight(effect: TreatmentEffect) -> float:
    """
    Convert a treatment effect into a signal weight for the signal engine.

    Weight = |ATE| * regime_stability * refutation_confidence, clamped
    to [0, 0.5]. The cap prevents any single causal signal from
    dominating the composite.

    refutation_confidence is 1.0 if refutation passed, 0.3 otherwise —
    we still use unrefuted signals but down-weight them heavily.
    """
    refutation_confidence = 1.0 if effect.refutation_passed else 0.3
    raw_weight = abs(effect.ate) * effect.regime_stability * refutation_confidence
    clamped = max(0.0, min(0.5, raw_weight))
    return clamped
