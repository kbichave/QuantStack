"""
Bootstrap-based regime augmentation for training data enrichment.

Tree-based models (LightGBM, XGBoost, CatBoost) learn decision boundaries,
not temporal patterns, so bootstrap resampling with perturbation is effective
for balancing regime representation without requiring heavy generative models
like TimeGAN or CTGAN.

Usage:
    augmenter = RegimeAugmenter()
    df_augmented = augmenter.augment(
        df, target_regime="crisis", n_synthetic=200, noise_std=0.05,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Regime identification helpers
# ---------------------------------------------------------------------------

_REGIME_RULES: dict[str, dict[str, float]] = {
    "crisis": {"vix_min": 35.0, "return_max": -0.01},
    "high_vol": {"vol_percentile_min": 90.0},
    "low_vol": {"vol_percentile_max": 10.0},
    "trending_up": {"return_min": 0.005, "adx_min": 25.0},
    "trending_down": {"return_max": -0.005, "adx_min": 25.0},
}


def _identify_regime_mask(
    df: pd.DataFrame,
    target_regime: str,
) -> pd.Series:
    """
    Return a boolean mask identifying rows belonging to ``target_regime``.

    Uses simple heuristics on columns available in the DataFrame.
    Falls back to volatility percentile if specific columns are missing.
    """
    n = len(df)
    mask = pd.Series(True, index=df.index)

    rules = _REGIME_RULES.get(target_regime)

    if rules is None:
        # Unknown regime label — treat as a column name if present
        if target_regime in df.columns:
            return df[target_regime].astype(bool)
        logger.warning(
            f"[Augmenter] Unknown regime '{target_regime}' and no matching column. "
            "Falling back to top-10% volatility windows."
        )
        returns = df["close"].pct_change()
        vol = returns.rolling(20).std()
        return vol >= vol.quantile(0.90)

    # Apply available rules
    if "vix_min" in rules and "vix" in df.columns:
        mask &= df["vix"] >= rules["vix_min"]
    if "return_max" in rules:
        returns = df["close"].pct_change(5)
        mask &= returns <= rules["return_max"]
    if "return_min" in rules:
        returns = df["close"].pct_change(5)
        mask &= returns >= rules["return_min"]
    if "vol_percentile_min" in rules:
        vol = df["close"].pct_change().rolling(20).std()
        threshold = vol.quantile(rules["vol_percentile_min"] / 100.0)
        mask &= vol >= threshold
    if "vol_percentile_max" in rules:
        vol = df["close"].pct_change().rolling(20).std()
        threshold = vol.quantile(rules["vol_percentile_max"] / 100.0)
        mask &= vol <= threshold
    if "adx_min" in rules and "adx" in df.columns:
        mask &= df["adx"] >= rules["adx_min"]

    matched = mask.sum()
    if matched == 0:
        logger.warning(
            f"[Augmenter] No rows matched regime '{target_regime}'. "
            f"Tried rules: {list(rules.keys())}. "
            "Returning empty mask."
        )
    else:
        logger.info(
            f"[Augmenter] Identified {matched}/{n} rows "
            f"({matched / n:.1%}) as '{target_regime}'"
        )
    return mask


# ---------------------------------------------------------------------------
# RegimeAugmenter
# ---------------------------------------------------------------------------


class RegimeAugmenter:
    """
    Generate synthetic samples for underrepresented regimes via
    bootstrap resampling with Gaussian perturbation.

    How it works:
      1. Identify rows belonging to ``target_regime`` using rule-based masks.
      2. Sample ``n_synthetic`` rows (with replacement) from the regime window.
      3. Add Gaussian noise (scaled by ``noise_std * column_std``) to numeric
         columns to create plausible variations.
      4. Concatenate synthetic rows to the original DataFrame.

    This is intentionally simple. Tree models care about feature value
    distributions, not temporal ordering, so bootstrap + noise is sufficient
    to improve class balance without introducing temporal artifacts.
    """

    def augment(
        self,
        df: pd.DataFrame,
        target_regime: str,
        n_synthetic: int = 100,
        noise_std: float = 0.05,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """
        Augment the DataFrame with synthetic samples from the target regime.

        Args:
            df: Training DataFrame with at least a 'close' column.
            target_regime: Regime to augment. Built-in options:
                "crisis", "high_vol", "low_vol", "trending_up", "trending_down".
                Can also be a column name containing boolean regime labels.
            n_synthetic: Number of synthetic rows to generate.
            noise_std: Noise scale as a fraction of each column's std dev.
                0.05 = 5% of column std. Higher values create more diverse
                samples but risk generating implausible data.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with original rows + n_synthetic augmented rows.
            Augmented rows have index prefixed with 'syn_'.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        rng = np.random.default_rng(seed)

        # Step 1: identify regime windows
        mask = _identify_regime_mask(df, target_regime)
        regime_rows = df.loc[mask]

        if len(regime_rows) == 0:
            logger.warning(
                f"[Augmenter] No regime rows found for '{target_regime}'. "
                "Returning original DataFrame unchanged."
            )
            return df.copy()

        # Step 2: bootstrap resample
        indices = rng.choice(len(regime_rows), size=n_synthetic, replace=True)
        synthetic = regime_rows.iloc[indices].copy()

        # Step 3: add noise to numeric columns
        numeric_cols = synthetic.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_std = df[col].std()
            if col_std > 0:
                noise = rng.normal(0, noise_std * col_std, size=n_synthetic)
                synthetic[col] = synthetic[col].values + noise

        # Step 4: assign synthetic index
        synthetic.index = pd.Index(
            [f"syn_{target_regime}_{i}" for i in range(n_synthetic)]
        )

        combined = pd.concat([df, synthetic], axis=0)

        logger.info(
            f"[Augmenter] Added {n_synthetic} synthetic '{target_regime}' rows. "
            f"Original: {len(df)}, Augmented: {len(combined)}"
        )
        return combined
