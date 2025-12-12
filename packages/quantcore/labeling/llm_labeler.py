"""
LLM-based labeling interface (pluggable, no direct LLM calls).

Provides a clean interface for LLM-generated trade labels:
- Loads precomputed labels from file (Parquet/JSONL)
- Attaches labels to DataFrames
- Falls back to deterministic mock for testing

For production, labels should be generated externally (via MCP, separate script, etc.)
and stored in a file for the labeler to load.
"""

from typing import Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


class LLMLabelProvider:
    """
    Provider for LLM-generated trade labels.

    Supports:
    - Loading precomputed labels from file
    - Mock mode for testing (deterministic heuristics)
    - Clean interface for attaching labels to DataFrames

    Labels provided:
    - label_llm_quality: 0-1 score indicating trade setup quality
    - label_llm_type: categorical (e.g., "ideal_pullback", "late_reversion", "weak_setup")
    """

    def __init__(
        self,
        label_file_path: Optional[str] = None,
        use_mock: bool = True,
    ):
        """
        Initialize LLM label provider.

        Args:
            label_file_path: Path to precomputed label file (Parquet or JSONL)
            use_mock: Whether to use mock mode if file not found
        """
        self.label_file_path = label_file_path
        self.use_mock = use_mock
        self.labels_df: Optional[pd.DataFrame] = None

        # Try to load labels from file
        if label_file_path is not None:
            self._load_labels_from_file()

    def _load_labels_from_file(self) -> None:
        """Load precomputed LLM labels from file."""
        if self.label_file_path is None:
            return

        file_path = Path(self.label_file_path)

        if not file_path.exists():
            logger.warning(f"LLM label file not found: {file_path}")
            if self.use_mock:
                logger.info("Using mock LLM labels for testing")
            return

        try:
            # Load based on file extension
            if file_path.suffix == ".parquet":
                self.labels_df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(self.labels_df)} LLM labels from Parquet")
            elif file_path.suffix == ".jsonl":
                self.labels_df = pd.read_json(file_path, lines=True)
                logger.info(f"Loaded {len(self.labels_df)} LLM labels from JSONL")
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return

            # Validate required columns
            required_cols = ["label_llm_quality", "label_llm_type"]
            missing_cols = [
                col for col in required_cols if col not in self.labels_df.columns
            ]
            if missing_cols:
                logger.error(f"Missing required columns in label file: {missing_cols}")
                self.labels_df = None

        except Exception as e:
            logger.error(f"Failed to load LLM labels: {e}")
            self.labels_df = None

    def attach_llm_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach LLM labels to DataFrame.

        Args:
            df: DataFrame with OHLCV and features

        Returns:
            DataFrame with LLM label columns added
        """
        result = df.copy()

        # If we have precomputed labels, use them
        if self.labels_df is not None:
            result = self._merge_precomputed_labels(result)
        # Otherwise, use mock mode if enabled
        elif self.use_mock:
            result = self._generate_mock_labels(result)
        else:
            # No labels available, set to NaN
            result["label_llm_quality"] = np.nan
            result["label_llm_type"] = None

        return result

    def _merge_precomputed_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge precomputed labels by index alignment.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with merged labels
        """
        result = df.copy()

        # Align labels to DataFrame index
        label_cols = ["label_llm_quality", "label_llm_type"]

        for col in label_cols:
            if col in self.labels_df.columns:
                # Reindex to match df index, forward fill if needed
                result[col] = self.labels_df[col].reindex(result.index)

        return result

    def _generate_mock_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate mock LLM labels using deterministic heuristics.

        This simulates what an LLM might produce based on:
        - Existing ATR TP/SL labels
        - Trend regime
        - Pattern features
        - Technical alignment

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with mock LLM labels
        """
        result = df.copy()

        # Initialize
        result["label_llm_quality"] = 0.0
        result["label_llm_type"] = "unknown"

        # Base quality on existing ATR label if available
        if "label_long" in result.columns:
            base_quality = result["label_long"].fillna(0.5)
        else:
            base_quality = pd.Series(0.5, index=result.index)

        # Adjust quality based on context features
        for i in range(len(result)):
            bq = base_quality.iloc[i]
            quality = bq if pd.notna(bq) else 0.5
            label_type = "neutral"

            # Check for QuantAgent trend features
            if "qa_trend_regime" in result.columns:
                trend_regime = result["qa_trend_regime"].iloc[i]

                # Uptrend + pullback = ideal long setup
                if trend_regime == 1 and "qa_pattern_is_pullback" in result.columns:
                    if result["qa_pattern_is_pullback"].iloc[i] == 1:
                        quality = min(quality + 0.3, 1.0)
                        label_type = "ideal_pullback"

                # Consolidation + mean reversion signal
                if "qa_pattern_consolidation" in result.columns:
                    if result["qa_pattern_consolidation"].iloc[i] == 1:
                        if "zscore_price" in result.columns:
                            zscore = result["zscore_price"].iloc[i]
                            if pd.notna(zscore) and abs(zscore) > 2.0:
                                quality = min(quality + 0.2, 1.0)
                                label_type = "mr_opportunity"

                # Strong trend without pullback = late entry
                if trend_regime != 0 and "qa_pattern_is_pullback" in result.columns:
                    if result["qa_pattern_is_pullback"].iloc[i] == 0:
                        quality = max(quality - 0.2, 0.0)
                        label_type = "late_entry"

            # Check trend quality
            if "qa_trend_quality_med" in result.columns:
                trend_quality = result["qa_trend_quality_med"].iloc[i]
                if pd.notna(trend_quality):
                    if trend_quality > 0.7:
                        quality = min(quality + 0.1, 1.0)
                    elif trend_quality < 0.3:
                        quality = max(quality - 0.1, 0.0)
                        if label_type == "neutral":
                            label_type = "choppy_market"

            # Breakout patterns
            if "qa_pattern_is_breakout" in result.columns:
                if result["qa_pattern_is_breakout"].iloc[i] != 0:
                    quality = min(quality + 0.15, 1.0)
                    label_type = "breakout_attempt"

            # Consolidation without catalyst = weak setup
            if "qa_pattern_consolidation" in result.columns:
                if result["qa_pattern_consolidation"].iloc[i] == 1:
                    if label_type == "neutral":
                        quality = max(quality - 0.1, 0.0)
                        label_type = "weak_consolidation"

            result.loc[result.index[i], "label_llm_quality"] = quality
            result.loc[result.index[i], "label_llm_type"] = label_type

        return result

    def create_hybrid_label(
        self,
        df: pd.DataFrame,
        atr_label_col: str = "label_long",
        quality_col: str = "label_llm_quality",
        quality_weight: float = 0.3,
    ) -> pd.Series:
        """
        Create hybrid label by weighting ATR label with LLM quality.

        Hybrid label combines:
        - Base signal from ATR TP/SL label (0 or 1)
        - Quality adjustment from LLM label (0-1)

        Args:
            df: DataFrame with both label types
            atr_label_col: Column name for ATR label
            quality_col: Column name for LLM quality
            quality_weight: Weight for quality adjustment (0-1)

        Returns:
            Series with hybrid labels (continuous 0-1)
        """
        if atr_label_col not in df.columns:
            logger.warning(f"ATR label column '{atr_label_col}' not found")
            return pd.Series(np.nan, index=df.index)

        if quality_col not in df.columns:
            logger.warning(f"Quality column '{quality_col}' not found")
            # Fall back to just the ATR label
            return df[atr_label_col].copy()

        atr_label = df[atr_label_col].fillna(0.5)
        llm_quality = df[quality_col].fillna(0.5)

        # Weighted combination
        # If ATR says WIN (1.0) and quality is high (0.9), hybrid = 1.0
        # If ATR says WIN (1.0) but quality is low (0.2), hybrid = 0.82 (deemphasized)
        # If ATR says LOSS (0.0) but quality is high (0.9), hybrid = 0.27 (some value)
        hybrid = (1 - quality_weight) * atr_label + quality_weight * llm_quality

        return hybrid

    def get_label_statistics(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Get statistics about LLM labels.

        Args:
            df: DataFrame with LLM labels

        Returns:
            Dictionary with label statistics
        """
        stats = {}

        if "label_llm_quality" in df.columns:
            quality = df["label_llm_quality"].dropna()
            stats["count"] = len(quality)
            stats["mean_quality"] = float(quality.mean())
            stats["median_quality"] = float(quality.median())
            stats["std_quality"] = float(quality.std())

        if "label_llm_type" in df.columns:
            types = df["label_llm_type"].dropna()
            stats["type_distribution"] = types.value_counts().to_dict()

        return stats
