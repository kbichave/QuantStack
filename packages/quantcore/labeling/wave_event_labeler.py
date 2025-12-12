"""
Wave-aware event labeling for trade analysis.

Extends standard event labeling to include wave context,
enabling analysis of MR trade performance by wave phase.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.labeling.event_labeler import EventLabeler, LabelConfig
from quantcore.features.waves import WaveFeatures, WaveRole
from quantcore.hierarchy.wave_context import WaveContextAnalyzer
from quantcore.config.timeframes import Timeframe


@dataclass
class WaveEventLabel:
    """
    Extended label including wave context at trade entry.

    Used for analyzing trade performance by wave phase.
    """

    # Standard trade outcome
    outcome: int  # 1 = win, 0 = loss
    bars_to_exit: int
    exit_type: str  # TP, SL, TIMEOUT
    pnl_pct: float

    # Wave context at entry
    wave_role: str
    wave_stage: int
    wave_conf: float
    prob_impulse_up: float
    prob_impulse_down: float
    prob_corr_down: float
    prob_corr_up: float

    # Derived flags
    entered_in_corr_down: bool
    entered_in_corr_up: bool
    entered_in_late_impulse: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome,
            "bars_to_exit": self.bars_to_exit,
            "exit_type": self.exit_type,
            "pnl_pct": self.pnl_pct,
            "wave_role": self.wave_role,
            "wave_stage": self.wave_stage,
            "wave_conf": self.wave_conf,
            "prob_impulse_up": self.prob_impulse_up,
            "prob_impulse_down": self.prob_impulse_down,
            "prob_corr_down": self.prob_corr_down,
            "prob_corr_up": self.prob_corr_up,
            "entered_in_corr_down": self.entered_in_corr_down,
            "entered_in_corr_up": self.entered_in_corr_up,
            "entered_in_late_impulse": self.entered_in_late_impulse,
        }


class WaveEventLabeler:
    """
    Labels trade events with wave context for analysis.

    Combines standard TP/SL outcome labeling with wave phase information,
    enabling performance attribution by wave context.
    """

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.H1,
        config: Optional[LabelConfig] = None,
    ):
        """
        Initialize wave event labeler.

        Args:
            timeframe: Primary timeframe for labeling
            config: Label configuration (uses timeframe defaults if None)
        """
        self.timeframe = timeframe
        self.event_labeler = EventLabeler(
            config or LabelConfig.from_timeframe(timeframe)
        )
        self.wave_features = WaveFeatures(timeframe)

    def label_with_wave_context(
        self,
        df: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
        atr_column: str = "atr",
    ) -> pd.DataFrame:
        """
        Label trades with wave context from 4H timeframe.

        Args:
            df: Primary timeframe DataFrame (e.g., 1H)
            df_h4: Optional 4H DataFrame for wave context
            atr_column: ATR column name

        Returns:
            DataFrame with trade labels and wave context columns
        """
        result = df.copy()

        # Standard event labeling
        result = self.event_labeler.label_trades(result, atr_column)

        # Add wave context columns
        result = self._add_wave_context(result, df_h4)

        return result

    def _add_wave_context(
        self,
        df: pd.DataFrame,
        df_h4: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Add wave context columns from 4H data."""
        result = df.copy()

        # Initialize wave context columns
        result["entry_wave_role"] = "none"
        result["entry_wave_stage"] = -1
        result["entry_wave_conf"] = 0.0
        result["entry_prob_impulse_up"] = 0.0
        result["entry_prob_impulse_down"] = 0.0
        result["entry_prob_corr_down"] = 0.0
        result["entry_prob_corr_up"] = 0.0
        result["entry_in_corr_down"] = False
        result["entry_in_corr_up"] = False
        result["entry_in_late_impulse"] = False

        if df_h4 is None or df_h4.empty:
            logger.warning("No 4H data provided for wave context")
            return result

        # Compute wave features on 4H if not present
        if "wave_role" not in df_h4.columns:
            h4_features = WaveFeatures(Timeframe.H4)
            df_h4 = h4_features.compute(df_h4)

        # Map wave context from 4H to 1H via forward fill
        wave_cols = [
            "wave_role",
            "wave_stage",
            "wave_conf",
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
        ]

        h4_reindexed = df_h4[wave_cols].reindex(result.index, method="ffill")

        # Assign with entry_ prefix
        for col in wave_cols:
            result[f"entry_{col}"] = h4_reindexed[col]

        # Compute derived flags
        result["entry_in_corr_down"] = (
            result["entry_wave_role"] == WaveRole.CORR_DOWN.value
        ) | (result["entry_prob_corr_down"] > 0.6)

        result["entry_in_corr_up"] = (
            result["entry_wave_role"] == WaveRole.CORR_UP.value
        ) | (result["entry_prob_corr_up"] > 0.6)

        result["entry_in_late_impulse"] = (
            result["entry_wave_role"].isin(
                [
                    WaveRole.IMPULSE_UP_TERMINAL.value,
                    WaveRole.IMPULSE_DOWN_TERMINAL.value,
                ]
            )
        ) | (result["entry_wave_stage"].isin([4, 5]))

        return result

    def get_wave_stratified_stats(
        self,
        df: pd.DataFrame,
        direction: str = "long",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get trade statistics stratified by wave context.

        Args:
            df: Labeled DataFrame from label_with_wave_context()
            direction: "long" or "short"

        Returns:
            Dict of statistics by wave category
        """
        label_col = f"label_{direction}"

        if label_col not in df.columns:
            return {}

        # Filter to valid labels
        labeled = df[df[label_col].notna()].copy()

        if len(labeled) == 0:
            return {}

        stats = {}

        # Overall stats
        stats["all"] = self._calc_stats(
            labeled, label_col, f"label_{direction}_pnl_pct"
        )

        # By corrective down
        corr_down = labeled[labeled["entry_in_corr_down"] == True]
        if len(corr_down) > 0:
            stats["corr_down"] = self._calc_stats(
                corr_down, label_col, f"label_{direction}_pnl_pct"
            )

        # By corrective up
        corr_up = labeled[labeled["entry_in_corr_up"] == True]
        if len(corr_up) > 0:
            stats["corr_up"] = self._calc_stats(
                corr_up, label_col, f"label_{direction}_pnl_pct"
            )

        # By late impulse
        late_impulse = labeled[labeled["entry_in_late_impulse"] == True]
        if len(late_impulse) > 0:
            stats["late_impulse"] = self._calc_stats(
                late_impulse, label_col, f"label_{direction}_pnl_pct"
            )

        # By wave stage
        for stage in [1, 2, 3, 4, 5, 10, 11, 12]:
            stage_df = labeled[labeled["entry_wave_stage"] == stage]
            if len(stage_df) > 5:  # Minimum sample
                stats[f"stage_{stage}"] = self._calc_stats(
                    stage_df, label_col, f"label_{direction}_pnl_pct"
                )

        # By wave role
        for role in WaveRole:
            role_df = labeled[labeled["entry_wave_role"] == role.value]
            if len(role_df) > 5:
                stats[f"role_{role.value}"] = self._calc_stats(
                    role_df, label_col, f"label_{direction}_pnl_pct"
                )

        return stats

    def _calc_stats(
        self,
        df: pd.DataFrame,
        label_col: str,
        pnl_col: str,
    ) -> Dict[str, Any]:
        """Calculate statistics for a subset."""
        labels = df[label_col]
        pnl = df[pnl_col] if pnl_col in df.columns else pd.Series(dtype=float)

        wins = (labels == 1).sum()
        losses = (labels == 0).sum()
        total = wins + losses

        return {
            "count": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl_pct": pnl.mean() if len(pnl) > 0 else 0,
            "total_pnl_pct": pnl.sum() if len(pnl) > 0 else 0,
            "avg_win_pnl": pnl[labels == 1].mean() if wins > 0 else 0,
            "avg_loss_pnl": pnl[labels == 0].mean() if losses > 0 else 0,
        }


class WavePerformanceAnalyzer:
    """
    Analyzes trade performance by wave context.

    Used for diagnostics and model validation.
    """

    def __init__(self):
        """Initialize performance analyzer."""
        self.wave_labeler = WaveEventLabeler()

    def analyze_performance_by_wave(
        self,
        trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create performance summary grouped by wave context.

        Args:
            trades_df: DataFrame with trade outcomes and wave context

        Returns:
            Summary DataFrame with metrics by wave category
        """
        if trades_df.empty:
            return pd.DataFrame()

        # Group by wave role
        role_groups = []

        for role in WaveRole:
            role_trades = trades_df[
                trades_df.get("entry_wave_role", "none") == role.value
            ]

            if len(role_trades) < 5:
                continue

            pnl_col = None
            for col in ["net_pnl", "pnl_pct", "return_pct"]:
                if col in role_trades.columns:
                    pnl_col = col
                    break

            metrics = {
                "category": f"role_{role.value}",
                "trade_count": len(role_trades),
                "win_rate": (
                    (role_trades["outcome"] == 1).mean()
                    if "outcome" in role_trades.columns
                    else np.nan
                ),
            }

            if pnl_col:
                metrics["avg_pnl"] = role_trades[pnl_col].mean()
                metrics["total_pnl"] = role_trades[pnl_col].sum()
                metrics["sharpe"] = self._calc_sharpe(role_trades[pnl_col])

            role_groups.append(metrics)

        return pd.DataFrame(role_groups)

    def _calc_sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean = returns.mean()
        std = returns.std()

        if std == 0 or np.isnan(std):
            return 0.0

        return (mean / std) * np.sqrt(periods)

    def get_wave_filter_recommendations(
        self,
        stats: Dict[str, Dict[str, Any]],
        min_trades: int = 20,
        min_win_rate_diff: float = 0.05,
    ) -> List[str]:
        """
        Get recommendations for wave-based filtering.

        Args:
            stats: Statistics from get_wave_stratified_stats()
            min_trades: Minimum trades for statistical significance
            min_win_rate_diff: Minimum win rate difference to recommend

        Returns:
            List of recommendations
        """
        if "all" not in stats:
            return ["Insufficient data for analysis"]

        recommendations = []
        base_wr = stats["all"]["win_rate"]

        # Check corrective down performance
        if "corr_down" in stats and stats["corr_down"]["count"] >= min_trades:
            corr_wr = stats["corr_down"]["win_rate"]
            if corr_wr > base_wr + min_win_rate_diff:
                recommendations.append(
                    f"✓ LONG trades in corrective_down show +{(corr_wr - base_wr)*100:.1f}% win rate vs baseline"
                )
            elif corr_wr < base_wr - min_win_rate_diff:
                recommendations.append(
                    f"✗ LONG trades in corrective_down underperform by {(base_wr - corr_wr)*100:.1f}%"
                )

        # Check late impulse performance
        if "late_impulse" in stats and stats["late_impulse"]["count"] >= min_trades:
            late_wr = stats["late_impulse"]["win_rate"]
            if late_wr < base_wr - min_win_rate_diff:
                recommendations.append(
                    f"⚠ Consider filtering out late_impulse entries (-{(base_wr - late_wr)*100:.1f}% win rate)"
                )

        # Check wave 2/4 performance (ideal for MR)
        for stage in [2, 4]:
            key = f"stage_{stage}"
            if key in stats and stats[key]["count"] >= min_trades:
                stage_wr = stats[key]["win_rate"]
                if stage_wr > base_wr + min_win_rate_diff:
                    recommendations.append(
                        f"✓ Wave {stage} entries show +{(stage_wr - base_wr)*100:.1f}% win rate"
                    )

        if not recommendations:
            recommendations.append(
                "No significant wave-based filtering recommendations"
            )

        return recommendations
