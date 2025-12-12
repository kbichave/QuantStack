"""
Feature orthogonalization and dimensionality reduction.

Removes redundant features to prevent overfitting.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class OrthogonalizationResult:
    """Result of feature orthogonalization."""

    original_features: int
    selected_features: int
    removed_features: List[str]
    feature_clusters: Dict[str, List[str]]
    explained_variance_ratio: Optional[float] = None


class CorrelationFilter:
    """
    Filter highly correlated features.

    Keeps one feature from each correlated cluster.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        method: str = "pearson",
    ):
        """
        Initialize correlation filter.

        Args:
            threshold: Correlation threshold for removal
            method: Correlation method (pearson, spearman, kendall)
        """
        self.threshold = threshold
        self.method = method
        self._selected_features: List[str] = []
        self._removed_features: List[str] = []
        self._clusters: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame) -> "CorrelationFilter":
        """
        Identify correlated feature groups.

        Args:
            X: Feature DataFrame

        Returns:
            self
        """
        # Calculate correlation matrix
        corr_matrix = X.corr(method=self.method).abs()

        # Track which features to keep
        n_features = len(X.columns)
        to_keep = set(range(n_features))
        clusters = {}

        for i in range(n_features):
            if i not in to_keep:
                continue

            col_i = X.columns[i]
            cluster = [col_i]

            for j in range(i + 1, n_features):
                if j not in to_keep:
                    continue

                col_j = X.columns[j]

                if corr_matrix.iloc[i, j] > self.threshold:
                    # Remove the one with higher avg correlation
                    avg_corr_i = corr_matrix.iloc[i, list(to_keep)].mean()
                    avg_corr_j = corr_matrix.iloc[j, list(to_keep)].mean()

                    if avg_corr_j > avg_corr_i:
                        to_keep.discard(j)
                        cluster.append(col_j)
                    else:
                        to_keep.discard(i)
                        cluster.append(col_j)
                        break

            if len(cluster) > 1:
                clusters[col_i] = cluster

        self._selected_features = [X.columns[i] for i in sorted(to_keep)]
        self._removed_features = [
            c for c in X.columns if c not in self._selected_features
        ]
        self._clusters = clusters

        logger.info(
            f"Correlation filter: {n_features} -> {len(self._selected_features)} features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame to selected features."""
        return X[self._selected_features]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def get_result(self) -> OrthogonalizationResult:
        """Get orthogonalization result."""
        return OrthogonalizationResult(
            original_features=len(self._selected_features)
            + len(self._removed_features),
            selected_features=len(self._selected_features),
            removed_features=self._removed_features,
            feature_clusters=self._clusters,
        )


class PCAReducer:
    """
    PCA-based dimensionality reduction.

    Transforms correlated features into orthogonal components.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        scale: bool = True,
    ):
        """
        Initialize PCA reducer.

        Args:
            n_components: Fixed number of components (overrides variance)
            variance_threshold: Target explained variance ratio
            scale: Whether to scale features before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scale = scale

        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "PCAReducer":
        """
        Fit PCA on data.

        Args:
            X: Feature DataFrame

        Returns:
            self
        """
        self._feature_names = list(X.columns)

        # Handle missing values
        X_clean = X.fillna(X.mean())

        # Scale if requested
        if self.scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean.values

        # Determine number of components
        if self.n_components is None:
            # Find components for target variance
            pca_full = PCA()
            pca_full.fit(X_scaled)

            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            n_components = max(1, min(n_components, X.shape[1]))
        else:
            n_components = min(self.n_components, X.shape[1])

        # Fit final PCA
        self._pca = PCA(n_components=n_components)
        self._pca.fit(X_scaled)

        logger.info(
            f"PCA: {X.shape[1]} -> {n_components} components "
            f"({self._pca.explained_variance_ratio_.sum():.1%} variance)"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to PCA components.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with PCA components
        """
        X_clean = X.fillna(X.mean())

        if self.scale and self._scaler:
            X_scaled = self._scaler.transform(X_clean)
        else:
            X_scaled = X_clean.values

        components = self._pca.transform(X_scaled)

        col_names = [f"PC{i+1}" for i in range(components.shape[1])]
        return pd.DataFrame(components, index=X.index, columns=col_names)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def get_loadings(self) -> pd.DataFrame:
        """Get feature loadings for each component."""
        if self._pca is None:
            return pd.DataFrame()

        loadings = pd.DataFrame(
            self._pca.components_.T,
            index=self._feature_names,
            columns=[f"PC{i+1}" for i in range(self._pca.n_components_)],
        )
        return loadings

    def get_top_features(
        self,
        n_features: int = 5,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top contributing features for each component.

        Args:
            n_features: Number of top features per component

        Returns:
            Dictionary of component -> [(feature, loading), ...]
        """
        loadings = self.get_loadings()

        result = {}
        for col in loadings.columns:
            sorted_loadings = loadings[col].abs().sort_values(ascending=False)
            top_features = [
                (feat, loadings.loc[feat, col])
                for feat in sorted_loadings.head(n_features).index
            ]
            result[col] = top_features

        return result


class FeatureOrthogonalizer:
    """
    Combined feature orthogonalization pipeline.

    Applies:
    1. Correlation filtering
    2. Optional PCA reduction
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        use_pca: bool = False,
        pca_variance_threshold: float = 0.95,
    ):
        """
        Initialize orthogonalizer.

        Args:
            correlation_threshold: Correlation threshold for filtering
            use_pca: Whether to apply PCA after correlation filter
            pca_variance_threshold: PCA variance threshold
        """
        self.corr_filter = CorrelationFilter(threshold=correlation_threshold)
        self.use_pca = use_pca
        self.pca_reducer = (
            PCAReducer(variance_threshold=pca_variance_threshold) if use_pca else None
        )

        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "FeatureOrthogonalizer":
        """
        Fit orthogonalization pipeline.

        Args:
            X: Feature DataFrame

        Returns:
            self
        """
        # Step 1: Correlation filter
        X_filtered = self.corr_filter.fit_transform(X)

        # Step 2: PCA (optional)
        if self.use_pca and self.pca_reducer:
            self.pca_reducer.fit(X_filtered)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.

        Args:
            X: Feature DataFrame

        Returns:
            Orthogonalized features
        """
        if not self._fitted:
            raise ValueError("Must fit before transform")

        # Step 1: Correlation filter
        X_filtered = self.corr_filter.transform(X)

        # Step 2: PCA (optional)
        if self.use_pca and self.pca_reducer:
            return self.pca_reducer.transform(X_filtered)

        return X_filtered

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def get_feature_importance_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping from output features to original features.

        Useful for interpreting model importance.
        """
        if self.use_pca and self.pca_reducer:
            return self.pca_reducer.get_top_features()
        else:
            # Direct mapping (1-to-1)
            return {f: [f] for f in self.corr_filter._selected_features}
