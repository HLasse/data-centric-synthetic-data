import pandas as pd
from pydantic import BaseModel
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    InverseKLDivergence,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)
from synthcity.plugins.core.dataloader import GenericDataLoader

from data_centric_synth.data_models.data_sculpting import AlphaPrecisionMetrics


class StatisticalFidelityMetrics(BaseModel):
    alpha_precision_metrics: AlphaPrecisionMetrics
    inv_kl_divergence: float
    wasserstein_distance: float
    mmd: float


class StatisticalFidelity:
    @staticmethod
    def calculate_metrics(
        X: pd.DataFrame, X_syn: pd.DataFrame
    ) -> StatisticalFidelityMetrics:
        return StatisticalFidelityMetrics(
            alpha_precision_metrics=StatisticalFidelity.calculate_alpha_precision(
                X, X_syn
            ),
            inv_kl_divergence=StatisticalFidelity.calculate_inv_kl_divergence(X, X_syn),
            wasserstein_distance=StatisticalFidelity.calculate_wasserstein_distance(
                X, X_syn
            ),
            mmd=StatisticalFidelity.calculate_mmd(X, X_syn),
        )

    @staticmethod
    def calculate_alpha_precision(
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> AlphaPrecisionMetrics:
        """Calculate the alpha precision metrics for synthetic data."""
        try:
            results = AlphaPrecision()._evaluate(
            X=GenericDataLoader(data=X, target_column="target"),
            X_syn=GenericDataLoader(data=X_syn, target_column="target"),
        )
        # if synthetic and generated data not the same length, e.g. after postprocessing
        except RuntimeError:
            return AlphaPrecisionMetrics(
                delta_precision_alpha_OC=0.0,
                delta_coverage_beta_OC=0.0,
                authenticity_OC=0.0,
                delta_coverage_beta_naive=0.0,
                delta_precision_alpha_naive=0.0,
                authenticity_naive=0.0,
            )
        return AlphaPrecisionMetrics(**results)

    @staticmethod
    def calculate_mmd(
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:
        """Calculate the MMD for synthetic data."""
        results = MaximumMeanDiscrepancy()._evaluate(
            X_gt=GenericDataLoader(data=X, target_column="target"),
            X_syn=GenericDataLoader(data=X_syn, target_column="target"),
        )
        return results["joint"]

    @staticmethod
    def calculate_wasserstein_distance(X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
        results = WassersteinDistance()._evaluate(
            X=GenericDataLoader(data=X, target_column="target"),
            X_syn=GenericDataLoader(data=X_syn, target_column="target"),
        )
        return results["joint"]

    @staticmethod
    def calculate_inv_kl_divergence(
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:
        results = InverseKLDivergence()._evaluate(
            X_gt=GenericDataLoader(data=X, target_column="target"),
            X_syn=GenericDataLoader(data=X_syn, target_column="target"),
        )
        return results["marginal"]
