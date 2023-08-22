"Dataclasses common to multiple experiments"

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data_centric_synth.data_models.data_sculpting import (
    AlphaPrecisionMetrics,
    DataSegments,
)
from data_centric_synth.experiments.statistical_fidelity import (
    StatisticalFidelityMetrics,
)


class Processing(BaseModel):
    """Baseclass to represent preprocessing or postprocessing strategies."""

    strategy_name: str
    uncertainty_threshold: Optional[float]
    uncertainty_percentile_threshold: Optional[int]
    data_segments: Optional[DataSegments]
    detection_auc: Optional[float]
    statistical_fidelity: Optional[
        Union[StatisticalFidelityMetrics, AlphaPrecisionMetrics]
    ]


class Metrics(BaseModel):
    """Metrics for a single experiment."""

    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def __sub__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            roc_auc=self.roc_auc - other.roc_auc,
            accuracy=self.accuracy - other.accuracy,
            precision=self.precision - other.precision,
            recall=self.recall - other.recall,
            f1=self.f1 - other.f1,
        )


class PerformanceEvaluation(BaseModel):
    """Results from evaluating a single experiment on a single data segment."""

    metrics: Optional[Metrics]
    feature_importances: Optional[Dict[str, float]]


class CausalPerformanceEvaluation(BaseModel):
    """Results from evaluating a causal discovery model on a single data segment."""

    fdr: float
    tpr: float
    fpr: float
    shd: float
    nnz: float


class DataSegmentEvaluation(BaseModel):
    """Results from evaluating a single experiment on all data segments."""

    full: PerformanceEvaluation
    easy: Optional[PerformanceEvaluation]
    ambiguous: Optional[PerformanceEvaluation]
    hard: Optional[PerformanceEvaluation]


class SimulatedData(BaseModel):
    dataset: pd.DataFrame
    flipped_indices: np.ndarray
    proportion_flipped: float
    n_samples: int
    n_features: int
    corr_range: Tuple[float, float]

    class Config:
        arbitrary_types_allowed = True
        arbitrary_types_allowed = True
