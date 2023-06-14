"""Dataclasses specific to the main experiment"""
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from data_centric_synth.data_models.experiments_common import (
    CausalPerformanceEvaluation,
    DataSegmentEvaluation,
    Processing,
)
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS
from data_centric_synth.experiments.statistical_fidelity import (
    StatisticalFidelityMetrics,
)


class Experiment3(BaseModel):
    """Results from a single experiment."""

    preprocessing: Processing
    postprocessing: Processing
    testprocessing: Processing
    synthetic_model_type: str
    classification_model_type: Optional[str]
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS
    classification_model_evaluation: Optional[DataSegmentEvaluation]
    normalized_evaluation: Optional[DataSegmentEvaluation] = None
    dag_evaluation: Optional[CausalPerformanceEvaluation]
    random_seed: int


class Experiment3Group(BaseModel):
    """Results from a group of experiments on the same dataset."""

    dataset_id: str
    experiments: List[Experiment3]


class Experiment3Suite(BaseModel):
    """Results from a suite of experiments on multiple datasets."""

    experiment_groups: List[Experiment3Group]


class Experiment3Dataset(BaseModel):
    """A dataset to run experiment 3 on"""

    name: str
    X: pd.DataFrame
    y: pd.Series

    class Config:
        arbitrary_types_allowed = True


class StatisticalFidelityExperiment(BaseModel):
    preprocessing_strategy: str
    postprocessing_strategy: str
    synthetic_model: str
    data_centric_method: str
    statistical_fidelity_metrics: StatisticalFidelityMetrics
    random_seed: int


class StatisticalFidelityExperimentGroup(BaseModel):
    """Results from a group of experiments on the same dataset."""

    dataset_id: str
    experiments: List[StatisticalFidelityExperiment]


class StatisticalFidelityExperimentSuite(BaseModel):
    """Results from a suite of experiments on multiple datasets."""

    experiment_groups: List[StatisticalFidelityExperimentGroup]
