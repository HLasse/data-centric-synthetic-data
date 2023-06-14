from pathlib import Path
from pickle import UnpicklingError
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel
from wasabi import msg

from data_centric_synth.data_models.experiment3 import (
    Experiment3,
    Experiment3Group,
    Experiment3Suite,
    StatisticalFidelityExperiment,
    StatisticalFidelityExperimentGroup,
    StatisticalFidelityExperimentSuite,
)
from data_centric_synth.data_models.experiments_common import (
    DataSegmentEvaluation,
    Metrics,
    PerformanceEvaluation,
)
from data_centric_synth.serialization.serialization import load_from_pickle


class ClassificationModelPerformance(BaseModel):
    """The performance of a classification model."""

    classification_model_type: str
    data_segment_evaluation: DataSegmentEvaluation
    preprocessing_strategy: str
    postprocessing_strategy: str
    random_seed: int

    def get_performance_by_data_segment(
        self,
        data_segment: Literal["full", "easy", "ambiguous", "hard"],
    ) -> Optional[PerformanceEvaluation]:
        """Get the performance for a specific data segment."""
        performance = getattr(self.data_segment_evaluation, data_segment)
        if performance.metrics is not None:
            return performance
        return PerformanceEvaluation(
            metrics=Metrics(
                roc_auc=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            ),
            feature_importances=None,
        )

    def get_performance_metric(
        self,
        data_segment: Literal["full", "easy", "ambiguous", "hard"],
        metric: Literal["roc_auc", "accuracy", "precision", "recall", "f1"],
    ) -> float:
        """Get the performance for a specific data segment and metric."""
        return getattr(
            self.get_performance_by_data_segment(data_segment=data_segment).metrics,
            metric,
        )


def classification_model_performances_to_df(
    classification_model_performances: List[ClassificationModelPerformance],
) -> pd.DataFrame:
    """Convert a list of ClassificationModelPerformance to a dataframe."""
    return pd.DataFrame(
        [
            {
                "random_seed": performance.random_seed,
                "preprocessing_strategy": performance.preprocessing_strategy,
                "postprocessing_strategy": performance.postprocessing_strategy,
                "classification_model_type": performance.classification_model_type,
                "data_segment": data_segment,
                "metric": metric,
                "value": performance.get_performance_metric(
                    data_segment=data_segment,  # type: ignore
                    metric=metric,  # type: ignore
                ),
            }
            for performance in classification_model_performances
            for data_segment in ["full", "easy", "ambiguous", "hard"]
            for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]
        ],
    )


def get_experiment3_suite(dataset_dirs: Path, verbose=False) -> Experiment3Suite:
    experiment_groups: List[Experiment3Group] = []
    for dataset_dir in dataset_dirs.iterdir():
        dataset_experiments: List[Experiment3] = []
        for seed_dir in dataset_dir.iterdir():
            if seed_dir.name == ".DS_Store":
                continue
            for synthetic_model_experiments in seed_dir.iterdir():
                try:
                    experiments = load_from_pickle(synthetic_model_experiments)
                except (EOFError, UnpicklingError) as e:
                    if verbose:
                        msg.warn(
                            f"Could not load experiments from {synthetic_model_experiments}: {e}",
                        )
                    continue
                if experiments is not None:
                    dataset_experiments.extend(
                        load_from_pickle(synthetic_model_experiments),
                    )
                else:
                    if verbose:
                        msg.warn(
                            f"Could not load experiments from {synthetic_model_experiments}",
                        )
        experiment_groups.append(
            Experiment3Group(
                dataset_id=dataset_dir.name,
                experiments=dataset_experiments,
            ),
        )
    return Experiment3Suite(experiment_groups=experiment_groups)


def get_statistical_fidelity_suite(
    dataset_dirs: Path, verbose=False,
) -> StatisticalFidelityExperimentSuite:
    experiment_groups: List[StatisticalFidelityExperimentGroup] = []
    for dataset_dir in dataset_dirs.iterdir():
        dataset_experiments: List[StatisticalFidelityExperiment] = []
        for seed_dir in dataset_dir.iterdir():
            if seed_dir.name == ".DS_Store":
                continue
            for synthetic_model_experiments in seed_dir.iterdir():
                try:
                    experiments = load_from_pickle(synthetic_model_experiments)
                except (EOFError, UnpicklingError) as e:
                    if verbose:
                        msg.warn(
                            f"Could not load experiments from {synthetic_model_experiments}: {e}",
                        )
                    continue
                if experiments is not None:
                    dataset_experiments.extend(
                        load_from_pickle(synthetic_model_experiments),
                    )
                else:
                    if verbose:
                        msg.warn(
                            f"Could not load experiments from {synthetic_model_experiments}",
                        )
        experiment_groups.append(
            StatisticalFidelityExperimentGroup(
                dataset_id=dataset_dir.name,
                experiments=dataset_experiments,
            ),
        )
    return StatisticalFidelityExperimentSuite(experiment_groups=experiment_groups)


def get_experiments_with_dag_evaluation(
    experiments: List[Experiment3],
) -> List[Experiment3]:
    """Get a list of experiments with DAG results."""
    return [
        experiment
        for experiment in experiments
        if experiment.dag_evaluation is not None
    ]


def get_experiments_with_classification_model_evaluation(
    experiments: List[Experiment3],
) -> List[Experiment3]:
    """Get a list of experiments with classification model results."""
    return [
        experiment
        for experiment in experiments
        if experiment.classification_model_evaluation is not None
    ]


def get_experiments_with_causal_estimation(
    experiments: List[Experiment3],
) -> List[Experiment3]:
    """Get a list of experiments with causal estimation results."""
    return [
        experiment
        for experiment in experiments
        if experiment.dag_evaluation is not None
    ]


def get_experiments_with_statistical_fidelity(
    experiments: List[Experiment3],
) -> List[Experiment3]:
    """Get a list of experiments with statistical fidelity results."""
    return [
        experiment
        for experiment in experiments
        if experiment.postprocessing.statistical_fidelity is not None
    ]


def get_experiments_with_feature_importance(
    experiments: List[Experiment3],
) -> List[Experiment3]:
    """Get a list of experiments with feature importance results."""
    return [
        experiment
        for experiment in experiments
        if experiment.classification_model_evaluation.full.feature_importances
        is not None
    ]


def get_data_centric_method(
    experiments: List[Experiment3],
    method: str,
) -> List[Experiment3]:
    """Get the baseline experiment from the group."""
    return [exp for exp in experiments if exp.data_centric_method == method]


def get_synthetic_model(
    experiments: List[Experiment3],
    model: str,
) -> List[Experiment3]:
    """Get the baseline experiment from the group."""
    return [exp for exp in experiments if exp.synthetic_model_type == model]


def get_unique_seeds(experiments: List[Experiment3]) -> set[int]:
    """Get the unique seeds from the experiments."""
    return {exp.random_seed for exp in experiments}


def subset_by_seed(experiments: List[Experiment3], seed: int) -> List[Experiment3]:
    """Get the experiments from a specific seed"""
    return [exp for exp in experiments if exp.random_seed == seed]


def get_classification_model_performance(
    experiments: List[Experiment3],
) -> List[ClassificationModelPerformance]:
    """Get the classification model performance from the experiments."""
    return [
        ClassificationModelPerformance(
            classification_model_type=exp.classification_model_type,  # type: ignore
            data_segment_evaluation=exp.classification_model_evaluation,  # type: ignore
            preprocessing_strategy=exp.preprocessing.strategy_name,
            postprocessing_strategy=exp.postprocessing.strategy_name,
            random_seed=exp.random_seed,
        )
        for exp in experiments
    ]


class FeatureSelectionRanking(BaseModel):
    """The feature selection ranking."""

    classification_model_type: str
    feature_importances: Dict[str, float]
    preprocessing_strategy: str
    postprocessing_strategy: str
    random_seed: int


def get_feature_importances(
    experiments: List[Experiment3],
) -> List[FeatureSelectionRanking]:
    """Get the feature importances from the experiments."""
    return [
        FeatureSelectionRanking(
            classification_model_type=exp.classification_model_type,  # type: ignore
            feature_importances=exp.classification_model_evaluation.full.feature_importances,  # type: ignore
            preprocessing_strategy=exp.preprocessing.strategy_name,
            postprocessing_strategy=exp.postprocessing.strategy_name,
            random_seed=exp.random_seed,
        )
        for exp in experiments
    ]


def feature_importances_to_df(
    feature_importances: List[FeatureSelectionRanking],
) -> pd.DataFrame:
    """Convert a list of FeatureSelectionRanking to a dataframe."""
    return pd.DataFrame(
        [
            {
                "random_seed": feature_importance.random_seed,
                "preprocessing_strategy": feature_importance.preprocessing_strategy,
                "postprocessing_strategy": feature_importance.postprocessing_strategy,
                "classification_model_type": feature_importance.classification_model_type,
                "feature": feature,
                "importance": importance,
            }
            for feature_importance in feature_importances
            for feature, importance in feature_importance.feature_importances.items()
        ],
    )
