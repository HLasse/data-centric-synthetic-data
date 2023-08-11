from data_centric_synth.data_models.experiment3 import Experiment3Suite
from data_centric_synth.evaluation.classification_performance import get_classification_performance_df
from data_centric_synth.evaluation.data_objects import NoisePerformanceDfs, PerformanceDfs
from data_centric_synth.evaluation.feature_selection import get_feature_selection_performance
from data_centric_synth.evaluation.model_selection import get_model_selection_performance_df
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS


from typing import List, Literal

from data_centric_synth.evaluation.statistical_fidelity import get_statistical_fidelity_performance_df


def get_performance_dfs(
    data_centric_methods: List[Literal[IMPLEMENTED_DATA_CENTRIC_METHODS]],
    synthetic_models: List[str],
    experiment_suite: Experiment3Suite,
) -> PerformanceDfs:
    classification_performance_df = get_classification_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    model_selection_df = get_model_selection_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    feature_selection_df = get_feature_selection_performance(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    return PerformanceDfs(
        classification=classification_performance_df,
        model_selection=model_selection_df,
        feature_selection=feature_selection_df,
    )


def get_noise_performance_dfs(
    data_centric_methods: List[Literal[IMPLEMENTED_DATA_CENTRIC_METHODS]],
    synthetic_models: List[str],
    experiment_suite: Experiment3Suite,
) -> NoisePerformanceDfs:
    classification_performance_df = get_classification_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    model_selection_df = get_model_selection_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    feature_selection_df = get_feature_selection_performance(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )
    statistical_fidelity_df = get_statistical_fidelity_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )

    return NoisePerformanceDfs(
        classification=classification_performance_df,
        model_selection=model_selection_df,
        feature_selection=feature_selection_df,
        statistical_fidelity=statistical_fidelity_df,
    )