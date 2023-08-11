from typing import Dict, Iterable, List, Union

import pandas as pd
import plotnine as pn
from pandas.io.json._normalize import nested_to_record # type: ignore
from wasabi import msg

from application.constants import RESULTS_DIR
from data_centric_synth.data_models.experiment3 import (
    Experiment3,
    Experiment3Suite,
    StatisticalFidelityExperiment,
    StatisticalFidelityExperimentGroup,
    StatisticalFidelityExperimentSuite,
)
from data_centric_synth.evaluation.extraction import (
    get_data_centric_method,

    get_experiments_with_statistical_fidelity,
    get_synthetic_model,
    get_unique_seeds,
    subset_by_seed,
)

from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

data_centric_methods = ["cleanlab", "dataiq", "datamaps"]
# synthetic_models = ["marginal_distributions", "tvae", "None", "ddpm"]
synthetic_models = [*get_default_synthetic_model_suite()]


def unpack_stat_fid_to_df(
    experiments: List[Experiment3],
) -> pd.DataFrame:
    """Get the performance of the causal estimation for each experiment."""
    statistical_fidelity_performances: List[Dict[str, Union[float, str]]] = [
        {
            **nested_to_record(experiment.postprocessing.statistical_fidelity.dict()),  # type: ignore
            "random_seed": experiment.random_seed,
            "preprocessing_strategy": experiment.preprocessing.strategy_name,
            "postprocessing_strategy": experiment.postprocessing.strategy_name,
        }
        for experiment in experiments
    ]
    return pd.DataFrame(statistical_fidelity_performances)


def get_statistical_fidelity_performance_df(
    experiment_suite: Experiment3Suite,
    data_centric_methods: Iterable[IMPLEMENTED_DATA_CENTRIC_METHODS],
    synthetic_models: List[str],
    verbose=False,
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = get_experiments_with_statistical_fidelity(dataset.experiments)
        if not exps:
            if verbose:
                msg.warn(f"No experiments for {dataset.dataset_id}")
            continue
        dataset_performances: List[pd.DataFrame] = []
        for method in data_centric_methods:
            data_centric_subset = get_data_centric_method(
                experiments=exps,
                method=method,
            )
            if not data_centric_subset:
                if verbose:
                    msg.warn(f"No experiments for {method} on {dataset.dataset_id}")
                continue
            method_performances: List[pd.DataFrame] = []
            for synthetic_model in synthetic_models:
                model_subset = get_synthetic_model(
                    experiments=data_centric_subset,
                    model=synthetic_model,
                )
                seed_performances: List[pd.DataFrame] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.append(
                        unpack_stat_fid_to_df(experiments=seed_subset),
                    )
                try:
                    synthetic_model_performance_df = pd.concat(seed_performances)
                    synthetic_model_performance_df[
                        "synthetic_model_type"
                    ] = synthetic_model
                    method_performances.append(synthetic_model_performance_df)
                except ValueError:
                    if verbose:
                        msg.warn(
                            f"No experiments for {dataset.dataset_id} {method} {synthetic_model}",
                        )
            method_performance_df = pd.concat(method_performances)
            method_performance_df["data_centric_method"] = method
            dataset_performances.append(method_performance_df)
        dataset_performance_df = pd.concat(dataset_performances)
        dataset_performance_df["dataset_id"] = dataset.dataset_id
        overall_performances.append(dataset_performance_df)

    return (
        pd.concat(overall_performances)
        .reset_index()
        .drop(columns=["index"])
        .drop_duplicates()
        .reset_index()
        .drop(columns=["index"])
    )


def unpack_stat_fid_exp_to_df(
    experiments: List[StatisticalFidelityExperiment],
) -> pd.DataFrame:
    """Get the performance of the causal estimation for each experiment."""
    statistical_fidelity_performances: List[Dict[str, Union[float, str]]] = [
        {
            **nested_to_record(experiment.statistical_fidelity_metrics.dict()),  # type: ignore
            "random_seed": experiment.random_seed,
            "preprocessing_strategy": experiment.preprocessing_strategy,
            "postprocessing_strategy": experiment.postprocessing_strategy,
        }
        for experiment in experiments
    ]
    return pd.DataFrame(statistical_fidelity_performances)


def get_statistical_fidelity_experiment_performance_df(
    experiment_suite: StatisticalFidelityExperimentSuite,
    data_centric_methods: Iterable[IMPLEMENTED_DATA_CENTRIC_METHODS],
    synthetic_models: List[str],
    verbose=False,
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = dataset.experiments
        if not exps:
            if verbose:
                msg.warn(f"No experiments for {dataset.dataset_id}")
            continue
        dataset_performances: List[pd.DataFrame] = []
        for method in data_centric_methods:
            data_centric_subset = get_data_centric_method(
                experiments=exps,  # type: ignore
                method=method,
            )
            if not data_centric_subset:
                if verbose:
                    msg.warn(f"No experiments for {method} on {dataset.dataset_id}")
                continue
            method_performances: List[pd.DataFrame] = []
            for synthetic_model in synthetic_models:
                model_subset = [
                    exp
                    for exp in data_centric_subset
                    if exp.synthetic_model == synthetic_model  # type: ignore
                ]
                seed_performances: List[pd.DataFrame] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.append(
                        unpack_stat_fid_exp_to_df(experiments=seed_subset),  # type: ignore
                    )
                try:
                    synthetic_model_performance_df = pd.concat(seed_performances)
                    synthetic_model_performance_df[
                        "synthetic_model_type"
                    ] = synthetic_model
                    method_performances.append(synthetic_model_performance_df)
                except ValueError:
                    if verbose:
                        msg.warn(
                            f"No experiments for {dataset.dataset_id} {method} {synthetic_model}",
                        )
            method_performance_df = pd.concat(method_performances)
            method_performance_df["data_centric_method"] = method
            dataset_performances.append(method_performance_df)
        dataset_performance_df = pd.concat(dataset_performances)
        dataset_performance_df["dataset_id"] = dataset.dataset_id
        overall_performances.append(dataset_performance_df)

    return (
        pd.concat(overall_performances)
        .reset_index()
        .drop(columns=["index"])
        .drop_duplicates()
        .reset_index()
        .drop(columns=["index"])
    )
