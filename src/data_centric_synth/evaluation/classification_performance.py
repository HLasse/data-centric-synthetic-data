"""Tools for extracting classification performance results."""
from typing import Dict, List

import pandas as pd
from plotnine.stats.stat_summary import mean_cl_boot

from application.constants import RESULTS_DIR
from data_centric_synth.data_models.experiment3 import Experiment3Suite
from data_centric_synth.evaluation.extraction import (
    ClassificationModelPerformance,
    classification_model_performances_to_df,
    get_classification_model_performance,
    get_data_centric_method,
    get_experiment3_suite,
    get_experiments_with_classification_model_evaluation,
    get_synthetic_model,
    get_unique_seeds,
    subset_by_seed,
)

from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)


def get_classification_performance_df(
    experiment_suite: Experiment3Suite,
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS],
    synthetic_models: List[str],
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = get_experiments_with_classification_model_evaluation(dataset.experiments)
        dataset_performances: List[pd.DataFrame] = []
        for method in data_centric_methods:
            data_centric_subset = get_data_centric_method(
                experiments=exps,
                method=method,
            )
            method_performances: List[pd.DataFrame] = []
            for synthetic_model in synthetic_models:
                model_subset = get_synthetic_model(
                    experiments=data_centric_subset,
                    model=synthetic_model,
                )
                seed_performances: List[ClassificationModelPerformance] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.extend(
                        get_classification_model_performance(experiments=seed_subset),
                    )
                synthetic_model_performance_df = (
                    classification_model_performances_to_df(
                        classification_model_performances=seed_performances,
                    )
                )
                synthetic_model_performance_df["synthetic_model_type"] = synthetic_model
                method_performances.append(synthetic_model_performance_df)
            method_performance_df = pd.concat(method_performances)
            method_performance_df["data_centric_method"] = method
            dataset_performances.append(method_performance_df)
        dataset_performance_df = pd.concat(dataset_performances)
        dataset_performance_df["dataset_id"] = dataset.dataset_id
        overall_performances.append(dataset_performance_df)

    overall_performance_df = pd.concat(overall_performances)
    # only keep data_segment == full
    overall_performance_df = overall_performance_df[
        overall_performance_df["data_segment"] == "full"
    ]

    return overall_performance_df


def make_categorical_cols(overall_performance_df) -> pd.DataFrame:
    # make data_segment a category so that it is ordered as full, easy, hard, ambiguous
    overall_performance_df["data_segment_cat"] = pd.Categorical(
        overall_performance_df["data_segment"],
        categories=["full", "easy", "hard", "ambiguous"],
    )
    # make preprocessing strategy a category so that it is ordered as org_data, baseline, easy_hard, easy_ambiguous_hard
    overall_performance_df["preprocessing_strategy_cat"] = pd.Categorical(
        overall_performance_df["preprocessing_strategy"],
        categories=["org_data", "baseline", "easy_hard", "easy_ambiguous_hard"],
    )
    # make postprocessing strategy a category so that it is orderes as org_data, baseline, easy_ambi
    overall_performance_df["postprocessing_strategy_cat"] = pd.Categorical(
        overall_performance_df["postprocessing_strategy"],
        categories=["org_data", "baseline", "easy_ambi"],
    )
    return overall_performance_df




if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
    "cleanlab",
    "dataiq",
    "datamaps",
]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    plot_save_dir = RESULTS_DIR / "experiment3" / "plots" / "classification_performance"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)

    # extract classification model performance
    overall_performance_df = get_classification_performance_df(
        experiment_suite=experiment_suite,
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
    )
