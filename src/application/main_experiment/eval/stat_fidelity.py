"""Code to generate the statistical fidelity results for the main experiment."""
from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.noise_eval import aggregate_and_make_pretty
from data_centric_synth.data_models.experiment3 import (
    StatisticalFidelityExperiment,
    StatisticalFidelityExperimentSuite,
)
from data_centric_synth.evaluation.extraction import (
    get_experiment3_suite,
    get_statistical_fidelity_suite,
)
from data_centric_synth.evaluation.statistical_fidelity import (
    get_statistical_fidelity_experiment_performance_df,
    get_statistical_fidelity_performance_df,
)
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

pd.set_option("mode.chained_assignment", None)


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    # dataset_dirs = RESULTS_DIR / "experiment3" / "statistical_fidelity"
    dataset_dirs = DATA_DIR / "main_experiment" / "statistical_fidelity"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "summary"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_statistical_fidelity_suite(dataset_dirs)
    # save_to_pickle(experiment_suite, RESULTS_DIR / "experiment3" / "experiment_suite.pkl")

    ### BASELINE CONDITION
    ## Get statistical fidelity results, aggregate and save to csv for combination with other results

    statistical_fidelity_df = get_statistical_fidelity_experiment_performance_df(
        experiment_suite=experiment_suite,
        synthetic_models=synthetic_models,
        data_centric_methods=data_centric_methods,
    )
    statistical_fidelity_df["postprocessing_strategy"] = statistical_fidelity_df[
        "postprocessing_strategy"
    ].replace({"easy_ambi": "no_hard"})

    aggregated_stat_fid_df = aggregate_and_make_pretty(
        performance_df=statistical_fidelity_df.query(
            "data_centric_method == 'cleanlab'"
        ),
        by_cols=[
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col="inv_kl_divergence",
        normalize=False,
        normalize_index_col="",
        digits=3,
    )
    aggregated_stat_fid_df = (
        aggregated_stat_fid_df.reset_index()
        .query(
            "(postprocessing_strategy == 'baseline' & preprocessing_strategy == 'baseline') | (preprocessing_strategy == 'org_data')"
        )
        .drop(columns=["postprocessing_strategy", "preprocessing_strategy"])
    )
    aggregated_stat_fid_df = aggregated_stat_fid_df.rename(
        {"synthetic_model_type": "Generative Model"}, axis=1
    )
    save_path = RESULTS_DIR / "main_experiment" / "tmp" / "aggregated_stat_fid.csv"
    # mkdir if not exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated_stat_fid_df.to_csv(
        save_path,
        index=False,
    )

    ### NORMALIZED PERFORMANCE
    normalized_stat_fid_df = aggregate_and_make_pretty(
        performance_df=statistical_fidelity_df.query(
            "data_centric_method == 'cleanlab'"
        ),
        by_cols=[
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col="inv_kl_divergence",
        normalize=True,
        normalize_index_col="synthetic_model_type",
        digits=3,
    )
    normalized_stat_fid_df = (
        normalized_stat_fid_df.reset_index()
        .query(
            "(postprocessing_strategy != 'baseline' | preprocessing_strategy != 'baseline') & preprocessing_strategy != 'org_data'"
        )
        .set_index(
            [
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ]
        )
    )
    normalized_stat_fid_df.index = normalized_stat_fid_df.index.rename(
        {
            "synthetic_model_type": "Generative Model",
            "preprocessing_strategy": "Preprocessing Strategy",
            "postprocessing_strategy": "Postprocessing Strategy",
        }
    )
    normalized_stat_fid_df.to_csv(
        RESULTS_DIR / "main_experiment" / "tmp" / "normalized_stat_fid.csv"
    )
