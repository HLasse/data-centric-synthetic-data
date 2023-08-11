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

from application.main_experiment.eval.summary_table import aggregated_table_to_latex

pd.set_option("mode.chained_assignment", None)


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    dataset_dirs = DATA_DIR / "main_experiment" / "statistical_fidelity"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "summary"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_statistical_fidelity_suite(dataset_dirs)

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

    metrics = {
        "inv_kl_divergence": "Inv. KL-D",
        "wasserstein_distance": "WD",
        "mmd": "MMD",
    }
    
    aggregated_dfs: List[pd.DataFrame] = []
    for col_name, pretty_name in metrics.items():
        aggregated_stat_fid_df = aggregate_and_make_pretty(
            performance_df=statistical_fidelity_df.query(
                "data_centric_method == 'cleanlab'"
            ),
            by_cols=[
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            value_col=col_name,
            normalize=False,
            normalize_index_col="",
            digits=3,
        ).rename({"mean (lower, upper)": pretty_name}, axis=1)
        aggregated_dfs.append(aggregated_stat_fid_df)
    aggregated_df = pd.concat(aggregated_dfs, axis=1)

    aggregated_table_to_latex(aggregated_table=aggregated_df)
