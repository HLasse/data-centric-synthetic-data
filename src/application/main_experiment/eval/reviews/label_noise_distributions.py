from typing import List

import pandas as pd
import plotnine as pn
from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.noise_eval import (
    aggregate_performance_across_tasks_no_merging,
    rename_easy_ambi_to_no_hard,
)
from application.main_experiment.eval.reviews.main_distributions import (
    fit_lm,
    fit_lm_and_get_processing_estimates,
    plot_density,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

from data_centric_synth.evaluation.summary_helpers import get_noise_performance_dfs

if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    plot_save_dir = (
        RESULTS_DIR / "main_experiment" / "plots" / "appendix" / "distributions"
    )
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    ## Performance on noisy data
    NOISY_DATASETS_DIR = DATA_DIR / "main_experiment" / "noise_data"
    noise_experiment_suite = get_experiment3_suite(NOISY_DATASETS_DIR)

    noise_performance_dfs = get_noise_performance_dfs(
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
        experiment_suite=noise_experiment_suite,
    )
    noise_performance_dfs = rename_easy_ambi_to_no_hard(noise_performance_dfs)

    # no aggregation happening, as all aggregation columns are supplied in by_cols
    noise_combined_unnormalized = (
        aggregate_performance_across_tasks_no_merging(
            classification_performance_df=noise_performance_dfs.classification.query(
                "data_centric_method == 'cleanlab' & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=noise_performance_dfs.model_selection.query(
                "data_centric_method == 'cleanlab'"
            ),
            feature_selection_df=noise_performance_dfs.feature_selection.query(
                "data_centric_method == 'cleanlab'"
            ),
            statistical_fidelity_df=noise_performance_dfs.statistical_fidelity.query(
                "data_centric_method == 'cleanlab'"
            ),
            by_cols=[
                "dataset_id",
                "synthetic_model_type",
                "random_seed",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
        )
        .reset_index()
        .query("synthetic_model_type != 'None'")
    )

    noise_combined_unnormalized["dataset_id"] = (
        noise_combined_unnormalized["dataset_id"]
        .astype(float)
        .multiply(100)
        .astype(int)
        .astype(str)
    )
    # make prop label noise a category so that it is ordered in the plot
    noise_combined_unnormalized["dataset_id"] = pd.Categorical(
        noise_combined_unnormalized["dataset_id"],
        categories=[
            "0",
            "2",
            "4",
            "6",
            "8",
            "10",
        ],
    )
    noise_combined_unnormalized["pre_post"] = (
        noise_combined_unnormalized["preprocessing_strategy"]
        + " - "
        + noise_combined_unnormalized["postprocessing_strategy"]
    )

    processing_estimates_dfs: List[pd.DataFrame] = []
    for task in noise_combined_unnormalized["tasks"].unique():
        x_str = "AUROC" if task == "Classification" else "Spearman's Rank Correlation"
        tmp_df = noise_combined_unnormalized.query("tasks == @task")
        (
            plot_density(tmp_df)
            + pn.labs(
                x=x_str,
                y="Density",
                color="Preprocessing - postprocessing",
                fill="Preprocessing - postprocessing",
                title=task,
            )
        ).save(
            plot_save_dir / f"density_noise_{task}.png",
            height=7,
            width=10,
            dpi=400,
        )
        # get estimates for the processing strategies
        processing_estimates = fit_lm_and_get_processing_estimates(
            df=tmp_df,
            formula="mean ~ dataset_id + synthetic_model_type + preprocessing_strategy + postprocessing_strategy",
        )
        processing_estimates["task"] = task
        processing_estimates_dfs.append(processing_estimates)

        mdl = fit_lm(
            df=tmp_df, formula="mean ~ synthetic_model_type + dataset_id * pre_post"
        )
        print(task)
        print(mdl.summary())

    processing_estimates_df = pd.concat(processing_estimates_dfs)
    processing_estimates_df["significant"] = (
        processing_estimates_df["p_value"] < 0.05
    ).astype(int)
    processing_estimates_df = processing_estimates_df.query(
        "task != 'Statistical Fidelity'"
    )
    print(processing_estimates_df.to_latex(index=False))
