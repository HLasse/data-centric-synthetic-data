from typing import List

import pandas as pd
import plotnine as pn

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.distributions_significance.main_distributions import (
    fit_lm,
    fit_lm_and_get_processing_estimates,
    plot_density,
)
from application.main_experiment.eval.summary_table import (
    aggregate_performance_across_tasks,
    aggregate_performance_across_tasks_no_merging,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.evaluation.summary_helpers import get_performance_dfs
from data_centric_synth.experiments.models import get_image_generative_model_suite

if __name__ == "__main__":
    data_centric_methods = ["cleanlab"]
    generative_models = [*get_image_generative_model_suite(), "None"]

    IMAGE_DATA_DIR = DATA_DIR / "image_experiment" / "breast_mnist"
    PLOT_SAVE_DIR = RESULTS_DIR / "image_experiment" / "plots"  
    experiment_suite =  get_experiment3_suite(IMAGE_DATA_DIR)

    performance_dfs = get_performance_dfs(data_centric_methods=data_centric_methods, synthetic_models=generative_models, experiment_suite=experiment_suite)
    
    
    combined_unnormalized = aggregate_performance_across_tasks(
        classification_performance_df=performance_dfs.classification.query(
            "classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
        ),
        model_selection_df=performance_dfs.model_selection,
        feature_selection_df=performance_dfs.feature_selection,
        by_cols=[
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        normalize=False,
        normalize_index_col="",
    )
    combined_unnormalized

    no_agg = (
        aggregate_performance_across_tasks_no_merging(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == 'cleanlab' & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == 'cleanlab'"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == 'cleanlab'"
            ),
            by_cols=[
                "synthetic_model_type",
                "random_seed",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
        )
        .reset_index()
        .query("synthetic_model_type != 'None'")
    )

    no_agg["pre_post"] = (
        no_agg["preprocessing_strategy"]
        + " - "
        + no_agg["postprocessing_strategy"]
    )

    processing_estimates_dfs: List[pd.DataFrame] = []
    for task in no_agg["tasks"].unique():
        x_str = "AUROC" if task == "Classification" else "Spearman's Rank Correlation"
        tmp_df = no_agg.query("tasks == @task")
        # get estimates for the processing strategies
        processing_estimates = fit_lm_and_get_processing_estimates(
            df=tmp_df,
            formula="mean ~ synthetic_model_type + preprocessing_strategy + postprocessing_strategy",
        )
        processing_estimates["task"] = task
        processing_estimates_dfs.append(processing_estimates)

        mdl = fit_lm(
            df=tmp_df,
            formula="mean ~ synthetic_model_type + preprocessing_strategy + postprocessing_strategy",
        )

        print(f"***************{task}***************")
        print(mdl.summary())
        print(f"As latex")
        print(mdl.summary().as_latex())

    processing_estimates_df = pd.concat(processing_estimates_dfs)
    processing_estimates_df["significant"] = (
        processing_estimates_df["p_value"] < 0.05
    ).astype(int)

    processing_estimates_df = processing_estimates_df.rename(
        {
            "task": "Task",
            "index": "Processing",
            "estimate": "Estimate",
            "std_err": "Std. Error",
            "p_value": "p-value",
            "significant": "Significant",
        },
        axis=1,
    )
    print(
        processing_estimates_df.set_index(["Task", "Processing"]).to_latex(
            float_format="%.3f"
        )
    )
