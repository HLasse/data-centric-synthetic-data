from typing import List

import pandas as pd
import plotnine as pn

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.noise_eval import (
    aggregate_performance_across_tasks,
    aggregate_performance_across_tasks_no_merging,
    aggregated_table_to_latex,
    rename_easy_ambi_to_no_hard,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.evaluation.summary_helpers import get_noise_performance_dfs
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

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

    # Table 1 equivalent - performance for baseline condition across all levels
    # of label noise

    performance_by_model = (
        aggregate_performance_across_tasks(
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
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            normalize=False,
            normalize_index_col="",
        )
        .reset_index()
        .rename({"synthetic_model_type": "Generative Model"}, axis=1)
        .set_index(
            ["Generative Model", "preprocessing_strategy", "postprocessing_strategy"]
        )
    )

    # re-order to match other tables
    performance_by_model = performance_by_model[
        [
            "Classification",
            "Feature Selection",
            "Model Selection",
            "Statistical Fidelity",
        ]
    ]
    aggregated_table_to_latex(aggregated_table=performance_by_model)

    # Performance across all generative models by label noise
    performance_by_label_noise = (
        aggregate_performance_across_tasks(
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
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            normalize=False,
            normalize_index_col="",
        )
        .reset_index()
        .rename({"dataset_id": "Prop. label noise"}, axis=1)
    )

    performance_by_label_noise = performance_by_label_noise.reset_index()
    performance_by_label_noise["preprocessing_strategy"] = pd.Categorical(
        performance_by_label_noise["preprocessing_strategy"],
        categories=["org_data", "baseline", "easy_hard"],
        ordered=True,
    )
    performance_by_label_noise = performance_by_label_noise.sort_values(
        ["Prop. label noise", "preprocessing_strategy"]
    )

    performance_by_label_noise = performance_by_label_noise.set_index(
        ["Prop. label noise", "preprocessing_strategy", "postprocessing_strategy"]
    )
    # re-order to match
    performance_by_label_noise = performance_by_label_noise[
        [
            "Classification",
            "Feature Selection",
            "Model Selection",
            "Statistical Fidelity",
        ]
    ]

    aggregated_table_to_latex(aggregated_table=performance_by_label_noise)

    # Performance across all levels of datasets and generative models
    performance_by_label_noise_and_dataset = aggregate_performance_across_tasks(
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
            "synthetic_model_type",
            "dataset_id",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        normalize=False,
        normalize_index_col="",
    )
