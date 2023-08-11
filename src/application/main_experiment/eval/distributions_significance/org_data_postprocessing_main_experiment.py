from typing import List

import pandas as pd

from application.constants import DATA_DIR
from application.main_experiment.eval.summary_table import (
    aggregate_performance_across_tasks,
    aggregated_table_to_latex,
    rename_easy_ambi_to_no_hard,
    rename_easy_ambiguous_hard_to_easy_ambi_hard,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.evaluation.postprocessing_real_data_only import (
    add_postprocessing_to_performance_dfs,
    load_postprocessing_performance_dfs,
)
from data_centric_synth.evaluation.summary_helpers import get_performance_dfs
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    # dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"

    experiment_suite = get_experiment3_suite(dataset_dirs)

    performance_dfs = get_performance_dfs(
        data_centric_methods, synthetic_models, experiment_suite
    )
    performances_dfs = rename_easy_ambi_to_no_hard(performance_dfs)
    performance_dfs = rename_easy_ambiguous_hard_to_easy_ambi_hard(performance_dfs)
    # add experiments with postprocessing of original data
    postprocessing_dfs = load_postprocessing_performance_dfs()
    performance_dfs = add_postprocessing_to_performance_dfs(
        performance_dfs=performance_dfs,
        postprocessing_dfs=postprocessing_dfs,
    )

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
    ).reset_index()

    combined_unnormalized["postprocessing_strategy"] = pd.Categorical(
        combined_unnormalized["postprocessing_strategy"],
        categories=[
            "org_data",
            "baseline",
            "no_hard",
        ],
    )
    combined_unnormalized = combined_unnormalized.sort_values(
        ["synthetic_model_type", "preprocessing_strategy", "postprocessing_strategy"]
    ).set_index(
        ["synthetic_model_type", "preprocessing_strategy", "postprocessing_strategy"]
    )

    aggregated_table_to_latex(combined_unnormalized[
        ["Classification", "Feature Selection", "Model Selection"]
    ])
