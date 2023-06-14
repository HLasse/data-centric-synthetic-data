"""Code to generate the plots for performance by dataset shown in Appendix C"""
from typing import List
import pandas as pd

import plotnine as pn

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.summary_table import (
    aggregate_performance_across_tasks_no_merging,
    get_performance_dfs,
    rename_easy_ambi_to_no_hard,
    rename_easy_ambiguous_hard_to_easy_ambi_hard,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)


def plot_line_facet(df: pd.DataFrame) -> pn.ggplot:
    return (
        pn.ggplot(
            df,
            pn.aes(
                x="processing",
                y="mean",
                color="synthetic_model_type",
                group="synthetic_model_type",
            ),
        )
        + pn.geom_point()
        + pn.geom_line()
        + pn.facet_wrap("~ dataset_id")
        + pn.theme_bw()
    )


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    #dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "appendix"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)

    performance_dfs = get_performance_dfs(
        data_centric_methods, synthetic_models, experiment_suite
    )
    performances_dfs = rename_easy_ambi_to_no_hard(performance_dfs)
    performance_dfs = rename_easy_ambiguous_hard_to_easy_ambi_hard(performance_dfs)

    combined = aggregate_performance_across_tasks_no_merging(
        classification_performance_df=performance_dfs.classification.query(
            "data_centric_method == 'cleanlab' & metric == 'roc_auc' & classification_model_type == 'XGBClassifier'"
        ),
        model_selection_df=performance_dfs.model_selection.query(
            "data_centric_method == 'cleanlab'"
        ),
        feature_selection_df=performance_dfs.feature_selection.query(
            "data_centric_method == 'cleanlab'"
        ),
        by_cols=[
            "dataset_id",
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
    ).reset_index()
    combined["processing"] = (
        combined["preprocessing_strategy"] + "-" + combined["postprocessing_strategy"]
    )

    (
        plot_line_facet(
            combined.query("tasks == 'Classification' & synthetic_model_type != 'None'")
        )
        + pn.labs(x="Preprocessing-Postprocessing", y="AUROC", title="Classification")
        + pn.theme(
            figure_size=(8, 7),
            axis_text_x=pn.element_text(rotation=90),
            legend_title=pn.element_blank(),
        )
    ).save(plot_save_dir / "classification_dataset.png", dpi=300)

    (
        plot_line_facet(combined.query("tasks == 'Model Selection'"))
        + pn.labs(
            x="Preprocessing-Postprocessing",
            y="Spearman's Rank Correlation",
            title="Model Selection",
        )
        + pn.theme(
            figure_size=(8, 7),
            axis_text_x=pn.element_text(rotation=90),
            legend_title=pn.element_blank(),
        )
    ).save(plot_save_dir / "model_selection_dataset.png", dpi=300)

    (
        plot_line_facet(combined.query("tasks == 'Feature Selection'"))
        + pn.labs(
            x="Preprocessing-Postprocessing",
            y="Spearman's Rank Correlation",
            title="Feature Selection",
        )
        + pn.theme(
            figure_size=(8, 7),
            axis_text_x=pn.element_text(rotation=90),
            legend_title=pn.element_blank(),
        )
    ).save(plot_save_dir / "feature_selection_dataset.png", dpi=300)
