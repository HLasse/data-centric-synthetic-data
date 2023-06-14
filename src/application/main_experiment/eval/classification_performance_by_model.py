"""Script to evaluate the classification performance of model trained on synthetic data
by the type of classification model. Shown in Appendix C"""
from typing import List

import plotnine as pn
from wasabi import msg

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.summary_table import (
    aggregate_and_make_pretty,
    aggregate_ranking,
)
from data_centric_synth.evaluation.classification_performance import (
    get_classification_performance_df,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
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

    # dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "appendix"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)
    performance_df = get_classification_performance_df(
        experiment_suite=experiment_suite,
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
    )
    performance_df["postprocessing_strategy"] = performance_df[
        "postprocessing_strategy"
    ].replace({"easy_ambi": "no_hard"})
    performance_df["preprocessing_strategy"] = performance_df[
        "preprocessing_strategy"
    ].replace({"easy_ambiguous_hard": "easy_ambi_hard"})

    agg_df_pretty = aggregate_and_make_pretty(
        performance_df=performance_df.query(
            "data_centric_method == 'cleanlab' & metric == 'roc_auc'"
        ),
        by_cols=[
            "preprocessing_strategy",
            "postprocessing_strategy",
            "classification_model_type",
        ],
        value_col="value",
        normalize=False,
        normalize_index_col="",
        digits=2,
    )
    msg.info(
        "Classification performance by supervised model and pre/post-processing strategy"
    )
    print(agg_df_pretty.to_latex())

    # plot
    agg_df = aggregate_ranking(
        performance_df=performance_df.query(
            "data_centric_method == 'cleanlab' & metric == 'roc_auc'"
        ),
        by_cols=[
            "synthetic_model_type",
            "classification_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col="value",
    )
    agg_df = agg_df.reset_index()
    agg_df["processing"] = (
        agg_df["preprocessing_strategy"] + "-" + agg_df["postprocessing_strategy"]
    )

    (
        pn.ggplot(
            agg_df.query("preprocessing_strategy != 'org_data'"),
            pn.aes(
                x="processing",
                y="mean",
                color="classification_model_type",
                group="classification_model_type",
            ),
        )
        + pn.geom_point()
        + pn.geom_line()
        + pn.labs(
            x="Preprocessing-postprocessing",
            y="AUROC",
            color="Classification model",
        )
        + pn.facet_wrap("~synthetic_model_type")
        + pn.theme_minimal()
        + pn.theme(
            legend_position="top",
            figure_size=(6, 4),
            legend_title=pn.element_blank(),
            axis_text_x=pn.element_text(angle=90, hjust=1),
        )
    ).save(
        plot_save_dir / "classification_performance_by_model.png",
        dpi=300,
    )
