"""Code to generate the main tables and plots for the main experiment."""
from typing import List, Optional

import numpy as np
import pandas as pd
import plotnine as pn
from plotnine.stats.stat_summary import mean_cl_boot
from wasabi import msg

from application.constants import DATA_DIR, RESULTS_DIR
from data_centric_synth.evaluation.data_objects import PerformanceDfs
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.evaluation.postprocessing_real_data_only import (
    add_postprocessing_to_performance_dfs,
    load_postprocessing_aggregated_statistical_fidelity,
    load_postprocessing_performance_dfs,
)
from data_centric_synth.evaluation.summary_helpers import get_performance_dfs
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)


def mean_cl_boot_wrapper(x: np.ndarray) -> pd.DataFrame:
    """Wrapper for mean_cl_boot."""
    bootstrapped = mean_cl_boot(x)
    return pd.DataFrame(
        {
            "mean": bootstrapped["y"],
            "lower": bootstrapped["ymin"],
            "upper": bootstrapped["ymax"],
        }
    )


def clean_multiindex(multiindex_df: pd.DataFrame) -> pd.DataFrame:
    """Removes the '0' column from running an apply operation that returns
    a dataframe"""
    index = multiindex_df.index
    # drop the index column
    index = index.droplevel(None)
    multiindex_df = multiindex_df.reset_index(drop=True)
    multiindex_df.index = index
    return multiindex_df


def aggregate_ranking(
    performance_df: pd.DataFrame, by_cols: List[str], value_col: str
) -> pd.DataFrame:
    bootstrapped = performance_df.groupby(by_cols)[value_col].apply(
        mean_cl_boot_wrapper
    )
    return clean_multiindex(bootstrapped)  # type: ignore


def round_and_combine(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = df.round(digits)

    df = df.astype(str)
    # paste lower and upper into the mean column
    df["mean"] = df["mean"] + " (" + df["lower"] + ", " + df["upper"] + ")"
    df = df.drop(columns=["lower", "upper"])
    df = df.rename(columns={"mean": "mean (lower, upper)"})
    return df


def find_max_values_by_index(df: pd.DataFrame) -> pd.DataFrame:
    """Find the max values by level 0 the index. Return the whole multiindex"""
    return df.groupby(level=0).max()


def normalize_to_baseline(
    aggregated_df: pd.DataFrame, normalize_index_col: str, by_cols: List[str]
) -> pd.DataFrame:
    normalized_dfs: List[pd.DataFrame] = []
    for name, group in aggregated_df.reset_index().groupby(normalize_index_col):
        if name == "None":
            normalized_dfs.append(group)
            continue
        baseline = group.query(
            "preprocessing_strategy == 'baseline' & postprocessing_strategy == 'baseline'"
        )[["mean", "lower", "upper"]]
        normalized_df: pd.DataFrame = group.copy()
        for row in group.itertuples():
            # get percent difference between mean, lower, and upper
            diff = (
                (row.mean - baseline["mean"].values[0])
                / baseline["mean"].values[0]
                * 100
            )
            lower = (
                (row.lower - baseline["mean"].values[0])
                / baseline["mean"].values[0]
                * 100
            )
            upper = (
                (row.upper - baseline["mean"].values[0])
                / baseline["mean"].values[0]
                * 100
            )
            # add to dataframe
            normalized_df.loc[row.Index, "mean"] = diff
            normalized_df.loc[row.Index, "lower"] = lower
            normalized_df.loc[row.Index, "upper"] = upper
        normalized_dfs.append(normalized_df)
    normalized = pd.concat(normalized_dfs)
    multi_index = aggregated_df.index
    normalized = normalized.drop(columns=by_cols)
    normalized.index = multi_index
    return normalized


def aggregate_and_make_pretty(
    performance_df: pd.DataFrame,
    by_cols: List[str],
    value_col: str,
    normalize: bool,
    normalize_index_col: str,
    digits: int,
) -> pd.DataFrame:
    """Aggregate the performance_df and make it pretty for latex"""
    aggregated = aggregate_ranking(
        performance_df=performance_df, by_cols=by_cols, value_col=value_col
    )
    if normalize:
        aggregated = normalize_to_baseline(
            aggregated_df=aggregated,
            normalize_index_col=normalize_index_col,
            by_cols=by_cols,
        )

    return round_and_combine(aggregated, digits=digits)


def aggregate_performance_across_tasks(
    classification_performance_df: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    feature_selection_df: pd.DataFrame,
    normalize: bool,
    normalize_index_col: str,
    by_cols: List[str],
    digits: int = 3,
) -> pd.DataFrame:
    classification = aggregate_and_make_pretty(
        performance_df=classification_performance_df,
        by_cols=by_cols,
        value_col="value",
        normalize=normalize,
        normalize_index_col=normalize_index_col,
        digits=digits,
    ).rename(columns={"mean (lower, upper)": "Classification"})

    model_selection = aggregate_and_make_pretty(
        performance_df=model_selection_df,
        by_cols=by_cols,
        value_col="rank_distance",
        normalize=normalize,
        normalize_index_col=normalize_index_col,
        digits=digits,
    ).rename(columns={"mean (lower, upper)": "Model Selection"})

    feature_selection = aggregate_and_make_pretty(
        performance_df=feature_selection_df,
        by_cols=by_cols,
        value_col="rank_distance",
        normalize=normalize,
        normalize_index_col=normalize_index_col,
        digits=digits,
    ).rename(columns={"mean (lower, upper)": "Feature Selection"})

    return classification.join(model_selection).join(feature_selection)


def aggregate_performance_across_tasks_no_merging(
    classification_performance_df: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    feature_selection_df: pd.DataFrame,
    by_cols: List[str],
) -> pd.DataFrame:
    classification = aggregate_ranking(
        performance_df=classification_performance_df,
        by_cols=by_cols,
        value_col="value",
    ).assign(tasks="Classification")

    model_selection = aggregate_ranking(
        performance_df=model_selection_df,
        by_cols=by_cols,
        value_col="rank_distance",
    ).assign(tasks="Model Selection")

    feature_selection = aggregate_ranking(
        performance_df=feature_selection_df,
        by_cols=by_cols,
        value_col="rank_distance",
    ).assign(tasks="Feature Selection")

    return pd.concat([classification, model_selection, feature_selection])


def find_dataset_with_max_classification_diff(performance_dfs: PerformanceDfs) -> str:
    dataset_performance = aggregate_ranking(
        performance_df=performance_dfs.classification.query(
            "data_centric_method == 'cleanlab' & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
        ),
        by_cols=[
            "dataset_id",
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col="value",
    ).reset_index()

    diffs = (  # type: ignore
        dataset_performance.query(
            "preprocessing_strategy == 'easy_hard' & postprocessing_strategy == 'easy_ambi'"
        )["mean"].values
        - dataset_performance.query(
            "preprocessing_strategy == 'baseline' & postprocessing_strategy == 'baseline'"
        )["mean"].values
    )

    diffs_df = (
        dataset_performance.query(
            "preprocessing_strategy == 'easy_hard' & postprocessing_strategy == 'easy_ambi'"
        )[
            [
                "dataset_id",
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ]
        ]
        .assign(diff=diffs)
        .sort_values("diff", ascending=False)
    )

    dataset_with_max_diff = diffs_df.iloc[0]["dataset_id"]
    return dataset_with_max_diff


def aggregated_table_to_latex(aggregated_table: pd.DataFrame) -> None:
    """Convert aggregated table to latex with a midrule after each synthetic model type
    and vpace to separate preprocessing conditions"""
    latex_str = aggregated_table.to_latex()
    add_midrule_before_occurences = [*get_default_synthetic_model_suite(), "None"]
    # add midrule before each occurence of synthetic model type
    for synthetic_model_type in add_midrule_before_occurences:
        latex_str = latex_str.replace(
            synthetic_model_type, "\\midrule\n " + synthetic_model_type
        )
    # add a vspace(3pt) after each line containing a synthetic model type
    latex_str = latex_str.replace(
        "\n     &           & easy\\_ambi",
        "\n\\vspace{3pt}\n     &           & easy\\_ambi",
    )

    print(latex_str)


def rename_easy_ambi_to_no_hard(performance_dfs: PerformanceDfs) -> PerformanceDfs:
    performance_dfs.classification[
        "postprocessing_strategy"
    ] = performance_dfs.classification["postprocessing_strategy"].replace(
        {"easy_ambi": "no_hard"}
    )
    performance_dfs.model_selection[
        "postprocessing_strategy"
    ] = performance_dfs.model_selection["postprocessing_strategy"].replace(
        {"easy_ambi": "no_hard"}
    )
    performance_dfs.feature_selection[
        "postprocessing_strategy"
    ] = performance_dfs.feature_selection["postprocessing_strategy"].replace(
        {"easy_ambi": "no_hard"}
    )
    return performance_dfs


def rename_easy_ambiguous_hard_to_easy_ambi_hard(
    performance_dfs: PerformanceDfs,
) -> PerformanceDfs:
    performance_dfs.classification[
        "preprocessing_strategy"
    ] = performance_dfs.classification["preprocessing_strategy"].replace(
        {"easy_ambiguous_hard": "easy_ambi_hard"}
    )
    performance_dfs.model_selection[
        "preprocessing_strategy"
    ] = performance_dfs.model_selection["preprocessing_strategy"].replace(
        {"easy_ambiguous_hard": "easy_ambi_hard"}
    )
    performance_dfs.feature_selection[
        "preprocessing_strategy"
    ] = performance_dfs.feature_selection["preprocessing_strategy"].replace(
        {"easy_ambiguous_hard": "easy_ambi_hard"}
    )
    return performance_dfs


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    # dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "summary"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)

    performance_dfs = get_performance_dfs(
        data_centric_methods, synthetic_models, experiment_suite
    )
    performances_dfs = rename_easy_ambi_to_no_hard(performance_dfs)
    performance_dfs = rename_easy_ambiguous_hard_to_easy_ambi_hard(performance_dfs)
    

    for data_centric_method in data_centric_methods:
        msg.divider(f"Data Centric Method: {data_centric_method}")
        ## Table 1: Performance across all datasets for baseline condition

        combined_unnormalized = aggregate_performance_across_tasks(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == @data_centric_method & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            by_cols=[
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            normalize=False,
            normalize_index_col="",
        )
        # subset to only the baseline and org_data conditions
        combined_unnormalized = (
            combined_unnormalized.reset_index()
            .query(
                "(postprocessing_strategy == 'baseline' & preprocessing_strategy == 'baseline') | (preprocessing_strategy == 'org_data')"
            )
            .drop(columns=["postprocessing_strategy", "preprocessing_strategy"])
        )
        combined_unnormalized = combined_unnormalized.rename(
            {"synthetic_model_type": "Generative Model"}, axis=1
        ).reset_index(drop=True)

        if data_centric_method == "cleanlab":
            stat_fid = pd.read_csv(
                RESULTS_DIR / "main_experiment" / "tmp" / "aggregated_stat_fid.csv"
            ).rename({"mean (lower, upper)": "Statistical fidelity"}, axis=1)
            postprocessing_stat_fid_df = (
                load_postprocessing_aggregated_statistical_fidelity(
                    metric="inv_kl_divergence"
                )
            )

            combined_unnormalized = combined_unnormalized.merge(
                stat_fid, how="left", on="Generative Model"
            )
        msg.info("Unnormalized performance -- baseline and org_data conditions")
        print(combined_unnormalized.to_latex(index=False))

        ## Table 2: Normalized performance across conditions
        combined_normalized = aggregate_performance_across_tasks(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == @data_centric_method & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            by_cols=[
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            normalize=True,
            normalize_index_col="synthetic_model_type",
            digits=2,
        )
        # subset to not include the baseline and org_data conditions
        combined_normalized = (
            combined_normalized.reset_index()
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
        combined_normalized.index = combined_normalized.index.rename(
            {
                "synthetic_model_type": "Generative Model",
                "preprocessing_strategy": "Preprocessing Strategy",
                "postprocessing_strategy": "Postprocessing Strategy",
            }
        )
        if data_centric_method == "cleanlab":
            stat_fid = (
                pd.read_csv(
                    RESULTS_DIR / "main_experiment" / "tmp" / "normalized_stat_fid.csv"
                )
                .rename({"mean (lower, upper)": "Statistical fidelity"}, axis=1)
                .set_index(
                    [
                        "Generative Model",
                        "Preprocessing Strategy",
                        "Postprocessing Strategy",
                    ]
                )
            )
            combined_normalized = combined_normalized.join(stat_fid)
        msg.info("Normalized performance -- all conditions")
        aggregated_table_to_latex(combined_normalized)

        # Figure 3
        ## Figure X: performance by pre/post processing strategy for each model averaged over datasets
        perf = aggregate_performance_across_tasks_no_merging(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == @data_centric_method & classification_model_type == 'XGBClassifier' & metric == 'roc_auc' & synthetic_model_type != 'None'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == @data_centric_method & synthetic_model_type != 'None'"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == @data_centric_method & synthetic_model_type != 'None'"
            ),
            by_cols=[
                "synthetic_model_type",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
        )
        perf.index = perf.index.rename(
            {
                "synthetic_model_type": "Generative Model",
                "preprocessing_strategy": "Preprocessing Strategy",
                "postprocessing_strategy": "Postprocessing Strategy",
            }
        )
        perf = perf.reset_index()
        perf["processing"] = (
            perf["Preprocessing Strategy"] + "-" + perf["Postprocessing Strategy"]
        )

        # make tasks categorical with order Classification, Model Selection, Feature Selection
        perf["tasks"] = pd.Categorical(perf["tasks"], categories=perf["tasks"].unique())

        (
            pn.ggplot(
                perf,
                pn.aes(
                    x="processing",
                    y="mean",
                    color="Generative Model",
                    group="Generative Model",
                ),
            )
            + pn.geom_line()
            + pn.geom_point(size=3)
            + pn.facet_wrap("~tasks", scales="free_y")
            + pn.scale_x_discrete(expand=(0.05, 0.05))
            + pn.theme_minimal()
            + pn.theme(
                legend_position="bottom",
                legend_title=pn.element_blank(),
                # axis_title_x=pn.element_blank(),
                legend_box_margin=120,
                figure_size=(10, 3),
                axis_text_x=pn.element_text(angle=45),
                panel_spacing=0.5,
            )
            + pn.scale_color_brewer(type="qual", palette="Set2")  # type: ignore
            + pn.labs(y="Performance", x="Preprocessing-Postprocessing")
        ).save(
            plot_save_dir
            / f"{data_centric_method}_pre_post_processing_performance.png",
            dpi=300,
        )
