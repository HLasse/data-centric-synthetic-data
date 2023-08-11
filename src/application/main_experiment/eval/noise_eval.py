"""Code to produce the plots for the experiment of adding label noise to the Covid mortality dataset"""
from typing import List

import numpy as np
import pandas as pd
import plotnine as pn
from application.constants import DATA_DIR, RESULTS_DIR
from data_centric_synth.evaluation.data_objects import NoisePerformanceDfs
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)
from plotnine.stats.stat_summary import mean_cl_boot

from data_centric_synth.evaluation.summary_helpers import get_noise_performance_dfs

pd.set_option("mode.chained_assignment", None)


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
    statistical_fidelity_df: pd.DataFrame,
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

    statistical_fidelity = aggregate_and_make_pretty(
        performance_df=statistical_fidelity_df,
        by_cols=by_cols,
        value_col="inv_kl_divergence",
        normalize=normalize,
        normalize_index_col=normalize_index_col,
        digits=digits,
    ).rename(columns={"mean (lower, upper)": "Statistical Fidelity"})

    return (
        classification.join(model_selection)
        .join(feature_selection)
        .join(statistical_fidelity)
    )


def aggregate_performance_across_tasks_no_merging(
    classification_performance_df: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    feature_selection_df: pd.DataFrame,
    statistical_fidelity_df: pd.DataFrame,
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

    statistical_fidelity = aggregate_ranking(
        performance_df=statistical_fidelity_df,
        by_cols=by_cols,
        value_col="inv_kl_divergence",
    ).assign(tasks="Statistical Fidelity")

    return pd.concat(
        [classification, model_selection, feature_selection, statistical_fidelity]
    )


def find_dataset_with_max_classification_diff(
    performance_dfs: NoisePerformanceDfs,
) -> str:
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


def rename_easy_ambi_to_no_hard(
    performance_dfs: NoisePerformanceDfs,
) -> NoisePerformanceDfs:
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
    performance_dfs.statistical_fidelity[
        "postprocessing_strategy"
    ] = performance_dfs.statistical_fidelity["postprocessing_strategy"].replace(
        {"easy_ambi": "no_hard"}
    )
    return performance_dfs


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "summary"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # experiment_suite = get_experiment_suite(dataset_dirs)
    # save_to_pickle(experiment_suite, RESULTS_DIR / "experiment3" / "experiment_suite.pkl")

    ## Performance on noisy data
    # NOISY_DATASETS_DIR = RESULTS_DIR / "experiment3" / "noise_data"
    NOISY_DATASETS_DIR = DATA_DIR / "main_experiment" / "noise_data"
    noise_experiment_suite = get_experiment3_suite(NOISY_DATASETS_DIR)

    noise_performance_dfs = get_noise_performance_dfs(
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
        experiment_suite=noise_experiment_suite,
    )
    noise_performance_dfs = rename_easy_ambi_to_no_hard(noise_performance_dfs)

    noise_combined_unnormalized = aggregate_performance_across_tasks_no_merging(
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
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
    )
    noise_combined_unnormalized.index = noise_combined_unnormalized.index.rename(
        {
            "synthetic_model_type": "Generative_Model",
            "preprocessing_strategy": "Preprocessing Strategy",
            "postprocessing_strategy": "Postprocessing Strategy",
            "dataset_id": "Prop label noise",
        }
    )
    noise_combined_unnormalized = noise_combined_unnormalized.reset_index()
    noise_combined_unnormalized["Prop label noise"] = (
        noise_combined_unnormalized["Prop label noise"]
        .astype(float)
        .multiply(100)
        .astype(int)
        .astype(str)
    )
    # make prop label noise a category so that it is ordered in the plot
    noise_combined_unnormalized["Prop label noise"] = pd.Categorical(
        noise_combined_unnormalized["Prop label noise"],
        categories=[
            "0",
            "2",
            "4",
            "6",
            "8",
            "10",
        ],
    )
    # make tasks a category so that it is ordered in the plot
    noise_combined_unnormalized["tasks"] = pd.Categorical(
        noise_combined_unnormalized["tasks"],
        categories=[
            "Classification",
            "Model Selection",
            "Feature Selection",
            "Statistical Fidelity",
        ],
    )

    noise_combined_unnormalized["processing"] = (
        noise_combined_unnormalized["Preprocessing Strategy"]
        + "-"
        + noise_combined_unnormalized["Postprocessing Strategy"]
    )

    (
        pn.ggplot(
            noise_combined_unnormalized.query("Generative_Model != 'None'"),
            pn.aes(
                x="Prop label noise",
                y="mean",
                fill="processing",
                color="processing",
                group="processing",
            ),
        )
        + pn.geom_point()
        + pn.geom_line()
        + pn.ylab("Performance")
        + pn.xlab("Percent label noise")
        + pn.facet_grid("tasks~Generative_Model", scales="free_y")
        + pn.theme_bw()
        + pn.theme(
            legend_position="bottom",
            legend_title=pn.element_blank(),
            legend_box_margin=40,
            figure_size=(8, 7),
            axis_text_x=pn.element_text(rotation=90),
        )
    ).save(plot_save_dir / "overall_noise_performance.png", dpi=300)

    # zoom in on ddpm across all tasks for clarity
    (
        pn.ggplot(
            noise_combined_unnormalized.query("Generative_Model == 'ddpm'"),
            pn.aes(
                x="Prop label noise",
                y="mean",
                fill="processing",
                color="processing",
                group="processing",
            ),
        )
        + pn.geom_line(alpha=0.5)
        + pn.geom_point(size=3)
        + pn.ylab("Performance")
        + pn.xlab("Percent label noise")
        + pn.facet_wrap("~tasks", scales="free_y")
        # + pn.scale_y_continuous(expand=(0.01, 0.01))
        + pn.theme_minimal()
        + pn.theme(
            legend_position="bottom",
            legend_title=pn.element_blank(),
            legend_box_margin=30,
            figure_size=(10, 4),
            panel_spacing=0.7,
        )
    ).save(plot_save_dir / "ddpm_noise_performance.png", dpi=300)
