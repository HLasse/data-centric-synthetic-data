import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_centric_synth.data_models.experiment3 import (
    Experiment3Group,
    Experiment3Suite,
)


def load_all_experiments_from_folder(folder: Path) -> Experiment3Suite:
    experiment_groups: List[Experiment3Group] = []
    for path in folder.glob("*.pkl"):
        with path.open("rb") as f:
            experiment_groups.append(pkl.load(f))
    return Experiment3Suite(experiment_groups=experiment_groups)


def melt_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the dataframe so that the preprocessing and postprocessing are in
    the same column."""
    plot_df = df.reset_index().melt(
        id_vars=["task_id", "preprocessing"],
        var_name="postprocessing",
    )
    # merge preprocessing and postprocessing to get a single column
    plot_df["pre+post"] = plot_df["preprocessing"].str.cat(
        plot_df["postprocessing"],
        sep=" + ",
    )
    return plot_df


def plot_and_save_lineplot(
    data: pd.DataFrame,
    save_path: Path,
    hue: str,
    normalized: bool,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    norm = "Normalized" if normalized else "Unnormalized"
    sns.lineplot(
        data=data,
        x="task_id",
        y="value",
        hue=hue,
        marker="o",
    ).set(title=f"{norm} performance by {title}", xlabel=xlabel, ylabel=ylabel)
    plt.savefig(save_path)
    plt.close()


def get_baselined_plotting_dfs(
    norm_plot_df: pd.DataFrame,
    unnorm_plot_df: pd.DataFrame,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    norm_preproc_baseline = norm_plot_df[norm_plot_df["postprocessing"] == "baseline"]
    norm_postproc_baseline = norm_plot_df[norm_plot_df["preprocessing"] == "baseline"]

    unnorm_preproc_baseline = unnorm_plot_df[
        unnorm_plot_df["postprocessing"] == "baseline"
    ]
    unnorm_postproc_baseline = unnorm_plot_df[
        unnorm_plot_df["preprocessing"] == "baseline"
    ]
    return {
        "normalized": {
            "preprocessing": norm_preproc_baseline,
            "postprocessing": norm_postproc_baseline,
        },
        "unnormalized": {
            "preprocessing": unnorm_preproc_baseline,
            "postprocessing": unnorm_postproc_baseline,
        },
    }


def plot_dataiq_proportions(
    experiments: Experiment3Suite,
    task_id_to_name: Optional[Dict[int, str]],
    path: Path,
    task_id_to_numeric: bool = False,
) -> None:
    dataiq_props = experiments.get_dataiq_proportions()
    # flatten to dataframe
    df = pd.DataFrame.from_dict(dataiq_props, orient="index").stack().to_frame()
    # to break out the lists into columns
    df = pd.DataFrame(df[0].values.tolist(), index=df.index).reset_index()
    df = df.rename(columns={"level_0": "task_id", "level_1": "processing"})
    df_counts = df.drop(
        ["proportion_easy", "proportion_ambiguous", "proportion_hard"],
        axis=1,
    )
    df_prop = df.drop(["n_easy", "n_ambiguous", "n_hard"], axis=1)

    if task_id_to_name is not None:
        df_counts["task_id"] = df_counts["task_id"].map(task_id_to_name)
        df_prop["task_id"] = df_prop["task_id"].map(task_id_to_name)
    if task_id_to_numeric:
        df_counts["task_id"] = df_counts["task_id"].astype(float).round(2)
        df_prop["task_id"] = df_prop["task_id"].astype(float).round(2)
        df_counts = df_counts.sort_values(by="task_id")
        df_prop = df_prop.sort_values(by="task_id")
    p_row = 0
    p_col = 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for df in [df_counts, df_prop]:
        for processing in ["preprocessing", "postprocessing"]:
            ax = axes[p_row, p_col]
            sub_df = df[df["processing"] == processing]
            sub_df = sub_df.set_index("task_id")
            sub_df.plot.bar(
                stacked=True,
                ax=ax,
                title=f"{processing} dataiq",
            )
            p_row += 1
        p_row = 0
        p_col = 1
    plt.savefig(path)
    plt.close()
        p_col = 1
    plt.savefig(path)
    plt.close()
