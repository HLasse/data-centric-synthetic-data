"""Plots and evaluation for the data-centric thresholds experiment."""


from pathlib import Path
from typing import Dict, List

import pandas as pd
import patchworklib as pw
import seaborn as sns

from application.constants import RESULTS_DIR
from application.data_centric_thresholds.run_data_centric_thresholds_exp import (
    DATASETS_DIR,
)
from data_centric_synth.data_models.experiment2 import (
    Experiment2Group,
    Experiment2Suite,
)
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS
from data_centric_synth.serialization.serialization import load_from_pickle


def get_experiments_for_datacentric_method(
    experiment_suite: Experiment2Suite,
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    max_proportion_flipped: float,
) -> Experiment2Suite:
    """Get the experiments for a particular data-centric method

    Args:
        experiment_suite (Experiment2Suite): The experiment suite to filter
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS): The data-centric
            method to filter by
        max_proportion_flipped (float): The maximum proportion of labels that were
            flipped

    Returns:
        Experiment2Suite: The filtered experiment suite
    """
    experiment_groups = []
    for experiment_group in experiment_suite.experiment_groups:
        if experiment_group.proportion_flipped > max_proportion_flipped:
            continue
        experiments = [
            experiment
            for experiment in experiment_group.experiments
            if experiment.data_centric_method == data_centric_method
        ]
        experiment_groups.append(
            Experiment2Group(
                experiments=experiments,
                proportion_flipped=experiment_group.proportion_flipped,
            ),
        )
    return Experiment2Suite(
        experiment_groups=experiment_groups,
        n_features=experiment_suite.n_features,
        n_samples=experiment_suite.n_samples,
        corr_range=experiment_suite.corr_range,
    )


def get_performance_from_experiment_group(
    experiment_group: Experiment2Group,
) -> pd.DataFrame:
    """Extract the recall and precision by threshold value and type from each
    experiment in an experiment group"""

    experiment_group_values = {
        "threshold_type": [],
        "threshold": [],
        "recall": [],
        "precision": [],
        "f1": [],
    }

    for experiment in experiment_group.experiments:
        if experiment.percentile_threshold is not None:
            experiment_group_values["threshold_type"].append("percentile")
            experiment_group_values["threshold"].append(experiment.percentile_threshold)
        else:
            experiment_group_values["threshold_type"].append("cutoff")
            experiment_group_values["threshold"].append(
                experiment.data_centric_threshold,
            )
        experiment_group_values["recall"].append(experiment.indices.recall)
        experiment_group_values["precision"].append(experiment.indices.precision)
        experiment_group_values["f1"].append(experiment.indices.f1)
    exp_group_df = pd.DataFrame(experiment_group_values)
    exp_group_df["proportion_flipped"] = experiment_group.proportion_flipped
    return exp_group_df


def get_dataframe_for_plotting(
    experiment_suite: Experiment2Suite,
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    max_proportion_flipped: float,
) -> pd.DataFrame:
    """Extract performance metrics from each experiment groups and return as
    a dataframe in long format ready for plotting"""
    experiments = get_experiments_for_datacentric_method(
        experiment_suite=experiment_suite,
        data_centric_method=data_centric_method,  # type: ignore
        max_proportion_flipped=max_proportion_flipped,
    )
    plot_dfs: List[pd.DataFrame] = []
    for experiment_group in experiments.experiment_groups:
        exp_group_df = get_performance_from_experiment_group(experiment_group)
        plot_dfs.append(exp_group_df)

    df = pd.concat(plot_dfs).melt(
        id_vars=["proportion_flipped", "threshold_type", "threshold"],
    )
    # default threshold for cleanlab is None which turns into NaN. Turning
    # into 0 for plotting
    df["threshold"] = df["threshold"].fillna(0)
    return df


def plot_catplot(
    df: pd.DataFrame,
    method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    save_dir: Path,
) -> pw.Brick:
    p = sns.catplot(
        data=df,
        x="threshold",
        y="value",
        hue="variable",
        row="threshold_type",
        col="proportion_flipped",
        kind="bar",
        sharex=False,
    )
    p.set(ylim=(0, 1))
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle(f"{method.title()} Thresholds")
    return pw.load_seaborngrid(p, figsize=(3, 6))
    # plt.savefig(save_dir / f"{method}_thresholds.png")


if __name__ == "__main__":
    MAX_PROPORTION_FLIPPED = 0.1
    PLOT_DIR = RESULTS_DIR / "data_centric_thresholds" / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir = DATASETS_DIR

    for experiment_d in experiment_dir.iterdir():
        experiment_path = experiment_d / "results" / "experiment2.pkl"
        experiments = load_from_pickle(experiment_path)

        method_dfs: Dict[str, pd.DataFrame] = {}
        for method in ["dataiq", "datamaps", "cleanlab"]:
            df = get_dataframe_for_plotting(
                experiment_suite=experiments,
                data_centric_method=method,  # type: ignore
                max_proportion_flipped=MAX_PROPORTION_FLIPPED,
            )
            method_dfs[method] = df
        ps = []
        for method, df in method_dfs.items():
            ps.append(plot_catplot(df=df, method=method, save_dir=experiment_d / "results"))  # type: ignore
