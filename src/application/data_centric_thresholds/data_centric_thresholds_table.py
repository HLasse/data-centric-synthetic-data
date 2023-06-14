from typing import List, Literal

import pandas as pd

from application.data_centric_thresholds.data_centric_thresholds_plots import (
    get_dataframe_for_plotting,
)
from application.data_centric_thresholds.run_data_centric_thresholds_exp import (
    DATASETS_DIR,
)
from data_centric_synth.data_models.experiment2 import Experiment2Suite
from data_centric_synth.serialization.serialization import load_from_pickle


def get_performance_by_metric(
    df: pd.DataFrame,
    metric: Literal["f1", "recall", "precision"],
) -> pd.DataFrame:
    df = df[df["variable"] == metric]
    df = df.groupby(["method", "threshold_type", "threshold"]).agg(
        {"value": ["std", "mean"]},
    )
    df.columns = df.columns.set_levels([metric], level=0)
    return df


def get_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    performance_tables: List[pd.DataFrame] = [
        get_performance_by_metric(df=df, metric=metric)
        for metric in df["variable"].unique()
    ]
    return pd.concat(performance_tables, axis=1)


def bold_values(df: pd.DataFrame, values_to_bold: List[float]) -> pd.DataFrame:
    """Bold the values in the dataframe that are in values_to_bold"""
    df = df.copy()
    for col in df.columns:
        for value in values_to_bold:
            df[col] = df[col].apply(lambda x: f"\\textbf{{{x}}}" if x == value else x)
    return df


def get_max_performance_by_metric(
    df: pd.DataFrame,
    metric: Literal["f1", "recall", "precision"],
) -> List[float]:
    return list(df[metric].groupby(["method"])["mean"].max().values)


def get_max_performance_values(df: pd.DataFrame) -> List[float]:
    max_values: List[float] = []
    for metric in ["recall", "precision", "f1"]:
        max_values.extend(get_max_performance_by_metric(df=df, metric=metric))
    return max_values


def std_to_parenthesis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in df.columns.get_level_values(1).unique():
        # Get the mean and std values for the column
        mean = df[("mean", metric)]
        std = df[("std", metric)]
        # Concatenate the mean and std values into a single string with parentheses
        new_values = [f"{m} ({s:.2f})" for m, s in zip(mean, std)]
        # Set the new values in the DataFrame
        df[metric] = new_values
    df = df.drop(columns=["mean", "std"])
    return df


if __name__ == "__main__":
    MAX_PROPORTION_FLIPPED = 0.1

    experiment_dir = DATASETS_DIR

    experiment_dfs: List[pd.DataFrame] = []
    for experiment_d in experiment_dir.iterdir():
        experiment_path = experiment_d / "results" / "experiment2.pkl"
        experiments: Experiment2Suite = load_from_pickle(experiment_path)

        method_dfs: List[pd.DataFrame] = []
        for method in ["dataiq", "datamaps", "cleanlab"]:
            df = get_dataframe_for_plotting(
                experiment_suite=experiments,
                data_centric_method=method,  # type: ignore
                max_proportion_flipped=MAX_PROPORTION_FLIPPED,
            )
            df["method"] = method
            df["n_features"] = experiments.n_features
            df["n_samples"] = experiments.n_samples
            method_dfs.append(df)
        experiment_dfs.append(pd.concat(method_dfs))

    df = pd.concat(experiment_dfs)
    performance_table = get_performance_table(df=df)
    max_values = get_max_performance_values(df=performance_table.round(3))

    # reorder for latex
    column_order = [
        ("mean", "f1"),
        ("mean", "recall"),
        ("mean", "precision"),
        ("std", "f1"),
        ("std", "recall"),
        ("std", "precision"),
    ]
    performance_table = performance_table.reorder_levels([1, 0], axis=1)
    performance_table = performance_table.reindex(columns=column_order)
    performance_table = bold_values(
        df=performance_table.round(3),
        values_to_bold=max_values,
    )
    performance_table = std_to_parenthesis(df=performance_table)
    print(performance_table.to_latex(escape=False, float_format="%.3f"))
