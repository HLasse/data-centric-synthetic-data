from pathlib import Path
from typing import List, Literal, Union

import pandas as pd

from application.constants import (
    DATA_DIR,
    MAIN_EXP_POSTPROCESSING_DIR,
    NOISE_DATASET_POSTPROCESSING_DIR,
    POSTPROCESSING_ONLY_SAVE_DIR,
)
from application.main_experiment.eval.noise_eval import aggregate_and_make_pretty
from data_centric_synth.data_models.experiment3 import (
    Experiment3,
    Experiment3Group,
    Experiment3Suite,
)
from data_centric_synth.evaluation.data_objects import (
    NoisePerformanceDfs,
    PerformanceDfs,
)
from data_centric_synth.evaluation.summary_helpers import (
    get_noise_performance_dfs,
    get_performance_dfs,
)
from data_centric_synth.serialization.serialization import load_from_pickle


def get_postprocessing_experiment3_suite(dataset_dirs: Path) -> Experiment3Suite:
    experiment_groups: list[Experiment3Group] = []

    for dataset_dir in dataset_dirs.iterdir():
        dataset_experiments: List[Experiment3] = []
        for seed_exp in dataset_dir.iterdir():
            dataset_experiments.extend(load_from_pickle(seed_exp))
        experiment_groups.append(
            Experiment3Group(
                dataset_id=dataset_dir.name,
                experiments=dataset_experiments,
            )
        )
    return Experiment3Suite(experiment_groups=experiment_groups)


def subset_performance_dfs_by_postprocessing(
    performance_dfs: PerformanceDfs,
) -> PerformanceDfs:
    """remove rows of the original dataset with no processing, as that is already
    contained in the other experiments"""
    return PerformanceDfs(
        classification=performance_dfs.classification.query(
            "postprocessing_strategy == 'no_hard'"
        ),
        model_selection=performance_dfs.model_selection.query(
            "postprocessing_strategy == 'no_hard'"
        ),
        feature_selection=performance_dfs.feature_selection.query(
            "postprocessing_strategy == 'no_hard'"
        ),
    )


def subset_noise_performance_dfs_by_postprocessing(
    performance_dfs: NoisePerformanceDfs,
) -> NoisePerformanceDfs:
    return NoisePerformanceDfs(
        classification=performance_dfs.classification.query(
            "postprocessing_strategy == 'no_hard'"
        ),
        model_selection=performance_dfs.model_selection.query(
            "postprocessing_strategy == 'no_hard'"
        ),
        feature_selection=performance_dfs.feature_selection.query(
            "postprocessing_strategy == 'no_hard'"
        ),
        statistical_fidelity=performance_dfs.statistical_fidelity.query(
            "postprocessing_strategy == 'no_hard'"
        ),
    )


def load_postprocessing_performance_dfs() -> PerformanceDfs:
    experiment_suite = get_postprocessing_experiment3_suite(
        dataset_dirs=MAIN_EXP_POSTPROCESSING_DIR
    )
    performance_dfs = get_performance_dfs(
        data_centric_methods=["cleanlab"],
        synthetic_models=["None"],
        experiment_suite=experiment_suite,
    )
    return subset_performance_dfs_by_postprocessing(performance_dfs=performance_dfs)


def load_postprocessing_aggregated_statistical_fidelity(
    metric="inv_kl_divergence",
) -> pd.DataFrame:
    experiment_suite = get_postprocessing_experiment3_suite(
        dataset_dirs=MAIN_EXP_POSTPROCESSING_DIR
    )
    performance_dfs = get_noise_performance_dfs(
        data_centric_methods=["cleanlab"],
        synthetic_models=["None"],
        experiment_suite=experiment_suite,
    )
    aggregated_stat_fid_df = aggregate_and_make_pretty(
        performance_df=performance_dfs.statistical_fidelity,
        by_cols=[
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col=metric,
        normalize=False,
        normalize_index_col="",
        digits=3,
    ).query("postprocessing_strategy == 'no_hard'")
    return aggregated_stat_fid_df


def load_noise_postprocessing_performance_dfs() -> NoisePerformanceDfs:
    experiment_suite = get_postprocessing_experiment3_suite(
        dataset_dirs=NOISE_DATASET_POSTPROCESSING_DIR
    )
    performance_dfs = get_noise_performance_dfs(
        data_centric_methods=["cleanlab"],
        synthetic_models=["None"],
        experiment_suite=experiment_suite,
    )
    return subset_noise_performance_dfs_by_postprocessing(
        performance_dfs=performance_dfs
    )


def add_postprocessing_to_performance_dfs(
    performance_dfs: PerformanceDfs, postprocessing_dfs: PerformanceDfs
) -> PerformanceDfs:
    return PerformanceDfs(
        classification=pd.concat(
            [performance_dfs.classification, postprocessing_dfs.classification],
            ignore_index=True,
        ),
        model_selection=pd.concat(
            [performance_dfs.model_selection, postprocessing_dfs.model_selection],
            ignore_index=True,
        ),
        feature_selection=pd.concat(
            [performance_dfs.feature_selection, postprocessing_dfs.feature_selection],
            ignore_index=True,
        ),
    )


if __name__ == "__main__":
    load_postprocessing_performance_dfs()
