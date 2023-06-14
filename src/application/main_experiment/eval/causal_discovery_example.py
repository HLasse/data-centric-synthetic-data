"""Script to evaluate an example of using the framework for causal discovery. Shown in Appendix C"""

from typing import Dict, List, Union

import pandas as pd
import plotnine as pn
from wasabi import msg

from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.summary_table import aggregate_ranking
from data_centric_synth.data_models.experiment3 import Experiment3, Experiment3Suite
from data_centric_synth.evaluation.extraction import (
    get_data_centric_method,
    get_experiment3_suite,
    get_experiments_with_causal_estimation,
    get_synthetic_model,
    get_unique_seeds,
    subset_by_seed,
)
from data_centric_synth.experiments.models import get_default_synthetic_model_suite

data_centric_methods = ["cleanlab", "dataiq", "datamaps"]
# synthetic_models = ["marginal_distributions", "tvae", "None", "ddpm"]
synthetic_models = [*get_default_synthetic_model_suite()]


def get_causal_estimation_performance_df(
    experiments: List[Experiment3],
) -> pd.DataFrame:
    """Get the performance of the causal estimation for each experiment."""
    causal_estimation_performances: List[Dict[str, Union[float, str]]] = [
        {
            **experiment.dag_evaluation.dict(),  # type: ignore
            "random_seed": experiment.random_seed,
            "preprocessing_strategy": experiment.preprocessing.strategy_name,
            "postprocessing_strategy": experiment.postprocessing.strategy_name,
        }
        for experiment in experiments
    ]
    return pd.DataFrame(causal_estimation_performances)


def get_causal_discovery_df(
    experiment_suite: Experiment3Suite, verbose=False
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = get_experiments_with_causal_estimation(dataset.experiments)
        dataset_performances: List[pd.DataFrame] = []
        for method in data_centric_methods:
            data_centric_subset = get_data_centric_method(
                experiments=exps,
                method=method,
            )
            method_performances: List[pd.DataFrame] = []
            for synthetic_model in synthetic_models:
                model_subset = get_synthetic_model(
                    experiments=data_centric_subset,
                    model=synthetic_model,
                )
                seed_performances: List[pd.DataFrame] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.append(
                        get_causal_estimation_performance_df(experiments=seed_subset),
                    )
                try:
                    synthetic_model_performance_df = pd.concat(seed_performances)
                    synthetic_model_performance_df[
                        "synthetic_model_type"
                    ] = synthetic_model
                    method_performances.append(synthetic_model_performance_df)
                except ValueError:
                    if verbose:
                        msg.warn(
                            f"No experiments for {dataset.dataset_id} {method} {synthetic_model}",
                        )
            method_performance_df = pd.concat(method_performances)
            method_performance_df["data_centric_method"] = method
            dataset_performances.append(method_performance_df)
        dataset_performance_df = pd.concat(dataset_performances)
        dataset_performance_df["dataset_id"] = dataset.dataset_id
        overall_performances.append(dataset_performance_df)

    return pd.concat(overall_performances).reset_index().drop(columns=["index"])


if __name__ == "__main__":
    # dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"
    plot_save_dir = RESULTS_DIR / "main_experiment" / "plots" / "appendix"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)
    # experiment_suite = get_experiment_suite_(RESULTS_DIR / "data")

    # extract experiments with
    performance_df = get_causal_discovery_df(experiment_suite)

    performance_df["postprocessing_strategy"] = performance_df[
        "postprocessing_strategy"
    ].replace({"easy_ambi": "no_hard"})
    performance_df["preprocessing_strategy"] = performance_df[
        "preprocessing_strategy"
    ].replace({"easy_ambiguous_hard": "easy_ambi_hard"})

    agg_df = aggregate_ranking(
        performance_df=performance_df.query(
            "data_centric_method == 'cleanlab' & dataset_id == '361055'"
        ),
        by_cols=[
            "dataset_id",
            "synthetic_model_type",
            "preprocessing_strategy",
            "postprocessing_strategy",
        ],
        value_col="shd",
    )
    agg_df = agg_df.reset_index()
    agg_df["processing"] = (
        agg_df["preprocessing_strategy"] + "-" + agg_df["postprocessing_strategy"]
    )

    (
        pn.ggplot(
            agg_df,
            pn.aes(
                x="processing",
                y="mean",
                color="synthetic_model_type",
                fill="synthetic_model_type",
                group="synthetic_model_type",
            ),
        )
        + pn.geom_point(size=3)
        + pn.geom_line()
        + pn.theme_minimal()
        + pn.labs(x="Preprocessing-Postprocessing", y="SHD")
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45, hjust=1),
            legend_position="top",
            legend_title=pn.element_blank(),
            figure_size=(4, 3),
        )
    ).save(plot_save_dir / "cleanlab_causal_discovery_performance.png", dpi=300)
