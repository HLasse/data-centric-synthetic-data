"""Script to evaluate the model selection results, i.e. how well do models trained
on synthetic data match the models selected on real data?"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import spearmanr

from application.constants import RESULTS_DIR
from data_centric_synth.data_models.experiment3 import Experiment3Suite
from data_centric_synth.evaluation.extraction import (
    ClassificationModelPerformance,
    classification_model_performances_to_df,
    get_classification_model_performance,
    get_data_centric_method,
    get_experiment3_suite,
    get_experiments_with_classification_model_evaluation,
    get_synthetic_model,
    get_unique_seeds,
    subset_by_seed,
)
from data_centric_synth.evaluation.plotting import (
    facet_pointplot_with_error_bars,
    pointplot_with_error_bars,
)
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)

CLASSIFICATION_MODEL_TO_ID_MAPPING = {
    "XGBClassifier": "1",
    "LogisticRegression": "2",
    "RandomForestClassifier": "3",
    "MLPClassifier": "4",
    "SVC": "5",
    "KNeighborsClassifier": "6",
    "GaussianNB": "7",
    "DecisionTreeClassifier": "8",
}
data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
    "cleanlab",
    "dataiq",
    "datamaps",
]
# synthetic_models = ["marginal_distributions", "tvae", "None", "ddpm"]
synthetic_models = [*get_default_synthetic_model_suite(), "None"]


pd.set_option("mode.chained_assignment", None)


def calculate_rank_distance(
    df: pd.DataFrame,
    rank_column: str,
    dataset_id_column: str,
    original_ranking: Dict[str, np.ndarray],
) -> np.ndarray:
    """Sorts the df by the value column and calculates the distance between the
    original ranking and the ranking of the id_column. The distance is calculated
    using spearmans rank correlation."""
    # duplicates are not allowed - keep only the max value (sometimes minor floating point errors lead to duplicates)
    max_df = df.groupby(["classification_model_type", rank_column], as_index=False)[
        "value"
    ].max()

    dataset_id = df[dataset_id_column].unique()
    if len(dataset_id) != 1:
        raise ValueError("Expected only one dataset_id")
    dataset_id = dataset_id[0]
    max_df = max_df.sort_values("value", ascending=False)
    return spearmanr(original_ranking[dataset_id], max_df[rank_column]).statistic  # type: ignore


def get_model_selection_performance_df(
    experiment_suite: Experiment3Suite,
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS],
    synthetic_models: List[str],
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = get_experiments_with_classification_model_evaluation(dataset.experiments)
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
                seed_performances: List[ClassificationModelPerformance] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.extend(
                        get_classification_model_performance(experiments=seed_subset),
                    )
                synthetic_model_performance_df = (
                    classification_model_performances_to_df(
                        classification_model_performances=seed_performances,
                    )
                )
                synthetic_model_performance_df["synthetic_model_type"] = synthetic_model
                method_performances.append(synthetic_model_performance_df)
            method_performance_df = pd.concat(method_performances)
            method_performance_df["data_centric_method"] = method
            dataset_performances.append(method_performance_df)
        dataset_performance_df = pd.concat(dataset_performances)
        dataset_performance_df["dataset_id"] = dataset.dataset_id
        overall_performances.append(dataset_performance_df)

    overall_performance_df = (
        pd.concat(overall_performances).reset_index().drop(columns=["index"])
    )

    # due to the implementation, the evaluation of models trained on real data and
    # evaluated on real data is repeated for each model within seed. I.e., there
    # will be 5 rows instead of 1 for each seed. They are identical, so we can
    # remove the duplicates
    # round value to 5 decimals to avoid floating point errors
    ranked_df = get_model_ranking(overall_performance_df=overall_performance_df)

    return ranked_df


def get_model_ranking(overall_performance_df: pd.DataFrame) -> pd.DataFrame:
    overall_performance_df["value"] = overall_performance_df["value"].round(5)
    overall_performance_df = overall_performance_df.drop_duplicates()
    overall_performance_df["classification_model_type_id"] = (
        overall_performance_df["classification_model_type"]
        .map(CLASSIFICATION_MODEL_TO_ID_MAPPING)
        .astype(int)
    )

    # in this experiment we only care about the performance on the full data segment
    # using auc roc as our metric
    overall_performance_df = overall_performance_df.query(
        "data_segment == 'full' & metric == 'roc_auc'",
    )

    # get model ranking of real data. Subset to only include the full data_segment
    # and evaluate based on roc auc
    real_data_model_ranking = (
        overall_performance_df.query("synthetic_model_type == 'None'")
        .groupby(["dataset_id", "classification_model_type_id"])
        .agg({"value": "mean"})
        .reset_index()
        .sort_values(["dataset_id", "value"], ascending=False)
    )
    # turn the ordering into a dict with dataset_id as key and the rank ordering
    # as numpy array
    real_data_model_ranking = (
        real_data_model_ranking.groupby("dataset_id")["classification_model_type_id"]
        .apply(lambda x: x.to_numpy())
        .to_dict()
    )

    # for each dataset, preprocessing_strategy, postprocessing_strategy, data_centric_method,
    # synthetic_model_type, and random_seed, get the rank of the classification models

    ranked_df = (
        overall_performance_df.groupby(
            [
                "dataset_id",
                "preprocessing_strategy",
                "postprocessing_strategy",
                "data_centric_method",
                "synthetic_model_type",
                "random_seed",
            ],
        )
        .apply(
            lambda x: calculate_rank_distance(  # type: ignore
                df=x,  # type: ignore
                rank_column="classification_model_type_id",
                original_ranking=real_data_model_ranking,
                dataset_id_column="dataset_id",
            ),
        )
        .reset_index()
    )
    ranked_df = ranked_df.rename(columns={0: "rank_distance"})
    # remove rows where synthetic_model_type is None and preprocessing_strategy is org_data
    # as their rank will always be 1
    ranked_df = ranked_df.query(
        "not (synthetic_model_type == 'None' & preprocessing_strategy == 'org_data')",
    )

    return ranked_df


if __name__ == "__main__":
    dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    plot_save_dir = RESULTS_DIR / "experiment3" / "plots" / "model_selection"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # experiment_suite = get_experiment_suite_(dataset_dirs)
    experiment_suite = get_experiment3_suite(dataset_dirs)

    # extract classification model performance
    ranked_df = get_model_selection_performance_df(
        experiment_suite=experiment_suite,
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
    )
