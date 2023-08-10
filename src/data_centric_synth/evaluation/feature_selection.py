"""Tools to extract feature selection performance"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import spearmanr

from application.constants import RESULTS_DIR
from data_centric_synth.data_models.experiment3 import Experiment3Suite
from data_centric_synth.evaluation.extraction import (
    FeatureSelectionRanking,
    feature_importances_to_df,
    get_data_centric_method,
    get_experiment3_suite,
    get_experiments_with_classification_model_evaluation,
    get_experiments_with_feature_importance,
    get_feature_importances,
    get_synthetic_model,
    get_unique_seeds,
    subset_by_seed,
)
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)


def calculate_rank_distance(
    df: pd.DataFrame,
    dataset_id_column: str,
    original_ranking: Dict[str, Dict[str, int]],
) -> np.ndarray:
    """Sorts the df by the importance column and maps the features to the ranking
    from the baseline results. Calculates the distance between the
    original ranking and the ranking of the feature column. The distance is calculated
    using spearmans rank correlation."""
    dataset_id = df[dataset_id_column].unique()
    if len(dataset_id) != 1:
        raise ValueError("Expected only one dataset_id")
    dataset_id = dataset_id[0]
    df = df.sort_values("importance", ascending=False)
    # map the feature to the rank in the original ranking
    df["feature_rank"] = df["feature"].replace(original_ranking[dataset_id])

    original_ranking_array = np.array(list(original_ranking[dataset_id].values()))
    return spearmanr(original_ranking_array, df["feature_rank"]).statistic  # type: ignore


def get_feature_selection_performance(
    experiment_suite: Experiment3Suite,
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS],
    synthetic_models: List[str],
    model_type: str = "XGBClassifier",
) -> pd.DataFrame:
    overall_performances: List[pd.DataFrame] = []
    for dataset in experiment_suite.experiment_groups:
        exps = get_experiments_with_classification_model_evaluation(dataset.experiments)
        exps = get_experiments_with_feature_importance(exps)
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
                seed_performances: List[FeatureSelectionRanking] = []
                for seed in get_unique_seeds(experiments=model_subset):
                    seed_subset = subset_by_seed(experiments=model_subset, seed=seed)
                    seed_performances.extend(
                        get_feature_importances(experiments=seed_subset),
                    )
                synthetic_model_performance_df = feature_importances_to_df(
                    feature_importances=seed_performances,
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

    ranked_df = calculate_feature_ranking_distance(
        overall_performance_df, model_type=model_type
    )
    return ranked_df


def calculate_feature_ranking_distance(
    overall_performance_df: pd.DataFrame,
    model_type: str,
) -> pd.DataFrame:
    # due to the implementation, the evaluation of models trained on real data and
    # evaluated on real data is repeated for each model within seed. I.e., there
    # will be 5 rows instead of 1 for each seed. They are identical, so we can
    # remove the duplicates
    overall_performance_df = overall_performance_df.drop_duplicates()

    # focusing on the feature importances from xgboost for now
    overall_performance_df = overall_performance_df.query(
        "classification_model_type == @model_type",
    )

    # get feature ranking of real data. Subset to only include the real data
    # set and get the feature ranking for each dataset
    # subsetting to only include one of the data centric methods, as none of them
    # have been applied at this point
    real_data_feature_ranking = (
        overall_performance_df.query(
            "synthetic_model_type == 'None' & postprocessing_strategy != 'no_hard' & data_centric_method == 'cleanlab'",
        )
        .groupby(["dataset_id", "feature"])
        .agg({"importance": "mean"})
        .sort_values(["dataset_id", "importance"], ascending=False)
        .reset_index()
    )

    # turn the ordering into a dict with dataset_id as key and the rank ordering
    # as numpy array
    real_data_feature_ranking_dict = (
        real_data_feature_ranking.groupby("dataset_id")["feature"]
        .apply(lambda x: x.values)
        .to_dict()
    )
    # create mapping from feature to rank for each dataset
    real_data_feature_mapping_dict = {
        dataset_id: {feature: rank for rank, feature in enumerate(features)}
        for dataset_id, features in real_data_feature_ranking_dict.items()
    }

    # for each dataset, preprocessing_strategy, postprocessing_strategy, data_centric_method,
    # synthetic_model_type, and random_seed, get the rank of the feature selection

    ranked_df = (
        overall_performance_df.query("synthetic_model_type != 'None' | postprocessing_strategy == 'no_hard'")
        .groupby(
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
                original_ranking=real_data_feature_mapping_dict,
                dataset_id_column="dataset_id",
            ),
        )
        .reset_index()
    )
    ranked_df = ranked_df.rename(columns={0: "rank_distance"})
    return ranked_df


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    plot_save_dir = RESULTS_DIR / "experiment3" / "plots" / "feature_selection"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # experiment_suite = get_experiment_suite_(dataset_dirs)
    experiment_suite = get_experiment3_suite(dataset_dirs)

    # extract classification model performance
    ranked_df = get_feature_selection_performance(
        experiment_suite=experiment_suite,
        data_centric_methods=data_centric_methods,
        synthetic_models=synthetic_models,
    )
