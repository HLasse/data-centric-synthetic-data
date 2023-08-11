from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from application.constants import DATA_CENTRIC_THRESHOLDS, SYNTHETIC_MODEL_PARAMS
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from wasabi import Printer

from data_centric_synth.causal_discovery.dagma import DAGMA_linear, evaluate_dag_model
from data_centric_synth.data_models.data_sculpting import (
    DataSegments,
    ProcessedData,
    SculptedData,
)
from data_centric_synth.data_models.experiment3 import Experiment3, Experiment3Dataset
from data_centric_synth.data_models.experiments_common import (
    DataSegmentEvaluation,
    Metrics,
    PerformanceEvaluation,
    Processing,
)
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    add_target_col,
    extract_subsets_from_sculpted_data,
    get_datacentric_segments_from_sculpted_data,
    sculpt_data_by_method,
)
from data_centric_synth.datasets.simulated.simulate_data import flip_outcome
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_classification_model_suite,
)
from data_centric_synth.experiments.statistical_fidelity import (
    StatisticalFidelity,
    StatisticalFidelityMetrics,
)
from data_centric_synth.serialization.serialization import save_to_pickle
from data_centric_synth.synthetic_data.synthetic_data_generation import (
    evaluate_detection_auc,
    fit_and_generate_synth_data,
    generate_synth_data_from_sculpted_data,
)
from data_centric_synth.utils import seed_everything


def _roc_auc_score(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_probs: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate the roc auc score using sklearn but return 0.0 if error occurs."""
    try:
        return roc_auc_score(y_true, y_pred_probs)  # type: ignore
    except:
        return 0.0


def calculate_metrics(
    y: Union[pd.Series, np.ndarray],
    y_pred_probs: Union[pd.Series, np.ndarray],
) -> Metrics:
    """Calculate metrics for the given predictions.

    Args:
        y (np.ndarray): The true labels.
        y_pred_probs (np.ndarray): The predicted probabilities for class 1.

    Returns:
        Dict[str, float]: The calculated metrics.
    """
    y_pred_int = np.where(y_pred_probs > 0.5, 1, 0)
    return Metrics(
        roc_auc=_roc_auc_score(y, y_pred_probs),  # type: ignore
        accuracy=accuracy_score(y, y_pred_int),  # type: ignore
        precision=precision_score(y, y_pred_int),  # type: ignore
        recall=recall_score(y, y_pred_int),  # type: ignore
        f1=f1_score(y, y_pred_int),  # type: ignore
    )


def reset_index_of_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def make_train_test_split(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Make a train/test split.

    Args:
        X (pd.DataFrame): The features.
        y (pd.Series): The labels.
        test_size (float): The size of the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test = reset_index_of_splits(train=X_train, test=X_test)
    y_train, y_test = (
        reset_index_of_splits(train=y_train, test=y_test)  # type: ignore
        if isinstance(y_train, pd.Series)
        else (pd.Series(y_train), pd.Series(y_test))
    )
    return X_train, X_test, y_train, y_test  # type: ignore


def evaluate_model_on_processed_data(
    model: ClassifierMixin,
    data: ProcessedData,
) -> PerformanceEvaluation:
    # if empty data, return empty evaluation
    if data.dataset.empty:
        return PerformanceEvaluation(metrics=None, feature_importances=None)

    pred_probs = model.predict_proba(data.dataset.drop("target", axis=1))[:, 1]  # type: ignore
    metrics = calculate_metrics(y=data.dataset["target"], y_pred_probs=pred_probs)
    feature_importances = (
        dict(
            zip(model.feature_names_in_, model.feature_importances_),  # type: ignore
        )
        if hasattr(model, "feature_importances_")
        else None
    )
    return PerformanceEvaluation(
        metrics=metrics,
        feature_importances=feature_importances,
    )


def evaluate_model_on_sculpted_data_splits(
    model: ClassifierMixin,
    sculpted_test_data: SculptedData,
) -> DataSegmentEvaluation:
    full = extract_subsets_from_sculpted_data(
        data=sculpted_test_data,
        subsets=["easy", "ambi", "hard"],
    )
    easy = extract_subsets_from_sculpted_data(data=sculpted_test_data, subsets=["easy"])
    ambiguous = extract_subsets_from_sculpted_data(
        data=sculpted_test_data,
        subsets=["ambi"],
    )
    hard = extract_subsets_from_sculpted_data(data=sculpted_test_data, subsets=["hard"])
    return DataSegmentEvaluation(
        full=evaluate_model_on_processed_data(model=model, data=full),
        easy=evaluate_model_on_processed_data(model=model, data=easy),
        ambiguous=evaluate_model_on_processed_data(model=model, data=ambiguous),
        hard=evaluate_model_on_processed_data(model=model, data=hard),
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessed_datasets: Dict[str, ProcessedData],
    postprocessed_datasets: Dict[str, Dict[str, ProcessedData]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    synthetic_model: str,
    classification_models: List[ClassifierMixin],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    random_state: int,
) -> List[Experiment3]:  # sourcery skip: for-append-to-extend, use-dict-items
    """Train XGBoost on the synthetic datasets and evaluate on holdout data.

    Args:
        X_train (pd.DataFrame): The features of the original dataset.
        y_train (pd.Series): The labels of the original dataset.
        preprocessed_datasets (Dict[str, ProcessedData]): The preprocessed
            original datasets. Used for passing the dataiq segments to the
            Experiment object.
        postprocessed_datasets (Dict[str, Dict[str, pd.DataFrame]]): The postprocessed
            synthetic datasets.
        X_test (pd.DataFrame): The holdout features.
        y_test (pd.Series): The holdout labels.
        synthetic_model (str): The name of the synthetic model.
        classification_models (List[ClassifierMixin]): The classification models to
            train and evaluate.
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS): The data
            centric method used for sculpting the data.
        percentile_threshold (int): The percentile used for the dataiq or datamaps
            processing.
        data_centric_threshold (float): The numeric threshold used for the data
            centric processing.
        random_state (int): The random state used for data generation.

    Returns:
        List[Experiment]: List of Experiment objects.
    """
    experiments: List[Experiment3] = []
    X_test.name = "target"
    # train models on the original data and evaluate on test data

    ## sculpt test data with data-centric method
    sculpted_test_data = sculpt_data_by_method(
        X=X_test,
        y=y_test,
        data_centric_method=data_centric_method,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )
    test_data_segments = get_datacentric_segments_from_sculpted_data(
        sculpted_data=sculpted_test_data,
        subsets=["easy", "ambi", "hard"],
    )
    ## train models on original data and evaluate on test data
    for model in classification_models:
        experiments.append(
            fit_and_evaluate_model_on_original_data(
                X_train=X_train,
                y_train=y_train,
                data_centric_method=data_centric_method,
                percentile_threshold=percentile_threshold,
                data_centric_threshold=data_centric_threshold,
                random_state=random_state,
                sculpted_test_data=sculpted_test_data,
                test_data_segments=test_data_segments,
                model=model,
            ),
        )
    ## train DAG model on original data as reference

    # train models on synthetic data and evaluate on test data
    for preprocessing_strategy in postprocessed_datasets:
        preprocessing_schema = Processing(
            strategy_name=preprocessing_strategy,
            uncertainty_percentile_threshold=percentile_threshold,
            uncertainty_threshold=data_centric_threshold,
            data_segments=preprocessed_datasets[preprocessing_strategy].data_segments,
            detection_auc=None,
            statistical_fidelity=None,
        )
        for postprocessing_strategy, postprocessed_data in postprocessed_datasets[
            preprocessing_strategy
        ].items():
            # make common post- and test-processing schemas
            postprocessing_schema = Processing(
                strategy_name=postprocessing_strategy,
                uncertainty_percentile_threshold=percentile_threshold,
                uncertainty_threshold=percentile_threshold,
                data_segments=postprocessed_data.data_segments,
                detection_auc=None,
                statistical_fidelity=StatisticalFidelity().calculate_metrics(
                    X=pd.concat([X_train, y_train], axis=1),
                    X_syn=postprocessed_data.dataset,
                ),
            )
            testprocessing_schema = Processing(
                strategy_name=data_centric_method,
                uncertainty_percentile_threshold=percentile_threshold,
                uncertainty_threshold=data_centric_threshold,
                data_segments=test_data_segments,
                detection_auc=None,
                statistical_fidelity=None,
            )
            # train classification model suite on synthetic data
            for model in classification_models:
                # train model on synthetic data and evaluate on test data
                model.fit(  # type: ignore
                    postprocessed_data.dataset.drop("target", axis=1),
                    postprocessed_data.dataset["target"],
                )
                # evaluate on test data, both full and data centric
                model_eval = evaluate_model_on_sculpted_data_splits(
                    model=model,
                    sculpted_test_data=sculpted_test_data,
                )
                experiments.append(
                    Experiment3(
                        preprocessing=preprocessing_schema,
                        postprocessing=postprocessing_schema,
                        testprocessing=testprocessing_schema,
                        data_centric_method=data_centric_method,
                        synthetic_model_type=synthetic_model,
                        classification_model_type=type(model).__name__,
                        classification_model_evaluation=model_eval,
                        dag_evaluation=None,
                        random_seed=random_state,
                    ),
                )

    return experiments


def fit_dagma_and_get_dag(X_train: pd.DataFrame) -> np.ndarray:
    model = DAGMA_linear(loss_type="l2")
    return model.fit(X_train.to_numpy(), lambda1=0.02)


def fit_and_evaluate_model_on_original_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    random_state: int,
    sculpted_test_data: SculptedData,
    test_data_segments: DataSegments,
    model: ClassifierMixin,
) -> Experiment3:
    """Fit a model on the original data and evaluate on the test data; both
    full and data centric splits."""
    # train model on original data
    model.fit(  # type: ignore
        X_train,
        y_train,
    )
    # evaluate on test data, both full and data centric splits
    baseline_model_test_results = evaluate_model_on_sculpted_data_splits(
        model=model,
        sculpted_test_data=sculpted_test_data,
    )
    # convert to Experiment object
    return real_data_model_evaluation_to_experiment(
        data_centric_method=data_centric_method,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
        random_state=random_state,
        test_data_segments=test_data_segments,
        model=model,
        model_eval_results=baseline_model_test_results,
    )


def real_data_model_evaluation_to_experiment(
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    random_state: int,
    test_data_segments: DataSegments,
    model: ClassifierMixin,
    model_eval_results: DataSegmentEvaluation,
    postprocessing_strategy: str = "org_data",
    statistical_fidelty: Optional[StatisticalFidelityMetrics] = None,
) -> Experiment3:
    """Save the results of a model trained on the original data to an Experiment
    object"""
    return Experiment3(
        preprocessing=Processing(
            strategy_name="org_data",
            uncertainty_percentile_threshold=None,
            uncertainty_threshold=None,
            data_segments=None,
            detection_auc=None,
            statistical_fidelity=None,
        ),
        postprocessing=Processing(
            strategy_name=postprocessing_strategy,
            uncertainty_percentile_threshold=None,
            uncertainty_threshold=None,
            data_segments=None,
            detection_auc=None,
            statistical_fidelity=statistical_fidelty,
        ),
        testprocessing=Processing(
            strategy_name=data_centric_method,
            uncertainty_percentile_threshold=percentile_threshold,
            uncertainty_threshold=data_centric_threshold,
            data_segments=test_data_segments,
            detection_auc=None,
            statistical_fidelity=None,
        ),
        synthetic_model_type="None",
        classification_model_type=type(model).__name__,
        data_centric_method=data_centric_method,
        classification_model_evaluation=model_eval_results,
        random_seed=random_state,
        dag_evaluation=None,
    )


def postprocess_synthetic_datasets(
    datasets: Dict[str, ProcessedData],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    data_centric_threshold: Optional[float],
    percentile_threshold: Optional[int],
) -> Dict[str, Dict[str, ProcessedData]]:
    """Apply different postprocessing strategies to the synthetic datasets.
    At present, this includes a baseline of not doing any postprocessing and
    any postprocessing strategy that removes hard examples.

    Args:
        datasets (Dict[str, pd.DataFrame]): The synthetic datasets.
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS):
            The data centric method to use for postprocessing.
        data_centric_threshold (float): The data centric threshold to use for
            postprocessing. If None, the threshold will be determined by the
            percentile threshold if the method is dataiq or datamaps, or be
            the cleanlab default threshold if the method is cleanlab.
        percentile_threshold (int): The percentile threshold to use for
            postprocessing. If None, the threshold will be determined by the
            data_centric_threshold if the method is dataiq or datamaps.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: The postprocessed synthetic datasets.
    """
    postprocessed_datasets: Dict[str, Dict[str, ProcessedData]] = defaultdict(dict)
    for name, processed_data in datasets.items():
        sculpted_synth_data = sculpt_data_by_method(
            X=processed_data.dataset.drop("target", axis=1),
            y=processed_data.dataset["target"],
            data_centric_method=data_centric_method,
            data_centric_threshold=data_centric_threshold,
            percentile_threshold=percentile_threshold,
        )

        # strategy 1: no postprocessing
        postprocessed_datasets[name]["baseline"] = extract_subsets_from_sculpted_data(
            data=sculpted_synth_data,
            subsets=["easy", "ambi", "hard"],
        )
        # strategy 2: remove hard examples
        easy_ambi = extract_subsets_from_sculpted_data(
            data=sculpted_synth_data,
            subsets=["easy", "ambi"],
        )
        postprocessed_datasets[name]["easy_ambi"] = easy_ambi
        # maybe TODO strategy 3: remove some amount of ambiguous examples?
    return postprocessed_datasets


def make_synthetic_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    synthetic_model: str,
    synthetic_model_params: Dict[str, Any],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    data_centric_threshold: Optional[float],
    percentile_threshold: Optional[int],
) -> Dict[str, ProcessedData]:
    """Prepare the data for the experiments. This includes different methods for
    preprocessing the data by sculpting with dataiq and creating synthetic data on
    different combinations of easy, ambiguous and hard data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        synthetic_model (str): The name of the synthetic data generator.
        synthetic_model_params (Dict[str, Any]): The parameters for the synthetic
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS): The method
            to use for data centric sculpting.
        data_centric_threshold (Optional[float]): The threshold to use for data
            centric sculpting. One of data_centric_threshold or percentile_threshold
            must be set if data_centric_method is dataiq or datamaps. If using
            the cleanlab method, this will override the cutoff for label quality.
        percentile_threshold (Optional[int]): The percentile threshold to use for
            data centric sculpting.

    Returns:
        Dict[str, pd.DataFrame]: The data for the experiments.
    """
    if (
        data_centric_method in ["dataiq", "datamaps"]
        and data_centric_threshold is None
        and percentile_threshold is None
    ):
        raise ValueError(
            "One of data_centric or percentile_threshold"
            + " must be set if data_centric_method is dataiq or datamaps.",
        )
    sculpted_data = sculpt_data_by_method(
        X=X_train,
        y=y_train,
        data_centric_threshold=data_centric_threshold,
        percentile_threshold=percentile_threshold,
        data_centric_method=data_centric_method,
    )
    # experiment 1: train synthetic data generator on all data
    train = add_target_col(X=X_train, target=y_train)
    full_synth_data = fit_and_generate_synth_data(
        data=train,
        model_name=synthetic_model,
        model_params=synthetic_model_params,
    )
    full_synthetic_data = ProcessedData(
        dataset=full_synth_data,
        data_segments=get_datacentric_segments_from_sculpted_data(
            sculpted_data=sculpted_data,
            subsets=["easy", "ambi", "hard"],
        ),
        statistical_likeness=None,
        detection_auc=None,
    )
    # experiment 2: train synthetic data generator on easy and hard data seperately
    # then combine
    easy_hard_synth_data = generate_synth_data_from_sculpted_data(
        data=sculpted_data,
        combine=["easy", "hard"],
        synthetic_model=synthetic_model,
        synthetic_model_params=synthetic_model_params,
    )
    # experiment 3: train synthetic data generator on easy, ambiguous and hard data seperately
    # then combine. Only makes makes sense for dataiq and datamaps
    if data_centric_method in ["dataiq", "datamaps"]:
        easy_ambiguous_hard_data = generate_synth_data_from_sculpted_data(
            data=sculpted_data,
            combine=["easy", "ambi", "hard"],
            synthetic_model=synthetic_model,
            synthetic_model_params=synthetic_model_params,
        )
        return {
            "baseline": full_synthetic_data,
            "easy_hard": easy_hard_synth_data,
            "easy_ambiguous_hard": easy_ambiguous_hard_data,
        }
    return {
        "baseline": full_synthetic_data,
        "easy_hard": easy_hard_synth_data,
    }


def run_experiment3(
    X: pd.DataFrame,
    y: pd.Series,
    synthetic_model: str,
    synthetic_model_params: Dict[str, Any],
    classification_models: List[ClassifierMixin],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    random_state: int,
) -> List[Experiment3]:  # sourcery skip: inline-immediately-returned-variable
    """Main function to run the experiments. Performs the following steps:
    1. Make train/test split
    2. Generate synthetic datasets with different preprocessing strategies
    3. Apply postprocessing to the synthetic datasets
    4. Train XGBoost on the synthetic datasets and evaluate on holdout data

    Args:
        X (pd.DataFrame): The features of the original data.
        y (pd.Series): The labels of the original data.
        synthetic_model (str): The synthetic model to use.
        synthetic_model_params (Dict[str, Any]): The parameters for the synthetic
        classification_models (List[ClassifierMixin]): The classification models to
            evaluate for classification. The models should be initialized with
            random_state=random_state.
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS): The method
            to use for data centric sculpting.
        percentile_threshold (int): The percentile threshold to use for the postprocessing.
        data_centric_threshold (Optional[float]): The threshold to use for data
            centric sculpting. One of data_centric_threshold or percentile_threshold
            must be set if data_centric_method is dataiq or datamaps. If using
            the cleanlab method, this will override the cutoff for label quality.
        random_state (int): The random seed that has been set globally.

    Returns:
        List[Experiment]: A list of Experiment objects.
    """
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # make synthetic datasets with different preprocessing strategies
    datasets = make_synthetic_datasets(
        X_train=X_train,
        y_train=y_train,
        synthetic_model=synthetic_model,
        synthetic_model_params=synthetic_model_params,
        data_centric_method=data_centric_method,
        data_centric_threshold=data_centric_threshold,
        percentile_threshold=percentile_threshold,
    )
    # apply postprocessing
    postprocessed_datasets = postprocess_synthetic_datasets(
        datasets=datasets,
        data_centric_method=data_centric_method,
        data_centric_threshold=data_centric_threshold,
        percentile_threshold=percentile_threshold,
    )
    # train classification models on the synth (and real) data and evaluate on holdout data
    experiments = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        preprocessed_datasets=datasets,
        postprocessed_datasets=postprocessed_datasets,
        X_test=X_test,
        y_test=y_test,
        synthetic_model=synthetic_model,
        percentile_threshold=percentile_threshold,
        data_centric_method=data_centric_method,
        classification_models=classification_models,
        data_centric_threshold=data_centric_threshold,
        random_state=random_state,
    )
    return experiments


def convert_uint_to_int(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        if X[col].dtype == "uint8":
            X[col] = X[col].astype("int64")
    return X


def run_main_experimental_loop(
    datasets: Iterable[Experiment3Dataset],
    save_dir: Path,
    random_seeds: Iterable[int],
    synthetic_model_suite: Iterable[str],
    data_centric_methods: Iterable[IMPLEMENTED_DATA_CENTRIC_METHODS] = (
        "cleanlab",
        "dataiq",
        "datamaps",
    ),
):
    """Run the main experimental loop"""
    msg = Printer(timestamp=True)

    for dataset in datasets:
        msg.info(f"Running experiments for dataset {dataset.name}")

        dataset_experiment_save_dir = save_dir / dataset.name
        msg.info(f"Saving to {dataset_experiment_save_dir}")
        dataset_experiment_save_dir.mkdir(exist_ok=True, parents=True)

        for random_seed in random_seeds:
            seed_dir = dataset_experiment_save_dir / f"seed_{random_seed}"
            seed_dir.mkdir(exist_ok=True, parents=True)
            msg.info(f"Running experiment for seed {random_seed}")
            seed_everything(seed=random_seed)
            classification_model_suite = get_default_classification_model_suite(
                random_state=random_seed,
            )
            for synthetic_model in synthetic_model_suite:
                synth_file_name = seed_dir / f"{synthetic_model}.pkl"
                if synth_file_name.exists():
                    msg.info(
                        f"Skipping synthetic model {synthetic_model} as it already exists in {synth_file_name}",
                    )
                    continue
                # create the file as a placeholder so that other jobs don't try to run it
                save_to_pickle(obj=None, path=synth_file_name)
                msg.info(f"Running experiment for synthetic model {synthetic_model}")
                syn_model_experiments: List[Experiment3] = []
                for data_centric_method in data_centric_methods:
                    try:
                        syn_model_experiments.extend(
                            run_experiment3(
                                X=dataset.X,  # type: ignore
                                y=dataset.y,  # type: ignore
                                synthetic_model=synthetic_model,
                                synthetic_model_params={
                                    **SYNTHETIC_MODEL_PARAMS[synthetic_model],
                                    "random_state": random_seed,
                                },
                                classification_models=classification_model_suite,
                                data_centric_method=data_centric_method,
                                percentile_threshold=DATA_CENTRIC_THRESHOLDS[
                                    data_centric_method
                                ].percentile_threshold,
                                data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
                                    data_centric_method
                                ].data_centric_threshold,
                                random_state=random_seed,
                            ),
                        )
                    except Exception as e:
                        msg.fail(
                            f"Failed to run experiment on dataset {dataset.name} "
                            + f"seed {random_seed} synthetic model {synthetic_model} "
                            + f"data centric method {data_centric_method}",
                        )
                        msg.fail(e)
                        continue
                save_to_pickle(syn_model_experiments, path=synth_file_name)


def run_noise_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    noise_level: float,
    synthetic_model: str,
    synthetic_model_params: Dict[str, Any],
    classification_models: List[ClassifierMixin],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    random_state: int,
) -> List[Experiment3]:  # sourcery skip: inline-immediately-returned-variable
    """Main function to run the experiments. Performs the following steps:
    1. Make train/test split
    2. Generate synthetic datasets with different preprocessing strategies
    3. Apply postprocessing to the synthetic datasets
    4. Train XGBoost on the synthetic datasets and evaluate on holdout data

    Args:
        X (pd.DataFrame): The features of the original data.
        y (pd.Series): The labels of the original data.
        synthetic_model (str): The synthetic model to use.
        synthetic_model_params (Dict[str, Any]): The parameters for the synthetic
        classification_models (List[ClassifierMixin]): The classification models to
            evaluate for classification. The models should be initialized with
            random_state=random_state.
        data_centric_method (IMPLEMENTED_DATA_CENTRIC_METHODS): The method
            to use for data centric sculpting.
        percentile_threshold (int): The percentile threshold to use for the postprocessing.
        data_centric_threshold (Optional[float]): The threshold to use for data
            centric sculpting. One of data_centric_threshold or percentile_threshold
            must be set if data_centric_method is dataiq or datamaps. If using
            the cleanlab method, this will override the cutoff for label quality.
        random_state (int): The random seed that has been set globally.

    Returns:
        List[Experiment]: A list of Experiment objects.
    """
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    # flip labels in the training data
    y_train, _ = flip_outcome(y_train, prop_to_flip=noise_level)  # type ignore

    # make synthetic datasets with different preprocessing strategies
    datasets = make_synthetic_datasets(
        X_train=X_train,
        y_train=y_train, # type: ignore
        synthetic_model=synthetic_model,
        synthetic_model_params=synthetic_model_params,
        data_centric_method=data_centric_method,
        data_centric_threshold=data_centric_threshold,
        percentile_threshold=percentile_threshold,
    )
    # apply postprocessing
    postprocessed_datasets = postprocess_synthetic_datasets(
        datasets=datasets,
        data_centric_method=data_centric_method,
        data_centric_threshold=data_centric_threshold,
        percentile_threshold=percentile_threshold,
    )
    # train classification models on the synth (and real) data and evaluate on holdout data
    experiments = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train, # type: ignore
        preprocessed_datasets=datasets,
        postprocessed_datasets=postprocessed_datasets,
        X_test=X_test,
        y_test=y_test,
        synthetic_model=synthetic_model,
        percentile_threshold=percentile_threshold,
        data_centric_method=data_centric_method,
        classification_models=classification_models,
        data_centric_threshold=data_centric_threshold,
        random_state=random_state,
    )
    return experiments


def run_noise_experimental_loop(
    datasets: Iterable[Experiment3Dataset],
    save_dir: Path,
    random_seeds: Iterable[int],
    synthetic_model_suite: Iterable[str],
    data_centric_methods: Iterable[IMPLEMENTED_DATA_CENTRIC_METHODS] = (
        "cleanlab",
        "dataiq",
        "datamaps",
    ),
    noise_levels: Iterable[float] = [0.02, 0.04, 0.06, 0.08, 0.1],
):
    """Run the main experimental loop"""
    msg = Printer(timestamp=True)

    for dataset in datasets:
        dataset_name = dataset.name
        for noise_level in noise_levels:
            dataset.name = dataset_name + f"_noise_{noise_level}"
            msg.info(
                f"Running experiments for dataset {dataset.name}, noise level {noise_level}"
            )

            dataset_experiment_save_dir = (
                save_dir / dataset.name / f"noise_{noise_level}"
            )
            msg.info(f"Saving to {dataset_experiment_save_dir}")
            dataset_experiment_save_dir.mkdir(exist_ok=True, parents=True)

            for random_seed in random_seeds:
                seed_dir = dataset_experiment_save_dir / f"seed_{random_seed}"
                seed_dir.mkdir(exist_ok=True, parents=True)
                msg.info(f"Running experiment for seed {random_seed}")
                seed_everything(seed=random_seed)
                classification_model_suite = get_default_classification_model_suite(
                    random_state=random_seed,
                )
                for synthetic_model in synthetic_model_suite:
                    synth_file_name = seed_dir / f"{synthetic_model}.pkl"
                    if synth_file_name.exists():
                        msg.info(
                            f"Skipping synthetic model {synthetic_model} as it already exists in {synth_file_name}",
                        )
                        continue
                    # create the file as a placeholder so that other jobs don't try to run it
                    save_to_pickle(obj=None, path=synth_file_name)
                    msg.info(
                        f"Running experiment for synthetic model {synthetic_model}"
                    )
                    syn_model_experiments: List[Experiment3] = []
                    for data_centric_method in data_centric_methods:
                        try:
                            syn_model_experiments.extend(
                                run_noise_experiment(
                                    X=dataset.X,  # type: ignore
                                    y=dataset.y,  # type: ignore
                                    noise_level=noise_level,
                                    synthetic_model=synthetic_model,
                                    synthetic_model_params={
                                        **SYNTHETIC_MODEL_PARAMS[synthetic_model],
                                        "random_state": random_seed,
                                    },
                                    classification_models=classification_model_suite,
                                    data_centric_method=data_centric_method,
                                    percentile_threshold=DATA_CENTRIC_THRESHOLDS[
                                        data_centric_method
                                    ].percentile_threshold,
                                    data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
                                        data_centric_method
                                    ].data_centric_threshold,
                                    random_state=random_seed,
                                ),
                            )
                        except Exception as e:
                            msg.fail(
                                f"Failed to run experiment on dataset {dataset.name} "
                                + f"seed {random_seed} synthetic model {synthetic_model} "
                                + f"data centric method {data_centric_method}",
                            )
                            msg.fail(e)
                            continue
                    save_to_pickle(syn_model_experiments, path=synth_file_name)
