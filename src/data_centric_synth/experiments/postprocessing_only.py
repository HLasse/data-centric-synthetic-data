from pathlib import Path
from typing import List, Literal

import pandas as pd
from application.constants import DATA_CENTRIC_THRESHOLDS, MAIN_EXP_POSTPROCESSING_DIR
from application.main_experiment.run_main_experiment import (
    load_main_experiment_datasets,
)
from application.main_experiment.run_noise_experiment import load_covid_data
from sklearn.base import ClassifierMixin
from wasabi import Printer

from data_centric_synth.data_models.data_sculpting import ProcessedData
from data_centric_synth.data_models.experiment3 import Experiment3
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    extract_subsets_from_sculpted_data,
    get_datacentric_segments_from_sculpted_data,
    sculpt_data_by_method,
)
from data_centric_synth.datasets.simulated.simulate_data import flip_outcome
from data_centric_synth.experiments.models import get_default_classification_model_suite
from data_centric_synth.experiments.run_experiments import (
    evaluate_model_on_sculpted_data_splits,
    fit_and_evaluate_model_on_original_data,
    make_train_test_split,
    real_data_model_evaluation_to_experiment,
)
from data_centric_synth.experiments.statistical_fidelity import StatisticalFidelity
from data_centric_synth.serialization.serialization import save_to_pickle
from data_centric_synth.utils import seed_everything


def run_postprocessing_experimental_loop(
    experiment_data: Literal["noise_data", "main_experiment"],
    save_dir: Path,
    random_seeds: List[int],
) -> None:
    """Run the postprocessing experimental loop"""
    if experiment_data == "noise_data":
        run_postprocessing_noise_data_loop(
            save_dir,
            random_seeds=random_seeds,
        )
    elif experiment_data == "main_experiment":
        run_postprocessing_main_exp_loop(
            random_seeds=random_seeds,
        )
    else:
        raise ValueError(f"Unknown experiment_data: {experiment_data}")


def run_postprocessing_main_exp_loop(random_seeds: List[int]) -> None:
    msg = Printer(timestamp=True)
    datasets = load_main_experiment_datasets()

    for dataset in datasets:
        msg.info(
            f"Running postprocessing, orignal data only experiments for dataset {dataset.name}"
        )

        dataset_experiment_save_dir = MAIN_EXP_POSTPROCESSING_DIR / dataset.name
        msg.info(f"Saving to {dataset_experiment_save_dir}")
        dataset_experiment_save_dir.mkdir(exist_ok=True, parents=True)

        for random_seed in random_seeds:
            file_name = dataset_experiment_save_dir / f"seed_{random_seed}.pkl"
            if file_name.exists():
                msg.info(f"Skipping {random_seed} as it already exists")
                continue
            msg.info(f"Running experiment for seed {random_seed}")

            seed_everything(seed=random_seed)
            classification_model_suite = get_default_classification_model_suite(
                random_state=random_seed,
            )
            # create the file as a placeholder so that other jobs don't try to run it
            save_to_pickle(obj=None, path=file_name)

            postprocessing_experiments = run_postprocessing_main_experiment(
                X=dataset.X,
                y=dataset.y,
                classification_models=classification_model_suite,
                random_state=random_seed,
            )
            save_to_pickle(obj=postprocessing_experiments, path=file_name)


def run_postprocessing_noise_data_loop(save_dir: Path, random_seeds: List[int]) -> None:
    msg = Printer(timestamp=True)

    noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.1]
    datasets = [load_covid_data()]

    for dataset in datasets:
        for noise_level in noise_levels:
            msg.info(
                f"Running postprocessing, original data only experiment for noise level {noise_level}"
            )
            dataset_experiment_save_dir = save_dir / dataset.name / f"{noise_level}"
            msg.info(f"Saving to {dataset_experiment_save_dir}")
            dataset_experiment_save_dir.mkdir(exist_ok=True, parents=True)
            for random_seed in random_seeds:
                file_name = dataset_experiment_save_dir / f"seed_{random_seed}.pkl"
                if file_name.exists():
                    msg.info(f"Skipping {random_seed} as it already exists")
                    continue
                msg.info(f"Running experiment for seed {random_seed}")

                seed_everything(seed=random_seed)
                classification_model_suite = get_default_classification_model_suite(
                    random_state=random_seed,
                )
                # create the file as a placeholder so that other jobs don't try to run it
                save_to_pickle(obj=None, path=file_name)

                postprocessing_experiments = run_postprocessing_noise_experiment(
                    X=dataset.X,
                    y=dataset.y,
                    noise_level=noise_level,
                    classification_models=classification_model_suite,
                    random_state=random_seed,
                )
                save_to_pickle(postprocessing_experiments, path=file_name)


def run_postprocessing_noise_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    noise_level: float,
    classification_models: List[ClassifierMixin],
    random_state: int,
) -> List[Experiment3]:
    """Main function to run the experiment assessing the impact of postprocessing
    the original data only."""
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    # flip labels in the training data
    y_train, _ = flip_outcome(y_train, prop_to_flip=noise_level)

    # postprocess the data
    postprocessed_data = postprocess_single_dataset(X_train=X_train, y_train=y_train)  # type: ignore

    # train and evaluate models
    experiments = train_and_evaluate_postprocessing_on_real_data_only(
        postprocessed_data=postprocessed_data,
        X_train=X_train,
        X_test=X_test,  # type: ignore
        y_train=y_train,  # type: ignore
        y_test=y_test,
        random_state=random_state,
        classification_models=classification_models,
    )
    return experiments


def run_postprocessing_main_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    classification_models: List[ClassifierMixin],
    random_state: int,
) -> List[Experiment3]:
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    # postprocess the data
    postprocessed_data = postprocess_single_dataset(X_train=X_train, y_train=y_train)

    # train and evaluate models
    experiments = train_and_evaluate_postprocessing_on_real_data_only(
        postprocessed_data=postprocessed_data,
        X_train=X_train,
        X_test=X_test,  # type: ignore
        y_train=y_train,  # type: ignore
        y_test=y_test,
        random_state=random_state,
        classification_models=classification_models,
    )
    return experiments


def train_and_evaluate_postprocessing_on_real_data_only(
    postprocessed_data: ProcessedData,
    X_train: pd.DataFrame,
    X_test: pd.Series,
    y_train: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    classification_models: List[ClassifierMixin],
) -> List[Experiment3]:
    experiments: List[Experiment3] = []
    X_test.name = "target"

    ## train models on original data and evaluate on test data
    sculpted_test_data = sculpt_data_by_method(
        X=X_test,  # type: ignore
        y=y_test,
        data_centric_method="cleanlab",
        percentile_threshold=DATA_CENTRIC_THRESHOLDS["cleanlab"].percentile_threshold,
        data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
            "cleanlab"
        ].data_centric_threshold,
    )
    test_data_segments = get_datacentric_segments_from_sculpted_data(
        sculpted_data=sculpted_test_data,
        subsets=["easy", "ambi", "hard"],
    )

    y_train.name = "target"
    statistical_fidelity = StatisticalFidelity().calculate_metrics(
        X=pd.concat([X_train, y_train], axis=1),
        X_syn=postprocessed_data.dataset,
    )

    for model in classification_models:
        ## train models on original data and evaluate on test data
        experiments.append(
            fit_and_evaluate_model_on_original_data(
                X_train=X_train,
                y_train=y_train,  # type: ignore
                data_centric_method="cleanlab",
                percentile_threshold=DATA_CENTRIC_THRESHOLDS[
                    "cleanlab"
                ].percentile_threshold,
                data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
                    "cleanlab"
                ].data_centric_threshold,
                random_state=random_state,
                sculpted_test_data=sculpted_test_data,
                test_data_segments=test_data_segments,
                model=model,
            ),
        )
        ## train models on postprocessed data and evaluate on test data
        model.fit(  # type: ignore
            postprocessed_data.dataset.drop(columns=["target"]),
            postprocessed_data.dataset["target"],
        )
        model_eval = evaluate_model_on_sculpted_data_splits(
            model=model, sculpted_test_data=sculpted_test_data
        )
        experiments.append(
            real_data_model_evaluation_to_experiment(
                data_centric_method="cleanlab",
                percentile_threshold=DATA_CENTRIC_THRESHOLDS[
                    "cleanlab"
                ].percentile_threshold,
                data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
                    "cleanlab"
                ].data_centric_threshold,
                random_state=random_state,
                test_data_segments=test_data_segments,
                model=model,
                model_eval_results=model_eval,
                postprocessing_strategy="no_hard",
                statistical_fidelty=statistical_fidelity,
            )
        )
    return experiments


def postprocess_single_dataset(
    X_train: pd.DataFrame, y_train: pd.Series
) -> ProcessedData:
    sculpted_data = sculpt_data_by_method(
        X=X_train,
        y=y_train,  # type: ignore
        data_centric_method="cleanlab",
        data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
            "cleanlab"
        ].data_centric_threshold,
        percentile_threshold=DATA_CENTRIC_THRESHOLDS["cleanlab"].percentile_threshold,
    )
    return extract_subsets_from_sculpted_data(
        data=sculpted_data, subsets=["easy", "ambi"]
    )
