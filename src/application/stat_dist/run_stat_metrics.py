"""Code to only run extraction of statistical fidelity results for the main experiment.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Literal

import numpy as np
import pandas as pd
from application.constants import (
    DATA_CENTRIC_THRESHOLDS,
    DATA_DIR,
    RESULTS_DIR,
    SYNTHETIC_MODEL_PARAMS,
)
from application.main_experiment.run_main_experiment import (
    load_main_experiment_datasets,
)
from data_centric_synth.data_models.data_sculpting import (
    AlphaPrecisionMetrics,
    ProcessedData,
)
from data_centric_synth.data_models.experiment3 import (
    Experiment3Dataset,
    StatisticalFidelityExperiment,
)
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)
from data_centric_synth.experiments.run_experiments import (
    make_synthetic_datasets,
    make_train_test_split,
    postprocess_synthetic_datasets,
)
from data_centric_synth.experiments.statistical_fidelity import (
    StatisticalFidelity,
    StatisticalFidelityMetrics,
)
from data_centric_synth.serialization.serialization import save_to_pickle
from data_centric_synth.utils import seed_everything
from wasabi import Printer


def run_stat_extraction_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    synthetic_model: str,
    synthetic_model_params: dict,
    data_centric_method: Literal[IMPLEMENTED_DATA_CENTRIC_METHODS],
    data_centric_threshold: float,
    percentile_threshold: int,
    random_state: int,
) -> List[StatisticalFidelityExperiment]:
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    y_train.name = "target"
    # make synthetic datasets with different preprocessing strategies
    datasets = make_synthetic_datasets(
        X_train=X_train,
        y_train=y_train,
        synthetic_model=synthetic_model,
        synthetic_model_params=synthetic_model_params,
        data_centric_method=data_centric_method,  #
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
    # evaluate statistical fidelity metrics
    results = evaluate_statistical_fidelity_on_postprocessed_datasets(
        postprocessed_datasets=postprocessed_datasets,
        real_dataset=pd.concat([X_train, y_train], axis=1),
        synthetic_model=synthetic_model,
        data_centric_method=data_centric_method,
        random_seed=random_state,
    )
    return results


def evaluate_statistical_fidelity_on_postprocessed_datasets(
    postprocessed_datasets: Dict[str, Dict[str, ProcessedData]],
    real_dataset: pd.DataFrame,
    synthetic_model: str,
    data_centric_method: str,
    random_seed: int,
) -> List[StatisticalFidelityExperiment]:
    """Evaluate statistical fidelity metrics for a given set of synthetic datasets and a real dataset"""
    results: List[StatisticalFidelityExperiment] = []
    for preprocessing_strategy in postprocessed_datasets.keys():
        for postprocessing_strategy, postprocessed_data in postprocessed_datasets[
            preprocessing_strategy
        ].items():
            results.append(
                StatisticalFidelityExperiment(
                    preprocessing_strategy=preprocessing_strategy,
                    postprocessing_strategy=postprocessing_strategy,
                    synthetic_model=synthetic_model,
                    data_centric_method=data_centric_method,
                    statistical_fidelity_metrics=StatisticalFidelity().calculate_metrics(
                        X=real_dataset, X_syn=postprocessed_data.dataset
                    ),
                    random_seed=random_seed,
                )
            )
    return results


def main_statistical_metrics_experiments(
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
                syn_model_experiments: List[StatisticalFidelityExperiment] = []
                for data_centric_method in data_centric_methods:
                    try:
                        syn_model_experiments.extend(
                            run_stat_extraction_experiment(
                                X=dataset.X,  # type: ignore
                                y=dataset.y,  # type: ignore
                                synthetic_model=synthetic_model,
                                synthetic_model_params={
                                    **SYNTHETIC_MODEL_PARAMS[synthetic_model],
                                    "random_state": random_seed,
                                },
                                data_centric_method=data_centric_method,
                                data_centric_threshold=DATA_CENTRIC_THRESHOLDS[  # type: ignore
                                    data_centric_method
                                ].data_centric_threshold,
                                percentile_threshold=DATA_CENTRIC_THRESHOLDS[  # type: ignore
                                    data_centric_method
                                ].percentile_threshold,
                                random_state=random_seed,
                            )
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


if __name__ == "__main__":
    # SAVE_DIR = RESULTS_DIR / "main_experiment" / "statistical_fidelity"
    SAVE_DIR = DATA_DIR / "main_experiment" / "statistical_fidelity"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    N_SEEDS = 10
    STARTING_SEED = 42
    seed_everything(seed=STARTING_SEED)

    random_seeds = np.random.randint(0, 10000, size=N_SEEDS)[0]
    if not isinstance(random_seeds, Iterable):
        random_seeds = [random_seeds]

    main_statistical_metrics_experiments(
        datasets=load_main_experiment_datasets(),
        save_dir=SAVE_DIR,
        random_seeds=random_seeds,
        synthetic_model_suite=get_default_synthetic_model_suite(),
        data_centric_methods=["cleanlab"],  # type: ignore
    )
