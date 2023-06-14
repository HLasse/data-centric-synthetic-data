"""Script to run the main experiment:

    1. Model selection. Do models trained on real/synhetic data choose the same model type?
    2. Model performance. Do models trained on real/synthetic data perform similarly?
    3. Feature selection. Do models trained on real/synthetic data choose the same features (feature importance)?
"""

from typing import Generator

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

from application.constants import DATA_DIR, RESULTS_DIR
from data_centric_synth.data_models.experiment3 import Experiment3Dataset
from data_centric_synth.datasets.openml.openml_loaders import (
    get_openml_benchmark_suite_task_ids,
    get_task_X_y,
)
from data_centric_synth.experiments.models import get_default_synthetic_model_suite
from data_centric_synth.experiments.run_experiments import run_main_experimental_loop
from data_centric_synth.utils import seed_everything


def dataset_iterator() -> Generator[Experiment3Dataset, None, None]:
    """Iterate over the datasets and yield X and y. Only yield if the dataset
    has less than 100_000 rows and less than 50 columns"""
    for task in get_openml_benchmark_suite_task_ids(suite_id=337):
        X, y = get_task_X_y(task_id=task)
        for col in X.columns:
            if X[col].dtype == "uint8":
                X[col] = X[col].astype(int)
        y = pd.Series(LabelEncoder().fit_transform(y))
        if X.shape[0] < 100_000 and X.shape[1] < 50:
            yield Experiment3Dataset(name=str(task), X=X, y=y.astype(int))


if __name__ == "__main__":
    # SAVE_DIR = RESULTS_DIR / "experiment3" / "data"
    SAVE_DIR = DATA_DIR / "main_experiment" / "data"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    N_SEEDS = 10
    STARTING_SEED = 42
    seed_everything(seed=STARTING_SEED)

    random_seeds = np.random.randint(0, 10000, size=N_SEEDS)

    run_main_experimental_loop(
        datasets=dataset_iterator(),
        save_dir=SAVE_DIR,
        random_seeds=random_seeds,
        synthetic_model_suite=get_default_synthetic_model_suite(),
    )
