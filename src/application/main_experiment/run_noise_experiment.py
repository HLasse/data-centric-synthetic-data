"""Script to run the noise test experiment, assessing the impact of added label noise
to the Covid mortality dataset
"""
import numpy as np

from application.constants import DATA_DIR, RESULTS_DIR
from application.figure1.run_figure1_exp import load_and_preprocess_covid_dataset
from data_centric_synth.data_models.experiment3 import Experiment3Dataset
from data_centric_synth.experiments.models import get_default_synthetic_model_suite
from data_centric_synth.experiments.run_experiments import run_noise_experimental_loop
from data_centric_synth.utils import seed_everything


def load_covid_data() -> Experiment3Dataset:
    X, y = load_and_preprocess_covid_dataset()
    return Experiment3Dataset(
        name="covid",
        X=X,
        y=y,
    )


if __name__ == "__main__":
    # SAVE_DIR = RESULTS_DIR / "experiment3" / "noise_data_covid"
    SAVE_DIR = DATA_DIR / "main_experiment" / "noise_data"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    N_SEEDS = 10
    STARTING_SEED = 42
    seed_everything(seed=STARTING_SEED)

    random_seeds = np.random.randint(0, 10000, size=N_SEEDS)[:5]

    dataset = load_covid_data()

    run_noise_experimental_loop(
        datasets=[dataset],
        save_dir=SAVE_DIR,
        random_seeds=random_seeds,
        synthetic_model_suite=get_default_synthetic_model_suite(),
        data_centric_methods=["cleanlab"],
        noise_levels=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
    )
