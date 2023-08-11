"""Script to run the experiment assessing the impact of the postprocessing strategy
on both the main experiment and the label noise experiment"""


import numpy as np
from application.constants import DATA_DIR, POSTPROCESSING_ONLY_SAVE_DIR
from data_centric_synth.experiments.postprocessing_only import (
    run_postprocessing_experimental_loop,
)
from data_centric_synth.utils import seed_everything

if __name__ == "__main__":
    SAVE_DIR = POSTPROCESSING_ONLY_SAVE_DIR
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    N_SEEDS = 10
    STARTING_SEED = 42
    seed_everything(seed=STARTING_SEED)

    random_seeds = list(np.random.randint(0, 10000, size=N_SEEDS)[:10])

    DATASET = "main_experiment"  # "noise_data" or "main_experiment"

    run_postprocessing_experimental_loop(
        experiment_data=DATASET,  # "noise_data" or "main_experiment"
        save_dir=SAVE_DIR,
        random_seeds=random_seeds,
    )
