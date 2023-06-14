"""Script to run the experiment to set data-centric thresholds shown in Appendix C
"""

import pickle as pkl
from pathlib import Path
from typing import List, Optional

from application.constants import ROOT_DIR
from data_centric_synth.data_models.experiment2 import (
    Experiment2,
    Experiment2Group,
    Experiment2Suite,
    OverlappingIndices,
)
from data_centric_synth.data_models.experiments_common import SimulatedData
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    sculpt_data_by_method,
)
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS
from data_centric_synth.serialization.serialization import save_to_pickle


def load_simulated_data(path: Path) -> SimulatedData:
    """Load the simulated data from a pickle file"""
    with open(path, "rb") as f:
        return pkl.load(f)


def sculpt_and_get_indices_by_method(
    simulated_data: SimulatedData,
    proportion_flipped: float,
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    uncertainty_threshold: Optional[float],
) -> Experiment2:
    """Sculpt the data using the data-centric method to get the indices of the
    hard/bad data segments and return an Experiment2 object.

    Args:
        simulated_data (SimulatedData): The simulated data to be sculpted
        proportion_flipped (float): The proportion of labels that were flipped
        synthetic_model (str): The synthetic model used to generate the data
        data_centric_method (Literal["dataiq", "datamaps", "cleanlab"]): The
            data-centric method to use.
        percentile_threshold (Optional[int]): The percentile threshold to use for
            the data-centric method.
        uncertainty_threshold (Optional[float]): The uncertainty threshold to use
            for the data-centric method

        Returns:
            Experiment2: An Experiment2 object"""
    sculpted_data = sculpt_data_by_method(
        X=simulated_data.dataset.drop(columns=["target"]),
        y=simulated_data.dataset["target"],
        data_centric_method=data_centric_method,  # type: ignore
        percentile_threshold=percentile_threshold,
        data_centric_threshold=uncertainty_threshold,
    )
    return Experiment2(
        data_centric_method=data_centric_method,  # type: ignore
        percentile_threshold=percentile_threshold,
        data_centric_threshold=uncertainty_threshold,
        proportion_flipped=proportion_flipped,
        indices=OverlappingIndices(
            flipped=simulated_data.flipped_indices,
            hard=sculpted_data.indices.hard,
        ),
    )


def run_experiment_2(
    simulated_data: SimulatedData,
) -> List[Experiment2]:  # sourcery skip: for-append-to-extend
    """Run experiment 2 on the given dataset. Could be done faster by training
    the data-centric methods on the dataset once and then using the trained
    models to get the indices of the hard/bad data segments. If too slow then
    this is the place to start.

    Steps:
    Iterate over the data-centric methods:
        Iterate over the percentile thresholds and uncertainty thresholds:
            Save the indices of the hard/bad data segments and the indices of
            the flipped labels.

    Args:
        simulated_data (SimulatedData): The simulated data to be sculpted
        proportion_flipped (float): The proportion of labels that were flipped"""
    experiments: List[Experiment2] = []
    proportion_flipped = simulated_data.proportion_flipped
    for data_centric_method in ["dataiq", "datamaps", "cleanlab"]:
        if data_centric_method in ["dataiq", "datamaps"]:
            for percentile_threshold in [20, 30, 40, 50, 60, 70, 80]:
                experiments.append(
                    sculpt_and_get_indices_by_method(
                        simulated_data=simulated_data,
                        proportion_flipped=proportion_flipped,
                        data_centric_method=data_centric_method,  # type: ignore
                        percentile_threshold=percentile_threshold,
                        uncertainty_threshold=None,
                    ),
                )
            for uncertainty_threshold in [0.1, 0.125, 0.15, 0.175, 0.2]:
                experiments.append(
                    sculpt_and_get_indices_by_method(
                        simulated_data=simulated_data,
                        proportion_flipped=proportion_flipped,
                        data_centric_method=data_centric_method,  # type: ignore
                        percentile_threshold=None,
                        uncertainty_threshold=uncertainty_threshold,
                    ),
                )
        else:
            for uncertainty_threshold in [None, 0.1, 0.125, 0.15, 0.175, 0.2]:
                experiments.append(
                    sculpt_and_get_indices_by_method(
                        simulated_data=simulated_data,
                        proportion_flipped=proportion_flipped,
                        data_centric_method=data_centric_method,  # type: ignore
                        percentile_threshold=None,
                        uncertainty_threshold=uncertainty_threshold,
                    ),
                )
    return experiments


DATASETS_DIR = ROOT_DIR / "data" / "simulated"
if __name__ == "__main__":
    for data_dir in DATASETS_DIR.iterdir():
        experiment_groups: List[Experiment2Group] = []
        save_dir = data_dir / "results"
        save_dir.mkdir(parents=True, exist_ok=True)
        for dataset_path in data_dir.glob("*.pkl"):
            print(f"Processing {dataset_path}...")
            experiments: List[Experiment2] = []
            simulated_data = load_simulated_data(path=dataset_path)

            experiment_group = Experiment2Group(
                proportion_flipped=simulated_data.proportion_flipped,
                experiments=run_experiment_2(
                    simulated_data=simulated_data,
                ),
            )
            experiment_groups.append(experiment_group)

        # n_samples, n_features, corr_range will be the same for each directory
        # so we just use the values from the last one from the loop
        experiment_suite = Experiment2Suite(
            experiment_groups=experiment_groups,
            n_samples=simulated_data.n_features,  # type: ignore
            n_features=simulated_data.n_features,  # type: ignore
            corr_range=simulated_data.corr_range,  # type: ignore
        )
        save_to_pickle(obj=experiment_suite, path=save_dir / "experiment2.pkl")
