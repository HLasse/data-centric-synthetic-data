from pathlib import Path
from typing import Optional

from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"
POSTPROCESSING_ONLY_SAVE_DIR = DATA_DIR / "main_experiment" / "org_data_postprocessing"
NOISE_DATASET_POSTPROCESSING_DIR = POSTPROCESSING_ONLY_SAVE_DIR / "covid"
MAIN_EXP_POSTPROCESSING_DIR = POSTPROCESSING_ONLY_SAVE_DIR / "main_experiment"


class DataCentricThresholds(BaseModel):
    data_centric_threshold: Optional[float]
    percentile_threshold: Optional[int]


DATA_CENTRIC_THRESHOLDS = {
    "cleanlab": DataCentricThresholds(
        data_centric_threshold=0.2,
        percentile_threshold=None,
    ),
    "dataiq": DataCentricThresholds(
        data_centric_threshold=0.2,
        percentile_threshold=None,
    ),
    "datamaps": DataCentricThresholds(
        data_centric_threshold=0.2,
        percentile_threshold=None,
    ),
}

SYNTHETIC_MODEL_PARAMS = {
    "marginal_distributions": {},
    "tvae": {
        "n_iter": 300,
        "lr": 0.0002,
        "decoder_n_layers_hidden": 4,
        "weight_decay": 0.001,
        "batch_size": 256,
        "n_units_embedding": 200,
        "decoder_n_units_hidden": 300,
        "decoder_nonlin": "elu",
        "decoder_dropout": 0.194325119117226,
        "encoder_n_layers_hidden": 1,
        "encoder_n_units_hidden": 450,
        "encoder_nonlin": "leaky_relu",
        "encoder_dropout": 0.04288563703094718,
    },
    "ctgan": {
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 50,
        "generator_nonlin": "tanh",
        "n_iter": 1000,
        "generator_dropout": 0.0574657940165757,
        "discriminator_n_layers_hidden": 4,
        "discriminator_n_units_hidden": 150,
        "discriminator_nonlin": "relu",
        "discriminator_n_iter": 3,
        "discriminator_dropout": 0.08727454632095322,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 500,
        "encoder_max_clusters": 14,
    },
    "bayesian_network": {
        "struct_learning_search_method": "hillclimb",
        "struct_learning_score": "bic",
    },
    "nflow": {
        "n_iter": 1000,
        "n_layers_hidden": 10,
        "n_units_hidden": 98,
        "dropout": 0.11496088236749386,
        "batch_norm": True,
        "lr": 0.0001,
        "linear_transform_type": "permutation",
        "base_transform_type": "rq-autoregressive",
        "batch_size": 512,
    },
    "ddpm": {
        "lr": 0.0009375080542687667,
        "batch_size": 2929,
        "num_timesteps": 998,
        "n_iter": 1051,
        "is_classification": True,
    },
}
