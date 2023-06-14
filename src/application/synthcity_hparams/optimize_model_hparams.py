"""Code to run the hyperparameter optimization of synthetic data generation models."""
import argparse
from functools import partial

import optuna
import pandas as pd
from application.constants import RESULTS_DIR
from data_centric_synth.datasets.adult.load_adult import load_adult_dataset
from data_centric_synth.serialization.serialization import save_to_pickle
from synthcity.benchmark import Benchmarks
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.optuna_sample import suggest_all


def objective(trial: optuna.Trial, model_name: str) -> float:
    """Optuna objective function for hyperparameter optimization of synthetic
    data generation models."""
    hp_space = Plugins().get(model_name).hyperparameter_space()
    params = suggest_all(trial=trial, distributions=hp_space)
    if "batch_size" in params:
        params["batch_size"] = 512
    if model_name == "ddpm":
        params["is_classification"] = True
    trial_id = f"trial_{trial.number}"
    try:
        report = Benchmarks.evaluate(
            [(trial_id, model_name, params)],
            loader,
            repeats=1,
            metrics={"detection": ["detection_xgb"]},
        )
    except Exception as e:  # invalid set of params
        print(f"{type(e).__name__}: {e}")
        print(params)
        raise optuna.TrialPruned
    return report[trial_id]["mean"].mean()


def uint_cols_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "uint8":
            df[col] = df[col].astype(int)
    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, required=True)
    args = argparser.parse_args()
    model_name = args.model_name

    df = load_adult_dataset()
    df = uint_cols_to_int(df)
    save_dir = RESULTS_DIR / "hparams"
    save_dir.mkdir(exist_ok=True, parents=True)

    loader = GenericDataLoader(
        df,
        target_column="salary",
    )

    print(f"Optimizing hyperparameters for {model_name}")
    study = optuna.create_study(direction="minimize")
    obj_fun = partial(objective, model_name=model_name)
    study.optimize(obj_fun, n_trials=20)
    save_to_pickle(study, path=save_dir / f"{model_name}.pkl")
