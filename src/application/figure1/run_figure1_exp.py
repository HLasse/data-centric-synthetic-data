"""Script to run the experiment to produce figure 1 in the main paper"""
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from pydantic import BaseModel
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from synthcity.metrics._utils import get_frequency
from xgboost import XGBClassifier

from application.constants import (
    DATA_CENTRIC_THRESHOLDS,
    DATA_DIR,
    RESULTS_DIR,
    SYNTHETIC_MODEL_PARAMS,
)
from application.synthcity_hparams.optimize_model_hparams import uint_cols_to_int
from data_centric_synth.data_models.data_sculpting import StratifiedIndices
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    sculpt_data_by_method,
    stratify_and_sculpt_with_dataiq,
)
from data_centric_synth.dataiq.dataiq_gradient_boosting import DataIQGradientBoosting
from data_centric_synth.datasets.adult.load_adult import load_adult_dataset
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS
from data_centric_synth.experiments.run_experiments import make_train_test_split
from data_centric_synth.serialization.serialization import save_to_pickle
from data_centric_synth.synthetic_data.synthetic_data_generation import (
    fit_and_generate_synth_data,
)


class POCExperiment(BaseModel):
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS
    synthetic_model: str
    dataset: str
    indices: StratifiedIndices
    inv_kl_divergence: float
    auc: float


def sculpt_data_and_get_indices_by_method(
    X: pd.DataFrame,
    y: pd.Series,
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
) -> StratifiedIndices:
    """Sculpt the data using the data-centric method to get the indices of the
    hard/bad data segments and return an Experiment2 object.

    Args:
        X (pd.DataFrame): The features of the dataset
        y (pd.Series): The target of the dataset
        proportion_flipped (float): The proportion of labels that were flipped
        synthetic_model (str): The synthetic model used to generate the data
        data_centric_method (Literal["dataiq", "datamaps", "cleanlab"]): The
            data-centric method to use.
        percentile_threshold (Optional[int]): The percentile threshold to use for
            the data-centric method.
        data_centric_threshold (Optional[float]): The uncertainty threshold to use
            for the data-centric method

        Returns:
            StratifiedIndices: The indices of the easy/ambiguous/hard data segments"""
    sculpted_data = sculpt_data_by_method(
        X=X,
        y=y,
        data_centric_method=data_centric_method,  # type: ignore
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )
    return sculpted_data.indices


def get_inv_kl_divergence(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> float:
    freqs = get_frequency(X_gt=real_data, X_synth=synthetic_data, n_histogram_bins=10)
    res = []
    for col in real_data.columns:
        real_freq, synth_freq = freqs[col]
        res.append(1 / (1 + np.sum(kl_div(real_freq, synth_freq))))
    return np.mean(res)  # type: ignore


def get_classification_performance(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Train XGBoost on the synthetic data and evaluate with roc_auc on the real data."""
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # type: ignore


def get_stratified_indices_from_trained_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: XGBClassifier,
    data_centric_method: Literal["cleanlab", "dataiq"],
) -> StratifiedIndices:
    """Sculpt the data using the data-centric method to get the indices of the
    hard/bad data segments.
    """
    if data_centric_method == "cleanlab":
        pred_probs = model.predict_proba(X)
        label_issues = find_label_issues(labels=y, pred_probs=pred_probs)
        indices = StratifiedIndices(
            easy=np.where(label_issues != True)[0],
            ambiguous=None,
            hard=np.where(label_issues)[0],
        )
    elif data_centric_method == "dataiq":
        dataiq = DataIQGradientBoosting()
        dataiq.evaluate_gradient_boosting(
            model=model,
            X=X,
            y=y,
        )
        indices = stratify_and_sculpt_with_dataiq(
            X=X,
            y=y,
            dataiq=dataiq,
            percentile_threshold=DATA_CENTRIC_THRESHOLDS[
                data_centric_method
            ].percentile_threshold,
            data_centric_threshold=DATA_CENTRIC_THRESHOLDS[
                data_centric_method
            ].data_centric_threshold,
            method="dataiq",
        ).indices
    return indices


def load_and_preprocess_covid_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the brazilian covid dataset from https://doi.org/10.1016/S2214-109X(20)30285-0.
    Preprocess to turn it into a classification problem."""
    time_horizon = 14
    df = pd.read_csv(
        "https://raw.githubusercontent.com/vanderschaarlab/synthetic-data-lab/main/data/Brazil_COVID/covid_normalised_numericalised.csv",
    )

    df.loc[
        (df["Days_hospital_to_outcome"] <= time_horizon) & (df["is_dead"] == 1),
        f"is_dead_at_time_horizon={time_horizon}",
    ] = 1
    df.loc[
        (df["Days_hospital_to_outcome"] > time_horizon),
        f"is_dead_at_time_horizon={time_horizon}",
    ] = 0
    df.loc[(df["is_dead"] == 0), f"is_dead_at_time_horizon={time_horizon}"] = 0
    df[f"is_dead_at_time_horizon={time_horizon}"] = df[
        f"is_dead_at_time_horizon={time_horizon}"
    ].astype(int)

    df = df.drop(
        columns=["is_dead", "Days_hospital_to_outcome"],
    )  # drop survival columns as they are not needed for a classification problem

    # y = is_dead_at_time_horizon=14
    # rename is_dead_at_time_horizon=14 to target
    df = df.rename(columns={f"is_dead_at_time_horizon={time_horizon}": "target"})
    y = df["target"]
    X = df.drop(columns=["target"])
    X = uint_cols_to_int(X)
    return X, y


def load_and_preprocess_adult_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = load_adult_dataset()
    df = uint_cols_to_int(df)
    # convert category columns to int
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].cat.codes
    # rename salary to target
    df = df.rename(columns={"salary": "target"})
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


dataset_loaders = {
    "adult": load_and_preprocess_adult_dataset,
    "covid": load_and_preprocess_covid_dataset,
}


def get_cleanlab_oof_indices(X: pd.DataFrame, y: pd.Series) -> StratifiedIndices:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 2))
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        cleanlab_model = XGBClassifier()
        cleanlab_model.fit(X_train, y_train)
        oof_preds[test_idx] = cleanlab_model.predict_proba(X_test)
    label_issues = find_label_issues(labels=y, pred_probs=oof_preds)
    cleanlab_indices = StratifiedIndices(
        easy=np.where(label_issues != True)[0],
        ambiguous=None,
        hard=np.where(label_issues)[0],
    )
    return cleanlab_indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to run POC on. One of 'adult' or 'covid'",
    )
    args = parser.parse_args()

    DATASET: Literal["adult", "covid"] = args.dataset

    dataset_loader = dataset_loaders[DATASET]
    X, y = dataset_loader()
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    #save_dir = RESULTS_DIR / "poc" / DATASET
    save_dir = DATA_DIR / "figure1" / DATASET

    save_dir.mkdir(exist_ok=True, parents=True)

    full_model = XGBClassifier()
    full_model.fit(X_train, y_train)

    # get OOF predictions for cleanlab
    # run 4-fold CV and save out of fold pred probs
    cleanlab_indices = get_cleanlab_oof_indices(X=X_train, y=y_train)
    dataiq_indicies = get_stratified_indices_from_trained_model(
        X=X_train,
        y=y_train,
        model=full_model,
        data_centric_method="dataiq",
    )

    original_data: List[POCExperiment] = []
    for data_centric_method, indices in zip(
        ["cleanlab", "dataiq"],
        [cleanlab_indices, dataiq_indicies],
    ):
        original_data.append(
            POCExperiment(
                data_centric_method=data_centric_method,  # type: ignore
                synthetic_model="Real data",
                dataset=DATASET,
                indices=indices,
                inv_kl_divergence=1.0,
                auc=get_classification_performance(
                    X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train
                ),
            )
        )

    save_to_pickle(original_data, save_dir / "original_data.pkl")

    # fit synthetic models to the original data and run cleanlab and dataiq on the
    # synthetic data
    for synthetic_model in [
        "marginal_distributions"
    ]:  # get_default_synthetic_model_suite():
        synth_experiments: List[POCExperiment] = []
        synthetic_data = fit_and_generate_synth_data(
            data=pd.concat([X_train, y_train], axis=1),
            model_name=synthetic_model,
            model_params={
                **SYNTHETIC_MODEL_PARAMS[synthetic_model],
                "random_state": 42,
            },
            target_column="target",
        )
        for data_centric_method in ["cleanlab", "dataiq"]:
            synth_experiments.append(
                POCExperiment(
                    data_centric_method=data_centric_method,  # type: ignore
                    synthetic_model=synthetic_model,
                    dataset=DATASET,
                    indices=get_stratified_indices_from_trained_model(
                        X=synthetic_data.drop(columns=["target"]),
                        y=synthetic_data["target"],
                        model=full_model,
                        data_centric_method=data_centric_method,  # type: ignore
                    ),
                    inv_kl_divergence=get_inv_kl_divergence(
                        real_data=pd.concat([X_train, y_train], axis=1),
                        synthetic_data=synthetic_data,
                    ),
                    auc=get_classification_performance(
                        X_train=synthetic_data.drop(columns=["target"]),
                        y_train=synthetic_data["target"],
                        X_test=X_test,
                        y_test=y_test,
                    ),
                )
            )
        save_to_pickle(synth_experiments, save_dir / f"{synthetic_model}.pkl")
