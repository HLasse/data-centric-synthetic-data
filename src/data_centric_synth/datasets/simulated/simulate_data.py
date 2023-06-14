"""Code for simulating data with varying correlation between features."""

import itertools
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_centric_synth.data_models.experiments_common import SimulatedData
from data_centric_synth.serialization.serialization import save_to_pickle


def simulate_data_with_outcome(
    n_features: int,
    y: np.ndarray,
    corr_range: Tuple[float, float],
) -> np.ndarray:
    """Simulate data with varying correlation with a binary outcome variable.

    Args:
        n_features (int): The number of features to simulate.
        y (np.ndarray): The outcome variable.
        corr_range (Tuple(float, float)): The range of correlation values to simulate.

    Returns:
        np.ndarray: The simulated data.
    """
    n_samples = y.shape[0]
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        corr = np.random.uniform(low=corr_range[0], high=corr_range[1])
        # the larger the correlation, the more the feature should be weighted
        feature = corr * y + np.random.normal(
            scale=np.sqrt(1 - corr**2),
            size=n_samples,
        )
        X[:, i] = feature
    return X


def test_performance_on_data(X: np.ndarray, y: np.ndarray) -> None:
    """Test the performance of a model on the simulated data."""
    model = XGBClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Model score: ")
    print(model.score(X_test, y_test))
    print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


def flip_outcome(
    y: Union[np.ndarray, pd.Series], prop_to_flip: float
) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]]:
    """Flip a proportion of the outcome variable from 0 to 1 and vice versa."""
    y_flipped = y.copy()
    indices_to_flip = np.random.choice(
        y.size,
        int(y.size * prop_to_flip),
        replace=False,
    )
    # flip the values at the selected indices
    y_flipped[indices_to_flip] = 1 - y_flipped[indices_to_flip]
    return y_flipped, indices_to_flip


def save_as_simulated_data(
    X: np.ndarray,
    y: np.ndarray,
    flipped_indices: np.ndarray,
    proportion_flipped: float,
    save_path: Path,
    n_samples: int,
    n_features: int,
    corr_range: Tuple[float, float],
) -> None:
    """Save the simulated data as a pandas DataFrame."""
    df = pd.DataFrame(X)
    df["target"] = y
    simulated_data = SimulatedData(
        dataset=df,
        flipped_indices=flipped_indices,
        proportion_flipped=proportion_flipped,
        n_samples=n_samples,
        n_features=n_features,
        corr_range=corr_range,
    )
    save_to_pickle(simulated_data, save_path)


def simulate_datasets(
    n_samples: int,
    n_features: int,
    save_dir: Path,
    corr_range: Tuple[float, float] = (-0.7, 0.7),
    max_flip_prop: float = 0.1,
    min_flip_prop: float = 0.02,
    step: float = 0.02,
):
    save_base_name = (
        f"features-{n_features}_samples-{n_samples}_corrange-{corr_range[1]}"
    )
    y = np.random.binomial(n=1, p=0.5, size=n_samples)
    X = simulate_data_with_outcome(
        n_features=n_features,
        y=y,
        corr_range=(-0.7, 0.7),
    )

    test_performance_on_data(X, y)
    # save the data
    save_as_simulated_data(
        X=X,
        y=y,
        flipped_indices=np.array([]),
        proportion_flipped=0.0,
        save_path=save_dir / f"{save_base_name}_flipped-0.pkl",
        n_samples=n_samples,
        n_features=n_features,
        corr_range=corr_range,
    )

    for prop_to_flip in np.arange(min_flip_prop, max_flip_prop + step, step):
        # flip some of the outcome values from 0 to 1 and vice versa
        y_flipped, flipped_indices = flip_outcome(y=y, prop_to_flip=prop_to_flip)
        print(f"Flipped {prop_to_flip * 100}% of the outcome values")
        test_performance_on_data(X, y_flipped)
        save_as_simulated_data(
            X=X,
            y=y_flipped,
            flipped_indices=flipped_indices,
            proportion_flipped=prop_to_flip,
            save_path=save_dir / f"{save_base_name}_flipped-{prop_to_flip}.pkl",
            n_samples=n_samples,
            n_features=n_features,
            corr_range=corr_range,
        )


if __name__ == "__main__":
    BASE_DATA_SAVE_PATH = Path("data") / "simulated"
    np.random.seed(42)
    n_samples_options = [10_000, 50_000]
    n_features_options = [10, 50]
    corr_range = (-0.7, 0.7)

    # make all combinations of samples and features with itertools
    for n_samples, n_features in itertools.product(
        n_samples_options,
        n_features_options,
    ):
        save_dir = (
            BASE_DATA_SAVE_PATH
            / f"features-{n_features}_samples-{n_samples}_corrange-{corr_range[1]}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        simulate_datasets(
            n_samples=n_samples,
            n_features=n_features,
            save_dir=save_dir,
            corr_range=corr_range,
        )
