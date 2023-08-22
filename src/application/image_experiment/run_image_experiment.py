from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import numpy as np
import torch
from cleanlab.classification import CleanLearning
from medmnist import BreastMNIST
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import ImageDataLoader
from torch import nn
from torchvision.transforms import Resize
from wasabi import Printer

from application.constants import DATA_DIR
from data_centric_synth.data_models.data_sculpting import (
    SculptedImageData,
    StratifiedIndices,
)
from data_centric_synth.data_models.experiment3 import Experiment3
from data_centric_synth.data_models.experiments_common import (
    DataSegmentEvaluation,
    PerformanceEvaluation,
    Processing,
)
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    get_cleanlab_label_issue_df,
)
from data_centric_synth.experiments.models import (
    get_default_classification_model_suite,
    get_image_generative_model_suite,
)
from data_centric_synth.experiments.run_experiments import (
    calculate_metrics,
    real_data_model_evaluation_to_experiment,
)
from data_centric_synth.serialization.serialization import save_to_pickle
from data_centric_synth.utils import seed_everything


@dataclass
class ImageExperimentDataset:
    X: np.ndarray
    y: np.ndarray
    org_height: int
    name: str


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X


model_skorch = NeuralNetClassifier(CNN, max_epochs=20)


class MedNistDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.images = torch.from_numpy(X)
        self.labels = torch.from_numpy(y)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_breast_mnist(split: Literal["train", "val", "test"]) -> ImageExperimentDataset:
    dataset = BreastMNIST(split=split, download=True)
    # scale to [0, 1]
    X = dataset.imgs.astype("float32") / 255
    # add channel dimension
    X = np.expand_dims(X, 1)
    # flatten labels to 1D
    y = dataset.labels.flatten()
    return ImageExperimentDataset(X=X, y=y, org_height=28, name="breast_mnist")


def make_image_train_test_split(
    dataset: ImageExperimentDataset, test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )
    return X_train, X_test, y_train, y_test

@dataclass
class ProcessedImageDataset:
    images: np.ndarray
    labels: np.ndarray



def get_sculpted_image_data(
        X: np.ndarray,
        y: np.ndarray,
        stratified_indices: StratifiedIndices,
) -> SculptedImageData:
    X_easy = X[stratified_indices.easy]
    y_easy = y[stratified_indices.easy]

    X_hard = X[stratified_indices.hard]
    y_hard = y[stratified_indices.hard]

    return SculptedImageData(
        X_easy=X_easy,
        X_hard=X_hard,
        y_easy=y_easy,
        y_hard=y_hard,
        indices=stratified_indices,
    )



def sculpt_image_data_with_cleanlab(
        X: np.ndarray, y: np.ndarray
) -> SculptedImageData:
    label_issues = get_cleanlab_label_issue_df(X=X, y=y, model=model_skorch)
    easy_indices = np.where(~label_issues["is_label_issue"])[0]
    hard_indices = np.where(label_issues["is_label_issue"])[0]

    return get_sculpted_image_data(
        X=X,
        y=y,
        stratified_indices=StratifiedIndices(
            easy=easy_indices, ambiguous=None, hard=hard_indices
        
        )
    )

def fit_and_generate_synth_image_data(
        images: np.ndarray,
        labels: np.ndarray,
        generative_model: str,
        generative_model_params: Dict[str, Any],
) -> ProcessedImageDataset:
    dataloader = ImageDataLoader(
        MedNistDataset(X=images, y=labels),
        height=32,
    )
    generator = Plugins().get(
        generative_model, **generative_model_params
    )
    generator.fit(dataloader)

    syn_samples, syn_labels = generator.generate(count=len(labels)).unpack().tensors()
    # resize to original size
    resizer = Resize((28, 28))
    syn_samples = resizer(syn_samples)
    return ProcessedImageDataset(images=syn_samples, labels=syn_labels)


def generate_synthetic_image_data_from_sculpted_images(
        sculpted_data: SculptedImageData,
        combine: List[str],
        generative_model: str,
        generative_model_params: Dict[str, Any],
) -> ProcessedImageDataset:
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    if "easy" in combine:
        easy_data = fit_and_generate_synth_image_data(
            images=sculpted_data.X_easy,
            labels=sculpted_data.y_easy,
            generative_model=generative_model,
            generative_model_params=generative_model_params,
        )
        images.append(easy_data.images)
        labels.append(easy_data.labels)
    if "hard" in combine:
        hard_data = fit_and_generate_synth_image_data(
            images=sculpted_data.X_hard,
            labels=sculpted_data.y_hard,
            generative_model=generative_model,
            generative_model_params=generative_model_params,
        )
        images.append(hard_data.images)
        labels.append(hard_data.labels)
    
    images = np.concatenate(images, axis=0) # type: ignore
    labels = np.concatenate(labels, axis=0) # type: ignore
    return ProcessedImageDataset(
        images=images, labels=labels # typre: ignore
    )


def make_synthetic_image_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    generative_model: str,
    generative_model_params: Dict[str, Any],
) -> Dict[str, ProcessedImageDataset]:
    
    sculpted_data = sculpt_image_data_with_cleanlab(
        X=X_train,
        y=y_train,
    )
    # train generative model on all data
    full_synth_data = fit_and_generate_synth_image_data(
        images=X_train,
        labels=y_train,
        generative_model=generative_model,
        generative_model_params=generative_model_params,
    )
    # train generative model on easy and hard data separately then combine
    easy_hard_synth_data = generate_synthetic_image_data_from_sculpted_images(
        sculpted_data=sculpted_data,
        combine=["easy", "hard"],
        generative_model=generative_model,
        generative_model_params=generative_model_params,
    )
    return {
        "baseline" : full_synth_data,
        "easy_hard" : easy_hard_synth_data,
    }


def extract_subset_from_sculpted_image_data(
    data: SculptedImageData,
    subsets: List[str],
) -> ProcessedImageDataset:
    """Extracts a subset of the SculptedImageData object and returns a ProcessedImageDataset object."""
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    if "easy" in subsets:
        images.append(data.X_easy)
        labels.append(data.y_easy)
    if "hard" in subsets:
        images.append(data.X_hard)
        labels.append(data.y_hard)
    images = np.concatenate(images, axis=0) # type: ignore
    labels = np.concatenate(labels, axis=0) # type: ignore
    return ProcessedImageDataset(images=images, labels=labels) # type: ignore


def postprocess_image_datasets(
    preprocessed_datasets: Dict[str, ProcessedImageDataset],
) -> Dict[str, Dict[str, ProcessedImageDataset]]:
    postprocessed_image_datasets: Dict[str, Dict[str, ProcessedImageDataset]] = defaultdict(dict)

    for name, preprocessed_data in preprocessed_datasets.items():
        sculpted_synth_data = sculpt_image_data_with_cleanlab(
            X=preprocessed_data.images.numpy(),
            y=preprocessed_data.labels.numpy(),
        )
        # strategy 1: no postprocessing
        postprocessed_image_datasets[name]["baseline"] = extract_subset_from_sculpted_image_data(
            data=sculpted_synth_data,
            subsets=["easy", "hard"],
        )
        # strategy 2: remove hard examples
        postprocessed_image_datasets[name]["no_hard"] = extract_subset_from_sculpted_image_data(
            data=sculpted_synth_data,
            subsets=["easy"],
        )
    return postprocessed_image_datasets

def flatten_images_to_1D(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def evaluate_image_model(
        model: ClassifierMixin,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> DataSegmentEvaluation:
    pred_probs = model.predict_proba(X_test)[:, 1] # type: ignore
    metrics = calculate_metrics(y=y_test, y_pred_probs=pred_probs)
    feature_importances = (
        dict(
            zip(model.feature_names_in_, model.feature_importances_),  # type: ignore
        )
        if hasattr(model, "feature_importances_")
        else None
    )
    return DataSegmentEvaluation(
        full=PerformanceEvaluation(
        metrics=metrics,
        feature_importances=feature_importances,
    ),
    easy=None,
    ambiguous=None,
    hard=None,
    )



def fit_and_evaluate_image_model_on_original_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: Any,
        random_state: int,
) -> Experiment3:
    model.fit(X_train, y_train)
    model_performance = evaluate_image_model(model=model, X_test=X_test, y_test=y_test)
    return real_data_model_evaluation_to_experiment(
        data_centric_method="cleanlab",
        percentile_threshold=None,
        data_centric_threshold=None,
        random_state=random_state,
        test_data_segments=None,
        model=model,
        model_eval_results=model_performance,
        postprocessing_strategy="org_data",
        statistical_fidelity=None,
    )


def train_and_evaluate_image_models(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        postprocessed_datasets: Dict[str, Dict[str, ProcessedImageDataset]],
        generative_model: str,
        random_state: int,
) -> List[Experiment3]:
    experiments: List[Experiment3] = []

    classification_models = get_default_classification_model_suite(random_state=random_state)

    # train models on original data nad evaluate on test data
    ## flatten to 1D
    X_train_1D = flatten_images_to_1D(X_train)
    X_test_1D = flatten_images_to_1D(X_test)

    for model in classification_models:
        experiments.append(
            fit_and_evaluate_image_model_on_original_data(
                X_train=X_train_1D, y_train=y_train, X_test=X_test_1D, y_test=y_test, model=model, random_state=random_state
            )
        )
    
    # train models on synthetic data and evaluate on test data
    for preprocessing_strategy in postprocessed_datasets:
        preprocessing_schema = Processing(
            strategy_name=preprocessing_strategy,
            uncertainty_percentile_threshold=None,
            uncertainty_threshold=None,
            data_segments=None,
            detection_auc=None,
            statistical_fidelity=None,
        )
        for postprocessing_strategy, postprocessed_data in postprocessed_datasets[
            preprocessing_strategy
        ].items():
            # make common post- and test-processing schemas
            postprocessing_schema = Processing(
                strategy_name=postprocessing_strategy,
                uncertainty_percentile_threshold=None,
                uncertainty_threshold=None,
                data_segments=None,
                detection_auc=None,
                statistical_fidelity=None,
            )
            testprocessing_schema = Processing(
                strategy_name="cleanlab",
                uncertainty_percentile_threshold=None,
                uncertainty_threshold=None,
                data_segments=None,
                detection_auc=None,
                statistical_fidelity=None,
            )
            # train classification model suite on synthetic data
            for model in classification_models:
                # train model on synthetic data and evaluate on test data
                model.fit(  # type: ignore
                    flatten_images_to_1D(postprocessed_data.images),
                    postprocessed_data.labels,
                )
                # evaluate on full test data
                model_eval = evaluate_image_model(model=model, X_test=X_test, y_test=y_test)
                
                experiments.append(
                    Experiment3(
                        preprocessing=preprocessing_schema,
                        postprocessing=postprocessing_schema,
                        testprocessing=testprocessing_schema,
                        data_centric_method="cleanlab",
                        synthetic_model_type=generative_model,
                        classification_model_type=type(model).__name__,
                        classification_model_evaluation=model_eval,
                        dag_evaluation=None,
                        random_seed=random_state,
                    ),
                )

    return experiments


def run_image_experiment(
    dataset: ImageExperimentDataset,
    generative_model: str,
    generative_model_params: Dict[str, Any],
    random_state: int,
) -> List[Experiment3]:
    X_train, X_test, y_train, y_test = make_image_train_test_split(
        dataset=dataset, test_size=0.2, random_state=random_state
    )
    # preprocess the data
    preprocessed_datasets = make_synthetic_image_datasets(
        X_train=X_train,
        y_train=y_train,
        generative_model=generative_model,
        generative_model_params=generative_model_params,
    )
    # apply postprocessing
    postprocessed_datasets = postprocess_image_datasets(
        preprocessed_datasets=preprocessed_datasets,
    )
    # train classification models on the synth real data and evaluate on holdout data
    experiments = train_and_evaluate_image_models(
        X_train=X_train,
        y_train=y_train,
        postprocessed_datasets=postprocessed_datasets,
        X_test=X_test,
        y_test=y_test,
        generative_model=generative_model,
        random_state=random_state,
    )
    return experiments


def run_image_experiment_loop(
    datasets: Iterable[ImageExperimentDataset],
    save_dir: Path,
    random_seeds: Iterable[int],
    generative_model_suite: Iterable[str],
):
    msg = Printer(timestamp=True)

    for dataset in datasets:
        msg.info(f"Running experiments for {dataset.name} dataset")

        dataset_experiment_save_dir = save_dir / dataset.name
        msg.info(f"Saving to {dataset_experiment_save_dir}")
        dataset_experiment_save_dir.mkdir(exist_ok=True, parents=True)

        for random_seed in random_seeds:
            seed_dir = dataset_experiment_save_dir / f"seed_{random_seed}"
            seed_dir.mkdir(exist_ok=True, parents=True)
            msg.info(f"Running experiment for seed {random_seed}")
            seed_everything(seed=random_seed)

            for generative_model in generative_model_suite:
                synth_file_name = seed_dir / f"{generative_model}.pkl"
                if synth_file_name.exists():
                    msg.info(
                        f"Skipping generative model {generative_model} as it already exists in {synth_file_name}",
                    )
                    continue
                # create the file as a placeholder so that other jobs don't try to run it
                save_to_pickle(obj=None, path=synth_file_name)
                msg.info(f"Running experiment for generative model {generative_model}")
                image_experiments: List[Experiment3] = []

                image_experiments.extend(
                    run_image_experiment(
                        dataset=dataset,
                        generative_model=generative_model,
                        generative_model_params={"n_iter": 10},
                        random_state=random_seed,
                    ),
                )

                save_to_pickle(image_experiments, path=synth_file_name)


if __name__ == "__main__":

    SAVE_DIR = DATA_DIR / "image_experiment" / "breast_mnist"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    N_SEEDS = 1
    STARTING_SEED = 42
    seed_everything(seed=STARTING_SEED)

    random_seeds = np.random.randint(0, 10000, size=N_SEEDS)

    run_image_experiment_loop(
        datasets=[load_breast_mnist(split="train")],
        save_dir=SAVE_DIR,
        random_seeds=random_seeds,
        generative_model_suite=get_image_generative_model_suite(),
    )

    # dataset = load_breast_mnist(split="train")
    # cl = CleanLearning(model_skorch)
    # _ = cl.fit(X=dataset.X, y=dataset.y)
    # label_issues = cl.get_label_issues()

    # dataloader = ImageDataLoader(
    #     MedNistDataset(X=dataset.X, y=dataset.y),
    #     random_state=42,
    #     height=32,
    # )
    # generator = Plugins().get(
    #     "image_cgan", batch_size=100, plot_progress=True, n_iter=1
    # )
    # generator.fit(dataloader)

    # syn_samples, syn_labels = generator.generate(count=5).unpack().tensors()
    # from torchvision.transforms import Resize

    # resizer = Resize((dataset.org_height, dataset.org_height))
    # syn_samples = resizer(syn_samples)
    # resizer = Resize((dataset.org_height, dataset.org_height))
    # syn_samples = resizer(syn_samples)
    # resizer = Resize((dataset.org_height, dataset.org_height))
    # syn_samples = resizer(syn_samples)
    # resizer = Resize((dataset.org_height, dataset.org_height))
    # syn_samples = resizer(syn_samples)
