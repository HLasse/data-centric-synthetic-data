# load most classifiers from sklearn
from typing import List, Literal

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_default_classification_model_suite(random_state: int) -> List[ClassifierMixin]:
    """Load a default suite of classification models."""

    return [
        XGBClassifier(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        KNeighborsClassifier(),
        SVC(random_state=random_state, probability=True),
        GaussianNB(),
        MLPClassifier(random_state=random_state),
    ]


def get_default_synthetic_model_suite() -> List[str]:
    """Loads the names of the default synthetic models to be used in experiments"""
    return [
        "tvae",
        "ctgan",
        "bayesian_network",
        "nflow",
        "ddpm",
    ]


IMPLEMENTED_DATA_CENTRIC_METHODS = Literal["dataiq", "datamaps", "cleanlab"]
