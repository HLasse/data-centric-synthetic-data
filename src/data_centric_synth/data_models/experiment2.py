"""Dataclasses specific to the experiment to determine which data-centric thresholds
to use."""
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS


class OverlappingIndices(BaseModel):
    flipped: np.ndarray
    hard: np.ndarray
    overlapping: Optional[np.ndarray] = None
    n_overlapping: Optional[int] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):  # noqa
        super().__init__(**data)
        if self.overlapping is None:
            self.overlapping = np.intersect1d(self.flipped, self.hard)
            self.n_overlapping = len(self.overlapping)
            self.recall = (
                self.n_overlapping / len(self.flipped) if len(self.flipped) > 0 else 0
            )
            self.precision = (
                self.n_overlapping / len(self.hard) if len(self.hard) > 0 else 0
            )
            self.f1 = (
                2 * self.recall * self.precision / (self.recall + self.precision)
                if self.recall + self.precision > 0
                else 0
            )


class Experiment2(BaseModel):
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS
    percentile_threshold: Optional[int]
    data_centric_threshold: Optional[float]
    proportion_flipped: float
    indices: OverlappingIndices

    class Config:
        arbitrary_types_allowed = True


class Experiment2Group(BaseModel):
    """An experiment group is a collection of experiments that are run on the same
    simulated dataset with different data-centric methods and thresholds."""

    proportion_flipped: float
    experiments: List[Experiment2]


class Experiment2Suite(BaseModel):
    experiment_groups: List[Experiment2Group]
    n_samples: int
    n_features: int
    corr_range: Tuple[float, float]
