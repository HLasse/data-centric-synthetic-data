import pandas as pd
from pydantic import BaseModel


class PerformanceDfs(BaseModel):
    classification: pd.DataFrame
    model_selection: pd.DataFrame
    feature_selection: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


class NoisePerformanceDfs(BaseModel):
    classification: pd.DataFrame
    model_selection: pd.DataFrame
    feature_selection: pd.DataFrame
    statistical_fidelity: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True