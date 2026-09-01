"""Interactive inference and explanation tools for megathrust seismicity models."""

from megathrust_xai.data import FEATURE_COLUMNS, REGION_NAMES, StudyData, load_study_data
from megathrust_xai.inference import Prediction, predict
from megathrust_xai.model import CheckpointSpec, PublishedNetwork, load_checkpoint

__all__ = [
    "FEATURE_COLUMNS",
    "REGION_NAMES",
    "CheckpointSpec",
    "Prediction",
    "PublishedNetwork",
    "StudyData",
    "load_checkpoint",
    "load_study_data",
    "predict",
]
__version__ = "0.1.0"
