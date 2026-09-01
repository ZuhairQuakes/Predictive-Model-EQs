from __future__ import annotations

from pathlib import Path

import numpy as np

from megathrust_xai.data import FEATURE_COLUMNS, load_study_data

ROOT = Path(__file__).resolve().parents[1]


def test_published_data_shape_and_transform() -> None:
    study = load_study_data(ROOT / "in-data")

    assert study.frame.shape[0] == 556
    assert len(study.feature_columns) == 49
    transformed = study.transform(study.frame.loc[:, FEATURE_COLUMNS])
    assert transformed.shape == (556, 49)
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-7)


def test_single_feature_row_transform() -> None:
    study = load_study_data(ROOT / "in-data")
    transformed = study.transform(study.frame.loc[0, list(FEATURE_COLUMNS)].to_numpy())
    assert transformed.shape == (1, 49)
