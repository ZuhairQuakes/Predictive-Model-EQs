from __future__ import annotations

from pathlib import Path

import numpy as np

from megathrust_xai.data import FEATURE_COLUMNS, load_study_data
from megathrust_xai.inference import explain_lrp, predict
from megathrust_xai.model import CheckpointSpec, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/ncls2-dex100/scenario0_epoch70.pt"


def test_checkpoint_metadata_and_prediction() -> None:
    model, spec = load_checkpoint(CHECKPOINT)
    study = load_study_data(ROOT / "in-data")
    values = study.transform(study.frame.loc[0, list(FEATURE_COLUMNS)].to_numpy())

    prediction = predict(model, values)

    assert spec == CheckpointSpec(2, 100, 0, 70, CHECKPOINT)
    assert prediction.scores.shape == (2,)
    assert np.all((prediction.scores >= 0) & (prediction.scores <= 1))
    assert prediction.predicted_class in (0, 1)


def test_lrp_returns_one_relevance_per_feature() -> None:
    model, _ = load_checkpoint(CHECKPOINT)
    study = load_study_data(ROOT / "in-data")
    values = study.transform(study.frame.loc[0, list(FEATURE_COLUMNS)].to_numpy())
    target = predict(model, values).predicted_class

    explanation = explain_lrp(model, values, target)

    assert explanation.relevance.shape == (49,)
    assert np.isfinite(explanation.relevance).all()
    assert np.isfinite(explanation.convergence_delta)


def test_all_published_checkpoints_load() -> None:
    checkpoints = sorted((ROOT / "models").glob("ncls*-dex*/*.pt"))

    assert len(checkpoints) == 20
    for checkpoint in checkpoints:
        model, spec = load_checkpoint(checkpoint)
        assert model[0].in_features == len(FEATURE_COLUMNS)
        assert model[-1].out_features == spec.classes
