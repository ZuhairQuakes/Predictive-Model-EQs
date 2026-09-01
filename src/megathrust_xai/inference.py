"""Prediction and Layer-wise Relevance Propagation interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from megathrust_xai.model import PublishedNetwork


@dataclass(frozen=True)
class Prediction:
    """Independent sigmoid class scores from a published checkpoint."""

    logits: np.ndarray
    scores: np.ndarray
    predicted_class: int


@dataclass(frozen=True)
class Explanation:
    """LRP relevance values and completeness delta for one target class."""

    relevance: np.ndarray
    convergence_delta: float
    target_class: int


def _input_tensor(transformed_features: np.ndarray) -> torch.Tensor:
    values = np.asarray(transformed_features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Expected one finite transformed feature row.")
    return torch.from_numpy(values)


def predict(model: PublishedNetwork, transformed_features: np.ndarray) -> Prediction:
    """Return the checkpoint's independent sigmoid scores and winning class."""
    tensor = _input_tensor(transformed_features)
    with torch.no_grad():
        logits = model(tensor)
        scores = torch.sigmoid(logits)
    score_values = scores[0].numpy()
    return Prediction(
        logits=logits[0].numpy(),
        scores=score_values,
        predicted_class=int(np.argmax(score_values)),
    )


def explain_lrp(
    model: PublishedNetwork,
    transformed_features: np.ndarray,
    target_class: int,
) -> Explanation:
    """Explain one class score with Captum's Layer-wise Relevance Propagation."""
    try:
        from captum.attr import LRP
    except ImportError as exc:  # pragma: no cover - only when optional extra is absent
        raise RuntimeError("Install the 'app' extra to calculate LRP explanations.") from exc
    if not 0 <= target_class < model[-1].out_features:
        raise ValueError("target_class is outside the model output range.")
    tensor = _input_tensor(transformed_features).requires_grad_(True)
    attribution, delta = LRP(model).attribute(
        tensor,
        target=target_class,
        return_convergence_delta=True,
    )
    return Explanation(
        relevance=attribution.detach().numpy()[0],
        convergence_delta=float(delta.detach().numpy()[0]),
        target_class=target_class,
    )
