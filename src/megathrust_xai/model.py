"""Loading of the selected PyTorch research checkpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_PATTERN = re.compile(
    r"ncls(?P<classes>[23])-dex(?P<distance>50|100)/"
    r"scenario(?P<scenario>[0-4])_epoch(?P<epoch>\d+)\.pt$"
)


@dataclass(frozen=True)
class CheckpointSpec:
    """Metadata encoded in a selected model checkpoint path."""

    classes: int
    exclusion_distance_km: int
    scenario: int
    epoch: int
    path: Path

    @classmethod
    def from_path(cls, path: str | Path) -> CheckpointSpec:
        checkpoint = Path(path)
        normalized = checkpoint.as_posix()
        match = CHECKPOINT_PATTERN.search(normalized)
        if match is None:
            raise ValueError(f"Unrecognized checkpoint path: {checkpoint}")
        return cls(
            classes=int(match.group("classes")),
            exclusion_distance_km=int(match.group("distance")),
            scenario=int(match.group("scenario")),
            epoch=int(match.group("epoch")),
            path=checkpoint,
        )


class PublishedNetwork(nn.Sequential):
    """Inference-equivalent two-hidden-layer network used by the checkpoints."""

    def __init__(self, inputs: int, hidden: tuple[int, int], outputs: int) -> None:
        super().__init__(
            nn.Linear(inputs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], outputs),
        )


def load_checkpoint(path: str | Path) -> tuple[PublishedNetwork, CheckpointSpec]:
    """Infer architecture from and safely load a weights-only checkpoint."""
    spec = CheckpointSpec.from_path(path)
    state = torch.load(spec.path, map_location="cpu", weights_only=True)
    required = {
        "inp.fc.weight",
        "inp.fc.bias",
        "layers.0.fc.weight",
        "layers.0.fc.bias",
        "out_cat.weight",
        "out_cat.bias",
    }
    if set(state) != required:
        missing = sorted(required - set(state))
        extra = sorted(set(state) - required)
        raise ValueError(f"Unexpected checkpoint parameters; missing={missing}, extra={extra}")

    input_count = state["inp.fc.weight"].shape[1]
    hidden = (
        state["inp.fc.weight"].shape[0],
        state["layers.0.fc.weight"].shape[0],
    )
    output_count = state["out_cat.weight"].shape[0]
    if output_count != spec.classes:
        raise ValueError("Checkpoint output width does not match its directory metadata.")
    model = PublishedNetwork(input_count, hidden, output_count)
    translated = {
        "0.weight": state["inp.fc.weight"],
        "0.bias": state["inp.fc.bias"],
        "2.weight": state["layers.0.fc.weight"],
        "2.bias": state["layers.0.fc.bias"],
        "4.weight": state["out_cat.weight"],
        "4.bias": state["out_cat.bias"],
    }
    model.load_state_dict(translated)
    model.eval()
    return model, spec
