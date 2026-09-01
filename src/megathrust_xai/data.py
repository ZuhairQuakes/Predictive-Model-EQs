"""Loading and deterministic preprocessing of the published regional features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

FEATURE_COLUMNS = (
    "DXT_200", "FDM_200", "SDM_200", "DXT_400", "FDM_400", "SDM_400",
    "DXT_600", "FDM_600", "SDM_600", "SED_AVE", "SED_STD", "SED_GRD",
    "V_UP", "V_TN", "AGE", "FRE_DG_AVE", "FRE_DG_STD", "EGR_DG_AVE",
    "EGR_DG_STD", "SRO_DG_AVE", "SRO_DG_STD", "IRO_DG_AVE", "IRO_DG_STD",
    "LRO_DG_AVE", "LRO_DG_STD", "CRD_UP_AVE", "CRD_UP_STD", "CRS_UP_AVE",
    "CRS_UP_STD", "CRM_UP_AVE", "CRM_UP_STD", "FRE_UP_AVE", "FRE_UP_STD",
    "EGO_UP_AVE", "EGO_UP_STD", "EGO_L_UP_AVE", "EGO_L_UP_STD",
    "EGO_SL_UP_AVE", "EGO_SL_UP_STD", "EGO_UM_UP_AVE", "EGO_UM_UP_STD",
    "EGR_UP_AVE", "EGR_UP_STD", "EGR_BG_UP_AVE", "EGR_BG_UP_STD",
    "INV_UP_AVE", "INV_UP_STD", "DLT_UP_AVE", "DLT_UP_STD",
)

REGION_NAMES = {
    "alu": "Alaska–Aleutians",
    "cam": "Central America",
    "izu": "Izu–Bonin–Mariana",
    "ker": "Tonga–Kermadec",
    "kur": "Japan–Kuriles–Kamchatka",
    "ryu": "Ryukyu–Nankai",
    "sam": "South America",
    "sum": "Southeast Asia",
}


@dataclass(frozen=True)
class StudyData:
    """Complete published samples and their reconstructed feature transformer."""

    frame: pd.DataFrame
    transformer: PowerTransformer
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def transform(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply the study's Yeo–Johnson power transformation to feature values."""
        if isinstance(values, pd.DataFrame):
            values = values.loc[:, self.feature_columns]
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(self.feature_columns):
            expected = len(self.feature_columns)
            raise ValueError(f"Expected {expected} features; got {array.shape[1]}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Feature values must all be finite.")
        return self.transformer.transform(array)


def load_study_data(data_directory: str | Path) -> StudyData:
    """Load the eight region files and reconstruct the published feature transform."""
    directory = Path(data_directory)
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        region = path.stem.lower()
        if region not in REGION_NAMES:
            continue
        regional = pd.read_csv(path, index_col=0)
        missing = sorted(set((*FEATURE_COLUMNS, "MR_GCMT")) - set(regional.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
        regional = regional.assign(
            REGION=region,
            REGION_NAME=REGION_NAMES[region],
            SEGMENT=regional.index.astype(int),
        )
        frames.append(regional)
    if len(frames) != len(REGION_NAMES):
        raise ValueError(f"Expected {len(REGION_NAMES)} regional CSV files; found {len(frames)}.")

    combined = pd.concat(frames, ignore_index=True)
    required = [*FEATURE_COLUMNS, "MR_GCMT"]
    complete = combined.dropna(subset=required).reset_index(drop=True)
    if complete.empty:
        raise ValueError("No complete feature rows are available.")
    transformer = PowerTransformer().fit(complete.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float))
    return StudyData(frame=complete, transformer=transformer)
