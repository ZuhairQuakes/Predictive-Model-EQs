# Explainable AI for megathrust seismicity

[![Repository quality](https://github.com/ZuhairQuakes/Predictive-Model-EQs/actions/workflows/repository-quality.yml/badge.svg)](https://github.com/ZuhairQuakes/Predictive-Model-EQs/actions/workflows/repository-quality.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13219207.svg)](https://doi.org/10.5281/zenodo.13219207)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

Research code, regional inputs, trained weights, and analysis notebooks for **“Testing Driving Mechanisms of Megathrust Seismicity With Explainable Artificial Intelligence.”** The project classifies subduction-zone segments by their largest observed earthquake and uses Layer-wise Relevance Propagation (LRP) to identify the features that drive each prediction.

## Interactive XAI explorer

The repository now includes an interactive research companion that lets users:

- select any of the 20 published checkpoints;
- explore 556 complete segments across eight subduction regions;
- inspect independent model class scores;
- calculate local Layer-wise Relevance Propagation explanations;
- perturb up to six features and observe the resulting prediction and relevance changes; and
- compare a selected segment with regional feature distributions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

The app is an explanatory research interface, not a forecast or operational
hazard product. Its fitted `PowerTransformer` is deterministically reconstructed
from the bundled 556 complete samples because the original serialized
transformer was not archived. Read [`SCIENTIFIC_METHOD.md`](SCIENTIFIC_METHOD.md)
for the resulting interpretation boundary.

## Scientific scope

The dataset represents 556 segments from eight subduction regions using 49 physical-state, dynamic, kinematic, and age features. The fully connected network separates:

- `C0`: maximum observed magnitude `Mw < 8.0`
- `C1`: maximum observed magnitude `Mw >= 8.0`

The analysis supports established links with interface curvature, sediment thickness, and long-wavelength bathymetric roughness. It also highlights slab-depth derivatives used as proxies for trench-parallel stress, particularly near slab steps and edges.

![Model architecture](images/model-architecture.png)

## Repository map

| Path | Purpose |
| --- | --- |
| `ml4szeq/src/` | preprocessing, model training, validation, and prediction code |
| `ml4szeq/parameters/` | default and sweep hyperparameters |
| `ml4szeq/environment.yml` | Conda environment for the modelling workflow |
| `models/` | selected trained PyTorch checkpoints by class setup and data exclusion scenario |
| `ntbk/` | sampling, classification-map, and LRP-map notebooks |
| `in-data/` | regional CSV inputs used by the mapping workflow |
| `helper_pkg/` | reusable geometry and focal-mechanism helpers |
| `images/` | figures used in this documentation |
| `src/megathrust_xai/` | tested checkpoint inference, preprocessing, LRP, and app code |
| `streamlit_app.py` | Streamlit Community Cloud and local application entry point |
| `tests/` | data, checkpoint, explanation, and application smoke tests |

## Quick start

Clone the repository and run the dependency-free integrity checks:

```bash
git clone https://github.com/ZuhairQuakes/Predictive-Model-EQs.git
cd Predictive-Model-EQs
python tools/validate_repository.py
```

Create the modelling environment:

```bash
conda env create -f ml4szeq/environment.yml
conda activate earthquakes
cd ml4szeq
python src/script.py --sep 0 --reg 0
```

The training pipeline expects prepared datasets under `ml4szeq/data/<dataset-name>/`. Download the archived research bundle from [Zenodo](https://doi.org/10.5281/zenodo.13219207) when reproducing the paper. Generated datasets, run logs, and outputs are intentionally ignored by Git.

`ml4szeq/config.json` now uses the repository-relative project root. To use a machine-specific configuration without changing the tracked file, point `ML4SZEQ_CONFIG` at your own JSON file:

```bash
ML4SZEQ_CONFIG=/path/to/config.json python src/script.py --sep 0 --reg 0
```

The mapping notebooks also depend on the geographic source files and map configuration used in the archived workflow. Copy the variable names from [`.env.example`](.env.example) into your shell environment, then start Jupyter from the repository root so tracked relative paths resolve consistently.

## Deploy the web page

For Streamlit Community Cloud, create an app from this repository and set the
entry point to `streamlit_app.py`; [`requirements.txt`](requirements.txt)
installs the app extra. No credentials are required. A container deployment is
also supported:

```bash
docker build -t megathrust-xai .
docker run --rm -p 8501:8501 megathrust-xai
```

Then open `http://localhost:8501`.

## Reproducibility notes

- Record the Git commit, environment export, dataset DOI/version, scenario, region split, separation distance, and random seed for every run.
- Weights & Biases is disabled in the tracked configuration. Authentication tokens must remain outside the repository.
- Tracked `.pt` files are research artifacts. New checkpoints belong in `ml4szeq/out/` unless deliberately selected for release.
- The quality workflow validates Python syntax, notebook/JSON structure, and local documentation links without downloading the full scientific environment.
- Class outputs are independent sigmoid scores from the original BCE-with-logits setup; they are not calibrated probabilities and need not sum to one.
- LRP relevance is local to the selected checkpoint, segment, target class, and preprocessing reconstruction. It does not imply causality.

## Citation

Use GitHub's **Cite this repository** menu or [`CITATION.cff`](CITATION.cff). The associated article is:

> Graciosa, J. C., Capitanio, F. A., Beall, A., Hargreaves, M., Gollapalli, T., Tang, T., & Zuhair, M. (2025). Testing Driving Mechanisms of Megathrust Seismicity With Explainable Artificial Intelligence. *Journal of Geophysical Research: Solid Earth, 130*(1), e2024JB028774. https://doi.org/10.1029/2024JB028774

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the validation and review expectations.

Software is available under the [MIT License](LICENSE). Dataset and trained-model
reuse must also follow the terms of the associated archive and source datasets.
