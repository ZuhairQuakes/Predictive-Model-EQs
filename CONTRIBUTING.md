# Contributing

Thank you for helping improve this research codebase.

## Before opening a change

1. Create the Conda environment described in [`ml4szeq/environment.yml`](ml4szeq/environment.yml), or install `.[dev,app]` for web-app work.
2. Keep generated data, run logs, and model outputs out of Git; the repository's `.gitignore` covers the standard locations.
3. Put tunable experiment settings in YAML or JSON rather than adding machine-specific paths to source files.

## Validation

Run the dependency-free repository checks before opening a pull request:

```bash
ruff check .
pytest
python tools/validate_repository.py
python -m build
```

Clear notebook execution state before committing regenerated notebooks:

```bash
python tools/strip_notebook_outputs.py ntbk/*.ipynb
```

For changes to the modelling pipeline, also run the affected workflow with a small representative dataset and record the command, random seed, and environment in the pull-request description. Do not commit credentials or Weights & Biases tokens.

Changes to predictions, preprocessing, class labels, or LRP must include a
scientific explanation, a deterministic test, and an update to
[`SCIENTIFIC_METHOD.md`](SCIENTIFIC_METHOD.md). Interactive changes should be
checked locally with `streamlit run streamlit_app.py`.

## Pull requests

Keep each pull request focused. Explain the scientific or engineering motivation, list affected datasets and scenarios, and distinguish regenerated outputs from hand-edited source. Changes that alter preprocessing, class definitions, trained weights, or reported figures should include evidence that the result remains reproducible.
