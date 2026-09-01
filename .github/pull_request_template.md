## Summary

Describe the change and its scientific or software motivation.

## Scientific integrity

- [ ] Model, data, preprocessing, and target-class assumptions are documented.
- [ ] Explanation changes include a deterministic validation case.
- [ ] Scores are not described as calibrated earthquake probabilities.
- [ ] Data and model provenance and reuse terms are respected.

## Verification

- [ ] `ruff check .`
- [ ] `pytest`
- [ ] `python tools/validate_repository.py`
- [ ] `python -m build`
- [ ] Interactive page inspected locally when UI behaviour changed.
