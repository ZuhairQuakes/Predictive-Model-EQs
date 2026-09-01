# Interactive XAI method and interpretation

The interactive explorer exposes selected models and feature data associated
with the published megathrust-seismicity study. It is intended for transparent
inspection and education, not prospective prediction.

## Data reconstruction

The eight files in `in-data/` contain 591 rows. The app retains the 556 rows
that are complete across the 49 published input features and `MR_GCMT`, matching
the sample count reported by the study. Region codes and along-trench segment
indices identify samples; they are not model inputs.

The original preprocessing code applies scikit-learn's Yeo–Johnson
`PowerTransformer` before selecting spatial train/test subsets. A serialized
transformer was not archived, so the app deterministically refits that
transformer on the bundled 556 complete rows. This is consistent with the
tracked algorithm and data but is explicitly described as a reconstruction,
not a bit-for-bit claim about an unarchived object.

## Model reconstruction

Each weights-only checkpoint contains three linear layers. The loader infers
the two hidden widths, reconstructs the original linear–ReLU–linear–ReLU–linear
inference graph, translates parameter names, and loads with PyTorch's
`weights_only=True` mode. Dropout is inactive during evaluation and batch
normalisation is absent from the selected checkpoints.

The study uses one-hot class labels with `BCEWithLogitsLoss`. The interface
therefore reports the original independent sigmoid class scores and selects
the maximum score for classification. These values are not a probability
distribution and are not calibrated earthquake probabilities.

Two-class labels are `Mw < 8.0` and `Mw ≥ 8.0`. Three-class labels follow the
paper notebook: `Mw < 6.4`, `6.4 ≤ Mw < 8.3`, and `Mw ≥ 8.3`.

## Layer-wise Relevance Propagation

Captum's default Layer-wise Relevance Propagation implementation attributes a
selected class output to the 49 transformed features. The interface defaults
to the winning class and reports signed relevance and Captum's completeness
delta. Positive relevance supports that score under the selected model;
negative relevance opposes it.

Relevance is not causal effect, feature sensitivity, uncertainty, or physical
importance outside that individual model evaluation. Relevance can change
under correlated features, alternative preprocessing, checkpoints, baselines,
or attribution rules.

## What-if controls

Sliders are limited to each feature's marginal 1st–99th percentile in the
bundled complete dataset. This reduces extreme extrapolation but does not keep
edited samples on the joint data manifold. A combination of individually
plausible values may still be physically impossible or absent from training.

## Responsible use

- The models classify segments using retrospective maximum observed magnitude.
- The app does not produce event times, locations, probabilities, or warnings.
- Spatial cross-validation scenario and exclusion distance must be reported.
- Input, label, and model uncertainty are not propagated by the interface.
- Scientific conclusions should use the article, archived workflow, sensitivity
  analyses, and independent validation—not the dashboard alone.
