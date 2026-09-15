# Allen reference t-type AP prediction

## Scope

This analysis predicts a compact median AP cycle and AP waveform features for
the Allen Mouse Whole Cortex and Hippocampus SMART-seq reference taxonomy. The
official cluster trimmed means are used because they share the SMART-seq assay
and 2021 CTX-HPF taxonomy with the Patch-seq labels more closely than the
550-gene whole-brain MERFISH panel.

Only neural reference clusters are retained. The local expression table
contains 358 neural leaves after removing Oligo, Astro, Endo, SMC-Peri, VLMC,
and Micro-PVM clusters.

Reference data:

- https://brain-map.org/our-research/cell-types-taxonomies/cell-types-database-rna-seq-data/mouse-whole-cortex-and-hippocampus-smart-seq
- https://doi.org/10.1016/j.cell.2021.04.021

## Prediction targets

Two related predictors are fit separately for each recording-temperature
cohort.

1. A coherent q50 spike-cycle predictor estimates resting voltage, ten Fourier
   harmonics, downstroke and upstroke durations, and period. Traces and phase
   loops are reconstructed from these parameters.
2. A direct AP-feature predictor estimates 15 waveform and phase-loop features.
   This avoids amplification of small Fourier errors when calculating dV/dt.

Both predictors use class-residualized sparse reduced-rank regression. Atlas
cluster means are affine-aligned gene by gene to Patch-seq t-type centroids.
All displayed predictions are clipped to the 0.5-99.5 percentile training
envelope. Transcriptomic support is measured in a whitened PCA space of
Patch-seq t-type centroids.

## Extrapolation test

**Exploratory benchmark:** hyperparameters were selected using full-cohort
donor cross-validation before the leave-type-out refits. Consequently the
held-out type influenced hyperparameter selection. A fully nested rerun is
required before interpreting these numbers as unbiased extrapolation scores.

True atlas-only leaves have expression but no ephys ground truth. Therefore,
their accuracy cannot be measured directly. The benchmark removes one complete
Patch-seq t-type at a time, refits gene selection and RRR without that type,
predicts its mean ephys from its RNA centroid, and compares the result with the
observed mean. Types with fewer than three paired cells are omitted from the
primary benchmark.

This is stricter than holding out cells while retaining other cells of the same
t-type in training.

Targets are features of fitted model templates, not independent measurements
freshly extracted from raw APs. Held-out inputs are Patch-seq RNA centroids;
this does not test atlas-to-Patch-seq transfer. Gene-wise atlas alignment and
support distances are heuristic, and intervals lack external calibration.
The q50 stimulus is the midpoint of each cell's sampled repetitive-current
range, not a common physical current or matched rheobase multiple.

AP amplitude here means peak-to-trough amplitude; half-width uses that midpoint.
Voltages at derivative extrema are not general geometric phase-loop inflections.
Direct feature predictions and reconstructed waveforms use separate regressions
and need not agree.

## Results

### Gouwens VISp, 34 C

- 366 paired cells, all q50 AP eligible.
- 42 t-types in the leave-one-t-type-out benchmark.
- Median direct-feature R2: 0.414 for class plus RNA versus 0.385 for class only.
- RNA adds substantial t-type-specific signal for upstroke duration (+0.277
  R2), AP amplitude (+0.274), AP peak relative to rest (+0.219), phase-loop area
  (+0.156), and AP half-width (+0.098).
- Direct-feature R2 is 0.838 for half-width, 0.414 for amplitude, 0.393 for
  maximum upstroke, and 0.320 for maximum downstroke.
- Voltage at maximum upstroke is not recovered (R2 = -0.142). Its atlas values
  should not be interpreted biologically.

### Scala, room temperature

- 1,208 paired cells; 714 have a complete q50 AP cycle.
- 40 t-types in the leave-one-t-type-out benchmark.
- Median direct-feature R2: 0.635 for class plus RNA versus 0.634 for class
  only.
- The median RNA increment over broad class is approximately zero. High total
  R2 values mostly reflect broad-family differences rather than demonstrated
  fine t-type prediction.

## Atlas interpretation policy

- Warm Gouwens predictions are preferred for Lamp5, Sncg, Vip, Sst, and Pvalb.
- Room-temperature Scala predictions are provided for cortical Glut_IT,
  Glut_ET, Glut_CT, Glut_NP, and Glut_L6b.
- The remaining 65 neural leaves, primarily hippocampal excitatory classes,
  are retained in the combined table but marked unsupported and have no
  endorsed AP-feature prediction.
- A reference leaf is called `atlas_only` when it is not the nearest atlas
  match of any observed Patch-seq t-type centroid. This is an inferred mapping,
  not an identity claim based only on the cluster name.
- `near` and `moderate` support are defined relative to the 95th percentile of
  nearest-neighbor distances observed among Patch-seq t-types. They are support
  diagnostics, not confidence probabilities.

The combined output contains 358 neural leaves, 293 with a supported broad
calibration class and 238 supported leaves inferred to be atlas-only. Warm
inhibitory predictions with non-far support are candidates for t-type-specific
validation. Remaining supported values are class-conditioned hypotheses.
The inferred `atlas_only` label does not establish absence from the published
Patch-seq datasets; an authoritative taxonomy crosswalk is still required.

## Main outputs

- `outputs/atlas_ttype_ephys_prediction/best_available_neural_ttype_ap_predictions.csv`
- `outputs/atlas_ttype_ephys_prediction/gouwens_visp/loto_feature_performance.csv`
- `outputs/atlas_ttype_ephys_prediction/scala_room_temperature/loto_feature_performance.csv`
- `outputs/atlas_ttype_ephys_prediction/*/atlas_neural_ttype_waveforms.csv`
- `outputs/atlas_ttype_ephys_prediction/*/loto_extrapolation_benchmark.png`
- `outputs/atlas_ttype_ephys_prediction/*/atlas_only_waveform_gallery.png`
