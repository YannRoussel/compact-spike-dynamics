# Time-phase adaptation pilot: 2026-09-15

## Scope and checks

All 168 selected cells completed: 60 Gouwens (54 donors) and 108 Scala
(84 donors). No donor appeared in more than one outer RNA fold. Each model
was fitted without the evaluation current. RNA prediction used three outer
donor folds and three inner folds for tuning, separately by dataset.

Implementation and reproduction commands: [time_phase_trajectory.md](time_phase_trajectory.md).
Code snapshot: `a1d69f7`. All 94 local tests passed, and the GitHub Actions run
for that snapshot passed. Figures were rendered and inspected locally.

## Held-out current results

Values are population medians. Lower is better. Adaptation ratio is mean
last-three ISIs divided by mean first-three ISIs; its error is dimensionless.

| Dataset | Model | Count error | Early ISI RMSE (ms) | Late ISI error (ms) | Adaptation ratio error |
| --- | --- | ---: | ---: | ---: | ---: |
| Gouwens | Recovery | 3.0 | 8.60 | 13.91 | 0.390 |
| Gouwens | ISI-only sequence | 4.0 | 5.26 | 14.12 | 0.269 |
| Gouwens | Joint sequence | 4.0 | 5.34 | 14.16 | 0.268 |
| Scala | Recovery | 4.5 | 11.46 | 10.97 | 0.943 |
| Scala | ISI-only sequence | 4.0 | 8.05 | 12.57 | 0.918 |
| Scala | Joint sequence | 4.0 | 8.40 | 12.93 | 0.857 |

The joint fit improves early-ISI error in 63% of Gouwens and 66% of Scala
cells, and adaptation-ratio error in 55% and 68%. It does not improve every
metric: Gouwens median count error increases; median late-ISI error increases
in both cohorts. Median paired differences and differences of population
medians are different statistics; both are provided in the aggregate tables.

The ISI-only and joint results are close. This experiment supports a more
flexible time-dependent timing description, but does not establish a clear
additional benefit from geometric supervision. The joint model uses geometry
in fitting the clock; it does not yet generate changing spike shapes.

## RNA-generated trains

These predictions use class plus RNA and the known injected-current protocol.
No fitted test-cell waveform, latency, or clock parameter is supplied to replay.

| Dataset | Model | Count error | Early ISI RMSE (ms) | Late ISI error (ms) | Adaptation ratio error |
| --- | --- | ---: | ---: | ---: | ---: |
| Gouwens | Recovery | 16.5 | 14.47 | 43.44 | 0.886 |
| Gouwens | ISI-only sequence | 10.5 | 12.71 | 20.43 | 0.402 |
| Gouwens | Joint sequence | 11.0 | 12.07 | 23.07 | 0.474 |
| Scala | Recovery | 9.5 | 13.16 | 19.73 | 1.068 |
| Scala | ISI-only sequence | 9.0 | 10.85 | 19.36 | 0.857 |
| Scala | Joint sequence | 8.5 | 10.62 | 18.62 | 0.872 |

Class-only predictions improve similarly. For the joint clock, class-only
adaptation-ratio errors are 0.506 (Gouwens) and 0.872 (Scala), compared with
0.474 and 0.872 for class plus RNA. Thus most gains cannot be attributed to
additional RNA information beyond broad class.

Median errors also hide large outliers. For joint RNA replays, spike-count
R2 is -0.050 in Gouwens and -5.668 in Scala; adaptation-ratio R2 is -0.078
and -0.241. These remain poor absolute predictions. The median parameter-wise
R2 for joint timing coefficients is only 0.063 and 0.039, versus 0.059 and
0.039 for class alone. Parameter R2 and generated-train R2 are distinct tests.

All 168 cells had finite primary timing metrics for all three prediction
sources. This does not remove bias from the preselected repetitive-spiking
cohort or from incomplete predicted trains; count and ISI errors must be read
together. The small, family-balanced sample is not representative of all
Patch-seq cells. No significance or causal genetic claim is made.

## Artifacts and recommendation

Local outputs under `outputs/time_phase_pilot/`:

- `performance.png`: median error comparison, including class-only controls.
- `gouwens_visp_examples.png` and `scala_room_temperature_examples.png`:
  traces, 3D trajectories, phase loops and ISI sequences. Cells chosen by ID
  ordering within family, without choosing for fit quality.
- `cells/*/training_cycle_features.csv`: all extracted training loops.
- `cells/*/fit.json`: fitted parameters, inner selections and protocol IDs.
- `cells/*/rna_replays.npz`: out-of-fold generated voltage traces.
- `rna_splits.csv` and `rna_hyperparameters.csv`: donor splits and tuning choices.
- Aggregate CSVs are also copied into `docs/results/time_phase_pilot/` for Git.

Keep this as an experimental branch. The ISI-only clock is a useful compact
timing comparator. Before expanding it, examine the worst train predictions
and establish repeat-sweep reliability of time-resolved waveform changes.
The evidence does not yet justify replacing the default with the joint model
or expanding transcriptomic claims.
