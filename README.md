# Inverse ephys: compact spike dynamics

The current project asks a compact inverse problem:

> Which low-dimensional waveform and timing parameters reproduce the spike
> loops of biological neurons, and which transcriptomic axes predict those
> parameters?

The latest fitted model is a channel-free phase template with a flexible
current-period relation, voltage/current/spike-driven recovery, and a fitted
current-step onset branch. The earlier exponential memory model remains a
comparison baseline. Models are linked to paired RNA expression with sparse
reduced-rank regression. Validation designs differ by experiment; atlas
leave-type-out scores are exploratory, not fully nested estimates. See
[`docs/abstract_phase_reduction.md`](docs/abstract_phase_reduction.md) and
[`docs/model_transcriptomics.md`](docs/model_transcriptomics.md).

## Start here

- [Reproducibility and current status](docs/reproducibility.md)
- [Independent validation candidates and evaluation plan](docs/independent_validation.md)
- [Experimental time-resolved phase trajectories](docs/time_phase_trajectory.md)
- [Atlas predictions and limitations](docs/allen_atlas_ephys_prediction.md)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[nwb,optim]'
MPLBACKEND=Agg python -m unittest discover -s tests -v
inverse-ephys inspect
```

Source code and synthetic tests are included. Biological recordings, expression
matrices, and generated outputs are excluded from Git. Historical scripts may
need explicit local input paths; see the reproducibility guide before a full run.

The repository also retains the earlier conductance-based route that motivated
the project:

It simulates a single-compartment neuron with sodium, potassium, and leak
currents. The six rate curves (`alpha_m`, `beta_m`, `alpha_h`, `beta_h`,
`alpha_n`, `beta_n`) and static membrane parameters are varied, models that do
not spike are rejected, and the retained traces are converted into fixed-width
feature vectors. A baseline regressor then predicts parameters from those
features.

## Parameterization

Each canonical HH rate curve has three transformed parameters:

- `log_rate_scale`: multiplicative rate/time-scale change
- `voltage_shift_mv`: horizontal shift of the curve
- `log_slope_scale`: stretch or compression of its voltage axis

This gives 18 parameters while preserving positive rates and retaining the
canonical HH model at the origin. The initial prior spans 0.5-2x rate,
-10 to +10 mV shift, and 0.7-1.4x voltage-axis scale. These are deliberately
conservative starting bounds, not a claim about all biologically possible
kinetics.

Eleven static coordinates are sampled alongside the kinetics:

- membrane area and specific capacitance
- maximal sodium, potassium, and leak conductances
- sodium, potassium, and leak reversal potentials
- separate `Q10` values for the `m`, `h`, and `n` gates

This gives 29 total inference coordinates. Physical values are stored in every
accepted-model row. `--kinetics-only` or `--static-only` can isolate either
parameter group. `--kinetic-sampling gate-correlated` samples coherent
alpha/beta pair movements with a smaller differential component while retaining
all 18 output coordinates. `--kinetic-sampling mixed` splits the noncanonical
draw evenly between independent and gate-correlated samples.

## Features

The built-in extractor includes spike count, firing rate, latency, ISI
statistics, threshold, amplitude, half-width, AHP, maximum/minimum `dV/dt`,
onset rapidness, acceleration extrema, AP-loop and full-cycle phase areas,
normalized phase-path length, and the `(V, dV/dt, -Iion)` coordinates where
`d2V/dt2 = 0` on the first spike's upstroke and downstroke. Observable
features are derived from a filtered voltage trace; exact `-Iion` coordinates
are retained only as model diagnostics.

eFEL is optional. Its features can be appended with `--efel` after installing
the optional dependency.

## Biological reference space

Before fitting alpha/beta parameters, the sampled models can be compared with
two author-maintained Patch-seq releases:

- Scala et al. M1: 29 scalar e-features for room-temperature and
  physiological-temperature recordings
- Gouwens et al. VISp: 24 curated IPFX features restored from the published
  z-scored matrix to physical units

The default `waveform-core` profile uses five quantities with a direct mapping
across both studies and the simulator: first-spike threshold, peak, half-width,
fast trough, and upstroke/downstroke ratio. `expanded-common` adds baseline
voltage and first-spike latency. `intrinsic` additionally includes input
resistance, membrane time constant, rheobase, and sag ratio.
The exact mappings are recorded in
[`docs/biological_feature_crosswalk.md`](docs/biological_feature_crosswalk.md).

Download the small processed releases and compare them with a generated model
dataset:

```bash
inverse-ephys fetch-biological
inverse-ephys generate \
  --protocol biological-screen \
  --screen-preset scala \
  --n-samples 256
inverse-ephys coverage outputs/hh_alpha_beta_dataset.csv --profile intrinsic
```

Coverage is evaluated in the full robustly standardized feature space. A
biological cell is called covered when its nearest HH model is closer than its
fifth-nearest biological neighbor. This local criterion adapts to dense and
sparse parts of the biological distribution. The PCA figure is a visualization
only and is not used to assign coverage. The command also writes split violin
plots and distribution tables for every feature in the selected profile.
Values in the violin plots are centered on the biological median and divided
by the biological IQR. Panels are ordered by normalized Wasserstein distance,
with biological recordings on the left and HH models on the right.

Raw local NWB files can also be used to add the voltage-derived spike-cycle
features that are absent from the processed releases. Install the `nwb` extra,
then point the extractor at either or both local release roots:

```bash
python3 -m pip install -e '.[nwb]'
inverse-ephys extract-local-nwb \
  --scala-root "/path/to/Scala_pseq_data" \
  --gouwens-root "/path/to/Patch-seq AIBS" \
  --output outputs/local_patchseq_spike_cycles.csv
```

The reader identifies long-square epochs from the injected-current waveform,
selects the lowest sampled positive step that spikes, and extracts the same
filtered-voltage feature block used for the HH models. Use
`--limit-per-dataset 3` for a quick format and protocol smoke test. See
[`docs/local_patchseq_nwb.md`](docs/local_patchseq_nwb.md) for the local cohort
inventory, identifier joins, and validation results.
The batch output is atomically checkpointed every 50 files; add `--resume` to
continue an interrupted extraction without repeating completed cells.

Use the raw feature table in coverage analysis with:

```bash
inverse-ephys coverage outputs/rt_models.csv \
  --profile spike-cycle \
  --raw-spike-cycles outputs/local_patchseq_spike_cycles.csv \
  --biological-dataset scala_room_temperature \
  --output-dir outputs/rt_spike_cycle_coverage
```

Run Gouwens separately against the physiological-temperature model population.
Audit the abstract coverage calls against the underlying voltage traces with:

```bash
PYTHONPATH=src python scripts/plot_covered_trace_pairs.py \
  outputs/rt_spike_cycle_coverage \
  outputs/rt_models.csv \
  outputs/local_patchseq_spike_cycles.csv \
  --preset fast \
  --temperature 22 \
  --max-covered 4 \
  --near-misses 1 \
  --title "Scala RT covered cells and nearest HH models" \
  --output outputs/covered_trace_pairs/scala_rt.png
```

The figure compares the complete rheobase responses, threshold-aligned first
spikes, and first-cycle phase planes. The companion CSV records current,
baseline, and spike-count mismatches. `--preset` must match the protocol that
generated the model feature table; the script refuses a mismatched replay.

The spatial extension gives the soma and AIS independent generic fast/slow
Na/K kinetics while preserving the selected effective-current model as an
exact parent. Its staged topology, memory-state, and guarded joint fit is
described in
[`docs/two_compartment_effective.md`](docs/two_compartment_effective.md).

Fit the existing population directly toward representative raw-NWB targets:

```bash
inverse-ephys direct-optimize outputs/rt_models.csv \
  --bounds outputs/rt_calibrated_prior.json \
  --raw-spike-cycles outputs/local_patchseq_spike_cycles.csv \
  --biological-dataset scala_room_temperature \
  --temperature 22 \
  --long-screen-preset scala \
  --n-targets 2 \
  --rounds 3 \
  --batch-size 32 \
  --long-promotions-per-round 4 \
  --workers 6 \
  --output-dir outputs/direct_optimization_rt
```

The objective gives equal weight to waveform, velocity, timing, inflection,
and normalized phase-geometry families. Onset and acceleration are reported as
held-out validation blocks. In every round the fast screen proposes a batch,
the best candidates are rescreened with the requested long protocol, and only
the accumulated long-protocol scores select the next round's elites.

## Quick start

```bash
cd /path/to/compact-spike-dynamics
python3 -m pip install -e .
inverse-ephys inspect
inverse-ephys fetch-biological
inverse-ephys generate --protocol biological-screen --n-samples 256 --workers 4
inverse-ephys coverage outputs/hh_alpha_beta_dataset.csv
inverse-ephys train outputs/hh_alpha_beta_dataset.csv
```

Without installing the package, prefix commands with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m inverse_ephys_alpha_beta.cli inspect
PYTHONPATH=src python3 -m inverse_ephys_alpha_beta.cli generate \
  --protocol biological-screen --n-samples 64
```

The calibration loop is:

```bash
inverse-ephys generate \
  --protocol biological-screen \
  --screen-preset fast \
  --static-only \
  --n-samples 64 \
  --output outputs/static_calibration.csv
inverse-ephys coverage \
  outputs/static_calibration.csv \
  --profile intrinsic \
  --output-dir outputs/static_calibration_intrinsic
inverse-ephys calibrate-prior \
  outputs/static_calibration.csv \
  outputs/static_calibration_intrinsic/model_plausibility.csv \
  --parameter-group static \
  --output outputs/calibrated_static_prior.json
inverse-ephys generate \
  --protocol biological-screen \
  --screen-preset fast \
  --bounds outputs/calibrated_static_prior.json \
  --n-samples 512 \
  --output outputs/combined_calibration.csv
inverse-ephys rescreen \
  outputs/combined_calibration.csv \
  --screen-preset scala \
  --output outputs/scala_long_sweep.csv
```

See [`docs/calibration_strategy.md`](docs/calibration_strategy.md) for the
staged expansion logic and the current pilot results.
See [`docs/spike_cycle_scope.md`](docs/spike_cycle_scope.md) for the focused
Na/K spike-cycle hypothesis and nested model comparison.
See [`docs/spike_cycle_features.md`](docs/spike_cycle_features.md) for the
measurement-matched waveform, inflection, curvature, and loop-area definitions.
See [`docs/rheobase_bisection.md`](docs/rheobase_bisection.md) for the exact
search algorithm and its interpretation.
See [`docs/local_patchseq_nwb.md`](docs/local_patchseq_nwb.md) for raw Scala and
Gouwens NWB extraction.
See [`docs/cell_specific_optimization.md`](docs/cell_specific_optimization.md)
for the passive-anchored, multi-sweep NSGA-II fitting workflow.
See [`docs/model_ladder.md`](docs/model_ladder.md) for the frozen-base CMA-ES
comparison of slow sodium, slow potassium, and soma/AIS extensions.
See [`docs/staged_ladder.md`](docs/staged_ladder.md) for the passive-corrected,
multi-parent fast/slow-K ladder and trust-region refinement.
See [`docs/phase_shape_experiment.md`](docs/phase_shape_experiment.md) for the
smoothed phase-plane concavity objective and matched one-compartment versus
soma-AIS adequacy experiment.
See [`docs/nested_phase_pareto.md`](docs/nested_phase_pareto.md) for the
three-objective comparison of shared alpha/beta, AIS-shifted activation, and
flexible monotone sodium-activation kinetics.
See [`docs/direct_kinetics_phase_fit.md`](docs/direct_kinetics_phase_fit.md)
for the direct steady-state/time-constant formalism, rheobase-guarded staged
fit, and four-cell Gouwens/Scala pilot results.
See [`docs/direct_slow_k_experiment.md`](docs/direct_slow_k_experiment.md)
for the one-state direct slow-potassium extension, absolute-current
spike-train replay, and the resulting state-dimension test.
See [`docs/effective_component_ladder.md`](docs/effective_component_ladder.md)
for the discrete gate-power screen, four-component effective Na/K model,
conditional memory-state selection, and 20% trust-region refinement.
See [`docs/abstract_phase_reduction.md`](docs/abstract_phase_reduction.md)
for the channel-free voltage-acceleration field, its stability diagnostics,
and the landmark phase-template model with a fitted slow adaptation state.

Run the abstract-model pilot with:

```bash
PYTHONPATH=src python3 scripts/run_abstract_phase_pilot.py \
  --output-root outputs/abstract_phase_pilot
```

Run the balanced 100-cell held-out-current test with:

```bash
PYTHONPATH=src python3 scripts/run_phase_template_population.py \
  --cells-per-dataset 50 \
  --workers 8 \
  --output-root outputs/phase_template_population_100_stable_cycles
```

Compare flexible current, spike-memory, and Izhikevich-like timing clocks on
that fixed cohort with:

```bash
PYTHONPATH=src python3 scripts/run_phase_timing_population.py \
  --workers 8 \
  --output-root outputs/phase_timing_population_100
```

Fit the selected compact phase-template plus spike-memory default to every
eligible local cell, then relate its fixed-width parameters to paired RNA
expression with donor-aware sparse reduced-rank regression:

```bash
PYTHONPATH=src python3 scripts/run_compact_population.py \
  --workers 16 \
  --resume \
  --output-root outputs/compact_population_all
PYTHONPATH=src python3 scripts/run_transcriptomic_rrr.py \
  --parameters outputs/compact_population_all/compact_model_parameters.csv \
  --output-root outputs/transcriptomic_rrr
PYTHONPATH=src python3 scripts/run_class_aware_rrr.py \
  --parameters outputs/compact_population_all/compact_model_parameters.csv \
  --output-root outputs/class_aware_rrr
PYTHONPATH=src python3 scripts/summarize_class_aware_rrr.py \
  --output-root outputs/class_aware_rrr
```

Gouwens and Scala are analyzed separately. Both the complete RNA-matched
cohort and a predeclared high-QC sensitivity cohort are reported. See
[`docs/model_transcriptomics.md`](docs/model_transcriptomics.md) for cell-ID
alignment, parameterization, grouped cross-validation, and interpretation
limits.

Fit the continuous recovery/onset extension to all compact models, reserve the
healthy family representatives from transcriptomic training, and replay models
predicted from their RNA profiles with:

```bash
PYTHONPATH=src python3 scripts/run_recovery_onset_targets.py \
  --workers 8 \
  --resume \
  --output-root outputs/recovery_onset_targets_all
PYTHONPATH=src python3 scripts/run_recovery_onset_heldout_srrr.py \
  --parameters outputs/recovery_onset_targets_all/recovery_onset_parameters.csv \
  --output-root outputs/recovery_onset_heldout_srrr
PYTHONPATH=src python3 scripts/plot_recovery_onset_rna_replays.py \
  --srrr-root outputs/recovery_onset_heldout_srrr \
  --output-root outputs/recovery_onset_rna_replays
```

Target transforms, class baselines, gene selection, scaling, and clipping
bounds are learned without the held-out cells. The ephys-fitted held-out models
are retained only as reconstruction targets and visual comparators.

Predict AP waveforms and direct AP features for the neural leaves of the Allen
CTX-HPF SMART-seq reference taxonomy with complete t-type holdouts:

```bash
PYTHONPATH=src python3 scripts/run_atlas_ephys_prediction.py \
  --datasets gouwens_visp scala_room_temperature \
  --output-root outputs/atlas_ttype_ephys_prediction
PYTHONPATH=src python3 scripts/summarize_atlas_ephys_predictions.py
```

The combined table keeps the 34 C Gouwens and room-temperature Scala
calibrations explicit and does not endorse unsupported reference classes. See
[`docs/allen_atlas_ephys_prediction.md`](docs/allen_atlas_ephys_prediction.md)
for the extrapolation design and interpretation limits.

Compare the fitted single adaptation time constant with a fixed multiscale
spike-history kernel on a balanced eligible-cell pilot:

```bash
PYTHONPATH=src python3 scripts/run_adaptation_kernel_population.py \
  --limit-per-dataset 100 \
  --workers 8 \
  --executor thread \
  --output-root outputs/adaptation_kernel_eligible_pilot_200
PYTHONPATH=src python3 scripts/run_kernel_transcriptomic_pilot.py
```

The fixed kernel is retained as an experimental alternative, not the default:
it did not consistently improve held-out timing or within-class RNA prediction
in the 200-cell pilot.

Run the dependency-free test suite with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Outputs

Dataset generation writes:

- accepted spiking models with parameters and features
- rejected parameter sets with a rejection reason
- JSON metadata describing the simulation and sampling configuration

Training writes held-out predictions, per-parameter metrics, global feature
importance, a serialized model, and a parameter-recovery figure.

The default 256 draws are enough to test the machinery, not to estimate
identifiability. A serious recovery experiment should target thousands of
accepted models and repeat the split across several random seeds. In particular,
negative held-out R2 values on a small smoke test mean only that the predictor
did worse than predicting the test-set mean; they are not evidence that a
parameter is fundamentally unrecoverable.

## Scientific cautions

Good prediction does not prove that a kinetic parameter is uniquely
identifiable. Correlated priors, the fixed conductances, and a single current
step can all make recovery look easier than it is. The next experimental
iterations should therefore add multiple current amplitudes, out-of-distribution
tests, noise, conductance variation, and recovery of the rate curves themselves
rather than judging only their parameter coordinates.

The biological screening protocol uses pA stimuli, computes a zero-current
resting state, brackets the first spiking current, and bisects that interval to
a configurable pA tolerance. Named presets provide a fast calibration screen,
Scala-like 600 ms sweeps, and Gouwens-like 1000 ms sweeps. A separate
hyperpolarizing sweep estimates input resistance, membrane time constant, and
sag. `rescreen` applies these protocols to an existing parameter table for
paired comparisons.

The classic HH rates are referenced to 6.3 degrees C. Simulations now apply
gate-specific `Q10` factors at an explicit recording temperature, but these
factors remain inference parameters rather than experimentally calibrated
constants. Preset defaults are 22 degrees C for Scala room temperature and
34 degrees C for Scala physiological temperature and Gouwens.
