# Smoothed phase-shape adequacy experiment

The `phase-shape-experiment` command asks a deliberately narrow question:
can optimized transient sodium and delayed-rectifier potassium kinetics recover
the first-spike phase-plane cycle, and does a minimal soma-AIS architecture
resolve mismatches that remain in a one-compartment model?

## Phase-cycle representation

The voltage trace is first filtered with a Savitzky-Golay window corresponding
to 0.15 ms. The first spike is split into an upstroke branch from threshold to
the voltage peak and a downstroke branch from the peak to the threshold return.
If the voltage does not return to the first threshold before the next spike,
the interspike trough is used as the downstroke endpoint.

Each branch is parameterized by its relative voltage and resampled on a fixed
64-point grid with a shape-preserving cubic interpolator. A cubic smoothing
spline is then fitted to `dV/dt` as a function of relative voltage. The
phase-plane slope and concavity are the first and second derivatives of this
spline:

```text
slope      = d(dV/dt) / dV
concavity  = d2(dV/dt) / dV2
```

This avoids estimating concavity as a noisy third time derivative of the raw
voltage trace. Biological and model traces pass through the same filtering,
spike detection, interpolation, and spline operations.

## Phase loss

The phase objective combines:

- physical `dV/dt` branch distance after threshold alignment;
- normalized branch-shape distance;
- phase-plane slope distance;
- pre-maximum-upstroke concavity magnitude and sign disagreement;
- explicit maximum, minimum, and span of `dV/dt`; and
- spike-amplitude, repolarization, and extremum-position landmarks.

The phase total is averaged over a rheobase sweep and one suprathreshold
training sweep. A third current step is held out and reported only after model
selection. Rheobase is recomputed for every candidate.

## Matched architecture comparison

Both architectures optimize the same 18 alpha/beta transformation parameters
and small sodium/potassium conductance corrections. The soma-AIS model adds
only three structural parameters:

- AIS area relative to soma;
- AIS sodium-density multiplier; and
- soma-AIS coupling conductance.

Both models receive the same number of parents, candidates, and generations
for a given cell. Candidate rate curves must remain monotone, have positive
finite gate time constants, and yield a stable resting state.

```bash
inverse-ephys phase-shape-experiment \
  outputs/local_patchseq_spike_cycles.csv \
  outputs/rt_models.csv \
  --bounds outputs/rt_calibrated_prior.json \
  --biological-dataset scala_room_temperature \
  --cell-id 20180920_sample_1 \
  --temperature 22 \
  --screen-preset scala \
  --population-size 8 \
  --generations 4 \
  --workers 4 \
  --output-dir outputs/phase_shape_pilot
```

The run writes best parameters, candidate histories, protocol-level scores,
phase-cycle comparisons, score bars, steady-state/time-constant curves, and
alpha/beta rate plots. Multiple cells can be summarized with:

```bash
python scripts/summarize_phase_experiments.py outputs/phase_shape_pilots
```

The combined outputs include a candidate-level concavity-versus-velocity plot.
The dotted reference box is a diagnostic target, not a biological acceptance
threshold: concavity loss below 0.25 and velocity-extent loss below 1.0.
