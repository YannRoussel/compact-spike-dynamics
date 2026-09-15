# Minimal model ladder

The `ladder-optimize` command compares three extensions of a fitted
single-compartment Na/K model while keeping its 20 fast kinetic and conductance
parameters frozen:

1. `na-slow`: one additional slow sodium-availability gate;
2. `k-slow`: one generic slow non-inactivating outward current;
3. `soma-ais`: a soma and effective AIS with shared fast kinetics.

Only extension-specific parameters are optimized. This makes improvements
attributable to the added temporal or spatial mechanism rather than to a
complete refit of the original model.

The slow gates are represented by complementary alpha/beta curves with
independent rate, voltage-shift, and voltage-slope transforms. The K-slow model
also optimizes its maximal conductance. The effective AIS model optimizes its
area fraction, sodium-density multiplier, and soma-AIS coupling.

CMA-ES minimizes a scalar objective containing the same passive, rheobase,
waveform, spike-dynamics, phase-geometry, firing-pattern, and parameter-prior
scores used in the NSGA-II experiment. Firing pattern receives twice the
weight, rheobase receives 1.5 times the weight, and the held-out sweep remains
excluded from optimization.

```bash
inverse-ephys ladder-optimize \
  outputs/local_patchseq_spike_cycles.csv \
  --base-model outputs/cell_optimization_rt_pilot/representative_model.csv \
  --biological-dataset scala_room_temperature \
  --cell-id 20180920_sample_1 \
  --temperature 22 \
  --screen-preset scala \
  --population-size 8 \
  --generations 4 \
  --workers 4 \
  --output-dir outputs/model_ladder_rt
```

Each variant writes its optimization history, best parameters, per-feature
comparison, held-out scores, metadata, and trace plot. The parent output
contains a combined summary table and comparison figure.
