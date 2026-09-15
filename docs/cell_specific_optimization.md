# Cell-specific alpha/beta optimization

The `cell-optimize` workflow turns the earlier population scan into a
cell-specific inverse problem while preserving the scan as a source of priors
and initial candidates.

## Experimental targets

For one local Patch-seq NWB file, the workflow selects:

- a hyperpolarizing sweep near -40 pA for resting voltage, input resistance,
  and membrane time constant;
- the sampled-rheobase sweep;
- a training sweep near 1.5 times rheobase;
- a held-out sweep near 2 times rheobase.

The held-out sweep is never used by NSGA-II. It measures whether a candidate
generalizes along the cell's firing-current response rather than merely matching
the two fitted traces.

The passive response anchors membrane area, total capacitance, and leak
conductance. Sodium and potassium maximal conductances are nuisance parameters
restricted to a narrow factor around classic HH values. The 18 alpha/beta
transform parameters remain the main search coordinates. Leak reversal is
solved for each candidate so the measured resting voltage is an exact
zero-current state.

When a prior population used variable gate-specific Q10 values, seed
alpha/beta rate scales are converted to preserve their effective rates at the
recording temperature under the optimizer's fixed Q10 of 3. This affects only
initialization and avoids confounding a temperature conversion with a kinetic
mutation.

## Objectives

The optimizer keeps seven objectives separate:

1. passive input resistance and membrane time constant;
2. rheobase;
3. first-spike waveform;
4. spike timing and voltage dynamics;
5. inflection points and phase-cycle geometry;
6. firing rate, spike count, latency, and adaptation;
7. a regularizer on the sodium and potassium conductance deviations.

Features are extracted from biological and simulated voltage with the same
filtering and detection code. Missing model features receive an explicit
penalty. Unstable resting states, divergent simulations, and non-spiking
candidates are rejected with a finite dominated score.

The implementation uses DEAP NSGA-II. The full Pareto front is retained; a
balanced representative minimizes the sum of the six electrophysiological
objectives plus a smaller conductance-prior weight.

## Example

```bash
inverse-ephys cell-optimize \
  outputs/local_patchseq_spike_cycles.csv \
  --biological-dataset scala_room_temperature \
  --cell-id 20180920_sample_1 \
  --temperature 22 \
  --screen-preset scala \
  --bounds outputs/rt_spike_cycle_calibrated_prior.json \
  --seed-models outputs/rt_cycle_checkpoint_population.csv \
  --population-size 32 \
  --generations 10 \
  --workers 4 \
  --output-dir outputs/cell_optimization_rt
```

Every generation atomically updates checkpoint CSVs. Final outputs contain the
history, population, Pareto front, representative parameters, target features,
held-out scores, metadata, a Pareto trade-off figure, and an overlaid
biological/model trace figure.
