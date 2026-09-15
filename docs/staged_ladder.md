# Staged cell-specific ladder

The `staged-ladder-optimize` command implements the pilot fitting strategy used
for two Gouwens VISp cells and two Scala room-temperature cells.

## Passive correction

The passive anchor is estimated from the smallest-magnitude negative current
step available for each cell. Resting voltage and the finite-step steady-state
voltage are then imposed simultaneously after accounting for the nonzero
steady sodium and potassium currents. This avoids the warm-cell failure caused
by treating the active currents as negligible at rest.

The implementation also rejects candidate models that:

- spike during the passive sweep;
- move in the wrong direction during a negative current step;
- imply a nonpositive input resistance; or
- are locally unstable at rest or at the passive endpoint.

## Optimization ladder

1. **Fast-channel fit.** NSGA-II fits transient sodium and potassium kinetics
   to the first-spike waveform and phase cycle. Absolute spike voltages are
   threshold-aligned for this stage, with a smaller penalty retaining
   information about large voltage translations.
2. **Diverse parent retention.** Several fast-channel models are retained,
   separated in normalized parameter space, so the next stage does not inherit
   a single arbitrary kinetic solution.
3. **Slow-potassium branches.** A generic slow non-inactivating potassium
   current is optimized independently from every retained parent with bounded
   CMA-ES. The branch with the best complete training objective is selected.
4. **Trust-region refinement.** Fast and slow parameters are jointly released
   within a narrow neighborhood: multiplicative parameters can move by 20%
   and voltage shifts by 4 mV by default. A regularization term discourages
   movement away from the frozen solution.

The held-out current step is never used for selection. It is reported beside
the training scores to expose overfitting.

```bash
inverse-ephys staged-ladder-optimize \
  outputs/local_patchseq_spike_cycles.csv \
  outputs/rt_models.csv \
  --bounds outputs/rt_calibrated_prior.json \
  --biological-dataset scala_room_temperature \
  --cell-id 20180920_sample_1 \
  --temperature 22 \
  --screen-preset scala \
  --fast-population-size 12 \
  --fast-generations 3 \
  --fast-elites 3 \
  --slow-population-size 8 \
  --slow-generations 3 \
  --final-population-size 10 \
  --final-generations 3 \
  --workers 4 \
  --output-dir outputs/staged_ladder_rt
```

Each run writes the three stage histories, retained Step 1 parents, best model
parameters, final per-feature residuals, a stage-score table, and trace,
first-spike phase-plane, and score-bar figures.

Multiple pilot directories can be collected into plot galleries with:

```bash
python scripts/summarize_staged_pilots.py outputs/staged_ladder_pilots
```

The summary records whether the slow-K or joint stage actually improves the
total score relative to the selected fast-only parent. A slow channel should be
treated as supported only when it improves the score and the trace-level
firing pattern, not merely because its optimized conductance is nonzero.
