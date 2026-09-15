# Nested sodium-activation Pareto experiment

The `phase-pareto-experiment` command tests which minimal sodium-activation
extension can improve the early phase-plane upstroke without sacrificing the
physical `dV/dt` range.

## Nested mechanisms

All three arms use the same soma-AIS structure, passive anchor, transient
sodium current, delayed-rectifier potassium current, conductance windows, and
optimized `h` and `n` alpha/beta families.

1. **Shared alpha/beta:** soma and AIS use the same six transformed HH rate
   curves.
2. **AIS m shift:** one additional parameter shifts only the AIS sodium
   activation rates along the voltage axis.
3. **Flexible shared m:** the six `alpha_m`/`beta_m` transforms are replaced by
   a monotone six-knot `m_inf(V)` curve and a positive six-knot `tau_m(V)`
   curve. The resulting activation kinetics remain shared between soma and
   AIS.

The fitted parameter counts are 23, 24, and 29 respectively. The flexible arm
replaces the original activation parameters rather than retaining redundant
copies.

## Pareto objectives

NSGA-II minimizes three objectives independently:

- smoothed branchwise concavity loss;
- maximum/minimum/total `dV/dt` extent loss; and
- all remaining phase-shape, landmark, rheobase, parameter-prior, and kinetic
  regularity costs.

The held-out sweep is evaluated only after selecting a representative from the
non-dominated front. A second diagnostic measures the closest distance to the
provisional focal target of concavity loss below 0.25 and velocity loss below
1.0.

Candidates are rejected before simulation if a gate steady-state curve is
non-monotone, a rate exceeds 2000/ms, or the minimum gate time constant at the
recording temperature is below:

```text
max(0.003 ms, 0.08 * simulation dt)
```

This prevents an optimizer from exploiting kinetics that the numerical
integration cannot resolve.

```bash
inverse-ephys phase-pareto-experiment \
  outputs/local_patchseq_spike_cycles.csv \
  --bounds outputs/rt_spike_cycle_calibrated_prior.json \
  --baseline-model outputs/phase_shape_pilots/scala_20180920_sample_1/soma_ais_best_model.csv \
  --biological-dataset scala_room_temperature \
  --cell-id 20180920_sample_1 \
  --temperature 22 \
  --screen-preset scala \
  --population-size 8 \
  --generations 3 \
  --workers 4 \
  --dt 0.05 \
  --output-dir outputs/phase_pareto_pilots/scala_20180920_sample_1
```

Cross-cell summaries are generated with:

```bash
python scripts/summarize_phase_pareto_pilots.py \
  outputs/phase_pareto_pilots
```

These pilot fronts test model adequacy, not parameter identifiability. A larger
population and repeated seeds are required before interpreting the density or
precise location of a front.

## Four-cell pilot

The first matched pilot used two Gouwens 34 C cells and two Scala
room-temperature cells, with 32 attempted candidates per mechanism and cell.
No mechanism reached both provisional focal thresholds in the same candidate.

- The AIS activation shift was near zero in three of four representative
  models and did not provide consistent evidence for compartment-specific
  activation voltage as the missing mechanism.
- Flexible shared activation strongly improved the two training cycles for
  Scala `20180920_sample_1`: phase total decreased from 7.64 to 4.17 and
  concavity loss from 1.34 to 0.44 while velocity-extent loss remained below
  one. Its held-out phase total was nevertheless worse than the shared model
  (7.79 versus 6.17), so this is a capacity result rather than a validated
  generalization result.
- Flexible activation moved the focal Pareto frontier closer to the target for
  Scala `20190425_sample_4` without improving the selected overall phase score.
- The shared alpha/beta family remained best for Gouwens `704047023`. Only two
  flexible candidates were valid for that cell, so the negative flexible result
  is under-sampled.

The next confirmatory run should use repeated seeds, seed every arm with the
same valid shared Pareto models, include the held-out sweep as a fourth
selection objective only after deciding that cross-current generalization is
part of the scientific target, and increase flexible-model viability through
narrower initialization rather than broader kinetic bounds.
