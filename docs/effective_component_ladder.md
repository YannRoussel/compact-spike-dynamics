# Effective-component topology ladder

## Question

This experiment asks whether a small number of theoretical channel components
can reproduce both the action-potential phase loop and spike-train behavior
when their gate kinetics and cooperativity are optimized.

The integer powers and the number of kinetic states are treated separately.
For example, `m^3` changes the nonlinear dependence of current on the existing
state `m(t)`. It does not create three independent activation memories.

## First-rung model

The soma currents are

```text
I_NaF = g_NaF m_f^a h_f^b (V - E_Na)
I_KF  = g_KF  n_f^c       (V - E_K)
I_NaS = g_NaS m_f^d s^e   (V - E_Na)
I_KS  = g_KS  p^f         (V - E_K)
```

where:

- `m_f`, `h_f`, and `n_f` are the independently fitted direct fast gates
- `s` is a slow sodium availability/recovery state
- `p` is a slow potassium activation state
- each exponent belongs to `{0, 1, 2, 3, 4}`
- an exponent of zero disables that gate or component

The slow Na component shares fast activation `m_f` and pays for one new
availability state. Slow K pays for one new activation state. This gives at
most four current components and five independent soma gate states.

Potassium inactivation is fixed to exponent zero in this rung. Allowing a
nonzero inactivation exponent would require an independently fitted
inactivation gate; it should be added only if the current rung still fails.
This avoids optimizing an unused or unidentifiable state.

Both slow curves use six direct voltage knots. Slow Na availability is
monotone decreasing, slow K activation is monotone increasing, and both time
constants are positive. The passive endpoint and coupled soma-AIS resting
state are reanchored for every candidate.

## Search ladder

1. Screen a representative factorial spine of fast exponent topologies.
2. Expand to all one-integer neighbors of the three best screen candidates.
3. Optimize fast Na and K conductance scales and retain three fast parents.
4. For the best parents, screen slow Na and slow K exponent choices at an
   intermediate conductance.
5. Recheck the two best slow topologies at low and high conductance, then fit
   the selected gate curve and conductance with CMA-ES.
6. Combine the best slow Na and slow K branches from the same fast parent.
7. Retain a slow mechanism only if total training objective improves and
   firing-pattern or spike-count loss improves by at least 5%.
8. Jointly refine all active fast parameters, static parameters, and retained
   slow parameters inside a parameter-aware 20% trust region.
   The refined child is rejected if firing-pattern or spike-count loss worsens
   by more than 10% relative to its seed.
9. Select using training loss, validation loss, rheobase feasibility, and a
   penalty for each added independent state.

The original direct parent is always retained alongside topology-refined fast
parents. Previously fitted slow-K gates can be supplied as explicit warm
starts. They are fully re-simulated and rescored; a warm start supplies a basin,
not a score exemption.

The full 32-member fast topology family remains available with
`--exhaustive-fast-topologies`. The default representative-plus-neighbor
screen is a computational pilot, not proof that distant exponent interactions
are irrelevant.

## Objective matching

As in the corrected slow-K experiment:

- phase shape is measured at matched relative excitability using each model's
  bisected rheobase
- spike-train features are measured at the exact biological current in pA
- the same recording window is used for biological and model traces
- spike-count loss is explicit

The current target construction supplies training sweeps plus one
`validation` sweep. That sweep participates in topology selection with a
weight of 0.25, so it is a model-selection validation set, not an untouched
final test set. A larger study should reserve an additional current amplitude
or cell-level split for final testing.

## Complexity interpretation

The complexity penalty counts independent states:

```text
fast parent:             m_f, h_f, n_f       -> 3 states
+ slow Na recovery:      add s               -> 4 states
+ slow K activation:     add p               -> 4 states
+ both slow components:  add s and p         -> 5 states
```

Changing `m^1` to `m^4` does not change this count. This distinction is
important for later transcriptomic interpretation: an exponent may describe
effective cooperativity, whereas an added state is evidence for another
kinetic timescale or molecular mixture.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/inverse-ephys-mpl \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
.venv/bin/python scripts/run_effective_component_pilots.py \
  --direct-root outputs/direct_phase_pilots_rheobase_guarded \
  --output-root outputs/effective_component_pilots \
  --fast-parent-count 3 \
  --mechanism-parent-count 2 \
  --joint-seed-count 2 \
  --fast-population-size 6 \
  --fast-generations 2 \
  --mechanism-population-size 8 \
  --mechanism-generations 2 \
  --joint-population-size 10 \
  --joint-generations 2 \
  --duration-ms 500 \
  --workers 4
```

Each cell writes:

- `candidate_summary.csv`
- `topology_screen.csv`
- `optimization_history.csv`
- `selected_parameters.csv`
- `firing_pattern_comparison.csv`
- phase-plane, voltage-trace, score, topology, and kinetic plots

The root output directory contains the combined candidate table, concise
selected metrics, parent deltas, combined firing-pattern features, cross-cell
loss and spike-count comparisons, and the selected state/topology plot.

## Four-cell pilot

The first pilot used the budgets in the command above. These are still shallow
for 13-dimensional slow-gate blocks and the final trust-region problem.

| Cell | Selected topology | States | Parent to selected objective | Parent to selected validation |
| --- | --- | ---: | ---: | ---: |
| Gouwens 704047023 | `NaF m3h1; KF n4` | 3 | 317.0 to 306.9 | 439.9 to 474.7 |
| Gouwens 674495385 | `NaF m3h1; KF n4; NaS m4h1; KS n1` | 5 | 870.7 to 677.2 | 847.7 to 737.3 |
| Scala 20180920_sample_1 | `NaF m3h1; KF n4; KS n1` | 4 | 317.1 to 292.0 | 257.6 to 253.9 |
| Scala 20190425_sample_4 | `NaF m3h1; KF n4` | 3 | 533.4 to 422.9 | 666.5 to 437.4 |

The selected spike counts at biological absolute currents are:

| Cell | Biological | Direct parent | Selected |
| --- | --- | --- | --- |
| Gouwens 704047023 | 1 / 10 / 16 | 0 / 1 / 1 | 1 / 1 / 1 |
| Gouwens 674495385 | 1 / 10 / 17 | 27 / 50 / 65 | 2 / 36 / 52 |
| Scala 20180920_sample_1 | 15 / 29 / 37 | 47 / 65 / 79 | 27 / 62 / 76 |
| Scala 20190425_sample_4 | 5 / 10 / 12 | 0 / 1 / 1 | 0 / 1 / 1 |

The three entries are sampled rheobase, suprathreshold training, and validation
currents. Added slow states are supported for two cells, but neither case
matches the full frequency-current relationship. The two plateauing cells
remain essentially one-spike models. The discrete fast-power search does not
select a noncanonical final exponent topology in this pilot.

Therefore:

- integer cooperativity alone is not the missing degree of freedom here
- independent slow memory improves subsets of cells
- slow Na plus slow K helps Gouwens 674 but still fires too rapidly
- slow K helps Scala 20180920 mainly near rheobase, not at larger currents
- Gouwens 704 and Scala 20190425 still require a recovery or
  compartment-specific mechanism that can restore repetitive firing

The next experiment should target AIS-specific sodium availability or a
second sodium recovery timescale for the plateauing pair, while preserving the
no-extra-state branch for cells that do not support it.
