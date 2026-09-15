# Independent soma-AIS effective-current experiment

This rung asks whether spatial localization can repair spike-train behavior
without abandoning the deliberately small channel vocabulary. The soma and
axon initial segment (AIS) both use:

```text
I_NaF = g_NaF m^a h^b (V - E_Na)
I_KF  = g_KF n^c (V - E_K)
I_NaS = g_NaS m^d s^e (V - E_Na)
I_KS  = g_KS p^f (V - E_K)
```

The integer powers belong to `{0, 1, 2, 3, 4}`. Fast sodium and potassium
always have activation states. Slow sodium adds an independent availability
state, and slow potassium adds an independent activation state. The soma and
AIS share reversal potentials and specific capacitance, but have independent
direct steady-state/time-constant curves and conductance densities.

## State vector

The simulator has 12 states:

```text
(V_s, m_s, h_s, n_s, s_s, p_s,
 V_a, m_a, h_a, n_a, s_a, p_a)
```

Applied current enters the soma. The compartments exchange axial current
`g_c (V_s - V_a)`. Both leak reversals are reanchored so the measured resting
voltage remains an exact coupled equilibrium.

## Staged fit

1. Lift the selected effective-component model into the 12-state model. The
   AIS slow currents are disabled and the AIS fast kinetics are copied from
   the soma. This parent is numerically identical to the previous simulator.
2. Screen AIS fast gate powers independently, normalize conductance at
   `-20 mV`, and fit AIS `m/h/n` curves plus fast conductances.
3. Test AIS slow sodium and slow potassium separately. A memory state is
   eligible only when total training loss improves and either firing-pattern
   or spike-count loss improves by at least 5%.
4. Combine eligible AIS slow mechanisms.
5. Jointly refine active soma and AIS parameters inside a 20% trust region.
   The result is rejected if firing-pattern or spike-count loss deteriorates
   by more than 10% from its seed.

The fitting objective is evaluated on soma voltage and retains the earlier
phase-shape, physical-scale, absolute-current spike-train, rheobase, validation,
and state-complexity terms. AIS voltage is saved as a diagnostic, not treated
as an observed target.

## Pilot

```bash
PYTHONPATH=src python scripts/run_two_compartment_effective_pilots.py \
  --workers 4
```

Use `--pilot gouwens_704047023` for one cell and `--resume` to retain completed
cells. Outputs include biological/parent/soma/AIS traces, soma phase planes,
score bars, kinetics, conductance densities, topologies, and all fitted
parameters.

## Four-cell pilot result

The initial pilot used one six-member CMA generation for each promoted fast,
slow, and joint stage. It is a mechanism screen, not a convergence study.

| Cell | Selected extension | Parent to selected objective | Spike counts (bio / parent / selected) |
| --- | --- | ---: | --- |
| Gouwens 704047023 | none | 307.0 to 307.0 | `1/1/1`, `10/1/1`, `16/1/1` |
| Gouwens 674495385 | AIS `NaF m4 h2`, `KF n4`, slow `K n1` | 677.2 to 634.8 | `1/2/1`, `10/36/33`, `17/52/50` |
| Scala 20180920-1 | none | 292.2 to 292.2 | `15/27/27`, `29/62/62`, `37/76/76` |
| Scala 20190425-4 | none | 423.0 to 423.0 | `5/0/0`, `10/1/1`, `12/1/1` |

The Gouwens 674495385 extension also improved phase-shape loss from `7.16` to
`3.75` and moved the model rheobase from `62.5` to `70.3 pA` for a biological
value of `70 pA`. Its AIS fast-sodium density reached `499.997 mS/cm2`, the
`500 mS/cm2` upper bound. The topology and localization are therefore useful
hypotheses, but the fitted AIS conductance is not an identified estimate.

The two depolarized one-spike plateau cells did not improve, and the
room-temperature over-firing cell retained its parent. At this budget,
spatial separation is not a general solution to the spike-train mismatch.
