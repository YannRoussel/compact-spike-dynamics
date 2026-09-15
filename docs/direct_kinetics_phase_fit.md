# Direct kinetics phase-plane fit

## Question

This experiment tests whether replacing transformed Hodgkin-Huxley
alpha/beta curves with directly optimized steady-state and time-constant
curves can reproduce the phase-plane cycle of real neurons while preserving
rheobase and physical action-potential scale.

The pilot cohort contains two Gouwens cells recorded near 34 degrees C and two
Scala cells recorded at room temperature:

- Gouwens 704047023
- Gouwens 674495385
- Scala 20180920_sample_1
- Scala 20190425_sample_4

## Five implemented changes

### 1. Direct gate kinetics

Each `m`, `h`, and `n` gate is represented by:

```text
dx/dt = (x_inf(V) - x) / tau_x(V)
alpha_x(V) = x_inf(V) / tau_x(V)
beta_x(V) = (1 - x_inf(V)) / tau_x(V)
```

Six voltage knots are used at -100, -70, -50, -30, 0, and 60 mV.
Monotone cubic interpolation enforces increasing `m_inf` and `n_inf`,
decreasing `h_inf`, and positive smooth time constants. The direct fit has
36 kinetic coordinates: six steady-state coordinates and six time-constant
coordinates for each gate.

### 2. Complete phase-cycle geometry

The repolarization branch now extends from the spike peak to the first
post-spike trough instead of ending at the downward threshold crossing.
Concavity is measured after voltage smoothing and spline regularization in
four separate regions:

- upstroke onset
- late upstroke
- high-voltage downstroke
- low-voltage downstroke

The low-voltage downstroke receives extra weight because this was the
systematic concave-versus-convex mismatch in the earlier fits.

### 3. Physical scale constraints

Normalized geometry alone can admit a well-shaped but incorrectly translated
or scaled phase loop. The objective therefore also constrains:

- maximum `dV/dt`
- minimum `dV/dt`
- peak voltage
- post-spike trough voltage

The velocity tolerance is 10% of the biological value, with numerical floors
of 20 mV/ms for the maximum and 15 mV/ms for the minimum. Peak and trough
tolerances are 4 mV.

### 4. Stable integration

Directly optimized time constants can be shorter than the simulation time
step. Gate states are therefore advanced with their exact Rush-Larsen update,
while soma and AIS voltages are advanced with an RK4 split step. This avoids
the numerical gate overshoot produced by an explicit all-state integrator.

### 5. Staged, rheobase-guarded fitting

The optimizer follows the agreed ladder:

1. Optimize direct fast kinetics for smoothed phase-cycle shape at rheobase
   and a suprathreshold current.
2. Retain two diverse shape elites.
3. Fit sodium/potassium conductance scale, reversal potentials, and soma-AIS
   static parameters while limiting inherited shape degradation to 20%.
4. Jointly refine all parameters inside a 20% trust region.
5. Evaluate a third current only after selection as held-out validation.

Model rheobase must remain within 15% of the biological value, with a 10 pA
minimum tolerance. This guard is essential: without it, the optimizer found
visually attractive cycles whose rheobases collapsed to 4-16 pA.

## Reproduction

```bash
PYTHONPATH=src python scripts/run_direct_phase_pilots.py \
  --raw-spike-cycles outputs/local_patchseq_spike_cycles.csv \
  --baseline-root outputs/phase_shape_pilots \
  --output-root outputs/direct_phase_pilots_rheobase_guarded \
  --workers 4
```

Use `--resume` to regenerate the cross-cell summaries and figures without
repeating the optimization.

## Four-cell result

The final scientific selection first restricts to rheobase-feasible stages and
then chooses the lowest objective. Three of four cells have a feasible stage:

| Cell | Selected stage | Model/bio rheobase (pA) | Shape loss | Low-V sign agreement | Physical constraint loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Gouwens 704047023 | shape | 113/100 | 1.96 | 0.86 | 8.86 |
| Gouwens 674495385 | scale | 47/70 | 6.62 | 0.67 | 34.57 |
| Scala 20180920_sample_1 | shape | 105/120 | 4.58 | 0.50 | 12.58 |
| Scala 20190425_sample_4 | joint | 207/200 | 1.82 | 0.71 | 20.51 |

No selected model satisfies all four physical constraints. Peak voltage is
21-28 mV too low in all four cells, and minimum `dV/dt` is too negative in all
four. The Scala 20190425 model illustrates the strongest conflict: its
normalized loop shape is good, but maximum `dV/dt` is about 134 mV/ms too
large while peak voltage is still about 21 mV too low.

The direct formalism therefore improves local phase geometry, including the
previous low-voltage repolarization mismatch, but the two-fast-channel model
does not independently control phase shape, physical scale, rheobase, and
spike-train timing. The traces also retain excessive firing at sustained
current.

## Interpretation and next experiment

Changing the kinetic curve family was worthwhile: the earlier transformed
alpha/beta family was not the only source of the mismatch. The remaining
tradeoff is now evidence about model structure rather than merely curve
rigidity.

The next minimal extension should preserve several direct-fast-channel shape
elites and add one optimized slow potassium gate. It should be fitted first to
adaptation, late interspike voltage, trough, and low-voltage repolarization,
then jointly refined in a narrow trust region. AIS-specific fast kinetics are
a second extension if the peak/upstroke scale conflict remains after slow
potassium is added.

The unguarded `outputs/direct_phase_pilots` run is retained only as a failure
diagnostic and must not be used for biological conclusions. The guarded
results are in `outputs/direct_phase_pilots_rheobase_guarded`.
