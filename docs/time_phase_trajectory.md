# Experimental time-resolved phase trajectories

## Implemented pilot

Completed results: [time_phase_pilot_results.md](time_phase_pilot_results.md).

Branch: `codex/time-phase-adaptation`. Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_time_phase_experiment.py \
  --workers 4 --resume --output-root outputs/time_phase_pilot
PYTHONPATH=src python scripts/plot_time_phase_results.py \
  --root outputs/time_phase_pilot
```

The default cohort is the existing family-stratified pilot: 60 Gouwens and
108 Scala cells. A smoke test uses `--limit 2 --skip-rna`. Cohort and source
paths are local data dependencies, excluded from Git.

Every cell is refitted on training currents only. The saved all-current
population fits are not used as held-out-current baselines. Each method uses
the same ten-harmonic waveform representation, latency and onset model.

Three methods are compared:

1. Existing voltage/current/spike-driven recovery clock.
2. ISI-only sequence clock: log period is a current-interpolated linear
   combination of 1 and `1-exp(-t/tau)` for tau = 20, 100, 500 ms.
3. Joint sequence clock: fits log ISI, log loop area, log amplitude, log maximum
   upstroke and log downstroke magnitude together, using a low-rank projection
   of fitted response trajectories. Rank and ridge are selected on late cycles
   of the training currents, using freely accumulated predicted event times.

The joint model uses geometry to constrain estimation of timing; its rendered
waveform is still the same median current-dependent template. It does not yet
generate spike-to-spike waveform accommodation. The first comparison tests a
different timing formalism; the ISI-only versus joint comparison isolates the
added geometry supervision. A 3D plot by itself adds no dynamical state.

`t` is elapsed time since the first spike. New events are generated recursively
from predicted periods. No observed event times or validation voltage are used
to generate a train. Periods are bounded to 1-2000 ms, with enough room for the
AP upstroke/downstroke segments. This domain is repetitive positive long-square
steps, not arbitrary inputs or predictions below rheobase.

RNA evaluation uses three outer donor folds and three inner donor folds,
separately by dataset. Broad-class means and preprocessing are estimated from
training donors. Ridge/rank and sparse penalty selection happen within each
outer training split. The fixed published input gene panel is reused; no new
genes are selected using test outcomes. Waveform/onset and timing are separate
prediction blocks, so waveform dimensionality does not dominate timing fitting.
All three clocks share the same RNA-predicted waveform/onset block. Compare
class-plus-RNA against a class-only baseline on exactly the same test cells.
The specified injected current is known at replay time; fitted test-cell
parameters are not supplied to the RNA simulator.

Reported train metrics include count error, first-three matched ISI error,
late-ISI error, and error in mean last-three/first-three ISI ratio. ISI metrics
with too few predicted events are missing, not zero; inspect eligible counts
and spike-count error together to avoid rewarding a truncated train. The
early metric uses as many of the first three intervals as both trains contain.
All comparisons are exploratory pilot analyses. Large donor-held-out errors
should not be described as successful RNA prediction merely because one model
outperforms another.

Per-cycle tables include acceleration-zero voltage/velocity coordinates.
These are temporal voltage inflections, not curvature sign changes of the
planar loop. They and scaled 3D arc length are diagnostic exports; the first
joint-fit experiment uses the five positive quantities listed above. Noise
robustness of all diagnostic coordinates has not been established.

## Idea

Represent each filtered recording as r(t) = (t, V(t), dV/dt). Repeated spikes
appear as successive loops advancing along time. Equal loop spacing corresponds
to regular firing; increasing spacing to adaptation. Changes in loop shape
show waveform accommodation, amplitude loss, or changing repolarization.

This representation adds no observations beyond V(t). It preserves the order
and evolution discarded by a median phase loop and may provide better fitting
targets. Time is an observation coordinate, not a physiological recovery state:
adding dt/dt = 1 does not identify the neuron's hidden memory. The curve is
open along time, not a closed 3D limit cycle.

## First feature set

- Per-spike ISI and first/late frequency ratios; loop spacing is essentially ISI.
- Per-spike peak, trough, max/min dV/dt, loop area and depolarization duration.
- Change in these quantities across spike number and physical time.
- Signed waveform-shape residuals after alignment at a fixed event.
- Coupling between preceding ISI and the next spike's area/width/upstroke.
- A few functional principal-component scores of aligned loop sequences,
  trained only within each training fold.

Keep physical amplitude and timing coordinates alongside normalized shape.
Do not independently standardize every cell and erase the biological variation
we want to predict. Use fixed training-derived scales for time, voltage, and
velocity when computing geometric distances. Comparing an early burst and a
late regular train requires preserving physical intervals during alignment.

## Curvature and torsion

For scaled coordinates r, curvature is |r' x r''| / |r'|^3, and torsion is
det(r',r'',r''') / |r' x r''|^2 when its denominator is nonzero. Because the
third coordinate is already V', curvature requires V''' and torsion V''''.
These derivatives amplify noise and depend on scale and smoothing choices.
Start with loop sequences and derivative extrema. Test curvature only after
sweep-to-sweep reliability and sensitivity to noise/filter bandwidth are known.

## Small experiment before another population fit

Use representative repeated long-square sweeps from both temperature cohorts.
Plot raw/smoothed traces, 3D curves, and per-spike feature trajectories. Compare
repeat-sweep agreement with between-cell variation. Then compare matched-size
feature blocks: current waveform/timing baseline versus added sequence features.
Evaluate donor-held-out RNA prediction and within-class gains with nested tuning.
No claim of a novel dynamical model follows from a new visualization alone.

If reliable sequence features add predictive signal, use them to constrain a
slow state controlling phase speed and waveform coefficients. Their dynamics
can motivate that state; they do not determine its equation uniquely.
