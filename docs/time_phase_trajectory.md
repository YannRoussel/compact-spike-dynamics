# Experimental time-resolved phase trajectories

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
