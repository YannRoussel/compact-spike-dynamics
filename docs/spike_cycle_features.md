# Spike-cycle feature definitions

The one-compartment baseline keeps the minimal NaT + KDR + leak conductance
model but uses a rich observation set. Every observable feature is computed
from time and voltage alone after the same measurement transform that will be
applied to experimental sweeps.

## Measurement transform

- require regularly sampled time points within 1% tolerance
- smooth voltage with a 0.15 ms, third-order Savitzky-Golay filter
- compute `dV/dt` numerically from the filtered voltage
- define threshold at 5% of the spike's maximum upstroke

The filter width and polynomial order are explicit `FeatureConfig` parameters.
Changing them creates a different feature protocol and must be recorded.

## Cycle boundaries

The **AP loop** begins at threshold and ends when voltage crosses the same
threshold voltage on the downstroke.

For spikes that remain above their detected threshold through the next cycle,
the AP-loop duration and area are undefined by construction. This is retained
as meaningful waveform behavior rather than silently closing the loop at an
arbitrary voltage.

The **full cycle** begins at the first threshold and ends at the next threshold.
For a single-spike sweep, it ends at most 12 ms after the peak. This captures
the early AHP while avoiding dependence on the complete stimulus duration.

## Feature blocks

### Waveform

- threshold, peak, amplitude, and half-width
- fast trough
- maximum upstroke and downstroke
- upstroke/downstroke ratio

### Timing and onset

- threshold-to-peak time
- peak-to-threshold-return time
- total AP-loop duration
- onset voltage and phase-plot slope at 10% of maximum upstroke
- the actual `dV/dt` target used for that fractional onset measurement

A fixed 20 mV/ms onset voltage and rapidness are also emitted as optional
features. They are undefined when threshold already exceeds 20 mV/ms.

### Curvature

- maximum positive and negative `d2V/dt2` and their voltages
- `d2V/dt2 = 0` coordinates nearest maximum upstroke
- `d2V/dt2 = 0` coordinates nearest maximum downstroke
- both inflection voltages relative to threshold or peak

### Loop geometry

- AP-loop and full-cycle polygon areas in the `(V, dV/dt)` plane
- area normalized by AP amplitude times the upstroke-to-downstroke `dV/dt` span
- full-cycle path length after separately normalizing voltage and `dV/dt`

## Current diagnostics

The simulator additionally reports `-Iion` at the upstroke and downstroke
inflections. These are model diagnostics, not core observables. An experimental
equivalent requires injected current and total capacitance:

```text
-Iion = C * dV/dt - Iapp
```

Series resistance, capacitance compensation, and current sign conventions must
be harmonized before using reconstructed current coordinates.

## Coverage policy

Twenty-nine simultaneous dimensions make local nearest-neighbor coverage highly
sensitive to biological sample density. Coverage should therefore be reported:

1. for standard waveform features;
2. separately for timing/onset, curvature, and loop geometry;
3. jointly after robust scaling and a prespecified dimension-reduction rule;
4. with held-out parameter recoverability, not coverage alone.

The processed Scala and Gouwens feature tables do not contain these derivative
features. Raw voltage sweeps are required before biological spike-cycle coverage
can be calculated.
