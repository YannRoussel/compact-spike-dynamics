# Spike-cycle project scope

## Central question

Can somatic action-potential and phase-plane geometry recover effective fast
sodium and potassium kinetics, and do those recovered kinetic phenotypes
associate with `Scn` and `Kcn` expression?

This is narrower than reproducing a complete electrophysiological type. Passive,
subthreshold, sag, latency, adaptation, and bursting properties require currents
that are not present in a transient-Na/delayed-rectifier-K model.

## Evidence from the 1,000-model screens

In the 11-feature pooled biological PCA, PC2 is primarily an
excitability/passive-property axis. Its strongest absolute correlations are
rheobase (0.75), input resistance (0.70), membrane time constant (0.59), AP peak
(0.58), baseline voltage (0.58), fast trough (0.51), and AP width (0.51). Sag is
moderate (0.34), while latency and threshold contribute little.

Restricting the comparison to the five harmonized spike-waveform features
changes the geometry. PC2 is then dominated by threshold opposed to AP width
and upstroke/downstroke ratio.

| Matched cohort | Intrinsic covered | Waveform covered | Intrinsic plausible | Waveform plausible |
| --- | ---: | ---: | ---: | ---: |
| Scala RT | 5/1,328 | 7/1,328 | 61/486 | 72/486 |
| Gouwens 34 degrees C | 3/3,398 | 31/3,399 | 12/115 | 23/115 |

The strict local coverage fraction remains low. This is a model-family and
space-filling diagnostic, not evidence that every accepted model is invalid.

## Proposed nested models

1. **M0: one compartment, NaT + KDR + leak.** Establish the recoverability of
   effective fast inward and outward kinetics from the spike cycle.
2. **M1: soma + axon initial segment, same channel families.** Test whether
   phase-plane onset and inflection structure requires spatial spike initiation
   rather than different channel kinetics.
3. **M2: add a high-threshold, fast K current.** Test whether a Kv3-like
   repolarizing component is needed to decouple threshold, width, downstroke,
   AHP, and high-frequency firing.
4. Add a Kv4-like transient K current only if latency or delayed firing returns
   to the target feature set.
5. Add `Ih`, M current, or calcium-dependent currents only in a separate
   whole-cell e-type branch that explicitly targets sag, adaptation, bursting,
   or slow AHP.

Each extension should be accepted only when it improves held-out biological
coverage and parameter recoverability enough to justify its additional
dimensions.

## Spike-cycle feature set

- threshold, onset rapidness, peak, amplitude, and half-width
- maximum upstroke and downstroke and their ratio
- fast trough and fast AHP
- phase-loop area
- voltage and `dV/dt` at all robust upstroke and downstroke curvature changes
- corresponding current coordinates when capacitance and injected current are
  sufficiently well measured
- the same features at rheobase and fixed offsets above rheobase

Raw sweeps are required for the derivative and curvature features. Derivatives
must use a common filtering, sampling-rate, and junction-potential policy.

## Static parameters

Resting voltage, input resistance, and membrane time constant should calibrate
or condition the nuisance passive parameters rather than enter the spike-cycle
target space. A practical approximation is to use measured resting voltage,
estimate total leak conductance from input resistance, and estimate total
capacitance from `tau / Rin`, while retaining uncertainty because active
subthreshold currents violate a purely passive estimate.

## Transcriptomic interpretation

The primary analysis can be restricted to `Scn` and `Kcn` genes, grouped by
biophysical family. Recovered model quantities should be described as effective
kinetic phenotypes, not direct estimates of a particular gene's conductance.
Auxiliary subunits, post-translational modulation, trafficking, axon-initial-
segment localization, morphology, and recording conditions can all separate
mRNA abundance from functional membrane conductance.

Inference should therefore retain posterior uncertainty or parameter
equivalence classes. Gate steady-state and time-constant descriptors over the
voltage range visited by the spike are likely more identifiable targets than
independent alpha and beta values outside that trajectory.
