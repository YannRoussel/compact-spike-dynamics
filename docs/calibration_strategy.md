# Calibration strategy

The goal is to cover a biologically measured e-feature space before asking
which e-features predict alpha/beta parameters. Coverage and identifiability are
different questions: a model family can cover the recordings while individual
channel parameters remain non-identifiable.

## Staged search

1. Sample static membrane parameters with canonical HH kinetics.
2. Rank retained models by nearest-neighbor distance in the 11-feature
   intrinsic biological space.
3. Build padded static bounds from the closest models.
4. Jointly sample those static bounds and conservative kinetic bounds.
5. Expand only kinetic directions associated with under-covered features.
6. Repeat at room and physiological temperatures with protocol-matched sweeps.
7. Train parameter-recovery models only after thousands of accepted models
   cover the biological support.

`calibrate-prior` uses all biologically plausible models when enough are
available. Otherwise it uses the nearest models up to `--minimum-models`.
Proposed ranges retain at least 35% of the original prior width and add padding,
so this is a conservative proposal rather than an optimizer collapsing onto a
few pilot samples.

## Pilot checkpoint

The initial static-only run retained 14 of 32 models. Five were plausible in
the five-dimensional waveform space and three in the 11-dimensional intrinsic
space.

A 64-model joint run using calibrated static ranges and conservative kinetic
ranges retained 25 models. Nine were waveform-plausible and five were
intrinsic-plausible after correcting the Scala sag convention. Coverage of the
pooled biological 1-99% interval was:

| Feature | Fraction covered |
| --- | ---: |
| Baseline voltage | 100% |
| Input resistance | 100% |
| Rheobase | 100% |
| Upstroke/downstroke ratio | 98.8% |
| AP peak | 93.8% |
| Fast trough | 67.2% |
| Sag ratio | 100% |
| AP width | 31.4% |
| AP threshold | 29.5% |
| Membrane time constant | 20.5% |
| Latency | 1.9% |

The full local biological coverage fraction is still below 1%, as expected
with only 25 model points against 4,910 complete recordings. At this scale,
per-feature interval coverage and the fraction of plausible models are more
diagnostic than raw point coverage.

Expanding all 18 kinetic dimensions by 1.5 times reduced acceptance to 17.2%
and produced no intrinsic-plausible survivors. The next expansion should
therefore be targeted. Slower gate time scales and capacitance/leak combinations
are candidates for AP width and membrane time constant. Threshold shifts should
be explored separately from global rate scaling.

## Refined protocol checkpoint

The screen now brackets rheobase on a coarse current grid and bisects the
non-spiking/spiking interval to a default precision of 2 pA. Adaptive event
integration stops a trial at its first spike. Full-resolution traces are
generated only for the final waveform and hyperpolarizing sweeps.

A paired 600 ms rescreen retained 24 of the earlier 25 joint models. Against
Scala room-temperature cells, 4/24 were locally plausible. In pooled space,
7/24 were plausible. The modeled latency 1-99% range expanded to approximately
0-33 ms, but the Scala room-temperature range extends to approximately 506 ms.

A denser 96-model calibration pass retained 42 models. It improved pooled
interval coverage for AP width to 71%, membrane time constant to 100%, and the
latency upper range to approximately 65 ms. Eight models were intrinsically
plausible before long-sweep rescreening. Rescreening the 12 nearest models at
600 ms retained all 12; five were locally plausible against Scala room
temperature.

The Gouwens Patch-seq recordings used 1 s steps at 34 degrees C. Only 2/12 of
the room-temperature-selected candidates remained spiking after transfer to
that condition. Separate warm-temperature calibration is therefore required;
Q10 cannot be treated as a harmless post-hoc rescaling.

A warm static/Q10 scan retained 6/64 models. Their `Q10_m` values ranged from
approximately 1.76 to 2.46. Using a padded prior from those survivors, an
independent 29-parameter warm scan retained 9/96 models, three of which were
locally Gouwens-plausible.

A gate-correlated 29-parameter scan retained 12/96 models. Compared with
independent sampling, it improved threshold, peak, trough, latency, and membrane
time-constant range coverage and covered three Gouwens cells rather than two.
It slightly reduced AP-width coverage and the fraction of plausible survivors.
Production sampling should therefore mix independent and gate-correlated draws.

The scalable workflow is:

1. Draw many candidates with the fast bisection preset.
2. Rank them in corrected intrinsic feature space.
3. Rescreen the nearest subset with the matched long protocol.
4. Calibrate room and warm temperature priors separately.
5. Retain cross-temperature models to constrain `Q10_m`, `Q10_h`, and `Q10_n`.

## Mixed 1,000-model checkpoint

Two separate fast screens used a 50:50 mixture of independent and
gate-correlated kinetic draws. The room-temperature screen used the corrected
Scala-calibrated prior at 22 degrees C. The warm screen used the calibrated
static/Q10 prior and broader kinetic windows at 34 degrees C.

| Screen | Accepted | Independent | Gate-correlated |
| --- | ---: | ---: | ---: |
| 22 degrees C | 486/1,000 | 207/501 | 279/499 |
| 34 degrees C | 116/1,000 | 45/501 | 71/499 |

Against the matched cohorts, 61/486 RT models were locally
Scala-room-temperature-plausible and 12/115 complete warm models were locally
Gouwens-plausible. The strict local coverage criterion covered 5/1,328 Scala
RT cells and 3/3,398 Gouwens cells. Gate correlation improved spiking yield at
both temperatures. Independent warm survivors had a higher plausible fraction,
so both strategies should remain in the search.

Coverage plots have two distinct meanings:

- Matched plots fit scaling, PCA, and local distances to one recording cohort.
  Use Scala room temperature for RT calibration and Gouwens VISp for 34-degree
  calibration.
- Pooled reference plots fit the geometry to all three cohorts and display
  Gouwens, Scala room temperature, and Scala physiological temperature
  together. They are useful context but are not used to calibrate either
  temperature-specific population.

The two-current HH model has no `Ih`. It can reproduce some numerical sag
values through passive and sodium/potassium transients, but broad biological
sag coverage will eventually require adding an `Ih` current or excluding sag
from the two-channel inverse problem.

## Raw spike-cycle checkpoint

Raw local NWB recordings added 23 measurement-matched waveform, onset,
inflection, curvature, and phase-loop features. A 6,000-attempt calibration
checkpoint mixed cycle-targeted and broad proposals at 22 and 34 degrees C.
After combining with the original populations and removing duplicate parameter
vectors, the coverage analysis used:

| Cohort | Accepted rows | Cycle-complete models | Biological cells covered |
| --- | ---: | ---: | ---: |
| Scala room temperature | 1,647 | 1,513 | 2 / 1,286 |
| Gouwens VISp | 431 | 378 | 4 / 464 |

The median nearest-model/local-radius ratio decreased from 5.07 to 4.72 over
92 to 1,513 RT models, and from 3.49 to 2.78 over 45 to 378 warm models.
Coverage itself plateaued. Cycle-targeted proposals improved median distances,
but adding more samples from either random prior is converging too slowly.

The next stage should use these models to train forward surrogates and acquire
parameters against diverse biological targets in rounds of approximately 500
accepted models. Continue toward 5,000 RT and 3,000 warm accepted models only
while each round materially lowers biological distance or adds covered cells.

Split violin comparisons supplement interval and local coverage. Each feature
is normalized by the biological median and IQR, with biological density on the
left and model density on the right. The panels are ordered by normalized
Wasserstein distance. This distinguishes merely spanning a biological interval
from placing substantial model probability in the same region. The outer 0.5%
of each density is clipped only for display; raw and normalized values remain
available in the exported distribution tables.

## Coverage bottleneck audit

The 6,000-attempt checkpoint did not justify further random scaling. The RT
population contained 1,513 spike-cycle-complete models and covered 2/1,286
Scala cells. The warm population contained 378 complete models and covered
4/464 local Gouwens cells.

This is not solely an artifact of the fifth-neighbor criterion. In 20
same-distribution controls, half of each biological cohort served as
pseudo-models for the other half. Mean coverage was 97.2% for Scala and 96.8%
for Gouwens, despite using only 643 and 232 pseudo-models respectively.

The mismatch is already present in five waveform-core dimensions: 16/1,328
Scala cells and 27/466 Gouwens cells were covered. Adding all spike-cycle
features reduces those counts to 2 and 4. Onset rapidness accounts for
approximately 40% of the squared nearest-model distance in Scala and 43% in
Gouwens. Removing the onset feature family lowers the median distance ratio
from 4.72 to 3.59 in Scala and from 2.78 to 2.12 in Gouwens, but does not close
the gap.

The strict feature vector also contains deterministic redundancies:

- onset dV/dt is a fixed fraction of maximum upstroke;
- AP duration is upstroke time plus repolarization time;
- relative inflection voltages are differences of absolute coordinates;
- upstroke/downstroke ratio is derived from the two velocities.

Removing these redundancies does not rescue strict coverage, but avoids
implicitly weighting some waveform properties multiple times. A preliminary
95%-variance PCA of 21 nonredundant features covers 4/1,286 Scala cells using
seven components and 74/464 Gouwens cells using five components. This reduced
manifold result is diagnostic rather than a replacement primary metric.

The next checkpoint should therefore:

1. separate channel-sensitive, spatial-initiation, and protocol-sensitive
   feature blocks;
2. verify model/recording filtering and derivative estimation;
3. optimize directly against representative biological medoids;
4. stop random population expansion until targeted fits establish which
   biological regions are reachable by a one-compartment Na/K model.

## Direct-fit pilots

A first targeted pilot selected two robust biological medoids per cohort. The
fit objective equally weighted five nonredundant channel-sensitive families:
waveform voltage, velocity, timing, inflection coordinates, and normalized
phase geometry. Onset and acceleration were held out for validation. Each
target started from its 12 nearest existing models and used three evolutionary
rounds of 32 direct simulations. In the first version, the three best
fast-screen candidates were evaluated with the full Scala or Gouwens protocol
only after the search.

The RT targets represented 761 and 525 cells. Long-protocol objective loss
improved by 13.4% and 7.6%, but target-local radius ratios decreased only from
5.29 to 4.83 and from 4.12 to 4.02. Neither target became locally covered.

After clipping extreme coordinates for medoid selection, the warm targets
represented 275 and 189 cells. One target worsened by 12.2% under the long
protocol despite improving on the fast screen. The second improved by 48.4%,
with its local-radius ratio decreasing from 3.21 to 2.20. It still remained
outside the biological neighborhood.

The pilot therefore establishes three points:

1. direct search improves representative fits more efficiently than random
   population expansion;
2. fast rheobase screening can mis-rank candidates relative to the matched
   long protocol;
3. onset remains poorly matched even when the fitted channel-sensitive
   families improve.

The implemented multi-fidelity version promotes four fast-screen candidates
from every round to the matched long protocol and selects the next round's
elites only from accumulated long-protocol results. With the same three rounds
of 32 proposals, RT long-protocol loss improved by 6.5% and 17.4%; local-radius
ratios changed from 5.29 to 5.08 and from 3.97 to 3.66. At 34 degrees C, one
target's apparent 39.6% fast-screen improvement shrank to 0.5% under the long
protocol, while the other retained a 49.3% improvement. Their local-radius
ratios ended at 2.80 and 2.25.

The multi-fidelity procedure therefore removes an important source of
fast-screen over-optimism, but it does not establish local biological coverage.
Scaling to dozens of medoids should wait until the remaining model-family and
measurement-filtering mismatches are tested explicitly.

## Trace-level coverage audit

Replaying the nearest models for all six spike-cycle-covered cells showed that
the local-radius cutoff is not the primary limitation. The current spike-cycle
profile measures first-cycle shape but excludes baseline voltage, rheobase, and
spike count. Three covered warm models therefore rested 12-42 mV away from the
recording and produced 46-72 spikes where the biological sweep produced one or
two. A warm cell just outside the cutoff at radius ratio 1.054 matched the full
response more plausibly than several cells below one.

Coverage should consequently be reported in two tiers:

1. `spike-shape covered`: the current first-cycle local-radius criterion;
2. `response covered`: spike-shape coverage plus matched long-protocol
   survival, baseline, rheobase, spike count, latency, and adaptation checks.

The ratio-one boundary remains a useful density-relative rule within a
well-defined feature block. It should not by itself be interpreted as
whole-cell electrophysiological coverage.
