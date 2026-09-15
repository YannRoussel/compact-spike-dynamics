# Abstract phase reduction

## Objective

This branch removes conductances and channel gates from the inferred model.
The target is a compact dynamical description of the voltage trace and its
phase-plane loop that can later be compared with Patch-seq expression.

A scalar autonomous equation

```text
dV/dt = f(V)
```

cannot generate a spike loop: the same voltage occurs on the upstroke and
downstroke with different derivatives. The smallest continuous observed state
is therefore `(V, u)`, where `u = dV/dt`. A third slow state `z` is tested
rather than assumed.

## Acceleration-field baseline

The first model is a generalized Lienard field:

```text
dV/dt = u
du/dt = A(V) + B(V)u + D(V)u^2
         + P(V)I + Q(V)Iu + C(V)z
dz/dt = S(V) - z/tau_z
```

`A`, `B`, `D`, `P`, `Q`, and optional `C` are cubic B-splines. Coefficients
are estimated by regularized linear regression against smoothed
`d2V/dt2`. The fit includes explicit dynamical constraints:

- zero-current rest is a fixed point;
- rest has negative restoring slope and damping;
- repetitive suprathreshold traces destabilize the zero-velocity state;
- average divergence along a repetitive spike cycle is negative;
- the field has a dissipative extension outside the sampled `(V, u)` domain.

Slow time constants are selected by leave-one-training-sweep-out prediction.
The held-out current step is not used for model selection.

This baseline is informative but insufficient. It predicts acceleration along
observed trajectories, yet its free biological orbit either expands beyond or
contracts inside the measured loop as the stability constraints change. Local
field regression does not determine transverse geometry reliably from one
thin observed orbit.

## Landmark phase-template model

The second model makes the observed periodic orbit the primary object. Each
peak-to-peak cycle is split at physiological landmarks:

1. peak to afterhyperpolarization trough;
2. trough to the `20 mV/ms` upstroke onset;
3. upstroke onset to the next peak.

Each segment is resampled separately. This keeps fast upstroke and
repolarization durations independent of the much more variable recovery/ISI.
Periodic voltage splines are interpolated across injected-current levels.

The reduced state is phase plus optional spike-driven memory:

```text
phase speed = segment length / segment duration
dz/dt = -z/tau_z, with one unit added per completed cycle
period = beta0 + betaI / (I - Irheo + epsilon) + betaz z
```

The current coordinate follows the type-I-like inverse distance from rheobase.
The slow term is additive, deliberately avoiding a current-by-memory
interaction with only two repetitive training current levels. The waveform
is also exported as Fourier coefficients for later population-level
prediction and transcriptomic association.

## Pilot

The default pilot uses:

- canonical HH at `6.3 C`, trained at `7`, `9`, `12`, and `15 uA/cm2`, with
  `10 uA/cm2` held out;
- Gouwens cell `674495385`, with four local repetitive/spiking sweeps for
  training and the `150 pA` sweep held out.

Results in `outputs/abstract_phase_pilot`:

| Source | Selected phase model | Observed / predicted spikes | Phase-plane Chamfer |
|---|---:|---:|---:|
| canonical HH | no slow state | 14 / 15 | 0.0014 |
| Gouwens 674495385 | `tau_z = 500 ms` | 28 / 29 | 0.0024 |

The Gouwens slow-state evidence is modest rather than definitive: its
late-cycle selection score improves from about `0.916` to `0.903`. It should
be tested across many cells and current levels before interpreting `500 ms`
as a stable biological timescale.

## Population generalization

The population experiment uses 50 Gouwens and 50 room-temperature Scala cells.
For each cell, one interior repetitive-spiking current is held out. It must be
bracketed by lower and higher repetitive training currents. Cells are selected
in a deterministic shuffled order based only on protocol eligibility, not
model quality.

Waveform metrics are calculated on complete cycles after the first cycle.
Maximum and minimum `dV/dt` are the median cycle extrema, which prevents
stimulus edges and one anomalous spike from defining the comparison. The NWB
reader also resolves Scala files whose response sweeps mix volt-like,
millivolt-like, and microvolt-like legacy storage.

Results in `outputs/phase_template_population_100_stable_cycles`:

| Dataset | Cells | Median Chamfer | Chamfer <= 0.01 | Median count error | Count within 1 | Median max/min `dV/dt` ratio |
|---|---:|---:|---:|---:|---:|---:|
| Gouwens VISp | 50 | 0.0021 | 92% | 10.0 | 12% | 0.974 / 1.014 |
| Scala RT | 50 | 0.0071 | 70% | 11.5 | 28% | 1.084 / 1.032 |

The compact template therefore generalizes much better for the shape and
scale of a typical spike cycle than for spike-train timing. The optional slow
state is selected in 74% of Gouwens cells and 66% of Scala cells, but improves
held-out absolute count error in only 40% and 46%, respectively. A single
inverse-current plus spike-memory timing module is therefore not a sufficient
population model. This result alone does not identify which component fails.

## Timing-state ladder

The preceding conclusion confounded the inverse current-period law with the
memory state. A nested clock experiment separates them while freezing the
validated phase template:

```text
current spline:
    log T = s0(I)

spike memory:
    du/dt = -u/tau
    u <- u + d after a spike
    log T = s0(I) + u

Izhikevich-like recovery:
    du/dt = (b (V - Vrest) / Vscale - u) / tau
    u <- u + d after a spike
    log T = s0(I) + u
```

Here `s0(I)` is a piecewise-linear spline through the training current levels.
The recovery state is defined directly in log-period units, removing an
otherwise arbitrary output-coupling scale. Candidate `tau` values are selected
from `20`, `50`, `100`, `200`, `500`, and `1000 ms` by late-cycle prediction
within training sweeps. The same interior current remains completely held out.

Results in `outputs/phase_timing_population_100`:

| Dataset | Clock | Median count error | First-ISI error (ms) | Late-ISI error (ms) | Adaptation-ratio error |
|---|---|---:|---:|---:|---:|
| Gouwens | legacy inverse/memory | 10.0 | 6.76 | 9.87 | 0.513 |
| Gouwens | current spline | 9.5 | 13.60 | 7.78 | 0.664 |
| Gouwens | spike memory | **4.0** | 6.66 | **6.55** | 0.355 |
| Gouwens | Izhikevich recovery | 5.0 | **5.60** | 7.71 | **0.353** |
| Scala RT | legacy inverse/memory | 11.5 | **5.90** | 8.98 | 0.851 |
| Scala RT | current spline | 2.0 | 13.42 | 8.51 | 0.943 |
| Scala RT | spike memory | **1.0** | 11.16 | 5.99 | 0.782 |
| Scala RT | Izhikevich recovery | 1.5 | 10.60 | **5.24** | **0.717** |

The flexible current law accounts for a substantial part of the Scala count
improvement, while explicit spike history is particularly important in
Gouwens. The Izhikevich-like state improves selected interval and adaptation
metrics but does not consistently outperform the simpler exponential
spike-memory clock. Its voltage-drive coefficient also reaches its fitting
bound in 24% of Gouwens and 16% of Scala cells, so it is not yet sufficiently
identified for transcriptomic association.

Training-only selection chooses Izhikevich recovery for 37/50 Gouwens cells
but only 11/50 Scala cells; Scala instead selects the current-only spline in
27/50 cells. Training late-cycle scores correlate only weakly with held-out
current performance. A final per-cell clock choice therefore needs nested
cross-current and within-sweep validation rather than late-cycle validation
alone.

## Outputs

- `candidate_comparison.csv`: acceleration-field model ladder
- `stability_diagnostics.csv`: rest eigenvalues and cycle volume contraction
- `phase_template_candidates.csv`: slow-state period-prediction ladder
- `phase_template_metrics.csv`: held-out trace and phase-plane metrics
- `phase_template_fourier_coefficients.csv`: compact waveform parameters
- `*_summary.png`: acceleration-field diagnostic
- `*_phase_template.png`: landmark phase-template held-out comparison
- `cell_model_metrics.csv`: population held-out phase, extrema, and count scores
- `population_summary.csv`: dataset-level population summary
- `population_generalization.png`: balanced Gouwens/Scala validation figure
- `timing_model_metrics.csv`: decomposed held-out clock errors
- `timing_parameters.csv`: fixed-width current and recovery parameters
- `paired_comparisons.csv`: paired model differences and Wilcoxon tests
- `timing_ladder_generalization.png`: four-clock population comparison

## Interpretation boundary

The phase-template model is currently a faithful reduced generator under
long-square stimulation, especially for spike-loop geometry. It is not yet a
globally identified vector field for arbitrary voltage perturbations.

A transverse amplitude coordinate and brief-current-kick experiment are only
needed if the scientific claim expands to off-cycle recovery and local
stability. They are not required for the present goal of learning compact
population spike-cycle coordinates. The immediate modeling target exposed by
the 100-cell test is instead the current-frequency and adaptation law.
