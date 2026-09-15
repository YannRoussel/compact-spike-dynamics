# Direct slow-potassium experiment

## Why one flexible potassium gate is still restricted

Freely varying `p_inf(V)` and `tau_p(V)` makes one first-order gate flexible
in voltage, but it does not give the gate two independent memories. A
one-state effective potassium current has the form

```text
I_K = g_K p(t) (V - E_K)
dp/dt = (p_inf(V) - p) / tau_p(V)
```

A mixture of fast and slow potassium components instead has, schematically,

```text
I_K = (g_fast n_fast(t) + g_slow n_slow(t)) (V - E_K)
```

The same weighted total activation can correspond to different
`(n_fast, n_slow)` pairs. Those pairs have different future derivatives when
their time constants differ. Therefore the mixture cannot generally be
collapsed into one Markovian first-order state, even if the effective
steady-state and time-constant curves are arbitrarily flexible.

It is still reasonable to call a multi-state construction one theoretical
potassium *channel module*. The important model-complexity count is then the
number of independent dynamical states, not the number of current labels.

## Model tested

The guarded direct `m/h/n` model selected for each pilot cell was frozen. One
additional soma-local, non-inactivating potassium state was added:

```text
I_slowK = g_slow p (V - E_K)
dp/dt = (p_inf(V) - p) / tau_p(V)
```

`p_inf` is monotone increasing and `tau_p` is positive. Both are represented
by monotone cubic interpolation at -100, -70, -50, -30, 0, and 60 mV. The
gate has 12 direct kinetic coordinates plus `g_slow`. Its reference
temperature is 6.3 C with a fixed Q10 of 2.3.

Every candidate is passive-reanchored to preserve:

- the measured resting voltage
- the finite-step input-resistance endpoint
- the coupled soma-AIS resting state

Fast gates and the new slow gate use exact Rush-Larsen updates.

## Measurement correction

The biological “rheobase” sweep is the lowest *sampled* current step that
spikes, not the exact rheobase. For example, Scala 20180920_sample_1 produces
15 spikes in its lowest sampled spiking sweep. Replaying a model at its
bisected rheobase necessarily produces about one spike and is not a matched
spike-train comparison.

The final experiment therefore separates two measurements:

- Phase shape is evaluated at matched relative excitability using each
  model's bisected rheobase.
- Spike-train features are evaluated at the exact biological current in pA
  over the same 500 ms window.

Spike-count loss is explicit rather than being diluted within the larger
firing-pattern feature group. Held-out current is excluded from optimization.

## Four-cell pilot

The pilot used 8 CMA candidates over 2 generations for each cell. This is a
mechanism screen, not a converged 13-dimensional parameter estimate.

| Cell | `g_slow` | Rheobase parent to slow K | Train-loss change | Count-loss change | Phase-shape change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gouwens 704047023 | 0.0049 | 113 to 113 pA | 0% | 0% | +2.8% |
| Gouwens 674495385 | 0.335 | 47 to 66 pA | -60.5% | -66.1% | +15.4% |
| Scala 20180920_sample_1 | 0.050 | 105 to 105 pA | -1.5% | -3.1% | +3.8% |
| Scala 20190425_sample_4 | 0.168 | 207 to 219 pA | approximately 0% | 0% | +1.8% |

Negative loss changes are improvements. Gouwens 674495385 is the only clear
train-level response. Its spike counts changed as follows:

| Protocol | Biological | Frozen fast | Direct slow K |
| --- | ---: | ---: | ---: |
| sampled rheobase | 1 | 27 | 1 |
| suprathreshold | 10 | 50 | 37 |
| held out | 17 | 65 | 54 |

The extension fixes the coarse rheobase behavior and reduces high-current
firing, but it remains much too fast. It also worsens first-spike phase shape
and physical constraints. The optimized activation is already strong near
-50 mV and has a recording-temperature tau of about 63 ms there, falling to
about 13 ms near -30 mV.

For mechanism selection, slow K is retained only when the total objective
improves and either firing-pattern or spike-count loss improves by at least
5%. Under this rule only Gouwens 674495385 retains the added state; the other
three return to the frozen fast parent.

For Scala 20180920_sample_1, counts change only from 47 to 46, 65 to 64, and
79 to 78 against biological counts of 15, 29, and 37. The improvement is too
small to support the mechanism by itself.

Gouwens 704047023 and Scala 20190425_sample_4 enter a depolarized plateau
after one spike. A dense conductance probe with an M-like gate found no
repetitive-firing interval: increasing `g_slow` moved each model directly from
one spike plus plateau to no spike.

## Conclusion

One extra slow outward state is useful for a subset of cells, particularly
for correcting low rheobase and excessive firing. It is not a universal
solution for the four pilots and does not solve the first-spike phase-plane
scale mismatch.

The result supports a compact **effective potassium module with at least two
states** rather than one all-purpose gate:

1. retain the optimized fast repolarizing `n` state
2. add an independently optimized slow/adaptive `p` state
3. fit the two jointly in a narrow trust region
4. retain a no-slow branch for cells where `g_slow` collapses

For plateauing cells, the next targeted degree of freedom should affect
sodium recovery or the soma-AIS interaction rather than add more purely
outward current. A slow sodium-inactivation/recovery state or AIS-specific
`h` kinetics is a more direct test.

## Reproduction

```bash
PYTHONPATH=src python scripts/run_direct_slow_k_pilots.py \
  --direct-root outputs/direct_phase_pilots_rheobase_guarded \
  --output-root outputs/direct_slow_k_absolute_pilots \
  --population-size 8 \
  --generations 2 \
  --duration-ms 500 \
  --workers 1
```

The main outputs are:

- `combined_stage_summary.csv`
- `combined_firing_pattern_comparison.csv`
- `cross_cell_slow_k_comparison.png`
- `cross_cell_spike_counts.png`
- per-cell phase, trace, score, and optimized slow-gate figures
