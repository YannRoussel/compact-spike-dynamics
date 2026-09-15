# Biological feature crosswalk

The coverage analysis uses semantic names rather than assuming that similarly
named source columns are interchangeable. This table records the exact mapping.

| Semantic feature | Scala M1 | Gouwens VISp | HH simulator | Default |
| --- | --- | --- | --- | --- |
| `ap_threshold_mv` | `AP threshold (mV)` | `ap_1_threshold_v_0_long_square` | `first_threshold_voltage_mv` | Yes |
| `ap_peak_mv` | threshold + `AP amplitude (mV)` | `ap_1_peak_v_0_long_square` | `first_peak_voltage_mv` | Yes |
| `ap_width_ms` | `AP width (ms)` | `1000 * ap_1_width_0_long_square` | `first_half_width_ms` | Yes |
| `fast_trough_mv` | threshold + `Afterhyperpolarization (mV)` | `ap_1_fast_trough_v_0_long_square` | `first_fast_trough_voltage_mv` | Yes |
| `upstroke_downstroke_ratio` | `Upstroke-to-downstroke ratio` | `ap_1_upstroke_downstroke_ratio_0_long_square` | `first_upstroke_downstroke_ratio` | Yes |
| `baseline_voltage_mv` | `Resting membrane potential (mV)` | `v_baseline` | `baseline_voltage_mv` | Expanded |
| `latency_ms` | `Latency (ms)` | `1000 * latency_0_long_square` | `first_spike_latency_ms` | Expanded |
| `input_resistance_mohm` | `Input resistance (MOhm)` | `input_resistance` | hyperpolarizing step estimate | Intrinsic |
| `membrane_tau_ms` | `Membrane time constant (ms)` | `1000 * tau` | hyperpolarizing step fit | Intrinsic |
| `rheobase_pa` | `Rheobase (pA)` | `rheobase_i` | upper bound of bisected firing boundary | Intrinsic |
| `sag_ratio` | `1 - 1 / Sag ratio` | `sag_nearest_minus_100` | hyperpolarizing step estimate | Intrinsic |
| `upstroke_mv_ms` | unavailable | `ap_1_upstroke_0_long_square` | `first_upstroke_mv_ms` | No |
| `downstroke_mv_ms` | unavailable | `ap_1_downstroke_0_long_square` | `first_downstroke_mv_ms` | No |

The Gouwens release stores these IPFX features as z-scores. The loader restores
physical values using `feature_mean` and `feature_std` from the same released
matrix.

## Protocol-sensitive features

Input resistance, membrane time constant, sag, and rheobase are available in
the `intrinsic` profile because the biological-screen simulator now runs
separate pA depolarizing and hyperpolarizing sweeps. They remain more
protocol-sensitive than the waveform core. The HH model has no `Ih`, so sag
should be treated as a passive or gating-transient descriptor rather than a
complete biological sag mechanism.

The Scala source defines its published ratio as peak hyperpolarizing deflection
divided by steady-state deflection, for which no sag equals 1. Gouwens and the
simulator use the recovered fraction, for which no sag equals 0. The Scala
values are therefore converted with `1 - 1 / ratio`.

Latency is particularly sensitive to sweep duration and distance above true
rheobase. The simulator measures latency from stimulus onset to AP threshold at
the spiking upper bound of a bisected rheobase interval. Long 600 and 1000 ms
presets still under-cover the biological long-latency tail, indicating a model
family limitation in addition to the earlier protocol bias.

The Scala `Max number of APs` feature is measured over a 600 ms sweep at the
highest usable current, whereas the Gouwens `avg_rate_0_long_square` feature is
measured near rheobase. They are retained under distinct semantic names and are
not treated as the same feature.

## Temperature

Scala provides room-temperature and physiological-temperature cohorts. The
simulator applies separate `Q10` values to `m`, `h`, and `n` rates relative to
the classic 6.3 degrees C reference. Coverage is reported separately for both
Scala temperature cohorts and Gouwens VISp.
