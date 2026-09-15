# Local Patch-seq NWB data

## Inventory

The local folders contain real current-clamp sweeps, not only the processed
e-feature tables:

| Cohort | Raw NWB files | Unique raw cell IDs | IDs in current processed table |
| --- | ---: | ---: | ---: |
| Scala room temperature | 1,328 | 1,328 | 1,328 |
| Gouwens VISp Patch-seq subset | 573 | 573 | 466 |

The Scala physiological-temperature processed cohort contains 184 cells, but
its raw sweeps are not present in the inspected `Scala_pseq_data/000008`
collection. The local Gouwens files are a 573-cell Patch-seq subset of the
3,411-cell processed VISp matrix used by the coverage analysis. The 107 local
Gouwens cells absent from that matrix can still contribute raw electrophysiology
but cannot yet be joined to that processed transcriptomic table.

## Identifier mapping

- Scala filenames encode `YYYYMMDD_sample_N`, which maps directly to the
  processed release's `Cell` field.
- Gouwens NWB filenames encode subject and ephys-session IDs. The accompanying
  `2020-07-08_mouse_file_manifest.csv` maps each filename to the Allen
  `cell_specimen_id` used by the processed table.

## Sweep selection

The raw reader pairs acquisition and stimulus series by NWB `sweep_number`.
This is necessary because Scala files can contain an `IZeroClampSeries` without
a corresponding stimulus series.

For each current-clamp sweep it:

1. converts voltage to mV and current to pA;
2. detects the longest non-baseline current epoch;
3. retains positive square steps at least 100 ms long;
4. orders them by amplitude;
5. selects the lowest sampled amplitude with at least one spike.

This produces a **sampled rheobase**, not the model pipeline's bisection-refined
rheobase. Its precision is limited by the experimental current-step spacing.

## Measurement transform

Experimental features use the same transform as simulated observations:

- Savitzky-Golay voltage filter, 0.15 ms window, polynomial order 3;
- `dV/dt` and `d2V/dt2` derived from filtered voltage;
- first-spike waveform, onset, timing, curvature, inflection, and phase-loop
  geometry features;
- no model-only ionic-current diagnostics.

## Full local extraction

The complete pass produced:

| Cohort | Raw files | Successfully extracted | Joined to processed metadata |
| --- | ---: | ---: | ---: |
| Scala room temperature | 1,328 | 1,328 | 1,328 |
| Gouwens VISp | 573 | 536 | 466 |

All 37 failed Gouwens files are outside the 466-cell processed-data join:
36 are zero-byte files and one is a truncated HDF5 file. Therefore no currently
joinable Gouwens cell is lost.

All 1,794 joined cells have complete waveform and curvature blocks. Timing/onset
and closed-loop geometry are complete for 1,750 cells: 1,286 Scala and 464
Gouwens cells.

## Validation

Against the published scalar features for the same cells, full-cohort median
absolute differences were:

| Feature | Scala | Gouwens |
| --- | ---: | ---: |
| First-spike threshold | 0.594 mV | 0.425 mV |
| First-spike peak | 0.002 mV | 0.048 mV |
| First-spike half-width | 0.046 ms | 0.101 ms |
| Upstroke/downstroke ratio | 0.009 | 0.035 |

These close matches validate the legacy NWB unit handling and sampled-rheobase
sweep selection on both formats.

## Command

```bash
inverse-ephys extract-local-nwb \
  --scala-root "/path/to/Scala_pseq_data" \
  --gouwens-root "/path/to/Patch-seq AIBS" \
  --limit-per-dataset 3 \
  --output outputs/local_patchseq_spike_cycles_smoke.csv
```

Remove `--limit-per-dataset` for the full local extraction. The output is
atomically checkpointed every 50 files; pass `--resume` to skip dataset/cell
pairs already written by an interrupted run.
