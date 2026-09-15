# Reproducibility and research status

## Snapshot: 2026-09-15

This is a research code snapshot, not a validated predictive atlas release.
The current population model is the phase-template recovery/onset model.
Historical HH and spatial models remain available for comparison. The Python
package keeps its original name to preserve script imports.

Latest local cohorts: 1,744 fitted cells (429 Gouwens, 1,315 Scala); paired RNA
is available for 366 and 1,208 respectively. Eligibility differs across analyses.
The atlas AP analysis retained 366 and 714 cells. Cohort counts must always be
reported with the relevant filtering rule.

## Installation and tests

Run from the cloned repository root with Python >=3.9:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[nwb,optim]'
MPLBACKEND=Agg python -m unittest discover -s tests -v
inverse-ephys inspect
```

The synthetic suite requires no biological downloads. GitHub Actions runs it
on Python 3.9 and 3.11. An Actions configuration is not evidence that remote CI
has already passed. The local environment is not a locked dependency snapshot;
full historical numerical reproducibility remains to be established in a fresh
environment before a tagged release.

## Main entry points

| Purpose | Script |
| --- | --- |
| Phase-template population | `scripts/run_phase_template_population.py` |
| Compact population export | `scripts/run_compact_population.py` |
| Latest recovery/onset targets | `scripts/run_recovery_onset_targets.py` |
| RNA held-out-cell experiment | `scripts/run_recovery_onset_heldout_srrr.py` |
| Trace, zoom, loop comparisons | `scripts/plot_recovery_onset_rna_replays.py` |
| Atlas prediction | `scripts/run_atlas_ephys_prediction.py` |
| Atlas combined tables | `scripts/summarize_atlas_ephys_predictions.py` |

Use each script's `--help` for input paths. Historical scripts contain local
path defaults and often consume earlier generated outputs. The commands in
the README document this sequence, but there is no single fresh-download
end-to-end pipeline yet. Do not treat missing cached files as model failures.

## Data and release boundaries

Git excludes NWBs, expression matrices, caches, fitted populations, and figures.
Obtain source data under the original releases' terms; this repository does
not relicense those data. The original alphas_and_betas repository's Apache 2.0
source-code license is preserved in `LICENSE` together with its Git history.

Before a scientific release, capture input checksums, specimen/donor splits,
taxonomy versions, environment versions, seeds, and command lines. Archive
small aggregate result tables alongside a provenance manifest; avoid committing
raw biological data or local credentials.

## What has and has not been established

- AP-loop geometry fits generalize across a substantial local population.
- RNA carries waveform-related signal; gains above broad class vary by cohort.
- Full RNA-predicted spike trains remain substantially less reliable.
- Atlas leave-type-out tuning was not nested; targets were fitted templates.
- Reference-to-Patch-seq expression alignment and taxonomy mapping need validation.
- General responses to arbitrary current and off-cycle stability are untested.

See `independent_validation.md` for the next evaluation design. Prior pilot
failure does not prove that any conductance-model family is incapable of fitting
the recordings.
