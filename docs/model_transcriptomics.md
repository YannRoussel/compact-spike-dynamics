# Compact-model transcriptomics

## Analysis question

The response variables are no longer maximal conductances or parameters of
named ion-channel equations. Each biological cell is represented by the
validated compact generator:

```text
V = W(phase, I)
log T = s0(I) + u
du/dt = -u/tau
u <- u + d after every spike
```

`W` is the phase-aligned spike waveform, `s0` is a piecewise-linear
current-period relation, and `u` is one spike-driven adaptation coordinate.
This is the smallest tested default that preserved spike-loop geometry and
improved spike-train timing across both local Patch-seq cohorts.

## All-cell fitting

For every NWB cell with at least three repetitive-spiking long-square current
levels:

1. one interior current is held out and bracketed by lower and higher training
   currents;
2. the compact model is fitted on the remaining currents and scored on the
   held-out current;
3. the final model is refitted on all usable spiking currents;
4. a fixed-width parameter vector is exported.

The fixed vector contains resting voltage, rheobase and current-range terms,
the timing decay and spike jump, baseline periods at five normalized current
levels, segment durations, and ten Fourier harmonics of the waveform at the
same five levels. Voltage Fourier coefficients are expressed relative to the
cell's resting voltage. Ten harmonics are used because eight already retain a
median 99.98% of template coefficient energy in the 100-cell validation
cohort.

Run or resume the fit with:

```bash
PYTHONPATH=src python scripts/run_compact_population.py \
  --workers 16 \
  --resume \
  --output-root outputs/compact_population_all
```

The parameter table retains held-out phase-loop, count, timing, and voltage
errors. These are diagnostics, not RRR response variables.

## RNA alignment

The transcriptomic inputs come from the local analysis repository accompanying
Kobak et al.:

- Gouwens: the preprocessed `T_dat` matrix in the coupled-autoencoder `.mat`
  file, aligned by `T_spec_id_label`. The `cells` entry in the derived Kobak
  pickle contains cluster IDs and must not be used as a specimen join key.
  Expressed `Scn*` and `Kcn*` rows absent from `T_dat` are restored from the
  local full CPM release and cached in the analysis output.
- Scala: raw read counts from `scala2020.pickle`, restricted to the published
  1000-gene mask plus expressed `Scn*`/`Kcn*` genes, transformed as
  `log2(CPM + 1)`.

Gouwens donor IDs come from the AIBS metadata table. Scala donor IDs come from
the `Mouse` column in the M1 metadata table.

## Reduced-rank regression

RNA expression is the predictor matrix `X`; compact-model parameters are the
multivariate response `Y`. The analysis is run separately for Gouwens and
Scala because recording temperature, protocol, region, expression processing,
and sampled cell classes differ.

The pipeline uses:

- donor-held-out cross-validation;
- fold-local centering and scaling;
- ridge RRR to select rank and ridge penalty;
- row-sparse elastic-net RRR with `l1_ratio = 0.5`;
- the one-standard-error rule to choose a sparser gene set;
- a relaxed ridge-RRR refit using only the selected genes.

The reported out-of-fold replay keeps donors separated, but it reuses the
rank and penalties selected from the same cross-validation folds. It is an
exploratory model-selection score, not a fully nested and unbiased estimate of
future-donor performance. A confirmatory analysis should use nested grouped
cross-validation or a locked external cohort.

Positive timing/current coordinates are log-transformed before fitting.
Redundant exported coordinates, such as both current maximum and span or
template extrema already encoded by Fourier coefficients, are excluded from
`Y`.

Two cohorts are reported:

- `all_eligible`: every RNA-matched fitted cell;
- `high_qc`: held-out phase Chamfer at most `0.01`, relative spike-count error
  at most `0.5`, and within-current late-period NRMSE at most `2.0`.

The second is a predeclared sensitivity analysis. It should not replace the
full cohort because fit quality can be correlated with biological cell class.

Run:

```bash
PYTHONPATH=src python scripts/run_transcriptomic_rrr.py \
  --parameters outputs/compact_population_all/compact_model_parameters.csv \
  --output-root outputs/transcriptomic_rrr
```

Each dataset/cohort directory contains donor-held-out CV tables, selected
genes, response loadings, per-parameter out-of-fold performance, cell
components, and summary figures. `Scn*` and `Kcn*` genes are flagged in the
selected-gene table, but the primary model does not restrict predictors to
those families. Targeted family inclusion only prevents the generic
high-variability filter from excluding the genes of direct interest. The
top-level cross-dataset tables report selected-gene
overlap relative to the shared gene universe and align latent components by
the cosine similarity of their compact-parameter loadings.

## Initial full-cohort results

The all-cell pass fitted 1744/1901 local NWB cells:

| Dataset | Compact fits | RNA matched | High-QC RNA matched |
|---|---:|---:|---:|
| Gouwens VISp | 429 | 366 | 274 |
| Scala RT | 1315 | 1208 | 743 |

Most failures were protocol ineligibility. Gouwens additionally contained 36
files that were not readable as HDF5/NWB. All failures and reasons are retained
in `outputs/compact_population_all/failures.csv`.

The selected relaxed sparse RRR models were:

| Dataset/cohort | Rank | Selected genes | CV replay R2 |
|---|---:|---:|---:|
| Gouwens all | 8 | 264 | 0.360 |
| Gouwens high QC | 8 | 262 | 0.370 |
| Scala all | 5 | 141 | 0.356 |
| Scala high QC | 8 | 392 | 0.322 |

These are cross-validated replay scores after hyperparameter selection, not
fully nested external-validation estimates.

Waveform harmonics and spike-segment durations carried most of the predictable
signal. Examples include Gouwens waveform coordinates with out-of-fold `R2`
around `0.55-0.59` and Scala downstroke duration at the low-current quarter
point with `R2 = 0.67`. Rheobase was also predictable (`0.53` in Gouwens and
`0.33` in Scala). In contrast, the discrete selected memory time constant was
not predictable (`R2 = -0.27` and `-0.12`), while the continuous spike-memory
jump retained modest signal (`0.23` and `0.34`). This argues against
interpreting the selected `tau` grid point as a stable cellular phenotype.

The shared predictor universe is only 341 genes because the two source
variable-gene panels differ. In the full cohorts, 38 selected genes overlap
versus 27 expected under a hypergeometric null (`p = 0.0027`). Recurrent
channel-family genes include `Scn1a`, `Scn3b`, `Kcnab3`, `Kcnc1`, and
`Kcnj5`. The first response component is similar across datasets
(`|cosine| = 0.72`), while later components are less stable. The high-QC
selected-gene overlap is not enriched (`p = 0.16`), so individual later
components and long gene lists should be treated as exploratory.

The `Scn`/`Kcn` families as a whole are not over-represented among selected
genes (hypergeometric `p = 0.78` for Gouwens and `0.18` for Scala in the full
cohorts). Their forced inclusion succeeded in making specific channel
candidates testable; it did not create evidence for global channel-family
enrichment.

Top predictors also include class markers such as `Lamp5`, `Vip`, `Pvalb`, and
`Sst`. The present analysis therefore captures both direct candidates and
broad transcriptomic-class structure. A next confirmatory analysis should
residualize broad class within each training fold or fit within sufficiently
large classes before making channel-specific mechanistic claims.

## Class-aware confirmatory analysis

The confirmatory analysis maps dataset-specific t-types to broad subclasses
(`Pvalb`, `Sst`, `Vip`, `Lamp5`, `Sncg`, and the major glutamatergic projection
classes). RNA and compact-model parameters are both residualized using class
means estimated exclusively from each training fold. Held-out donors are never
used to estimate class means, scaling, rank, or ridge penalty.

Prediction and gene selection are deliberately separated:

- nested donor-grouped ridge RRR provides the primary unbiased prediction
  estimate;
- class-residualized sparse RRR on the complete cohort provides exploratory
  gene loadings;
- separate within-class sparse RRR is reported where sample size permits.

Run and summarize with:

```bash
PYTHONPATH=src python scripts/run_class_aware_rrr.py \
  --parameters outputs/compact_population_all/compact_model_parameters.csv \
  --output-root outputs/class_aware_rrr
PYTHONPATH=src python scripts/summarize_class_aware_rrr.py \
  --output-root outputs/class_aware_rrr
```

The nested results are:

| Dataset/cohort | Class-only R2 | Class + RNA R2 | Delta R2 | Residual variance explained |
|---|---:|---:|---:|---:|
| Gouwens all | 0.315 | 0.396 | 0.081 | 0.119 |
| Gouwens high QC | 0.329 | 0.392 | 0.063 | 0.094 |
| Scala all | 0.346 | 0.371 | 0.025 | 0.039 |
| Scala high QC | 0.361 | 0.387 | 0.025 | 0.040 |

Broad class therefore explains most of the original RNA prediction, especially
in Scala, but it does not explain everything. Within-class RNA adds a
replicable positive increment in both datasets, with a substantially larger
effect in Gouwens.

For the all-eligible cohorts, RNA beyond class increases rheobase `R2` by
`0.072` in Gouwens and `0.057` in Scala. The spike-memory jump increases by
`0.062` and `0.011`. The tau-grid increment is only `0.025` and `0.012`, and
its absolute nested prediction remains below zero (`-0.248` and `-0.114`).
Thus class adjustment does not rescue the fitted tau as a stable molecular
phenotype. Separate Gouwens within-class CV replay gives `R2 = 0.071` for
Pvalb and `0.112` for Sst; these are supportive sensitivity results rather
than nested estimates.

## Adaptation identifiability experiment

The single-tau model was compared with a fixed four-timescale spike-history
kernel:

```text
log T = s0(I) + sum_j a_j z_j
dz_j/dt = -z_j/tau_j
z_j <- z_j + 1 after every spike
tau_j = 25, 100, 400, 1600 ms
```

The fixed timescales are shared by all cells. Cell-specific targets are the
kernel values at 0, 50, 200, and 800 ms, avoiding a discrete cell-specific tau
selection.

A balanced pilot used 100 previously eligible compact-model cells per dataset:

```bash
PYTHONPATH=src python scripts/run_adaptation_kernel_population.py \
  --limit-per-dataset 100 \
  --workers 8 \
  --executor thread \
  --output-root outputs/adaptation_kernel_eligible_pilot_200
PYTHONPATH=src python scripts/run_kernel_transcriptomic_pilot.py
```

The single-tau objective is nearly flat. The median relative score margin
between its best and second-best tau grid points is only `0.016` in Gouwens
and `0.0029` in Scala. This directly supports parameter non-identifiability.

The fixed kernel does not improve the held-out timing default:

| Dataset | Single-tau late-period NRMSE | Kernel NRMSE | Cells with lower late-ISI error |
|---|---:|---:|---:|
| Gouwens | 0.687 | 0.695 | 42% |
| Scala | 0.746 | 0.770 | 53% |

It improves adaptation-ratio error in 60% of Scala cells but only 47% of
Gouwens cells, with no consistent improvement in accumulated spike timing.
Kernel values fitted separately at individual currents are also only modestly
correlated with the full-cell fit, particularly at long lags.

The matched transcriptomic pilot contains 88 Gouwens and 90 Scala cells. The
nested within-class RNA increment is negative for both target families:

| Dataset | Single tau/jump Delta R2 | Fixed-kernel Delta R2 |
|---|---:|---:|
| Gouwens | -0.007 | -0.001 |
| Scala | -0.023 | -0.043 |

Consequently the fixed kernel was not scaled to all 1744 cells. More hidden
adaptation coordinates are not justified by the present long-square data.
The next timing experiment should first add a conditioning train followed by
variable silent recovery intervals, or sufficiently rich fluctuating current,
so that decay timescales are observed directly rather than inferred from one
short within-sweep transient.

## Interpretation

RRR finds low-rank predictive covariation. It does not prove that a selected
gene directly implements a fitted waveform or timing parameter. Cell-type,
developmental, and recording covariates can induce indirect associations.
Robust claims require donor-held-out prediction, agreement between datasets,
stability under the high-QC filter, and targeted follow-up within
transcriptomic classes.

## Reference

Kobak D, Bernaerts Y, Weis MA, Scala F, Tolias AS, Berens P (2021).
[Sparse reduced-rank regression for exploratory visualisation of paired
multivariate data](https://doi.org/10.1111/rssc.12494), *Journal of the Royal
Statistical Society: Series C* 70:980-1000. The accompanying
[analysis repository](https://github.com/berenslab/patch-seq-rrr) supplies the
local expression matrices used here.
