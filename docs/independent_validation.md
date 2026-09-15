# Independent validation candidates

Literature and public metadata checked 2026-09-15. This is a shortlist, not a
completed external validation. New publication dates do not prove new specimens.

## Best first candidate

Sorensen, Gouwens and colleagues, *Connecting single-cell transcriptomes to
projectomes in the mouse visual cortex*, Nature (2026), reports 1,528 neurons
with electrophysiology and transcriptomics, focusing on excitatory populations.

- Paper: https://doi.org/10.1038/s41586-026-10424-8
- Raw electrophysiology: https://dandiarchive.org/dandiset/001455
- Frozen release: `0.250625.2239`
- Public DANDI API verified 1,528 assets, 52,219,233,046 bytes (~52.2 GB).
- RNA location linked by the paper: https://data.nemoarchive.org/other/grant/aibs_internal/zeng/transcriptome/scell/PATCHseq/mouse/raw/

This is the strongest candidate for extending cortical validation, especially
excitatory types. It is from the Allen pipeline, so it is not an independent-lab
replication. Its 2023 preprint and 2025 data release predate the final paper.
Verify specimen and donor overlap with every training source, confirm a usable
processed expression table and specimen join, and inspect temperature, current
protocols, filtering, and liquid-junction correction before selecting cells.
Our warm training cohort is predominantly inhibitory; warm excitatory prediction
is a domain-transfer test, not an already supported calibration class.

## Additional candidates

| Dataset | Useful test | Limitation |
| --- | --- | --- |
| Budzillo et al., mouse basal ganglia, 2026 preprint; 904 paired ephys/RNA cells | Strong region/type shift and AP kinetics | Beyond our cortical taxonomy; data joins/access not yet verified |
| Berg et al., human L2/3 cortex, Nature 2021 | Cross-species representation and waveform fitting | Mouse RNA predictor cannot be assumed transferable |
| Cadwell/Fuzik early mouse Patch-seq cohorts | Independent laboratories, shared scalar feature checks | Small cohorts, RNA QC and taxonomy uncertainty; raw waveform access needs checking |

Sources:

- Basal ganglia preprint: https://doi.org/10.64898/2026.06.24.734325
- Berg: https://doi.org/10.1038/s41586-021-03813-8
- Early datasets and QC: https://doi.org/10.3389/fnmol.2018.00363

Reanalyses of Gouwens/Scala do not constitute independent samples. The early
dataset QC study obtained summarized ephys tables from authors; availability
of such tables does not imply public raw traces suitable for phase geometry.

## Evaluation to freeze before inspecting outcomes

1. Define the claim: waveform reconstruction, RNA-to-ephys prediction, or
   prediction of genuinely unobserved t-types. Evaluate these separately.
2. Freeze specimen/donor exclusion lists and a taxonomy crosswalk derived from
   RNA alone. For unseen-type tests, exclude the type from all training sources.
3. Extract targets directly from raw biological traces using the same physical
   filtering and definitions. Report failures and eligible fractions. Use a
   specified current protocol; q50 of each cell's range is not matched drive.
4. Freeze preprocessing, normalization and model selection on training data.
   In leave-type-out validation, nest all tuning and gene selection inside the
   outer split. Do not use held-out ephys for temperature/batch calibration.
5. Compare RNA predictions with class means, nearest-RNA-centroid transfer,
   ridge regression, and RRR. Report improvement over class, within-class R2,
   absolute errors, and donor/type-bootstrap intervals.
6. Evaluate both direct feature predictions and features extracted from generated
   traces. Their agreement must be measured, not assumed. Distinguish negative
   results caused by RNA transfer from limitations of waveform reconstruction.
7. For atlas claims, test predictions from independent atlas RNA centroids
   against external type-mean raw ephys. Patch-seq centroid prediction alone
   does not validate this transfer.

## Fallback without a compatible external cohort

Use fully nested leave-type-out and donor-held-out validation, with within-class
RNA permutation controls. Test Gouwens-to-Scala transfer on shared inhibitory
types, but predeclare temperature/region shift and keep any calibration subset
separate from the test donors. Split repeated sweeps to estimate measurement
reliability. These are valuable internal/transfer checks, not replacements for
new experimental ground truth. Definitive atlas extrapolation remains a set of
predictions for prospective recordings of prespecified unobserved types.
