#!/usr/bin/env python3
"""Paired held-out-current and nested donor-held-out RNA adaptation pilot."""

from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from inverse_ephys_alpha_beta.abstract_phase_model import AbstractObservedTrace
from inverse_ephys_alpha_beta.compact_model import _add_waveform_parameters
from inverse_ephys_alpha_beta.phase_population import select_phase_population_protocols
from inverse_ephys_alpha_beta.phase_template_model import fit_phase_template_ladder
from inverse_ephys_alpha_beta.recovery_onset import (
    OnsetModel, fit_recovery_timing, fit_onset_model, simulate_recovery_onset_model,
)
from inverse_ephys_alpha_beta.recovery_onset_compact import (
    recovery_onset_model_from_row, recovery_onset_parameter_row,
)
from inverse_ephys_alpha_beta.reduced_rank import (
    _folds, fit_class_residualized_rrr_analysis, predict_class_residualized_rrr,
)
from inverse_ephys_alpha_beta.transcriptomics import (
    align_expression_to_parameters, expression_matrix, broad_transcriptomic_class,
)
from inverse_ephys_alpha_beta.time_phase import (
    FRACTIONS, extract_time_phase_features, fit_sequence_clock, clock_parameter_row,
    clock_from_row, simulate_sequence, train_metrics,
)
from run_recovery_onset_heldout_srrr import (
    _load_expression, DEFAULT_KOBAK_ROOT, DEFAULT_PATCHSEQ_ROOT,
)


MODELS = ("recovery", "sequence_timing", "sequence_joint")
METRICS = ("count_error", "early_isi_rmse_ms", "late_isi_error_ms", "adaptation_ratio_error")


def save_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def trace_from_cache(path):
    with np.load(path) as a:
        t, v, current = a["time"], a["bio"], a["current"]
        start, end = a["epoch"]
        u = a["velocity"]
    return AbstractObservedTrace("held_out", "validation", t, v, u, np.gradient(u, t),
        current, np.ones_like(t, dtype=bool), float(start), float(end))


def fit_cell(task):
    row, root = task
    key = str(row["dataset"]) + "__" + str(row["cell_id"])
    destination = Path(root) / "cells" / key
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            protocols = select_phase_population_protocols(row["nwb_path"])
            first = protocols.training[0]
            rest = float(np.median(first.voltage_mv[first.time_ms < first.stimulus_start_ms]))
            fit = fit_phase_template_ladder(protocols.training, rest)
            waveform = fit.no_slow_model
            timing, _ = fit_recovery_timing(fit.cycles, rest)
            try:
                onset = fit_onset_model(protocols.training)
            except ValueError:
                onset = OnsetModel(timing.current_levels,
                    tuple(1.0 for _ in timing.current_levels), tuple(np.nan for _ in timing.current_levels))
            parameters = dict(resting_voltage_mv=rest, rheobase_current_pa=waveform.rheobase_current,
                repetitive_current_min_pa=waveform.current_levels[0],
                repetitive_current_span_pa=waveform.current_levels[-1]-waveform.current_levels[0])
            _add_waveform_parameters(parameters, waveform, FRACTIONS, 10,
                                     lambda current: timing.period(current, 0))
            parameters = recovery_onset_parameter_row(parameters, timing, onset)
            tables = [extract_time_phase_features(trace) for trace in protocols.training]
            tables = [table for table in tables if len(table) >= 3]
            if not tables:
                raise ValueError("No usable cycle feature sequences")
            save_csv(pd.concat(tables), destination / "training_cycle_features.csv")
            # Evaluate the same fixed-width waveform representation later predicted by RNA.
            wave, timing, onset = recovery_onset_model_from_row(parameters)
            validation = protocols.validation
            simulations = {"recovery": simulate_recovery_onset_model(wave, timing, onset,
                validation.time_ms, validation.input_value, rest)}
            all_parameters, settings = {}, {}
            for model in MODELS[1:]:
                clock, setting = fit_sequence_clock(tables, joint=model == "sequence_joint")
                seq = clock_parameter_row(clock, wave.current_levels[0], wave.current_levels[-1])
                all_parameters[model] = dict(parameters, **seq)
                settings[model] = setting
                simulations[model] = simulate_sequence(wave, onset, clock_from_row(all_parameters[model]), validation)
            all_parameters["recovery"] = parameters
            metrics = [{"dataset": row["dataset"], "cell_id": str(row["cell_id"]),
                "model": model, "source": "ephys", **train_metrics(validation, sim.voltage_mv)}
                for model, sim in simulations.items()]
            np.savez_compressed(destination / "traces.npz", time=validation.time_ms,
                bio=validation.voltage_mv, velocity=validation.velocity_mv_ms,
                current=validation.input_value, epoch=[validation.stimulus_start_ms, validation.stimulus_end_ms],
                **{name: sim.voltage_mv for name, sim in simulations.items()})
            save_csv(extract_time_phase_features(validation), destination / "validation_cycle_features.csv")
            payload = {"status": "ok", "dataset": row["dataset"], "cell_id": str(row["cell_id"]),
                "parameters": all_parameters, "settings": settings, "metrics": metrics,
                "training_sweeps": list(protocols.training_sweep_numbers),
                "validation_sweep": protocols.validation_sweep_number,
                "validation_current_pa": protocols.validation_current}
    except Exception as error:
        payload = {"status": "failed", "dataset": row["dataset"], "cell_id": str(row["cell_id"]),
                   "reason": f"{type(error).__name__}: {error}"}
    (destination / "fit.json").write_text(json.dumps(payload, indent=2))
    return payload


def expression_args():
    return Namespace(
        gouwens_mat=DEFAULT_KOBAK_ROOT / "data/gouwens2020/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat",
        gouwens_metadata=DEFAULT_PATCHSEQ_ROOT / "Patch-seq AIBS/20200711_patchseq_metadata_mouse.csv",
        gouwens_cpm=DEFAULT_PATCHSEQ_ROOT / "Patch-seq AIBS/transcriptomes/20200513_Mouse_PatchSeq_Release_cpm.v2/20200513_Mouse_PatchSeq_Release_cpm.v2.csv",
        scala_pickle=DEFAULT_KOBAK_ROOT / "data/scala2020.pickle",
        scala_metadata=DEFAULT_PATCHSEQ_ROOT / "Scala_pseq_data/m1_patchseq_meta_data.csv",
        expression_cache=Path("outputs/transcriptomic_rrr/expression_cache"))


def rna_evaluation(root, successes):
    metrics, hyper, splits, parameter_scores = [], [], [], []
    for dataset in ("gouwens_visp", "scala_room_temperature"):
        records = [s for s in successes if s["dataset"] == dataset]
        table = pd.DataFrame([dict(dataset=dataset, cell_id=s["cell_id"]) for s in records])
        expression = _load_expression(dataset, expression_args(), set(table.cell_id))
        expression, aligned = align_expression_to_parameters(expression, table)
        lookup = {s["cell_id"]: s for s in records}
        records = [lookup[str(cell)] for cell in aligned.cell_id]
        x = expression_matrix(expression)
        labels = np.array([broad_transcriptomic_class(t) for t in expression.transcriptomic_types])
        # The sign-log transform is fixed analytically, independent of the cohort.
        frames = {model: pd.DataFrame([s["parameters"][model] for s in records]) for model in MODELS}
        baseline = frames["recovery"]
        common = [c for c in baseline if not c.startswith("recovery_") and
                  not c.startswith("template_min_") and not c.startswith("template_max_") and
                  c != "onset_fit_rmse_median_mv"]
        timing = [c for c in baseline if c.startswith("recovery_")]
        blocks = {"common": baseline[common], "recovery": baseline[timing]}
        blocks.update({m: frames[m][[c for c in frames[m] if c.startswith("sequence_")]] for m in MODELS[1:]})
        predictions = {source: {block: np.empty_like(frame.to_numpy(float)) for block, frame in blocks.items()}
                       for source in ("rna", "class")}
        folds = list(_folds(expression.donor_ids, 3))
        for fold, (train, test) in enumerate(folds):
            for index in test:
                splits.append(dict(dataset=dataset, cell_id=records[index]["cell_id"],
                    donor_id=str(expression.donor_ids[index]), broad_class=str(labels[index]), outer_fold=fold))
            for block, frame in blocks.items():
                raw = frame.to_numpy(float)
                y = np.sign(raw) * np.log1p(np.abs(raw))
                variable = np.std(y[train], axis=0) > 1e-9
                if not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite RNA targets in {dataset}/{block}")
                with threadpool_limits(limits=1):
                    analysis = fit_class_residualized_rrr_analysis(x[train], y[train][:,variable],
                        expression.donor_ids[train], labels[train], ranks=(1, 2, 3),
                        penalties=(10.0, 100.0), sparse_ratios=(0.1, 0.3), fold_count=3)
                    pred, cls = predict_class_residualized_rrr(analysis, x[train], y[train][:,variable],
                        labels[train], x[test], labels[test])
                for source, values in (("rna", pred), ("class", cls)):
                    full = np.tile(np.mean(y[train], axis=0), (len(test), 1))
                    full[:, variable] = values
                    full = np.clip(full, np.quantile(y[train], 0.005, axis=0),
                                   np.quantile(y[train], 0.995, axis=0))
                    predictions[source][block][test] = np.sign(full) * np.expm1(np.abs(full))
                hyper.append(dict(dataset=dataset, outer_fold=fold, block=block,
                    rank=analysis.rank, ridge_penalty=analysis.ridge_penalty,
                    sparse_ratio=analysis.sparse_ratio,
                    training_cells=len(train), test_cells=len(test)))
                print(f"RNA {dataset} fold {fold+1}/3 {block}", flush=True)
        for block, frame in blocks.items():
            raw = frame.to_numpy(float)
            denom = np.sum((raw - raw.mean(axis=0))**2, axis=0)
            for source in predictions:
                score = 1 - np.sum((raw-predictions[source][block])**2,axis=0)/np.maximum(denom,1e-12)
                parameter_scores.extend(dict(dataset=dataset, block=block, source=source,
                    parameter=name, r2=float(value)) for name,value in zip(frame.columns,score))
        for i, record in enumerate(records):
            directory = root / "cells" / (dataset + "__" + record["cell_id"])
            trace = trace_from_cache(directory / "traces.npz")
            replay = {}
            for source in predictions:
                row = {name: value for block in ("common", "recovery")
                       for name,value in zip(blocks[block].columns, predictions[source][block][i])}
                wave, recovery, onset = recovery_onset_model_from_row(row)
                sims = {"recovery": simulate_recovery_onset_model(wave, recovery, onset,
                    trace.time_ms, trace.input_value, wave.resting_voltage_mv)}
                for model in MODELS[1:]:
                    seqrow = dict(row, **dict(zip(blocks[model].columns, predictions[source][model][i])))
                    sims[model] = simulate_sequence(wave, onset, clock_from_row(seqrow), trace)
                for model, sim in sims.items():
                    replay[source + "_" + model] = sim.voltage_mv
                    metrics.append(dict(dataset=dataset,cell_id=record["cell_id"],source=source,
                        model=model,**train_metrics(trace, sim.voltage_mv)))
            np.savez_compressed(directory / "rna_replays.npz", **replay)
    save_csv(pd.DataFrame(splits), root / "rna_splits.csv")
    save_csv(pd.DataFrame(hyper), root / "rna_hyperparameters.csv")
    save_csv(pd.DataFrame(parameter_scores), root / "rna_parameter_scores.csv")
    return metrics


def plots(root, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axes_row, (dataset, group) in zip(axes, metrics.groupby("dataset")):
        for ax, metric in zip(axes_row, METRICS):
            for j, source in enumerate(("ephys", "rna", "class")):
                vals = [group.loc[(group.source==source)&(group.model==m),metric].median() for m in MODELS]
                ax.bar(np.arange(3)+(j-1)*0.24, vals, 0.23, label=source)
            ax.set_xticks(range(3), ["Recovery", "ISI clock", "Joint clock"], rotation=20)
            ax.set_title(dataset.replace("_", " ")+"\n"+metric.replace("_", " "), fontsize=9)
            ax.legend(fontsize=7)
    fig.savefig(root / "performance.png", dpi=150)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("outputs/recovery_onset_population_168/cohort.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/time_phase_pilot"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rna", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, dtype={"cell_id":str})
    if args.limit:
        cohort = cohort.groupby("dataset", group_keys=False).head(args.limit)
    save_csv(cohort, root / "cohort.csv")
    results, tasks = [], []
    for row in cohort.to_dict("records"):
        cache = root / "cells" / (row["dataset"]+"__"+str(row["cell_id"])) / "fit.json"
        if args.resume and cache.exists():
            results.append(json.loads(cache.read_text()))
        else:
            tasks.append((row, str(root)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i,result in enumerate(executor.map(fit_cell,tasks)):
            results.append(result)
            print(f"Fit {i+1}/{len(tasks)} {result['dataset']} {result['cell_id']}: {result['status']}",flush=True)
    successes = [r for r in results if r["status"] == "ok"]
    save_csv(pd.DataFrame([r for r in results if r["status"] != "ok"]), root / "failures.csv")
    metrics = [m for r in successes for m in r["metrics"]]
    save_csv(pd.DataFrame(metrics), root / "ephys_metrics.csv")
    if not args.skip_rna:
        metrics += rna_evaluation(root, successes)
    frame = pd.DataFrame(metrics)
    save_csv(frame, root / "metrics.csv")
    summary = frame.groupby(["dataset","source","model"])[list(METRICS)].median().reset_index()
    save_csv(summary, root / "summary.csv")
    plots(root, frame)
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
