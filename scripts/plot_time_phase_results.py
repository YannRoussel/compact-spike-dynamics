#!/usr/bin/env python3
"""Paired pilot summaries and trace/time-phase/ISI examples."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/time_phase_pilot"))
    args = parser.parse_args()
    root = args.root
    metrics = pd.read_csv(root / "metrics.csv", dtype={"cell_id":str})
    summary = []
    for (dataset, source), group in metrics.groupby(["dataset", "source"]):
        for metric in ("count_error", "early_isi_rmse_ms", "late_isi_error_ms", "adaptation_ratio_error"):
            pivot = group.pivot(index="cell_id", columns="model", values=metric)
            for model in ("sequence_timing", "sequence_joint"):
                paired = pivot[["recovery", model]].dropna()
                delta = paired[model] - paired.recovery
                summary.append(dict(dataset=dataset, source=source, metric=metric, model=model,
                    eligible_cells=len(paired), total_cells=len(pivot),
                    baseline_median=paired.recovery.median(), new_median=paired[model].median(),
                    median_paired_delta=delta.median(), improved_fraction=(delta < 0).mean()))
    pd.DataFrame(summary).to_csv(root / "paired_comparisons.csv", index=False)

    behavior_scores = []
    for (dataset, source, model), group in metrics.groupby(["dataset", "source", "model"]):
        for name in ("spike_count", "adaptation_ratio"):
            columns = ["observed_" + name, "predicted_" + name]
            valid = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
            actual, predicted = valid.to_numpy().T
            denominator = np.sum((actual - actual.mean()) ** 2)
            behavior_scores.append(dict(dataset=dataset, source=source, model=model,
                feature=name, eligible_cells=len(valid), total_cells=len(group),
                r2=1-np.sum((actual-predicted)**2)/denominator if denominator > 0 else np.nan))
    pd.DataFrame(behavior_scores).to_csv(root / "behavior_r2.csv", index=False)

    cohort = pd.read_csv(root / "cohort.csv", dtype={"cell_id":str})
    for dataset, cells in cohort.groupby("dataset"):
        selected = []
        # Predeclared cell-ID ordering, without selecting on fit performance.
        for family in ("Pvalb", "Sst", "Vip", "Glut_IT"):
            for _, row in cells.loc[cells.broad_class.eq(family)].sort_values("cell_id").iterrows():
                directory = root / "cells" / (dataset + "__" + row.cell_id)
                if (directory / "rna_replays.npz").exists():
                    selected.append((family, row.cell_id, directory))
                    break
        if not selected:
            continue
        fig = plt.figure(figsize=(18, 4 * len(selected)), constrained_layout=True)
        for index, (family, cell, directory) in enumerate(selected):
            with np.load(directory / "traces.npz") as a, np.load(directory / "rna_replays.npz") as b:
                t = a["time"] - a["epoch"][0]
                stop = a["epoch"][1] - a["epoch"][0]
                curves = [("Biology", a["bio"], "#222222"),
                          ("Recovery fit", a["recovery"], "#999999"),
                          ("Joint fit", a["sequence_joint"], "#008577"),
                          ("Joint RNA", b["rna_sequence_joint"], "#c03d65")]
                ax = fig.add_subplot(len(selected), 4, 4 * index + 1)
                ax3 = fig.add_subplot(len(selected), 4, 4 * index + 2, projection="3d")
                axp = fig.add_subplot(len(selected), 4, 4 * index + 3)
                axi = fig.add_subplot(len(selected), 4, 4 * index + 4)
                mask = (t >= 0) & (t <= min(stop, 120))
                for name,v,color in curves:
                    u = np.gradient(v, t)
                    ax.plot(t, v, color=color, lw=0.7, alpha=0.85, label=name)
                    if name != "Recovery fit":
                        ax3.plot(t[mask][::2], v[mask][::2], u[mask][::2], color=color, lw=0.65)
                    axp.plot(v[mask], u[mask], color=color, lw=0.65, alpha=0.75)
                    active = (t >= 0) & (t < stop)
                    spikes = t[active][find_peaks(v[active], height=0,
                        distance=max(1,int(round(1 / np.median(np.diff(t))))))[0]]
                    axi.plot(spikes[:-1], np.diff(spikes), color=color, marker=".",ms=3,lw=0.8)
                ax.set_title(f"{family}: {cell}", fontsize=10)
                ax.set(xlabel="Time from step (ms)",ylabel="V (mV)",xlim=(-10,stop+10))
                ax.legend(fontsize=7,loc="upper right")
                ax3.set(xlabel="Time (ms)",ylabel="V (mV)",zlabel="dV/dt (mV/ms)")
                ax3.set_title("First 120 ms", fontsize=10)
                ax3.view_init(22, -65)
                axp.set(xlabel="V (mV)",ylabel="dV/dt (mV/ms)")
                axi.set(xlabel="Spike time (ms)",ylabel="Following ISI (ms)")
        fig.suptitle(dataset.replace("_", " ")+": held-out current and donor-held-out RNA", fontsize=14)
        fig.savefig(root / (dataset + "_examples.png"), dpi=150)
        plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
