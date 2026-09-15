#!/usr/bin/env python3
"""Collect nested phase Pareto pilots and build cross-cell comparisons."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


VARIANTS = (
    "shared-alpha-beta",
    "ais-m-shift",
    "flexible-shared-m",
)
COLORS = {
    "shared-alpha-beta": "#e76f51",
    "ais-m-shift": "#7b2cbf",
    "flexible-shared-m": "#1976a3",
}
LABELS = {
    "shared-alpha-beta": "Shared alpha/beta",
    "ais-m-shift": "AIS m shift",
    "flexible-shared-m": "Flexible shared m",
}
GALLERIES = {
    "nested_phase_gallery.png": "nested_phase_comparison.png",
    "nested_pareto_gallery.png": "nested_pareto_fronts.png",
    "nested_score_gallery.png": "nested_score_bars.png",
    "nested_activation_gallery.png": "nested_activation_kinetics.png",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _label(name: str) -> str:
    dataset, cell_id = name.split("_", maxsplit=1)
    return f"{dataset.title()}: {cell_id}"


def collect(
    root: Path,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    directories = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "nested_variant_summary.csv").exists()
    )
    if not directories:
        raise ValueError(f"No nested phase outputs found under {root}")
    summary_frames = []
    target_rows = []
    for directory in directories:
        summary = pd.read_csv(directory / "nested_variant_summary.csv")
        summary.insert(0, "pilot", directory.name)
        for variant in VARIANTS:
            history = pd.read_csv(directory / f"{variant}_history.csv")
            selected = summary["variant"].eq(variant)
            summary.loc[selected, "evaluated_candidates"] = len(history)
            summary.loc[selected, "valid_candidates"] = int(
                history["valid"].sum()
            )
            summary.loc[selected, "valid_candidate_fraction"] = float(
                history["valid"].mean()
            )
        summary_frames.append(summary)
        for variant in VARIANTS:
            front = pd.read_csv(directory / f"{variant}_pareto.csv")
            front = front.loc[front["valid"]].copy()
            front["joint_target_distance"] = np.sqrt(
                (front["phase__concavity"] / 0.25) ** 2
                + (front["phase__velocity_extent"] / 1.0) ** 2
            )
            nearest = front.loc[front["joint_target_distance"].idxmin()]
            min_concavity = front.loc[front["phase__concavity"].idxmin()]
            min_velocity = front.loc[
                front["phase__velocity_extent"].idxmin()
            ]
            target_rows.append(
                {
                    "pilot": directory.name,
                    "variant": variant,
                    "valid_pareto_models": len(front),
                    "minimum_joint_target_distance": nearest[
                        "joint_target_distance"
                    ],
                    "concavity_at_minimum_joint_distance": nearest[
                        "phase__concavity"
                    ],
                    "velocity_at_minimum_joint_distance": nearest[
                        "phase__velocity_extent"
                    ],
                    "remaining_fit_at_minimum_joint_distance": nearest[
                        "objective_remaining_fit"
                    ],
                    "minimum_concavity": min_concavity[
                        "phase__concavity"
                    ],
                    "velocity_at_minimum_concavity": min_concavity[
                        "phase__velocity_extent"
                    ],
                    "minimum_velocity": min_velocity[
                        "phase__velocity_extent"
                    ],
                    "concavity_at_minimum_velocity": min_velocity[
                        "phase__concavity"
                    ],
                }
            )
    return (
        directories,
        pd.concat(summary_frames, ignore_index=True),
        pd.DataFrame(target_rows),
    )


def make_gallery(
    directories: list[Path],
    source_name: str,
    destination: Path,
) -> None:
    columns = 2
    panel_width = 1800
    image_height = (
        1350 if source_name == "nested_phase_comparison.png" else 900
    )
    label_height = 64
    gap = 22
    rows = (len(directories) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns + 1) * gap,
            rows * (image_height + label_height) + (rows + 1) * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34)
    for index, directory in enumerate(directories):
        row, column = divmod(index, columns)
        left = gap + column * (panel_width + gap)
        top = gap + row * (image_height + label_height + gap)
        draw.text(
            (left + 8, top + 6),
            _label(directory.name),
            fill="#202124",
            font=title_font,
        )
        image = Image.open(directory / source_name).convert("RGB")
        fitted = ImageOps.contain(
            image,
            (panel_width, image_height),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(
            fitted,
            (
                left + (panel_width - fitted.width) // 2,
                top + label_height + (image_height - fitted.height) // 2,
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94)


def make_cross_cell_fronts(
    directories: list[Path],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.5),
        constrained_layout=True,
    )
    for axis, directory in zip(axes.flat, directories):
        for variant in VARIANTS:
            front = pd.read_csv(directory / f"{variant}_pareto.csv")
            front = front.loc[front["valid"]]
            axis.scatter(
                front["phase__velocity_extent"],
                np.maximum(front["phase__concavity"], 1e-5),
                color=COLORS[variant],
                s=40,
                alpha=0.78,
                edgecolors="none",
                label=LABELS[variant],
            )
        axis.axvline(1.0, color="#8d99ae", linestyle=":", linewidth=1.0)
        axis.axhline(0.25, color="#8d99ae", linestyle=":", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_title(_label(directory.name), loc="left")
        axis.set_xlabel("Velocity-extent loss")
        axis.set_ylabel("Smoothed concavity loss")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_metric_comparison(
    summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    destination: Path,
) -> None:
    pilots = summary["pilot"].drop_duplicates().tolist()
    labels = [_label(pilot).replace(": ", "\n") for pilot in pilots]
    metrics = (
        (summary, "total", "Representative phase total", False),
        (summary, "concavity", "Representative concavity", False),
        (summary, "velocity_extent", "Representative velocity extent", False),
        (
            target_summary,
            "minimum_joint_target_distance",
            "Closest focal-target distance",
            False,
        ),
    )
    x = np.arange(len(pilots))
    width = 0.24
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 8.0),
        constrained_layout=True,
    )
    for axis, (frame, metric, title, higher_is_better) in zip(
        axes.flat,
        metrics,
    ):
        for index, variant in enumerate(VARIANTS):
            values = (
                frame.loc[
                    frame["variant"].eq(variant),
                    ["pilot", metric],
                ]
                .set_index("pilot")
                .reindex(pilots)
            )
            axis.bar(
                x + (index - 1.0) * width,
                values[metric],
                width,
                color=COLORS[variant],
                label=LABELS[variant],
            )
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_title(title, loc="left")
        axis.set_ylabel(
            "Higher is better" if higher_is_better else "Lower is better"
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_generalization_plot(
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    pilots = summary["pilot"].drop_duplicates().tolist()
    labels = [_label(pilot).replace(": ", "\n") for pilot in pilots]
    x = np.arange(len(pilots))
    width = 0.24
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.5),
        constrained_layout=True,
    )
    for index, variant in enumerate(VARIANTS):
        values = (
            summary.loc[
                summary["variant"].eq(variant),
                ["pilot", "total", "held_out__total"],
            ]
            .set_index("pilot")
            .reindex(pilots)
        )
        axes[0].bar(
            x + (index - 1.0) * width,
            values["held_out__total"],
            width,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        axes[1].bar(
            x + (index - 1.0) * width,
            values["held_out__total"] - values["total"],
            width,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    axes[0].set_title("Held-out phase total", loc="left")
    axes[0].set_ylabel("Lower is better")
    axes[1].set_title("Held-out minus training total", loc="left")
    axes[1].set_ylabel("Positive values indicate worse generalization")
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main(root: Path, output_dir: Path | None = None) -> None:
    output = output_dir or root
    directories, summary, target_summary = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "combined_nested_summary.csv", index=False)
    target_summary.to_csv(
        output / "combined_frontier_targets.csv",
        index=False,
    )
    for destination_name, source_name in GALLERIES.items():
        make_gallery(
            directories,
            source_name,
            output / destination_name,
        )
    make_cross_cell_fronts(
        directories,
        output / "cross_cell_pareto_fronts.png",
    )
    make_metric_comparison(
        summary,
        target_summary,
        output / "nested_metric_comparison.png",
    )
    make_generalization_plot(
        summary,
        output / "nested_heldout_generalization.png",
    )


def arguments() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**arguments())
