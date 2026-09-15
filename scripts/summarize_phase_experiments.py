#!/usr/bin/env python3
"""Collect phase-adequacy experiments and build cross-cell figures."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PLOT_FILES = {
    "phase_shape_gallery.png": "phase_shape_comparison.png",
    "phase_score_gallery.png": "phase_score_bar_plot.png",
    "gate_kinetics_gallery.png": "gate_kinetics.png",
    "alpha_beta_gallery.png": "alpha_beta_rates.png",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _label(name: str) -> str:
    dataset, cell_id = name.split("_", maxsplit=1)
    return f"{dataset.title()}: {cell_id}"


def collect(root: Path) -> tuple[list[Path], pd.DataFrame]:
    directories = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "architecture_summary.csv").exists()
    )
    if not directories:
        raise ValueError(f"No phase experiment outputs found under {root}")
    frames = []
    for directory in directories:
        frame = pd.read_csv(directory / "architecture_summary.csv")
        frame.insert(0, "pilot", directory.name)
        frames.append(frame)
    return directories, pd.concat(frames, ignore_index=True)


def collect_histories(
    directories: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for directory in directories:
        for architecture, filename in (
            ("one-compartment", "one_compartment_history.csv"),
            ("soma-ais", "soma_ais_history.csv"),
        ):
            frame = pd.read_csv(directory / filename)
            frame.insert(0, "architecture", architecture)
            frame.insert(0, "pilot", directory.name)
            frames.append(frame)
    histories = pd.concat(frames, ignore_index=True)
    valid = histories.loc[
        histories["valid"].astype(bool)
        & histories["phase__concavity"].notna()
        & histories["phase__velocity_extent"].notna()
    ].copy()

    rows = []
    for (pilot, architecture), group in valid.groupby(
        ["pilot", "architecture"],
        sort=False,
    ):
        best_concavity = group.loc[group["phase__concavity"].idxmin()]
        best_velocity = group.loc[group["phase__velocity_extent"].idxmin()]
        best_sign = group.loc[
            group["phase__onset_concavity_sign_agreement"].idxmax()
        ]
        jointly_close = (
            group["phase__concavity"].lt(0.25)
            & group["phase__velocity_extent"].lt(1.0)
        )
        rows.append(
            {
                "pilot": pilot,
                "architecture": architecture,
                "valid_candidates": len(group),
                "minimum_concavity_loss": best_concavity[
                    "phase__concavity"
                ],
                "velocity_loss_at_minimum_concavity": best_concavity[
                    "phase__velocity_extent"
                ],
                "minimum_velocity_loss": best_velocity[
                    "phase__velocity_extent"
                ],
                "concavity_loss_at_minimum_velocity": best_velocity[
                    "phase__concavity"
                ],
                "maximum_sign_agreement": best_sign[
                    "phase__onset_concavity_sign_agreement"
                ],
                "velocity_loss_at_maximum_sign_agreement": best_sign[
                    "phase__velocity_extent"
                ],
                "candidates_with_concavity_lt_0_25_and_velocity_lt_1": int(
                    jointly_close.sum()
                ),
            }
        )
    return valid, pd.DataFrame(rows)


def make_gallery(
    directories: list[Path],
    source_name: str,
    destination: Path,
) -> None:
    columns = 2
    panel_width = 1900
    image_height = (
        950 if source_name == "phase_score_bar_plot.png" else 1500
    )
    label_height = 70
    gap = 24
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
    title_font = _font(36)
    for index, directory in enumerate(directories):
        row, column = divmod(index, columns)
        left = gap + column * (panel_width + gap)
        top = gap + row * (image_height + label_height + gap)
        draw.text(
            (left + 10, top + 8),
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


def make_architecture_comparison(
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    metrics = (
        ("total", "Training phase total", False),
        ("concavity", "Smoothed concavity loss", False),
        ("velocity_extent", "Velocity-extent loss", False),
        (
            "onset_concavity_sign_agreement",
            "Onset concavity sign agreement",
            True,
        ),
    )
    pilots = summary["pilot"].drop_duplicates().tolist()
    labels = [_label(pilot).replace(": ", "\n") for pilot in pilots]
    colors = ("#e76f51", "#7b2cbf")
    architectures = ("one-compartment", "soma-ais")
    x = np.arange(len(pilots))
    width = 0.34
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 8.0),
        constrained_layout=True,
    )
    for axis, (metric, title, higher_is_better) in zip(
        axes.flat,
        metrics,
    ):
        for index, (architecture, color) in enumerate(
            zip(architectures, colors)
        ):
            subset = (
                summary.loc[
                    summary["architecture"].eq(architecture),
                    ["pilot", metric],
                ]
                .set_index("pilot")
                .reindex(pilots)
            )
            axis.bar(
                x + (index - 0.5) * width,
                subset[metric],
                width,
                color=color,
                label=architecture.replace("-", " ").title(),
            )
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_title(title, loc="left")
        axis.set_ylabel(
            "Higher is better" if higher_is_better else "Lower is better"
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_attainability_plot(
    histories: pd.DataFrame,
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    pilots = histories["pilot"].drop_duplicates().tolist()
    colors = {
        "one-compartment": "#e76f51",
        "soma-ais": "#7b2cbf",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.5),
        constrained_layout=True,
    )
    for axis, pilot in zip(axes.flat, pilots):
        pilot_history = histories.loc[histories["pilot"].eq(pilot)]
        for architecture, color in colors.items():
            values = pilot_history.loc[
                pilot_history["architecture"].eq(architecture)
            ]
            axis.scatter(
                values["phase__velocity_extent"],
                values["phase__concavity"],
                s=28,
                alpha=0.55,
                color=color,
                edgecolors="none",
                label=architecture.replace("-", " ").title(),
            )
            selected = summary.loc[
                summary["pilot"].eq(pilot)
                & summary["architecture"].eq(architecture)
            ]
            if not selected.empty:
                axis.scatter(
                    selected["velocity_extent"],
                    selected["concavity"],
                    s=110,
                    marker="*",
                    color=color,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                )
        axis.axvline(1.0, color="#8d99ae", linestyle=":", linewidth=1.0)
        axis.axhline(0.25, color="#8d99ae", linestyle=":", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_title(_label(pilot), loc="left")
        axis.set_xlabel("Velocity-extent loss")
        axis.set_ylabel("Smoothed concavity loss")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main(root: Path, output_dir: Path | None = None) -> None:
    output = output_dir or root
    directories, summary = collect(root)
    histories, attainability = collect_histories(directories)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "combined_architecture_summary.csv", index=False)
    attainability.to_csv(output / "attainability_summary.csv", index=False)
    for destination_name, source_name in PLOT_FILES.items():
        make_gallery(
            directories,
            source_name,
            output / destination_name,
        )
    make_architecture_comparison(
        summary,
        output / "architecture_comparison.png",
    )
    make_attainability_plot(
        histories,
        summary,
        output / "concavity_velocity_tradeoff.png",
    )


def arguments() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**arguments())
