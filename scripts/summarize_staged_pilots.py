#!/usr/bin/env python3
"""Collect staged-ladder pilot scores and build readable plot galleries."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PLOT_FILES = {
    "trace_gallery.png": "trace_comparison.png",
    "phase_plane_gallery.png": "phase_plane_comparison.png",
    "score_gallery.png": "score_bar_plot.png",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _pilot_label(name: str) -> str:
    dataset, cell_id = name.split("_", maxsplit=1)
    return f"{dataset.title()}: {cell_id}"


def collect_summaries(pilot_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for pilot_dir in pilot_dirs:
        summary = pd.read_csv(pilot_dir / "stage_summary.csv")
        summary.insert(0, "pilot", pilot_dir.name)
        frames.append(summary)
    combined = pd.concat(frames, ignore_index=True)
    baseline = combined.groupby("pilot")["scalar_score"].transform("first")
    combined["delta_vs_step1"] = combined["scalar_score"] - baseline
    combined["relative_improvement_vs_step1"] = (
        baseline - combined["scalar_score"]
    ) / baseline
    combined["improved_vs_step1"] = combined["scalar_score"] < baseline
    return combined


def make_gallery(
    pilot_dirs: list[Path],
    source_name: str,
    destination: Path,
    combined: pd.DataFrame,
) -> None:
    columns = 2
    panel_width = 1900
    image_height = 900 if source_name == "score_bar_plot.png" else 1500
    label_height = 90
    gap = 24
    rows = (len(pilot_dirs) + columns - 1) // columns
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
    note_font = _font(27)
    for index, pilot_dir in enumerate(pilot_dirs):
        row, column = divmod(index, columns)
        left = gap + column * (panel_width + gap)
        top = gap + row * (image_height + label_height + gap)
        draw.text(
            (left + 12, top + 8),
            _pilot_label(pilot_dir.name),
            fill="#202124",
            font=title_font,
        )
        if source_name == "score_bar_plot.png":
            rows_for_pilot = combined.loc[combined["pilot"].eq(pilot_dir.name)]
            slow_row = rows_for_pilot.loc[
                rows_for_pilot["stage"].eq("step2_kslow")
            ].iloc[0]
            verdict = (
                "slow K improves total"
                if bool(slow_row["improved_vs_step1"])
                else "fast-only parent remains better"
            )
            draw.text(
                (left + 12, top + 51),
                verdict,
                fill="#555555",
                font=note_font,
            )
        image = Image.open(pilot_dir / source_name).convert("RGB")
        fitted = ImageOps.contain(
            image,
            (panel_width, image_height),
            Image.Resampling.LANCZOS,
        )
        image_left = left + (panel_width - fitted.width) // 2
        image_top = top + label_height + (image_height - fitted.height) // 2
        canvas.paste(fitted, (image_left, image_top))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94)


def main(root: Path, output_dir: Path | None = None) -> None:
    output = output_dir or root
    pilot_dirs = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "stage_summary.csv").exists()
    )
    if not pilot_dirs:
        raise ValueError(f"No staged pilot outputs found under {root}")
    combined = collect_summaries(pilot_dirs)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "pilot_summary.csv", index=False)
    for destination_name, source_name in PLOT_FILES.items():
        make_gallery(
            pilot_dirs,
            source_name,
            output / destination_name,
            combined,
        )


def get_arguments() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing one staged-ladder result directory per cell",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the pilot root",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**get_arguments())
