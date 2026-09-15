#!/usr/bin/env python3
"""Fit recovery-onset dynamics for every existing compact population model."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd

from inverse_ephys_alpha_beta.recovery_onset_targets import (
    recovery_onset_target_task,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(
            "outputs/compact_population_all/compact_model_parameters.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recovery_onset_targets_all"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()
    output_root = args["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        args["parameters"],
        dtype={"cell_id": "string"},
    )
    if args["limit"] is not None:
        source = source.iloc[: args["limit"]].copy()
    target_path = output_root / "recovery_onset_parameters.csv"
    failure_path = output_root / "failures.csv"
    fitted: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if args["resume"] and target_path.exists():
        fitted = pd.read_csv(
            target_path,
            dtype={"cell_id": "string"},
        ).to_dict("records")
        try:
            failures = pd.read_csv(
                failure_path,
                dtype={"cell_id": "string"},
            ).to_dict("records")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            failures = []
    completed = {
        (str(row["dataset"]), str(row["cell_id"]))
        for row in (*fitted, *failures)
    }
    tasks = [
        row
        for row in source.to_dict("records")
        if (str(row["dataset"]), str(row["cell_id"])) not in completed
    ]

    def checkpoint() -> None:
        _write_csv(pd.DataFrame(fitted), target_path)
        _write_csv(pd.DataFrame(failures), failure_path)

    executor_class = (
        ProcessPoolExecutor
        if args["executor"] == "process"
        else ThreadPoolExecutor
    )
    with executor_class(max_workers=args["workers"]) as executor:
        for index, result in enumerate(
            executor.map(
                recovery_onset_target_task,
                tasks,
                chunksize=1,
            ),
            start=1,
        ):
            if result["status"] == "ok":
                fitted.append(result["row"])
            else:
                failures.append(result)
            if index % args["checkpoint_every"] == 0:
                checkpoint()
                print(
                    f"completed {len(completed) + index}/{len(source)} "
                    f"(ok={len(fitted)}, failed={len(failures)})",
                    flush=True,
                )
    checkpoint()
    summary = (
        pd.DataFrame(fitted)
        .groupby(["dataset", "onset_fit_status"], dropna=False)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    _write_csv(summary, output_root / "fit_summary.csv")
    (output_root / "metadata.json").write_text(
        json.dumps(
            {
                "source_parameters": str(args["parameters"]),
                "requested_cell_count": len(source),
                "successful_cell_count": len(fitted),
                "failed_cell_count": len(failures),
                "workers": args["workers"],
                "executor": args["executor"],
                "model": "phase_template_recovery_onset",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
