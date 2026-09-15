#!/usr/bin/env python3
"""Identify abstract voltage oscillators from HH and Patch-seq traces."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.abstract_phase_model import (
    AbstractLadderResult,
    AbstractObservedTrace,
    AbstractPhaseConfig,
    AbstractPhaseModel,
    fit_abstract_model_ladder,
    memory_from_voltage,
    model_stability,
    prepare_abstract_trace,
    trace_field_metrics,
    trace_simulation_metrics,
)
from inverse_ephys_alpha_beta.cell_targets import (
    build_cell_optimization_target,
    load_raw_spike_cycle_table,
)
from inverse_ephys_alpha_beta.hh_model import (
    SimulationConfig,
    Stimulus,
    equilibrium_state,
    simulate,
)
from inverse_ephys_alpha_beta.kinetics import KineticParameters
from inverse_ephys_alpha_beta.phase_template_model import (
    fit_phase_template_ladder,
    phase_plane_chamfer_distance,
    simulate_phase_template_model,
    template_fourier_coefficients,
)
from inverse_ephys_alpha_beta.raw_patchseq import (
    CurrentClampSweep,
    count_sweep_spikes,
    read_current_clamp_sweeps,
)
from inverse_ephys_alpha_beta.static_parameters import BiophysicalParameters


def _prepare_observed_sweep(
    sweep: CurrentClampSweep,
    name: str,
    role: str,
    margin_ms: float,
    config: AbstractPhaseConfig,
) -> AbstractObservedTrace:
    if sweep.stimulus_start_ms is None or sweep.stimulus_end_ms is None:
        raise ValueError(f"Sweep {sweep.sweep_number} has no stimulus epoch")
    start = float(sweep.stimulus_start_ms)
    end = float(sweep.stimulus_end_ms)
    keep = (sweep.time_ms >= start - margin_ms) & (
        sweep.time_ms <= end + margin_ms
    )
    time = sweep.time_ms[keep] - (start - margin_ms)
    return prepare_abstract_trace(
        name=name,
        role=role,
        time_ms=time,
        voltage_mv=sweep.voltage_mv[keep],
        input_value=sweep.current_pa[keep],
        stimulus_start_ms=margin_ms,
        stimulus_end_ms=margin_ms + end - start,
        config=config,
    )


def _biological_traces(
    raw_table: pd.DataFrame,
    dataset: str,
    cell_id: str,
    temperature_c: float,
    config: AbstractPhaseConfig,
) -> tuple[list[AbstractObservedTrace], list[AbstractObservedTrace], dict]:
    target = build_cell_optimization_target(
        raw_table,
        dataset=dataset,
        cell_id=cell_id,
        temperature_c=temperature_c,
        local_passive=True,
    )
    sweeps = {
        sweep.sweep_number: sweep
        for sweep in read_current_clamp_sweeps(
            target.nwb_path,
            long_square_only=True,
        )
    }
    target_training = [
        _prepare_observed_sweep(
            sweeps[protocol.sweep_number],
            protocol.name,
            protocol.role,
            margin_ms=20.0,
            config=config,
        )
        for protocol in target.training_protocols
    ]
    validation = [
        _prepare_observed_sweep(
            sweeps[protocol.sweep_number],
            protocol.name,
            protocol.role,
            margin_ms=20.0,
            config=config,
        )
        for protocol in target.validation_protocols
    ]
    held_out_sweeps = {
        protocol.sweep_number
        for protocol in target.validation_protocols
    }
    additional_sweeps = [
        sweep
        for sweep in sweeps.values()
        if sweep.sweep_number not in held_out_sweeps
        and sweep.stimulus_amplitude_pa is not None
        and sweep.stimulus_amplitude_pa > 0.0
        and count_sweep_spikes(sweep) > 0
        and sweep.sweep_number
        not in {
            protocol.sweep_number
            for protocol in target.training_protocols
        }
    ]
    training = [
        *target_training,
        *[
            _prepare_observed_sweep(
                sweep,
                (
                    f"train_{sweep.stimulus_amplitude_pa:g}_pa"
                    f"_s{sweep.sweep_number}"
                ),
                "training",
                margin_ms=20.0,
                config=config,
            )
            for sweep in additional_sweeps
        ],
    ]
    metadata = {
        "dataset": target.dataset,
        "cell_id": target.cell_id,
        "nwb_path": target.nwb_path,
        "temperature_c": target.temperature_c,
        "rheobase_pa": target.sampled_rheobase_pa,
        "training_sweeps": [
            trace.name for trace in training
        ],
        "validation_sweeps": [
            protocol.sweep_number
            for protocol in target.validation_protocols
        ],
    }
    return training, validation, metadata


def _hh_traces(
    config: AbstractPhaseConfig,
) -> tuple[list[AbstractObservedTrace], list[AbstractObservedTrace], dict]:
    kinetics = KineticParameters.canonical()
    biophysics = BiophysicalParameters()
    initial = equilibrium_state(kinetics, biophysics)
    traces = []
    currents = (7.0, 9.0, 10.0, 12.0, 15.0)
    held_out_index = 2
    for index, current in enumerate(currents):
        simulation_config = SimulationConfig(
            duration_ms=250.0,
            dt_ms=0.025,
            initial_voltage_mv=float(initial[0]),
            stimulus=Stimulus(
                amplitude_ua_cm2=current,
                start_ms=25.0,
                end_ms=225.0,
            ),
            conductances=biophysics,
            temperature_c=6.3,
        )
        trace = simulate(
            kinetics,
            simulation_config,
            initial_state=initial,
        )
        observed = prepare_abstract_trace(
            name=(
                "held_out"
                if index == held_out_index
                else f"train_{current:g}"
            ),
            role=(
                "validation"
                if index == held_out_index
                else "training"
            ),
            time_ms=trace.time_ms,
            voltage_mv=trace.voltage_mv,
            input_value=trace.applied_current_ua_cm2,
            stimulus_start_ms=25.0,
            stimulus_end_ms=225.0,
            config=config,
        )
        traces.append(observed)
    training = [
        trace
        for index, trace in enumerate(traces)
        if index != held_out_index
    ]
    validation = [traces[held_out_index]]
    return training, validation, {
        "dataset": "canonical_hh",
        "cell_id": "canonical_6.3C",
        "temperature_c": 6.3,
        "training_current_ua_cm2": [7.0, 9.0, 12.0, 15.0],
        "validation_current_ua_cm2": [10.0],
    }


def _model_label(model: AbstractPhaseModel) -> str:
    return (
        "two_state"
        if not model.has_slow_state
        else f"slow_{model.slow_tau_ms:g}_ms"
    )


def _metric_rows(
    source: str,
    result: AbstractLadderResult,
    traces: list[AbstractObservedTrace],
    config: AbstractPhaseConfig,
) -> tuple[list[dict], dict[tuple[str, str], object]]:
    rows = []
    simulations = {}
    models = {
        "two_state": result.two_state,
        "selected": result.selected,
    }
    for model_name, model in models.items():
        for trace in traces:
            field = trace_field_metrics(model, trace)
            simulation_metric, simulation = trace_simulation_metrics(
                model,
                trace,
                config,
            )
            rows.append(
                {
                    "source": source,
                    "model": model_name,
                    "model_family": _model_label(model),
                    "trace": trace.name,
                    "role": trace.role,
                    **asdict(field),
                    "voltage_rmse_mv": simulation_metric.voltage_rmse_mv,
                    "velocity_rmse_mv_ms": (
                        simulation_metric.velocity_rmse_mv_ms
                    ),
                    "simulation_valid": simulation_metric.simulation_valid,
                    "simulation_reason": simulation.reason,
                }
            )
            simulations[(model_name, trace.name)] = simulation
    return rows, simulations


def _coefficient_rows(
    source: str,
    name: str,
    model: AbstractPhaseModel,
) -> list[dict]:
    rows = []
    for component, values in model.coefficients.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "source": source,
                    "model": name,
                    "model_family": _model_label(model),
                    "component": component,
                    "basis_index": index,
                    "coefficient": value,
                    "slow_tau_ms": model.slow_tau_ms,
                    "training_voltage_min_mv": (
                        model.training_voltage_min_mv
                    ),
                    "training_voltage_max_mv": (
                        model.training_voltage_max_mv
                    ),
                }
            )
    return rows


def _plot_source(
    source: str,
    result: AbstractLadderResult,
    training: list[AbstractObservedTrace],
    validation: list[AbstractObservedTrace],
    simulations: dict[tuple[str, str], object],
    path: Path,
) -> None:
    selected = result.selected
    held_out = validation[0]
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(14.5, 11.0),
        constrained_layout=True,
    )
    colors = {
        "observed": "#202124",
        "two_state": "#d55e00",
        "selected": "#0072b2",
    }

    for trace in (*training, *validation):
        relative_time = trace.time_ms - trace.stimulus_start_ms
        axis = axes[0, 0] if trace.role == "training" else axes[0, 1]
        axis.plot(
            relative_time,
            trace.voltage_mv,
            linewidth=0.8,
            label=trace.name,
        )
    axes[0, 0].set_title("Training voltage traces", loc="left")
    axes[0, 1].set_title("Held-out voltage trace", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8)

    held_time = held_out.time_ms - held_out.stimulus_start_ms
    for model_name in ("two_state", "selected"):
        simulation = simulations[(model_name, held_out.name)]
        axes[0, 1].plot(
            held_time,
            simulation.voltage_mv,
            color=colors[model_name],
            linewidth=0.9,
            alpha=0.85,
            label=model_name.replace("_", " "),
        )
    axes[0, 1].legend(frameon=False, fontsize=8)

    candidates = pd.DataFrame(result.candidate_table)
    x = np.arange(len(candidates))
    axes[0, 2].bar(
        x,
        candidates["selection_cross_validation_nrmse"],
        color="#8d99ae",
        label="training CV",
    )
    axes[0, 2].scatter(
        x,
        candidates["validation_acceleration_nrmse"],
        color="#c1121f",
        zorder=3,
        label="held out",
    )
    axes[0, 2].set_xticks(
        x,
        candidates["label"].str.replace("_", "\n"),
        fontsize=7,
    )
    axes[0, 2].set_ylabel("Acceleration NRMSE")
    axes[0, 2].set_title("Slow-memory selection", loc="left")
    axes[0, 2].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        held_out.voltage_mv,
        held_out.velocity_mv_ms,
        color=colors["observed"],
        linewidth=1.1,
        label="observed",
    )
    for model_name in ("two_state", "selected"):
        simulation = simulations[(model_name, held_out.name)]
        axes[1, 0].plot(
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
            color=colors[model_name],
            linewidth=0.9,
            alpha=0.8,
            label=model_name.replace("_", " "),
        )
    axes[1, 0].set_title("Free-running held-out phase plane", loc="left")
    axes[1, 0].set_xlabel("V (mV)")
    axes[1, 0].set_ylabel("dV/dt (mV/ms)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    memory = (
        memory_from_voltage(
            held_out,
            float(selected.slow_tau_ms),
            selected.memory_vhalf_mv,
            selected.memory_slope_mv,
            selected.memory_source_per_ms,
        )
        if selected.has_slow_state
        else np.zeros_like(held_out.time_ms)
    )
    predicted = selected.acceleration(
        held_out.voltage_mv,
        held_out.velocity_mv_ms,
        held_out.input_value,
        memory,
    )
    sample = np.linspace(
        0,
        len(held_out.time_ms) - 1,
        min(6000, len(held_out.time_ms)),
    ).astype(int)
    axes[1, 1].scatter(
        held_out.acceleration_mv_ms2[sample],
        predicted[sample],
        s=3,
        c=held_out.voltage_mv[sample],
        cmap="viridis",
        alpha=0.35,
        rasterized=True,
    )
    limits = np.nanquantile(
        np.concatenate(
            (
                held_out.acceleration_mv_ms2[sample],
                predicted[sample],
            )
        ),
        (0.01, 0.99),
    )
    axes[1, 1].plot(limits, limits, color="#555555", linestyle="--")
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_xlabel("Observed d²V/dt²")
    axes[1, 1].set_ylabel("Predicted d²V/dt²")
    axes[1, 1].set_title("Teacher-forced field fit", loc="left")

    axes[1, 2].plot(
        held_time,
        memory,
        color="#6a4c93",
        linewidth=1.0,
    )
    axes[1, 2].set_title(
        (
            f"Observed slow coordinate, tau={selected.slow_tau_ms:g} ms"
            if selected.has_slow_state
            else "No slow coordinate selected"
        ),
        loc="left",
    )
    axes[1, 2].set_ylabel("z")

    voltage_grid = np.linspace(
        selected.training_voltage_min_mv,
        selected.training_voltage_max_mv,
        400,
    )
    for axis, component, label in zip(
        axes[2],
        ("A", "B", "P"),
        (
            "Restoring field A(V)",
            "Linear damping B(V)",
            "Input P(V); dashed slow C(V)",
        ),
    ):
        axis.plot(
            voltage_grid,
            selected.component(component, voltage_grid),
            color="#0072b2",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_xlabel("V (mV)")
        axis.set_title(label, loc="left")
    if selected.has_slow_state:
        twin = axes[2, 2].twinx()
        twin.plot(
            voltage_grid,
            selected.component("C", voltage_grid),
            color="#b23a48",
            linestyle="--",
            label="C(V)",
        )
        twin.set_ylabel("Slow field C(V)", color="#b23a48")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{source}: abstract voltage oscillator",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _spike_count(
    voltage_mv: np.ndarray,
    time_ms: np.ndarray,
    start_ms: float,
    end_ms: float,
) -> int:
    during = (time_ms >= start_ms) & (time_ms < end_ms)
    above = voltage_mv >= 0.0
    return int(np.sum(above[1:] & ~above[:-1] & during[1:]))


def _plot_phase_template_source(
    source: str,
    result,
    held_out: AbstractObservedTrace,
    no_slow_simulation,
    selected_simulation,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 7.6),
        constrained_layout=True,
    )
    relative_time = held_out.time_ms - held_out.stimulus_start_ms
    axes[0, 0].plot(
        relative_time,
        held_out.voltage_mv,
        color="#202124",
        linewidth=0.9,
        label="observed",
    )
    axes[0, 0].plot(
        relative_time,
        no_slow_simulation.voltage_mv,
        color="#d55e00",
        linewidth=0.8,
        label="phase only",
    )
    axes[0, 0].plot(
        relative_time,
        selected_simulation.voltage_mv,
        color="#0072b2",
        linewidth=0.8,
        label="selected",
    )
    axes[0, 0].set_title("Held-out voltage train", loc="left")
    axes[0, 0].set_xlabel("Time from stimulus onset (ms)")
    axes[0, 0].set_ylabel("V (mV)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    during = (
        (held_out.time_ms >= held_out.stimulus_start_ms)
        & (held_out.time_ms < held_out.stimulus_end_ms)
    )
    upward = np.flatnonzero(
        (held_out.voltage_mv[1:] >= 0.0)
        & (held_out.voltage_mv[:-1] < 0.0)
        & during[1:]
    ) + 1
    cycle_start = (
        held_out.time_ms[upward[1]]
        if len(upward) >= 2
        else held_out.stimulus_start_ms
    )
    stable_cycle = during & (held_out.time_ms >= cycle_start)
    axes[0, 1].plot(
        held_out.voltage_mv[stable_cycle],
        held_out.velocity_mv_ms[stable_cycle],
        color="#202124",
        linewidth=1.0,
        label="observed",
    )
    axes[0, 1].plot(
        no_slow_simulation.voltage_mv[stable_cycle],
        no_slow_simulation.velocity_mv_ms[stable_cycle],
        color="#d55e00",
        linewidth=0.8,
        label="phase only",
    )
    axes[0, 1].plot(
        selected_simulation.voltage_mv[stable_cycle],
        selected_simulation.velocity_mv_ms[stable_cycle],
        color="#0072b2",
        linewidth=0.8,
        label="selected",
    )
    axes[0, 1].set_title("Held-out phase plane", loc="left")
    axes[0, 1].set_xlabel("V (mV)")
    axes[0, 1].set_ylabel("dV/dt (mV/ms)")

    candidates = pd.DataFrame(result.candidate_table)
    x = np.arange(len(candidates))
    axes[0, 2].bar(
        x,
        candidates["late_period_nrmse"],
        color="#8d99ae",
    )
    axes[0, 2].set_xticks(
        x,
        candidates["label"].str.replace("_", "\n"),
        fontsize=7,
    )
    axes[0, 2].set_ylabel("Late-cycle period NRMSE")
    axes[0, 2].set_title("Slow-state selection", loc="left")

    phase = np.asarray(result.selected.phase_grid)
    for current, template in zip(
        result.selected.current_levels,
        result.selected.voltage_templates_mv,
    ):
        axes[1, 0].plot(
            phase,
            template,
            linewidth=1.0,
            label=f"I={current:g}",
        )
    axes[1, 0].set_title("Learned periodic waveforms", loc="left")
    axes[1, 0].set_xlabel("Cycle phase")
    axes[1, 0].set_ylabel("V (mV)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for cycle_set in result.cycles:
        axes[1, 1].plot(
            np.arange(len(cycle_set.periods_ms)),
            cycle_set.periods_ms,
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            label=f"I={cycle_set.current_value:g}",
        )
    axes[1, 1].set_title("Training interspike periods", loc="left")
    axes[1, 1].set_xlabel("Cycle index")
    axes[1, 1].set_ylabel("Period (ms)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].plot(
        relative_time,
        selected_simulation.memory,
        color="#6a4c93",
        linewidth=1.0,
    )
    axes[1, 2].set_title(
        (
            f"Slow state, tau={result.selected.slow_tau_ms:g} ms"
            if result.selected.has_slow_state
            else "No slow state selected"
        ),
        loc="left",
    )
    axes[1, 2].set_xlabel("Time from stimulus onset (ms)")
    axes[1, 2].set_ylabel("z")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{source}: landmark phase-template reduction",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-spike-cycles",
        default="outputs/local_patchseq_spike_cycles.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/abstract_phase_pilot",
    )
    parser.add_argument("--dataset", default="gouwens_visp")
    parser.add_argument("--cell-id", default="674495385")
    parser.add_argument("--temperature-c", type=float, default=34.0)
    parser.add_argument("--skip-hh", action="store_true")
    parser.add_argument("--skip-biological", action="store_true")
    args = parser.parse_args()

    config = AbstractPhaseConfig()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = []
    if not args.skip_hh:
        experiments.append(("canonical_hh", *_hh_traces(config)))
    if not args.skip_biological:
        raw = load_raw_spike_cycle_table(args.raw_spike_cycles)
        experiments.append(
            (
                f"{args.dataset}_{args.cell_id}",
                *_biological_traces(
                    raw,
                    args.dataset,
                    args.cell_id,
                    args.temperature_c,
                    config,
                ),
            )
        )

    candidate_frames = []
    metric_rows = []
    coefficient_rows = []
    stability_rows = []
    phase_candidate_frames = []
    phase_metric_rows = []
    fourier_rows = []
    metadata = {"config": asdict(config), "experiments": {}}
    for source, training, validation, source_metadata in experiments:
        result = fit_abstract_model_ladder(
            training,
            validation,
            config,
        )
        candidates = pd.DataFrame(result.candidate_table)
        candidates.insert(0, "source", source)
        candidates["selected"] = candidates["label"].eq(
            _model_label(result.selected)
        )
        candidate_frames.append(candidates)
        rows, simulations = _metric_rows(
            source,
            result,
            [*training, *validation],
            config,
        )
        metric_rows.extend(rows)
        coefficient_rows.extend(
            _coefficient_rows(source, "two_state", result.two_state)
        )
        coefficient_rows.extend(
            _coefficient_rows(source, "selected", result.selected)
        )
        for name, model in (
            ("two_state", result.two_state),
            ("selected", result.selected),
        ):
            stability_rows.append(
                {
                    "source": source,
                    "model": name,
                    "model_family": _model_label(model),
                    **asdict(model_stability(model, training[0])),
                }
            )
        _plot_source(
            source,
            result,
            training,
            validation,
            simulations,
            output_root / f"{source}_summary.png",
        )
        resting_voltage = float(
            np.median(
                training[0].voltage_mv[
                    training[0].time_ms
                    < training[0].stimulus_start_ms
                ]
            )
        )
        phase_result = fit_phase_template_ladder(
            training,
            resting_voltage_mv=resting_voltage,
        )
        phase_candidates = pd.DataFrame(
            phase_result.candidate_table
        )
        phase_candidates.insert(0, "source", source)
        selected_label = (
            "phase_only"
            if not phase_result.selected.has_slow_state
            else (
                f"phase_slow_"
                f"{phase_result.selected.slow_tau_ms:g}_ms"
            )
        )
        phase_candidates["selected"] = phase_candidates[
            "label"
        ].eq(selected_label)
        phase_candidate_frames.append(phase_candidates)
        held_out = validation[0]
        no_slow_simulation = simulate_phase_template_model(
            phase_result.no_slow_model,
            held_out.time_ms,
            held_out.input_value,
            float(held_out.voltage_mv[0]),
        )
        selected_phase_simulation = simulate_phase_template_model(
            phase_result.selected,
            held_out.time_ms,
            held_out.input_value,
            float(held_out.voltage_mv[0]),
        )
        for name, model, simulation in (
            (
                "phase_only",
                phase_result.no_slow_model,
                no_slow_simulation,
            ),
            (
                "selected",
                phase_result.selected,
                selected_phase_simulation,
            ),
        ):
            during = (
                (held_out.time_ms >= held_out.stimulus_start_ms)
                & (held_out.time_ms < held_out.stimulus_end_ms)
            )
            phase_metric_rows.append(
                {
                    "source": source,
                    "model": name,
                    "model_family": (
                        "phase_only"
                        if not model.has_slow_state
                        else f"phase_slow_{model.slow_tau_ms:g}_ms"
                    ),
                    "held_out_voltage_rmse_mv": float(
                        np.sqrt(
                            np.mean(
                                (
                                    simulation.voltage_mv[during]
                                    - held_out.voltage_mv[during]
                                )
                                ** 2
                            )
                        )
                    ),
                    "held_out_phase_chamfer": (
                        phase_plane_chamfer_distance(
                            held_out,
                            simulation,
                        )
                    ),
                    "observed_spike_count": _spike_count(
                        held_out.voltage_mv,
                        held_out.time_ms,
                        held_out.stimulus_start_ms,
                        held_out.stimulus_end_ms,
                    ),
                    "simulated_spike_count": _spike_count(
                        simulation.voltage_mv,
                        held_out.time_ms,
                        held_out.stimulus_start_ms,
                        held_out.stimulus_end_ms,
                    ),
                    "observed_max_dvdt_mv_ms": float(
                        np.max(held_out.velocity_mv_ms[during])
                    ),
                    "simulated_max_dvdt_mv_ms": float(
                        np.max(simulation.velocity_mv_ms[during])
                    ),
                    "observed_min_dvdt_mv_ms": float(
                        np.min(held_out.velocity_mv_ms[during])
                    ),
                    "simulated_min_dvdt_mv_ms": float(
                        np.min(simulation.velocity_mv_ms[during])
                    ),
                }
            )
            for row in template_fourier_coefficients(model):
                fourier_rows.append(
                    {
                        "source": source,
                        "model": name,
                        **row,
                    }
                )
        _plot_phase_template_source(
            source,
            phase_result,
            held_out,
            no_slow_simulation,
            selected_phase_simulation,
            output_root / f"{source}_phase_template.png",
        )
        metadata["experiments"][source] = {
            **source_metadata,
            "selected_model": _model_label(result.selected),
            "selected_phase_model": selected_label,
        }

    pd.concat(candidate_frames, ignore_index=True).to_csv(
        output_root / "candidate_comparison.csv",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(
        output_root / "trace_metrics.csv",
        index=False,
    )
    pd.DataFrame(coefficient_rows).to_csv(
        output_root / "model_coefficients.csv",
        index=False,
    )
    pd.DataFrame(stability_rows).to_csv(
        output_root / "stability_diagnostics.csv",
        index=False,
    )
    pd.concat(phase_candidate_frames, ignore_index=True).to_csv(
        output_root / "phase_template_candidates.csv",
        index=False,
    )
    pd.DataFrame(phase_metric_rows).to_csv(
        output_root / "phase_template_metrics.csv",
        index=False,
    )
    pd.DataFrame(fourier_rows).to_csv(
        output_root / "phase_template_fourier_coefficients.csv",
        index=False,
    )
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
