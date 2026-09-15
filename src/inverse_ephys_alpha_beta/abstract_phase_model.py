"""Data-derived generalized Lienard models for neuronal voltage dynamics.

The model deliberately avoids channel and conductance parameters. Its observed
state is voltage and voltage velocity:

    dV/dt = u
    du/dt = A(V) + B(V) u + D(V) u^2
            + P(V) I + Q(V) I u + C(V) z
    dz/dt = S(V) - z / tau_z

The slow state is optional. A, B, P, and C are smooth cubic B-spline
functions. Their coefficients are estimated by regularized linear regression
against smoothed voltage acceleration. The resulting model is a continuous
oscillator with no spike reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class AbstractPhaseConfig:
    """Numerical and regularization settings for field identification."""

    spline_basis_count: int = 9
    spline_degree: int = 3
    derivative_window_ms: float = 0.35
    derivative_polynomial_order: int = 3
    smoothness_penalty: float = 0.002
    ridge_penalty: float = 1e-5
    velocity_weight: float = 2.0
    acceleration_weight: float = 2.0
    maximum_samples_per_trace: int = 12_000
    onset_exclusion_ms: float = 0.6
    memory_vhalf_mv: float = -20.0
    memory_slope_mv: float = 5.0
    memory_source_per_ms: float = 1.0
    slow_tau_candidates_ms: tuple[float, ...] = (
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
    )
    slow_complexity_penalty: float = 0.01
    rest_anchor_weight: float = 100.0
    rest_stability_weight: float = 1_000.0
    rest_damping_per_ms: float = -0.15
    rest_restoring_per_ms2: float = -0.02
    repetitive_spiking_instability_weight: float = 200.0
    repetitive_spiking_damping_per_ms: float = 0.15
    cycle_contraction_weight: float = 500.0
    cycle_mean_damping_per_ms: float = -0.05
    velocity_domain_margin: float = 1.25
    boundary_velocity_damping: float = 5.0
    boundary_voltage_restoring: float = 1.0
    simulation_dt_ms: float = 0.025
    voltage_min_mv: float = -140.0
    voltage_max_mv: float = 100.0
    velocity_limit_mv_ms: float = 2_000.0


@dataclass(frozen=True)
class AbstractObservedTrace:
    """Smoothed voltage data and derivatives for one current protocol."""

    name: str
    role: str
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    velocity_mv_ms: np.ndarray
    acceleration_mv_ms2: np.ndarray
    input_value: np.ndarray
    fit_mask: np.ndarray
    stimulus_start_ms: float
    stimulus_end_ms: float


@dataclass(frozen=True)
class AbstractSimulation:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    velocity_mv_ms: np.ndarray
    memory: np.ndarray
    valid: bool
    reason: str


@dataclass(frozen=True)
class AbstractFieldMetrics:
    acceleration_nrmse: float
    acceleration_correlation: float
    voltage_rmse_mv: float = float("nan")
    velocity_rmse_mv_ms: float = float("nan")
    simulation_valid: bool = False


@dataclass(frozen=True)
class AbstractStability:
    resting_voltage_mv: float
    resting_acceleration_mv_ms2: float
    restoring_slope_per_ms2: float
    damping_per_ms: float
    max_rest_eigenvalue_real_per_ms: float
    locally_stable: bool
    observed_cycle_log_volume_change: float
    volume_contracting_on_observed_cycle: bool


@dataclass(frozen=True)
class AbstractPhaseModel:
    """Spline-parameterized two-state or one-memory-state oscillator."""

    knots_mv: tuple[float, ...]
    degree: int
    coefficients: Mapping[str, tuple[float, ...]]
    voltage_center_mv: float
    voltage_scale_mv: float
    velocity_scale_mv_ms: float
    acceleration_scale_mv_ms2: float
    input_scale: float
    memory_scale: float
    slow_tau_ms: float | None
    memory_vhalf_mv: float
    memory_slope_mv: float
    memory_source_per_ms: float
    velocity_domain_margin: float
    boundary_velocity_damping: float
    boundary_voltage_restoring: float
    training_voltage_min_mv: float
    training_voltage_max_mv: float

    @property
    def has_slow_state(self) -> bool:
        return self.slow_tau_ms is not None

    @property
    def basis_count(self) -> int:
        return len(self.knots_mv) - self.degree - 1

    @property
    def parameter_count(self) -> int:
        return self.basis_count * len(self.coefficients)

    def basis(self, voltage_mv: np.ndarray | float) -> np.ndarray:
        voltage = np.asarray(voltage_mv, dtype=float)
        clipped_voltage = np.clip(
            voltage,
            self.training_voltage_min_mv,
            self.training_voltage_max_mv,
        )
        identity = np.eye(self.basis_count)
        return np.asarray(
            BSpline(
                self.knots_mv,
                identity,
                self.degree,
                extrapolate=True,
            )(clipped_voltage),
            dtype=float,
        )

    def basis_derivative(
        self,
        voltage_mv: np.ndarray | float,
    ) -> np.ndarray:
        voltage = np.asarray(voltage_mv, dtype=float)
        clipped = np.clip(
            voltage,
            self.training_voltage_min_mv,
            self.training_voltage_max_mv,
        )
        identity = np.eye(self.basis_count)
        return np.asarray(
            BSpline(
                self.knots_mv,
                identity,
                self.degree,
                extrapolate=True,
            ).derivative(1)(clipped),
            dtype=float,
        )

    def component(
        self,
        name: str,
        voltage_mv: np.ndarray | float,
    ) -> np.ndarray:
        coefficients = np.asarray(self.coefficients[name], dtype=float)
        return np.asarray(self.basis(voltage_mv) @ coefficients)

    def memory_activation(
        self,
        voltage_mv: np.ndarray | float,
    ) -> np.ndarray:
        argument = (
            np.asarray(voltage_mv, dtype=float) - self.memory_vhalf_mv
        ) / self.memory_slope_mv
        return self.memory_source_per_ms / (
            1.0 + np.exp(np.clip(-argument, -80.0, 80.0))
        )

    def acceleration(
        self,
        voltage_mv: np.ndarray | float,
        velocity_mv_ms: np.ndarray | float,
        input_value: np.ndarray | float,
        memory: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        basis = self.basis(voltage_mv)
        raw_velocity = (
            np.asarray(velocity_mv_ms, dtype=float)
            / self.velocity_scale_mv_ms
        )
        velocity = np.clip(
            raw_velocity,
            -self.velocity_domain_margin,
            self.velocity_domain_margin,
        )
        normalized_input = (
            np.asarray(input_value, dtype=float) / self.input_scale
        )
        normalized = (
            basis @ np.asarray(self.coefficients["A"])
            + (basis @ np.asarray(self.coefficients["B"])) * velocity
            + (basis @ np.asarray(self.coefficients["D"])) * velocity**2
            + (basis @ np.asarray(self.coefficients["P"]))
            * normalized_input
            + (basis @ np.asarray(self.coefficients["Q"]))
            * normalized_input
            * velocity
        )
        if self.has_slow_state:
            normalized_memory = (
                np.asarray(memory, dtype=float) / self.memory_scale
            )
            normalized = normalized + (
                basis @ np.asarray(self.coefficients["C"])
            ) * normalized_memory
        normalized = (
            normalized
            - self.boundary_velocity_damping
            * (raw_velocity - velocity)
            - self.boundary_voltage_restoring
            * (
                np.asarray(voltage_mv, dtype=float)
                - np.clip(
                    np.asarray(voltage_mv, dtype=float),
                    self.training_voltage_min_mv,
                    self.training_voltage_max_mv,
                )
            )
            / self.voltage_scale_mv
        )
        return self.acceleration_scale_mv_ms2 * normalized

    def physical_damping(
        self,
        voltage_mv: np.ndarray | float,
        velocity_mv_ms: np.ndarray | float = 0.0,
        input_value: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        normalized_velocity = (
            np.asarray(velocity_mv_ms, dtype=float)
            / self.velocity_scale_mv_ms
        )
        normalized_input = (
            np.asarray(input_value, dtype=float) / self.input_scale
        )
        return (
            self.acceleration_scale_mv_ms2
            / self.velocity_scale_mv_ms
            * (
                self.component("B", voltage_mv)
                + 2.0
                * self.component("D", voltage_mv)
                * normalized_velocity
                + self.component("Q", voltage_mv)
                * normalized_input
            )
        )


@dataclass(frozen=True)
class AbstractLadderResult:
    two_state: AbstractPhaseModel
    slow_state_candidates: tuple[AbstractPhaseModel, ...]
    selected: AbstractPhaseModel
    training_metrics: Mapping[str, AbstractFieldMetrics]
    validation_metrics: Mapping[str, AbstractFieldMetrics]
    candidate_table: tuple[Mapping[str, object], ...]


def _window_samples(
    dt_ms: float,
    window_ms: float,
    polynomial_order: int,
    sample_count: int,
) -> int:
    window = max(
        polynomial_order + 2,
        int(round(window_ms / dt_ms)),
    )
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        window = sample_count if sample_count % 2 else sample_count - 1
    if window <= polynomial_order:
        raise ValueError("Trace is too short for derivative smoothing")
    return window


def prepare_abstract_trace(
    name: str,
    role: str,
    time_ms: Sequence[float],
    voltage_mv: Sequence[float],
    input_value: Sequence[float] | float,
    stimulus_start_ms: float,
    stimulus_end_ms: float,
    config: AbstractPhaseConfig | None = None,
) -> AbstractObservedTrace:
    """Smooth voltage and calculate first and second time derivatives."""
    config = config or AbstractPhaseConfig()
    time = np.asarray(time_ms, dtype=float)
    voltage = np.asarray(voltage_mv, dtype=float)
    if time.ndim != 1 or voltage.shape != time.shape:
        raise ValueError("Time and voltage must be equal one-dimensional arrays")
    if len(time) < 9 or not np.all(np.isfinite(time + voltage)):
        raise ValueError("Trace must contain at least nine finite samples")
    intervals = np.diff(time)
    if np.any(intervals <= 0.0):
        raise ValueError("Time must be strictly increasing")
    dt_ms = float(np.median(intervals))
    if np.max(np.abs(intervals - dt_ms)) > 0.01 * dt_ms:
        raise ValueError("Trace must be regularly sampled within one percent")
    if np.isscalar(input_value):
        current = np.full_like(time, float(input_value))
    else:
        current = np.asarray(input_value, dtype=float)
        if current.shape != time.shape:
            raise ValueError("Input must be scalar or match the time array")
    window = _window_samples(
        dt_ms,
        config.derivative_window_ms,
        config.derivative_polynomial_order,
        len(time),
    )
    filtered_voltage = savgol_filter(
        voltage,
        window_length=window,
        polyorder=config.derivative_polynomial_order,
        mode="interp",
    )
    velocity = savgol_filter(
        voltage,
        window_length=window,
        polyorder=config.derivative_polynomial_order,
        deriv=1,
        delta=dt_ms,
        mode="interp",
    )
    acceleration = savgol_filter(
        voltage,
        window_length=window,
        polyorder=config.derivative_polynomial_order,
        deriv=2,
        delta=dt_ms,
        mode="interp",
    )
    margin = window // 2 + 1
    fit_mask = np.ones(len(time), dtype=bool)
    fit_mask[:margin] = False
    fit_mask[-margin:] = False
    fit_mask &= (
        np.abs(time - stimulus_start_ms)
        > config.onset_exclusion_ms
    )
    fit_mask &= (
        np.abs(time - stimulus_end_ms)
        > config.onset_exclusion_ms
    )
    fit_mask &= np.isfinite(
        filtered_voltage + velocity + acceleration + current
    )
    return AbstractObservedTrace(
        name=name,
        role=role,
        time_ms=time,
        voltage_mv=np.asarray(filtered_voltage),
        velocity_mv_ms=np.asarray(velocity),
        acceleration_mv_ms2=np.asarray(acceleration),
        input_value=current,
        fit_mask=fit_mask,
        stimulus_start_ms=float(stimulus_start_ms),
        stimulus_end_ms=float(stimulus_end_ms),
    )


def memory_from_voltage(
    trace: AbstractObservedTrace,
    tau_ms: float,
    vhalf_mv: float,
    slope_mv: float,
    source_per_ms: float = 1.0,
) -> np.ndarray:
    """Calculate the deterministic slow state driven by observed voltage."""
    if tau_ms <= 0.0 or slope_mv <= 0.0 or source_per_ms <= 0.0:
        raise ValueError("Slow-state scales must be positive")
    time = trace.time_ms
    dt = float(np.median(np.diff(time)))
    activation = source_per_ms / (
        1.0
        + np.exp(
            np.clip(
                -(trace.voltage_mv - vhalf_mv) / slope_mv,
                -80.0,
                80.0,
            )
        )
    )
    decay = math.exp(-dt / tau_ms)
    source_gain = tau_ms * (1.0 - decay)
    state = np.empty_like(time)
    state[0] = tau_ms * activation[0]
    for index in range(len(time) - 1):
        state[index + 1] = (
            decay * state[index] + source_gain * activation[index]
        )
    return state


def _clamped_knots(
    voltage: np.ndarray,
    basis_count: int,
    degree: int,
) -> np.ndarray:
    if basis_count < degree + 1:
        raise ValueError("Spline basis count is too small for the degree")
    lower = float(np.min(voltage))
    upper = float(np.max(voltage))
    if upper - lower < 20.0:
        raise ValueError("Training voltage does not span a spike trajectory")
    internal_count = basis_count - degree - 1
    internal = (
        np.linspace(lower, upper, internal_count + 2)[1:-1]
        if internal_count
        else np.asarray([], dtype=float)
    )
    return np.concatenate(
        (
            np.repeat(lower, degree + 1),
            internal,
            np.repeat(upper, degree + 1),
        )
    )


def _basis_matrix(
    voltage_mv: np.ndarray,
    knots_mv: np.ndarray,
    degree: int,
) -> np.ndarray:
    basis_count = len(knots_mv) - degree - 1
    clipped = np.clip(voltage_mv, knots_mv[degree], knots_mv[-degree - 1])
    return np.asarray(
        BSpline(
            knots_mv,
            np.eye(basis_count),
            degree,
            extrapolate=True,
        )(clipped),
        dtype=float,
    )


def _subsample_indices(
    indices: np.ndarray,
    maximum: int,
) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return indices[np.unique(np.round(positions).astype(int))]


def _second_difference_penalty(
    block_count: int,
    basis_count: int,
) -> np.ndarray:
    if basis_count < 3:
        return np.zeros((0, block_count * basis_count), dtype=float)
    difference = np.diff(np.eye(basis_count), n=2, axis=0)
    rows = []
    for block in range(block_count):
        row = np.zeros(
            (difference.shape[0], block_count * basis_count),
            dtype=float,
        )
        start = block * basis_count
        row[:, start : start + basis_count] = difference
        rows.append(row)
    return np.vstack(rows)


def fit_abstract_phase_model(
    traces: Sequence[AbstractObservedTrace],
    config: AbstractPhaseConfig | None = None,
    slow_tau_ms: float | None = None,
) -> AbstractPhaseModel:
    """Fit a two-state or one-slow-state acceleration field."""
    config = config or AbstractPhaseConfig()
    if not traces:
        raise ValueError("At least one training trace is required")
    selected_rows = []
    memories = []
    for trace in traces:
        indices = np.flatnonzero(trace.fit_mask)
        indices = _subsample_indices(
            indices,
            config.maximum_samples_per_trace,
        )
        if not len(indices):
            raise ValueError(f"Trace {trace.name} has no fitted samples")
        selected_rows.append(indices)
        memories.append(
            (
                memory_from_voltage(
                    trace,
                    slow_tau_ms,
                    config.memory_vhalf_mv,
                    config.memory_slope_mv,
                    config.memory_source_per_ms,
                )
                if slow_tau_ms is not None
                else np.zeros(len(trace.time_ms), dtype=float)
            )
        )
    voltage = np.concatenate(
        [
            trace.voltage_mv[indices]
            for trace, indices in zip(traces, selected_rows)
        ]
    )
    velocity = np.concatenate(
        [
            trace.velocity_mv_ms[indices]
            for trace, indices in zip(traces, selected_rows)
        ]
    )
    acceleration = np.concatenate(
        [
            trace.acceleration_mv_ms2[indices]
            for trace, indices in zip(traces, selected_rows)
        ]
    )
    input_values = np.concatenate(
        [
            trace.input_value[indices]
            for trace, indices in zip(traces, selected_rows)
        ]
    )
    memory = np.concatenate(
        [
            state[indices]
            for state, indices in zip(memories, selected_rows)
        ]
    )
    velocity_scale = max(
        1.0,
        float(np.max(np.abs(velocity))),
    )
    acceleration_scale = max(
        1.0,
        float(np.quantile(np.abs(acceleration), 0.98)),
    )
    input_scale = max(
        1.0,
        float(np.max(np.abs(input_values))),
    )
    memory_scale = (
        max(1e-6, float(np.quantile(np.abs(memory), 0.98)))
        if slow_tau_ms is not None
        else 1.0
    )
    knots = _clamped_knots(
        voltage,
        config.spline_basis_count,
        config.spline_degree,
    )
    basis = _basis_matrix(voltage, knots, config.spline_degree)
    blocks = [
        basis,
        basis * (velocity / velocity_scale)[:, None],
        basis * (velocity / velocity_scale)[:, None] ** 2,
        basis * (input_values / input_scale)[:, None],
        basis
        * (
            (input_values / input_scale)
            * (velocity / velocity_scale)
        )[:, None],
    ]
    names = ["A", "B", "D", "P", "Q"]
    if slow_tau_ms is not None:
        blocks.append(basis * (memory / memory_scale)[:, None])
        names.append("C")
    design = np.hstack(blocks)
    target = acceleration / acceleration_scale
    velocity_importance = np.clip(
        np.abs(velocity) / velocity_scale,
        0.0,
        1.0,
    )
    acceleration_importance = np.clip(
        np.abs(acceleration) / acceleration_scale,
        0.0,
        1.0,
    )
    weights = (
        1.0
        + config.velocity_weight * velocity_importance
        + config.acceleration_weight * acceleration_importance
    )
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    data_normalizer = math.sqrt(len(weighted_target))
    weighted_design = weighted_design / data_normalizer
    weighted_target = weighted_target / data_normalizer
    anchor_rows = []
    anchor_targets = []
    anchor_weights = []
    spline = BSpline(
        knots,
        np.eye(config.spline_basis_count),
        config.spline_degree,
        extrapolate=True,
    )
    for trace, state in zip(traces, memories):
        baseline = trace.time_ms < trace.stimulus_start_ms
        if not np.any(baseline):
            continue
        rest = float(np.median(trace.voltage_mv[baseline]))
        rest_basis = _basis_matrix(
            np.asarray([rest]),
            knots,
            config.spline_degree,
        )[0]
        rest_derivative = np.asarray(
            spline.derivative(1)(
                np.clip(
                    rest,
                    knots[config.spline_degree],
                    knots[-config.spline_degree - 1],
                )
            ),
            dtype=float,
        )
        rest_memory = (
            float(np.median(state[baseline])) / memory_scale
            if slow_tau_ms is not None
            else 0.0
        )

        fixed_point_row = np.zeros(design.shape[1], dtype=float)
        a_start = names.index("A") * config.spline_basis_count
        fixed_point_row[
            a_start : a_start + config.spline_basis_count
        ] = rest_basis
        if slow_tau_ms is not None:
            c_start = names.index("C") * config.spline_basis_count
            fixed_point_row[
                c_start : c_start + config.spline_basis_count
            ] = rest_basis * rest_memory
        anchor_rows.append(fixed_point_row)
        anchor_targets.append(0.0)
        anchor_weights.append(config.rest_anchor_weight)

        damping_row = np.zeros(design.shape[1], dtype=float)
        b_start = names.index("B") * config.spline_basis_count
        damping_row[
            b_start : b_start + config.spline_basis_count
        ] = rest_basis
        anchor_rows.append(damping_row)
        anchor_targets.append(
            config.rest_damping_per_ms
            * velocity_scale
            / acceleration_scale
        )
        anchor_weights.append(config.rest_stability_weight)

        restoring_row = np.zeros(design.shape[1], dtype=float)
        restoring_row[
            a_start : a_start + config.spline_basis_count
        ] = rest_derivative
        if slow_tau_ms is not None:
            restoring_row[
                c_start : c_start + config.spline_basis_count
            ] = rest_derivative * rest_memory
        anchor_rows.append(restoring_row)
        anchor_targets.append(
            config.rest_restoring_per_ms2 / acceleration_scale
        )
        anchor_weights.append(config.rest_stability_weight)

        during = (
            (trace.time_ms >= trace.stimulus_start_ms)
            & (trace.time_ms < trace.stimulus_end_ms)
        )
        above = trace.voltage_mv >= 0.0
        spike_count = int(
            np.sum(
                above[1:]
                & ~above[:-1]
                & during[1:]
            )
        )
        if spike_count >= 3:
            plateau_input = float(
                np.median(trace.input_value[during])
            )
            instability_row = np.zeros(
                design.shape[1],
                dtype=float,
            )
            instability_row[
                b_start : b_start + config.spline_basis_count
            ] = rest_basis
            q_start = names.index("Q") * config.spline_basis_count
            instability_row[
                q_start : q_start + config.spline_basis_count
            ] = rest_basis * plateau_input / input_scale
            anchor_rows.append(instability_row)
            anchor_targets.append(
                config.repetitive_spiking_damping_per_ms
                * velocity_scale
                / acceleration_scale
            )
            anchor_weights.append(
                config.repetitive_spiking_instability_weight
            )

            cycle_indices = np.flatnonzero(
                during & trace.fit_mask
            )
            cycle_basis = _basis_matrix(
                trace.voltage_mv[cycle_indices],
                knots,
                config.spline_degree,
            )
            cycle_velocity = (
                trace.velocity_mv_ms[cycle_indices]
                / velocity_scale
            )
            cycle_input = (
                trace.input_value[cycle_indices] / input_scale
            )
            contraction_row = np.zeros(
                design.shape[1],
                dtype=float,
            )
            contraction_row[
                b_start : b_start + config.spline_basis_count
            ] = np.mean(cycle_basis, axis=0)
            d_start = names.index("D") * config.spline_basis_count
            contraction_row[
                d_start : d_start + config.spline_basis_count
            ] = np.mean(
                2.0 * cycle_basis * cycle_velocity[:, None],
                axis=0,
            )
            contraction_row[
                q_start : q_start + config.spline_basis_count
            ] = np.mean(
                cycle_basis * cycle_input[:, None],
                axis=0,
            )
            anchor_rows.append(contraction_row)
            anchor_targets.append(
                config.cycle_mean_damping_per_ms
                * velocity_scale
                / acceleration_scale
            )
            anchor_weights.append(config.cycle_contraction_weight)
    if anchor_rows:
        anchor_weight_array = np.sqrt(
            np.asarray(anchor_weights, dtype=float)
        )
        weighted_design = np.vstack(
            (
                weighted_design,
                np.asarray(anchor_rows) * anchor_weight_array[:, None],
            )
        )
        weighted_target = np.concatenate(
            (
                weighted_target,
                np.asarray(anchor_targets) * anchor_weight_array,
            )
        )
    penalty = _second_difference_penalty(
        len(blocks),
        config.spline_basis_count,
    )
    normal = weighted_design.T @ weighted_design
    normal += config.smoothness_penalty * (penalty.T @ penalty)
    normal += config.ridge_penalty * np.eye(normal.shape[0])
    right = weighted_design.T @ weighted_target
    coefficients = np.linalg.solve(normal, right)
    mapping = {
        name: tuple(
            coefficients[
                index * config.spline_basis_count :
                (index + 1) * config.spline_basis_count
            ]
        )
        for index, name in enumerate(names)
    }
    return AbstractPhaseModel(
        knots_mv=tuple(float(value) for value in knots),
        degree=config.spline_degree,
        coefficients=mapping,
        voltage_center_mv=float(np.median(voltage)),
        voltage_scale_mv=max(1.0, float(np.std(voltage))),
        velocity_scale_mv_ms=velocity_scale,
        acceleration_scale_mv_ms2=acceleration_scale,
        input_scale=input_scale,
        memory_scale=memory_scale,
        slow_tau_ms=slow_tau_ms,
        memory_vhalf_mv=config.memory_vhalf_mv,
        memory_slope_mv=config.memory_slope_mv,
        memory_source_per_ms=config.memory_source_per_ms,
        velocity_domain_margin=config.velocity_domain_margin,
        boundary_velocity_damping=config.boundary_velocity_damping,
        boundary_voltage_restoring=config.boundary_voltage_restoring,
        training_voltage_min_mv=float(knots[config.spline_degree]),
        training_voltage_max_mv=float(knots[-config.spline_degree - 1]),
    )


def trace_field_metrics(
    model: AbstractPhaseModel,
    trace: AbstractObservedTrace,
) -> AbstractFieldMetrics:
    memory = (
        memory_from_voltage(
            trace,
            float(model.slow_tau_ms),
            model.memory_vhalf_mv,
            model.memory_slope_mv,
            model.memory_source_per_ms,
        )
        if model.has_slow_state
        else np.zeros(len(trace.time_ms), dtype=float)
    )
    mask = trace.fit_mask
    predicted = model.acceleration(
        trace.voltage_mv[mask],
        trace.velocity_mv_ms[mask],
        trace.input_value[mask],
        memory[mask],
    )
    observed = trace.acceleration_mv_ms2[mask]
    residual = predicted - observed
    scale = max(
        1.0,
        float(np.sqrt(np.mean(observed**2))),
    )
    correlation = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if np.std(predicted) > 0.0 and np.std(observed) > 0.0
        else float("nan")
    )
    return AbstractFieldMetrics(
        acceleration_nrmse=float(
            np.sqrt(np.mean(residual**2)) / scale
        ),
        acceleration_correlation=correlation,
    )


def simulate_abstract_phase_model(
    model: AbstractPhaseModel,
    time_ms: Sequence[float],
    input_value: Sequence[float] | float,
    initial_voltage_mv: float,
    initial_velocity_mv_ms: float,
    initial_memory: float = 0.0,
    config: AbstractPhaseConfig | None = None,
) -> AbstractSimulation:
    """Integrate the learned continuous oscillator with fixed-step RK4."""
    config = config or AbstractPhaseConfig()
    requested_time = np.asarray(time_ms, dtype=float)
    if len(requested_time) < 2 or np.any(np.diff(requested_time) <= 0.0):
        raise ValueError("Simulation time must be strictly increasing")
    dt = min(
        config.simulation_dt_ms,
        float(np.median(np.diff(requested_time))),
    )
    start = float(requested_time[0])
    stop = float(requested_time[-1])
    count = int(math.ceil((stop - start) / dt)) + 1
    internal_time = np.linspace(start, stop, count)
    if np.isscalar(input_value):
        internal_input = np.full(count, float(input_value))
    else:
        values = np.asarray(input_value, dtype=float)
        if values.shape != requested_time.shape:
            raise ValueError("Input array must match requested simulation time")
        internal_input = np.interp(
            internal_time, requested_time, values
        )
    state = np.asarray(
        (initial_voltage_mv, initial_velocity_mv_ms, initial_memory),
        dtype=float,
    )
    states = np.empty((count, 3), dtype=float)
    states[0] = state

    def rhs(t_index: int, candidate: np.ndarray) -> np.ndarray:
        current = float(internal_input[min(t_index, count - 1)])
        voltage, velocity, memory = candidate
        acceleration = float(
            model.acceleration(
                voltage,
                velocity,
                current,
                memory,
            )
        )
        memory_rhs = (
            float(model.memory_activation(voltage))
            - memory / float(model.slow_tau_ms)
            if model.has_slow_state
            else 0.0
        )
        return np.asarray((velocity, acceleration, memory_rhs))

    valid = True
    reason = "accepted"
    for index in range(count - 1):
        k1 = rhs(index, state)
        k2 = rhs(index, state + 0.5 * dt * k1)
        k3 = rhs(index, state + 0.5 * dt * k2)
        k4 = rhs(index + 1, state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        states[index + 1] = state
        if (
            not np.all(np.isfinite(state))
            or not config.voltage_min_mv <= state[0] <= config.voltage_max_mv
            or abs(state[1]) > config.velocity_limit_mv_ms
        ):
            valid = False
            reason = f"diverged at {internal_time[index + 1]:.3f} ms"
            states[index + 1 :] = np.nan
            break
    voltage = np.interp(requested_time, internal_time, states[:, 0])
    velocity = np.interp(requested_time, internal_time, states[:, 1])
    memory = np.interp(requested_time, internal_time, states[:, 2])
    return AbstractSimulation(
        time_ms=requested_time,
        voltage_mv=voltage,
        velocity_mv_ms=velocity,
        memory=memory,
        valid=valid,
        reason=reason,
    )


def trace_simulation_metrics(
    model: AbstractPhaseModel,
    trace: AbstractObservedTrace,
    config: AbstractPhaseConfig | None = None,
) -> tuple[AbstractFieldMetrics, AbstractSimulation]:
    initial_memory = (
        float(
            memory_from_voltage(
                trace,
                float(model.slow_tau_ms),
                model.memory_vhalf_mv,
                model.memory_slope_mv,
                model.memory_source_per_ms,
            )[0]
        )
        if model.has_slow_state
        else 0.0
    )
    simulation = simulate_abstract_phase_model(
        model,
        trace.time_ms,
        trace.input_value,
        float(trace.voltage_mv[0]),
        float(trace.velocity_mv_ms[0]),
        initial_memory,
        config,
    )
    field = trace_field_metrics(model, trace)
    mask = trace.fit_mask
    return (
        AbstractFieldMetrics(
            acceleration_nrmse=field.acceleration_nrmse,
            acceleration_correlation=field.acceleration_correlation,
            voltage_rmse_mv=(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                simulation.voltage_mv[mask]
                                - trace.voltage_mv[mask]
                            )
                            ** 2
                        )
                    )
                )
                if simulation.valid
                else float("nan")
            ),
            velocity_rmse_mv_ms=(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                simulation.velocity_mv_ms[mask]
                                - trace.velocity_mv_ms[mask]
                            )
                            ** 2
                        )
                    )
                )
                if simulation.valid
                else float("nan")
            ),
            simulation_valid=simulation.valid,
        ),
        simulation,
    )


def model_stability(
    model: AbstractPhaseModel,
    trace: AbstractObservedTrace,
    resting_voltage_mv: float | None = None,
) -> AbstractStability:
    """Evaluate fixed-point and observed-cycle contraction diagnostics."""
    baseline = trace.time_ms < trace.stimulus_start_ms
    rest = (
        float(resting_voltage_mv)
        if resting_voltage_mv is not None
        else float(np.median(trace.voltage_mv[baseline]))
    )
    rest_memory = (
        float(model.slow_tau_ms) * float(model.memory_activation(rest))
        if model.has_slow_state
        else 0.0
    )
    acceleration = float(
        model.acceleration(rest, 0.0, 0.0, rest_memory)
    )
    derivative_basis = model.basis_derivative(rest)
    restoring = float(
        model.acceleration_scale_mv_ms2
        * (
            derivative_basis
            @ np.asarray(model.coefficients["A"], dtype=float)
            + (
                (
                    derivative_basis
                    @ np.asarray(
                        model.coefficients["C"],
                        dtype=float,
                    )
                )
                * rest_memory
                / model.memory_scale
                if model.has_slow_state
                else 0.0
            )
        )
    )
    damping = float(model.physical_damping(rest))
    if model.has_slow_state:
        slow_gain = float(
            model.acceleration_scale_mv_ms2
            * model.component("C", rest)
            / model.memory_scale
        )
        activation = float(model.memory_activation(rest))
        activation_derivative = (
            activation
            * (
                model.memory_source_per_ms - activation
            )
            / (
                model.memory_source_per_ms
                * model.memory_slope_mv
            )
        )
        jacobian = np.asarray(
            (
                (0.0, 1.0, 0.0),
                (restoring, damping, slow_gain),
                (
                    activation_derivative,
                    0.0,
                    -1.0 / float(model.slow_tau_ms),
                ),
            )
        )
    else:
        jacobian = np.asarray(((0.0, 1.0), (restoring, damping)))
    max_eigenvalue = float(
        np.max(np.real(np.linalg.eigvals(jacobian)))
    )
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
        & trace.fit_mask
    )
    divergence = float(
        np.trapz(
            model.physical_damping(
                trace.voltage_mv[during],
                trace.velocity_mv_ms[during],
                trace.input_value[during],
            ),
            trace.time_ms[during],
        )
    )
    if model.has_slow_state:
        divergence -= float(
            (trace.time_ms[during][-1] - trace.time_ms[during][0])
            / float(model.slow_tau_ms)
        )
    return AbstractStability(
        resting_voltage_mv=rest,
        resting_acceleration_mv_ms2=acceleration,
        restoring_slope_per_ms2=restoring,
        damping_per_ms=damping,
        max_rest_eigenvalue_real_per_ms=max_eigenvalue,
        locally_stable=bool(max_eigenvalue < 0.0),
        observed_cycle_log_volume_change=divergence,
        volume_contracting_on_observed_cycle=bool(divergence < 0.0),
    )


def fit_abstract_model_ladder(
    training_traces: Sequence[AbstractObservedTrace],
    validation_traces: Sequence[AbstractObservedTrace],
    config: AbstractPhaseConfig | None = None,
) -> AbstractLadderResult:
    """Compare two- and three-state fields without selecting on held-out data."""
    config = config or AbstractPhaseConfig()
    if not training_traces:
        raise ValueError("At least one training trace is required")

    candidate_taus: tuple[float | None, ...] = (
        None,
        *config.slow_tau_candidates_ms,
    )
    fitted_models = tuple(
        fit_abstract_phase_model(
            training_traces,
            config,
            slow_tau_ms=tau,
        )
        for tau in candidate_taus
    )
    two_state = fitted_models[0]
    slow_models = fitted_models[1:]
    candidates = (two_state, *slow_models)
    rows = []
    model_training_metrics = {}
    model_validation_metrics = {}
    best_model = two_state
    best_score = float("inf")
    for model, slow_tau_ms in zip(candidates, candidate_taus):
        label = (
            "two_state"
            if not model.has_slow_state
            else f"slow_{model.slow_tau_ms:g}_ms"
        )
        training = {
            trace.name: trace_field_metrics(model, trace)
            for trace in training_traces
        }
        validation = {
            trace.name: trace_field_metrics(model, trace)
            for trace in validation_traces
        }
        train_loss = float(
            np.mean(
                [
                    metric.acceleration_nrmse
                    for metric in training.values()
                ]
            )
        )
        validation_loss = float(
            np.mean(
                [
                    metric.acceleration_nrmse
                    for metric in validation.values()
                ]
            )
        ) if validation else float("nan")
        if len(training_traces) >= 2:
            cross_validation_losses = []
            for held_out_index, held_out in enumerate(training_traces):
                fold_training = tuple(
                    trace
                    for index, trace in enumerate(training_traces)
                    if index != held_out_index
                )
                fold_model = fit_abstract_phase_model(
                    fold_training,
                    config,
                    slow_tau_ms=slow_tau_ms,
                )
                cross_validation_losses.append(
                    trace_field_metrics(
                        fold_model,
                        held_out,
                    ).acceleration_nrmse
                )
            cross_validation_loss = float(
                np.mean(cross_validation_losses)
            )
        else:
            cross_validation_loss = train_loss
        score = cross_validation_loss + (
            config.slow_complexity_penalty
            if model.has_slow_state
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "slow_tau_ms": model.slow_tau_ms,
                "parameter_count": model.parameter_count,
                "training_acceleration_nrmse": train_loss,
                "validation_acceleration_nrmse": validation_loss,
                "selection_cross_validation_nrmse": (
                    cross_validation_loss
                ),
                "selection_score": score,
            }
        )
        if model is two_state:
            model_training_metrics = training
            model_validation_metrics = validation
        if score < best_score:
            best_score = score
            best_model = model
    selected_training = {
        trace.name: trace_field_metrics(best_model, trace)
        for trace in training_traces
    }
    selected_validation = {
        trace.name: trace_field_metrics(best_model, trace)
        for trace in validation_traces
    }
    return AbstractLadderResult(
        two_state=two_state,
        slow_state_candidates=slow_models,
        selected=best_model,
        training_metrics={
            **{
                f"two_state__{name}": metric
                for name, metric in model_training_metrics.items()
            },
            **{
                f"selected__{name}": metric
                for name, metric in selected_training.items()
            },
        },
        validation_metrics={
            **{
                f"two_state__{name}": metric
                for name, metric in model_validation_metrics.items()
            },
            **{
                f"selected__{name}": metric
                for name, metric in selected_validation.items()
            },
        },
        candidate_table=tuple(rows),
    )
