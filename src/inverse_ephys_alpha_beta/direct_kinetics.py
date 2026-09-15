"""Direct steady-state and time-constant kinetics for the m, h, and n gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .activation_kinetics import (
    _hermite_array,
    _hermite_scalar,
    _pchip_slopes,
)


DIRECT_VOLTAGE_KNOTS_MV = (-100.0, -70.0, -50.0, -30.0, 0.0, 60.0)
DIRECT_GATES = ("m", "h", "n")
_GATE_DIRECTIONS = {"m": 1.0, "h": -1.0, "n": 1.0}
_KNOT_COUNT = len(DIRECT_VOLTAGE_KNOTS_MV)


def _gate_parameter_names(gate: str) -> tuple[str, ...]:
    return (
        f"param__direct_{gate}__start_logit",
        *(
            f"param__direct_{gate}__log_logit_increment_{index}"
            for index in range(1, _KNOT_COUNT)
        ),
        *(
            f"param__direct_{gate}__log_tau_knot_{index}"
            for index in range(_KNOT_COUNT)
        ),
    )


DIRECT_KINETIC_PARAMETER_NAMES = tuple(
    name
    for gate in DIRECT_GATES
    for name in _gate_parameter_names(gate)
)
DIRECT_GATE_PARAMETER_COUNT = 2 * _KNOT_COUNT
DIRECT_SLOW_K_PARAMETER_NAMES = (
    "param__slow_k__start_logit",
    *(
        f"param__slow_k__log_logit_increment_{index}"
        for index in range(1, _KNOT_COUNT)
    ),
    *(
        f"param__slow_k__log_tau_knot_{index}"
        for index in range(_KNOT_COUNT)
    ),
)


def direct_kinetic_parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return broad but numerically resolvable bounds at 6.3 degrees C."""
    lower: list[float] = []
    upper: list[float] = []
    for gate in DIRECT_GATES:
        if gate == "h":
            lower.append(0.2)
            upper.append(16.0)
        else:
            lower.append(-16.0)
            upper.append(-0.2)
        lower.extend([-3.0] * (_KNOT_COUNT - 1))
        upper.extend([3.0] * (_KNOT_COUNT - 1))
        tau_upper = 20.0 if gate == "m" else 100.0
        lower.extend([math.log(0.05)] * _KNOT_COUNT)
        upper.extend([math.log(tau_upper)] * _KNOT_COUNT)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def direct_slow_k_parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Bounds for a slow monotone potassium activation gate at 6.3 C."""
    lower = np.asarray(
        (
            -12.0,
            *([-3.0] * (_KNOT_COUNT - 1)),
            *([math.log(20.0)] * _KNOT_COUNT),
        ),
        dtype=float,
    )
    upper = np.asarray(
        (
            -1.0,
            *([3.0] * (_KNOT_COUNT - 1)),
            *([math.log(2_000.0)] * _KNOT_COUNT),
        ),
        dtype=float,
    )
    return lower, upper


def direct_slow_k_initial() -> np.ndarray:
    """Return an M-like activation curve with a 300 ms reference tau."""
    voltage = np.asarray(DIRECT_VOLTAGE_KNOTS_MV, dtype=float)
    logits = (voltage + 35.0) / 10.0
    increments = np.maximum(np.diff(logits), 1e-6)
    values = np.asarray(
        (
            logits[0],
            *np.log(increments),
            *([math.log(300.0)] * _KNOT_COUNT),
        ),
        dtype=float,
    )
    lower, upper = direct_slow_k_parameter_bounds()
    return np.clip(values, lower + 1e-9, upper - 1e-9)


def direct_kinetic_initial(kinetics: object) -> np.ndarray:
    """Sample any alpha/beta-compatible kinetics into the direct coordinates."""
    voltage = np.asarray(DIRECT_VOLTAGE_KNOTS_MV, dtype=float)
    rates = kinetics.rates(voltage)
    values: list[float] = []
    for gate in DIRECT_GATES:
        alpha = np.asarray(rates[f"alpha_{gate}"], dtype=float)
        beta = np.asarray(rates[f"beta_{gate}"], dtype=float)
        total = np.maximum(alpha + beta, 1e-12)
        steady = np.clip(alpha / total, 1e-8, 1.0 - 1e-8)
        logits = np.log(steady / (1.0 - steady))
        direction = _GATE_DIRECTIONS[gate]
        oriented = direction * logits
        oriented = np.maximum.accumulate(
            oriented + np.arange(_KNOT_COUNT) * 1e-8
        )
        increments = np.maximum(np.diff(oriented), 1e-6)
        values.extend(
            (
                float(logits[0]),
                *np.log(increments),
                *np.log(1.0 / total),
            )
        )
    lower, upper = direct_kinetic_parameter_bounds()
    return np.clip(
        np.asarray(values, dtype=float),
        lower + 1e-9,
        upper - 1e-9,
    )


@dataclass(frozen=True)
class DirectGateCurves:
    direction: float
    logits: tuple[float, ...]
    logit_slopes: tuple[float, ...]
    log_tau_values: tuple[float, ...]
    log_tau_slopes: tuple[float, ...]


@dataclass(frozen=True)
class DirectKinetics:
    """Monotone steady states and positive smooth time constants."""

    voltage_knots_mv: tuple[float, ...]
    gates: Mapping[str, DirectGateCurves]
    parameters: tuple[float, ...]

    @classmethod
    def from_parameters(cls, parameters: ArrayLike) -> "DirectKinetics":
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(DIRECT_KINETIC_PARAMETER_NAMES),):
            raise ValueError(
                "Direct kinetic vector has the wrong shape: "
                f"{values.shape}"
            )
        knots = np.asarray(DIRECT_VOLTAGE_KNOTS_MV, dtype=float)
        gates: dict[str, DirectGateCurves] = {}
        for gate_index, gate in enumerate(DIRECT_GATES):
            start = gate_index * DIRECT_GATE_PARAMETER_COUNT
            gate_values = values[
                start : start + DIRECT_GATE_PARAMETER_COUNT
            ]
            direction = _GATE_DIRECTIONS[gate]
            logits = np.empty(_KNOT_COUNT, dtype=float)
            logits[0] = gate_values[0]
            increments = np.exp(gate_values[1:_KNOT_COUNT])
            logits[1:] = logits[0] + direction * np.cumsum(increments)
            log_tau = gate_values[_KNOT_COUNT:]
            gates[gate] = DirectGateCurves(
                direction=direction,
                logits=tuple(logits),
                logit_slopes=tuple(_pchip_slopes(knots, logits)),
                log_tau_values=tuple(log_tau),
                log_tau_slopes=tuple(_pchip_slopes(knots, log_tau)),
            )
        return cls(
            voltage_knots_mv=tuple(knots),
            gates=gates,
            parameters=tuple(values),
        )

    def gate_curves(
        self,
        gate: str,
        voltage_mv: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if gate not in self.gates:
            raise KeyError(f"Unknown HH gate: {gate}")
        curves = self.gates[gate]
        logits = _hermite_array(
            voltage_mv,
            self.voltage_knots_mv,
            curves.logits,
            curves.logit_slopes,
        )
        steady = 1.0 / (1.0 + np.exp(np.clip(-logits, -80.0, 80.0)))
        log_tau = _hermite_array(
            voltage_mv,
            self.voltage_knots_mv,
            curves.log_tau_values,
            curves.log_tau_slopes,
        )
        tau = np.exp(np.clip(log_tau, -20.0, 20.0))
        return steady, tau

    def _gate_curves_scalar(
        self,
        gate: str,
        voltage_mv: float,
    ) -> tuple[float, float]:
        curves = self.gates[gate]
        logit = _hermite_scalar(
            voltage_mv,
            self.voltage_knots_mv,
            curves.logits,
            curves.logit_slopes,
        )
        steady = 1.0 / (
            1.0 + math.exp(max(-80.0, min(80.0, -logit)))
        )
        log_tau = _hermite_scalar(
            voltage_mv,
            self.voltage_knots_mv,
            curves.log_tau_values,
            curves.log_tau_slopes,
        )
        return steady, math.exp(max(-20.0, min(20.0, log_tau)))

    def rates(
        self,
        voltage_mv: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        rates: dict[str, NDArray[np.float64]] = {}
        for gate in DIRECT_GATES:
            steady, tau = self.gate_curves(gate, voltage_mv)
            rates[f"alpha_{gate}"] = steady / tau
            rates[f"beta_{gate}"] = (1.0 - steady) / tau
        return rates

    def rates_scalar(
        self,
        voltage_mv: float,
    ) -> tuple[float, float, float, float, float, float]:
        values: list[float] = []
        for gate in DIRECT_GATES:
            steady, tau = self._gate_curves_scalar(gate, voltage_mv)
            values.extend((steady / tau, (1.0 - steady) / tau))
        return tuple(values)

    def steady_state(
        self,
        voltage_mv: float,
    ) -> tuple[float, float, float]:
        return tuple(
            self._gate_curves_scalar(gate, voltage_mv)[0]
            for gate in DIRECT_GATES
        )

    def to_mapping(self) -> dict[str, float]:
        return dict(zip(DIRECT_KINETIC_PARAMETER_NAMES, self.parameters))

    def smoothness_penalty(self) -> float:
        """Penalize unnecessary knot-to-knot bending in direct coordinates."""
        penalties: list[float] = []
        for gate in DIRECT_GATES:
            curves = self.gates[gate]
            penalties.extend(
                (
                    float(np.mean(np.diff(curves.logits, n=2) ** 2)),
                    float(
                        np.mean(
                            np.diff(curves.log_tau_values, n=2) ** 2
                        )
                    ),
                )
            )
        return float(np.mean(penalties))


@dataclass(frozen=True)
class DirectSlowKinetics:
    """One monotone slow activation state with a positive flexible tau."""

    logits: tuple[float, ...]
    logit_slopes: tuple[float, ...]
    log_tau_values: tuple[float, ...]
    log_tau_slopes: tuple[float, ...]
    parameters: tuple[float, ...]
    q10: float = 2.3
    reference_temperature_c: float = 6.3

    @classmethod
    def from_parameters(cls, parameters: ArrayLike) -> "DirectSlowKinetics":
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(DIRECT_SLOW_K_PARAMETER_NAMES),):
            raise ValueError(
                "Direct slow-K vector has the wrong shape: "
                f"{values.shape}"
            )
        knots = np.asarray(DIRECT_VOLTAGE_KNOTS_MV, dtype=float)
        logits = np.empty(_KNOT_COUNT, dtype=float)
        logits[0] = values[0]
        logits[1:] = logits[0] + np.cumsum(
            np.exp(values[1:_KNOT_COUNT])
        )
        log_tau = values[_KNOT_COUNT:]
        return cls(
            logits=tuple(logits),
            logit_slopes=tuple(_pchip_slopes(knots, logits)),
            log_tau_values=tuple(log_tau),
            log_tau_slopes=tuple(_pchip_slopes(knots, log_tau)),
            parameters=tuple(values),
        )

    def gate_curves(
        self,
        voltage_mv: ArrayLike,
        temperature_c: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        logits = _hermite_array(
            voltage_mv,
            DIRECT_VOLTAGE_KNOTS_MV,
            self.logits,
            self.logit_slopes,
        )
        steady = 1.0 / (1.0 + np.exp(np.clip(-logits, -80.0, 80.0)))
        log_tau = _hermite_array(
            voltage_mv,
            DIRECT_VOLTAGE_KNOTS_MV,
            self.log_tau_values,
            self.log_tau_slopes,
        )
        tau = np.exp(np.clip(log_tau, -20.0, 20.0))
        if temperature_c is not None:
            tau = tau / self.temperature_factor(temperature_c)
        return steady, tau

    def gate_curves_scalar(
        self,
        voltage_mv: float,
        temperature_c: float | None = None,
    ) -> tuple[float, float]:
        logit = _hermite_scalar(
            voltage_mv,
            DIRECT_VOLTAGE_KNOTS_MV,
            self.logits,
            self.logit_slopes,
        )
        steady = 1.0 / (
            1.0 + math.exp(max(-80.0, min(80.0, -logit)))
        )
        log_tau = _hermite_scalar(
            voltage_mv,
            DIRECT_VOLTAGE_KNOTS_MV,
            self.log_tau_values,
            self.log_tau_slopes,
        )
        tau = math.exp(max(-20.0, min(20.0, log_tau)))
        if temperature_c is not None:
            tau /= self.temperature_factor(temperature_c)
        return steady, tau

    def temperature_factor(self, temperature_c: float) -> float:
        exponent = (
            temperature_c - self.reference_temperature_c
        ) / 10.0
        return float(self.q10**exponent)

    def smoothness_penalty(self) -> float:
        return float(
            0.5 * np.mean(np.diff(self.logits, n=2) ** 2)
            + 0.5
            * np.mean(np.diff(self.log_tau_values, n=2) ** 2)
        )

    def to_mapping(self) -> dict[str, float]:
        return dict(zip(DIRECT_SLOW_K_PARAMETER_NAMES, self.parameters))
