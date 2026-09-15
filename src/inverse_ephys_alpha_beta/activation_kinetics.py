"""Alternative sodium-activation parameterizations for nested phase fits."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kinetics import KineticParameters


FLEXIBLE_M_VOLTAGE_KNOTS_MV = (-90.0, -65.0, -45.0, -25.0, 20.0, 60.0)
FLEXIBLE_M_PARAMETER_NAMES = (
    "param__flex_m__start_logit",
    *(
        f"param__flex_m__log_logit_increment_{index}"
        for index in range(1, len(FLEXIBLE_M_VOLTAGE_KNOTS_MV))
    ),
    *(
        f"param__flex_m__log_tau_knot_{index}"
        for index in range(len(FLEXIBLE_M_VOLTAGE_KNOTS_MV))
    ),
)


def _endpoint_slope(
    first_h: float,
    second_h: float,
    first_delta: float,
    second_delta: float,
) -> float:
    slope = (
        (2.0 * first_h + second_h) * first_delta
        - first_h * second_delta
    ) / (first_h + second_h)
    if np.sign(slope) != np.sign(first_delta):
        return 0.0
    if (
        np.sign(first_delta) != np.sign(second_delta)
        and abs(slope) > 3.0 * abs(first_delta)
    ):
        return 3.0 * first_delta
    return float(slope)


def _pchip_slopes(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Return shape-preserving cubic Hermite slopes at fixed knots."""
    h = np.diff(x_values)
    delta = np.diff(y_values) / h
    slopes = np.zeros_like(y_values)
    for index in range(1, len(y_values) - 1):
        left = delta[index - 1]
        right = delta[index]
        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            continue
        weight_left = 2.0 * h[index] + h[index - 1]
        weight_right = h[index] + 2.0 * h[index - 1]
        slopes[index] = (
            weight_left + weight_right
        ) / (weight_left / left + weight_right / right)
    slopes[0] = _endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _endpoint_slope(
        h[-1],
        h[-2],
        delta[-1],
        delta[-2],
    )
    return slopes


def _hermite_scalar(
    x: float,
    knots: tuple[float, ...],
    values: tuple[float, ...],
    slopes: tuple[float, ...],
) -> float:
    clipped = min(knots[-1], max(knots[0], float(x)))
    index = min(len(knots) - 2, max(0, bisect_right(knots, clipped) - 1))
    width = knots[index + 1] - knots[index]
    position = (clipped - knots[index]) / width
    p2 = position * position
    p3 = p2 * position
    return float(
        (2.0 * p3 - 3.0 * p2 + 1.0) * values[index]
        + (p3 - 2.0 * p2 + position) * width * slopes[index]
        + (-2.0 * p3 + 3.0 * p2) * values[index + 1]
        + (p3 - p2) * width * slopes[index + 1]
    )


def _hermite_array(
    x: ArrayLike,
    knots: tuple[float, ...],
    values: tuple[float, ...],
    slopes: tuple[float, ...],
) -> NDArray[np.float64]:
    x_array = np.clip(np.asarray(x, dtype=float), knots[0], knots[-1])
    knot_array = np.asarray(knots)
    value_array = np.asarray(values)
    slope_array = np.asarray(slopes)
    indices = np.clip(
        np.searchsorted(knot_array, x_array, side="right") - 1,
        0,
        len(knots) - 2,
    )
    widths = knot_array[indices + 1] - knot_array[indices]
    positions = (x_array - knot_array[indices]) / widths
    p2 = positions**2
    p3 = positions**3
    return (
        (2.0 * p3 - 3.0 * p2 + 1.0) * value_array[indices]
        + (p3 - 2.0 * p2 + positions) * widths * slope_array[indices]
        + (-2.0 * p3 + 3.0 * p2) * value_array[indices + 1]
        + (p3 - p2) * widths * slope_array[indices + 1]
    )


def flexible_m_parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    knot_count = len(FLEXIBLE_M_VOLTAGE_KNOTS_MV)
    lower = [
        -14.0,
        *([-2.3] * (knot_count - 1)),
        *([math.log(0.003)] * knot_count),
    ]
    upper = [
        -1.0,
        *([2.5] * (knot_count - 1)),
        *([math.log(5.0)] * knot_count),
    ]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def flexible_m_initial(
    kinetics: KineticParameters,
) -> np.ndarray:
    knots = np.asarray(FLEXIBLE_M_VOLTAGE_KNOTS_MV)
    rates = kinetics.rates(knots)
    steady = rates["alpha_m"] / (rates["alpha_m"] + rates["beta_m"])
    logits = np.log(
        np.clip(steady, 1e-8, 1.0 - 1e-8)
        / np.clip(1.0 - steady, 1e-8, 1.0)
    )
    logits = np.maximum.accumulate(logits + np.arange(len(logits)) * 1e-8)
    increments = np.maximum(np.diff(logits), 1e-6)
    tau = 1.0 / (rates["alpha_m"] + rates["beta_m"])
    values = np.concatenate(
        (
            logits[:1],
            np.log(increments),
            np.log(np.maximum(tau, 1e-8)),
        )
    )
    lower, upper = flexible_m_parameter_bounds()
    return np.clip(values, lower + 1e-9, upper - 1e-9)


@dataclass(frozen=True)
class ShiftedActivationKinetics:
    """Apply one horizontal voltage shift to sodium activation only."""

    base: object
    activation_shift_mv: float

    def rates(self, voltage_mv: ArrayLike) -> dict[str, NDArray[np.float64]]:
        voltage = np.asarray(voltage_mv, dtype=float)
        rates = self.base.rates(voltage)
        shifted = self.base.rates(voltage - self.activation_shift_mv)
        rates["alpha_m"] = shifted["alpha_m"]
        rates["beta_m"] = shifted["beta_m"]
        return rates

    def rates_scalar(
        self,
        voltage_mv: float,
    ) -> tuple[float, float, float, float, float, float]:
        unshifted = self.base.rates_scalar(voltage_mv)
        shifted = self.base.rates_scalar(
            voltage_mv - self.activation_shift_mv
        )
        return (
            shifted[0],
            shifted[1],
            unshifted[2],
            unshifted[3],
            unshifted[4],
            unshifted[5],
        )

    def steady_state(self, voltage_mv: float) -> tuple[float, float, float]:
        rates = self.rates_scalar(voltage_mv)
        return (
            rates[0] / (rates[0] + rates[1]),
            rates[2] / (rates[2] + rates[3]),
            rates[4] / (rates[4] + rates[5]),
        )


@dataclass(frozen=True)
class FlexibleActivationKinetics:
    """Use monotone m_inf and positive tau_m PCHIP curves."""

    base: KineticParameters
    voltage_knots_mv: tuple[float, ...]
    activation_logits: tuple[float, ...]
    activation_slopes: tuple[float, ...]
    log_tau_values: tuple[float, ...]
    log_tau_slopes: tuple[float, ...]

    @classmethod
    def from_parameters(
        cls,
        base: KineticParameters,
        parameters: ArrayLike,
    ) -> "FlexibleActivationKinetics":
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(FLEXIBLE_M_PARAMETER_NAMES),):
            raise ValueError(
                "Flexible activation vector has the wrong shape: "
                f"{values.shape}"
            )
        knot_count = len(FLEXIBLE_M_VOLTAGE_KNOTS_MV)
        logits = np.empty(knot_count, dtype=float)
        logits[0] = values[0]
        logits[1:] = logits[0] + np.cumsum(np.exp(values[1:knot_count]))
        log_tau = values[knot_count:]
        knots = np.asarray(FLEXIBLE_M_VOLTAGE_KNOTS_MV, dtype=float)
        return cls(
            base=base,
            voltage_knots_mv=tuple(knots),
            activation_logits=tuple(logits),
            activation_slopes=tuple(_pchip_slopes(knots, logits)),
            log_tau_values=tuple(log_tau),
            log_tau_slopes=tuple(_pchip_slopes(knots, log_tau)),
        )

    def _activation_scalar(self, voltage_mv: float) -> tuple[float, float]:
        logit = _hermite_scalar(
            voltage_mv,
            self.voltage_knots_mv,
            self.activation_logits,
            self.activation_slopes,
        )
        m_inf = 1.0 / (1.0 + math.exp(max(-80.0, min(80.0, -logit))))
        log_tau = _hermite_scalar(
            voltage_mv,
            self.voltage_knots_mv,
            self.log_tau_values,
            self.log_tau_slopes,
        )
        return m_inf, math.exp(max(-20.0, min(20.0, log_tau)))

    def activation_curves(
        self,
        voltage_mv: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        logits = _hermite_array(
            voltage_mv,
            self.voltage_knots_mv,
            self.activation_logits,
            self.activation_slopes,
        )
        m_inf = 1.0 / (1.0 + np.exp(np.clip(-logits, -80.0, 80.0)))
        log_tau = _hermite_array(
            voltage_mv,
            self.voltage_knots_mv,
            self.log_tau_values,
            self.log_tau_slopes,
        )
        return m_inf, np.exp(np.clip(log_tau, -20.0, 20.0))

    def rates(self, voltage_mv: ArrayLike) -> dict[str, NDArray[np.float64]]:
        rates = self.base.rates(voltage_mv)
        steady, tau = self.activation_curves(voltage_mv)
        rates["alpha_m"] = steady / tau
        rates["beta_m"] = (1.0 - steady) / tau
        return rates

    def rates_scalar(
        self,
        voltage_mv: float,
    ) -> tuple[float, float, float, float, float, float]:
        base_rates = self.base.rates_scalar(voltage_mv)
        steady, tau = self._activation_scalar(voltage_mv)
        return (
            steady / tau,
            (1.0 - steady) / tau,
            base_rates[2],
            base_rates[3],
            base_rates[4],
            base_rates[5],
        )

    def steady_state(self, voltage_mv: float) -> tuple[float, float, float]:
        rates = self.rates_scalar(voltage_mv)
        return (
            rates[0] / (rates[0] + rates[1]),
            rates[2] / (rates[2] + rates[3]),
            rates[4] / (rates[4] + rates[5]),
        )

    def to_mapping(self) -> Mapping[str, tuple[float, ...]]:
        return {
            "voltage_knots_mv": self.voltage_knots_mv,
            "activation_logits": self.activation_logits,
            "log_tau_values": self.log_tau_values,
        }
