"""Parameterized alpha/beta rate curves for the classic HH gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

RATE_NAMES = (
    "alpha_m",
    "beta_m",
    "alpha_h",
    "beta_h",
    "alpha_n",
    "beta_n",
)
PARAMETER_NAMES = tuple(
    f"param__{rate}__{field}"
    for rate in RATE_NAMES
    for field in ("log_rate_scale", "voltage_shift_mv", "log_slope_scale")
)


def _safe_exp(value: ArrayLike) -> NDArray[np.float64]:
    return np.exp(np.clip(np.asarray(value, dtype=float), -80.0, 80.0))


def _vtrap(x: ArrayLike, scale: float) -> NDArray[np.float64]:
    """Stable x / (exp(x / scale) - 1), including the removable singularity."""
    x_array = np.asarray(x, dtype=float)
    ratio = x_array / scale
    denominator = np.expm1(np.clip(ratio, -80.0, 80.0))
    near_zero = np.abs(ratio) < 1e-7
    safe_denominator = np.where(near_zero, 1.0, denominator)
    quotient = x_array / safe_denominator
    return np.where(near_zero, scale * (1.0 - ratio / 2.0), quotient)


def _alpha_m(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 0.1 * _vtrap(-(voltage + 40.0), 10.0)


def _beta_m(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 4.0 * _safe_exp(-(voltage + 65.0) / 18.0)


def _alpha_h(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 0.07 * _safe_exp(-(voltage + 65.0) / 20.0)


def _beta_h(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 1.0 / (1.0 + _safe_exp(-(voltage + 35.0) / 10.0))


def _alpha_n(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 0.01 * _vtrap(-(voltage + 55.0), 10.0)


def _beta_n(voltage_mv: ArrayLike) -> NDArray[np.float64]:
    voltage = np.asarray(voltage_mv, dtype=float)
    return 0.125 * _safe_exp(-(voltage + 65.0) / 80.0)


_BASE_RATES: Mapping[str, Callable[[ArrayLike], NDArray[np.float64]]] = {
    "alpha_m": _alpha_m,
    "beta_m": _beta_m,
    "alpha_h": _alpha_h,
    "beta_h": _beta_h,
    "alpha_n": _alpha_n,
    "beta_n": _beta_n,
}

_RATE_PIVOTS_MV = {
    "alpha_m": -40.0,
    "beta_m": -65.0,
    "alpha_h": -65.0,
    "beta_h": -35.0,
    "alpha_n": -55.0,
    "beta_n": -65.0,
}


def _vtrap_scalar(x: float, scale: float) -> float:
    ratio = x / scale
    if abs(ratio) < 1e-7:
        return scale * (1.0 - ratio / 2.0)
    return x / math.expm1(max(-80.0, min(80.0, ratio)))


def _base_rate_scalar(rate_name: str, voltage_mv: float) -> float:
    if rate_name == "alpha_m":
        return 0.1 * _vtrap_scalar(-(voltage_mv + 40.0), 10.0)
    if rate_name == "beta_m":
        return 4.0 * math.exp(max(-80.0, min(80.0, -(voltage_mv + 65.0) / 18.0)))
    if rate_name == "alpha_h":
        return 0.07 * math.exp(max(-80.0, min(80.0, -(voltage_mv + 65.0) / 20.0)))
    if rate_name == "beta_h":
        exponent = max(-80.0, min(80.0, -(voltage_mv + 35.0) / 10.0))
        return 1.0 / (1.0 + math.exp(exponent))
    if rate_name == "alpha_n":
        return 0.01 * _vtrap_scalar(-(voltage_mv + 55.0), 10.0)
    if rate_name == "beta_n":
        return 0.125 * math.exp(max(-80.0, min(80.0, -(voltage_mv + 65.0) / 80.0)))
    raise KeyError(f"Unknown HH rate: {rate_name}")


@dataclass(frozen=True)
class RateTransform:
    """Amplitude, horizontal shift, and voltage-axis stretch of one HH rate."""

    log_rate_scale: float = 0.0
    voltage_shift_mv: float = 0.0
    log_slope_scale: float = 0.0

    def apply(self, rate_name: str, voltage_mv: ArrayLike) -> NDArray[np.float64]:
        if rate_name not in _BASE_RATES:
            raise KeyError(f"Unknown HH rate: {rate_name}")

        voltage = np.asarray(voltage_mv, dtype=float)
        pivot = _RATE_PIVOTS_MV[rate_name]
        slope_scale = np.exp(self.log_slope_scale)
        transformed_voltage = pivot + (
            voltage - pivot - self.voltage_shift_mv
        ) / slope_scale
        return np.exp(self.log_rate_scale) * _BASE_RATES[rate_name](transformed_voltage)

    def apply_scalar(self, rate_name: str, voltage_mv: float) -> float:
        pivot = _RATE_PIVOTS_MV[rate_name]
        transformed_voltage = pivot + (
            voltage_mv - pivot - self.voltage_shift_mv
        ) / math.exp(self.log_slope_scale)
        return math.exp(self.log_rate_scale) * _base_rate_scalar(
            rate_name, transformed_voltage
        )


@dataclass(frozen=True)
class KineticParameters:
    """Transforms for all six rates of the m, h, and n gates."""

    alpha_m: RateTransform = RateTransform()
    beta_m: RateTransform = RateTransform()
    alpha_h: RateTransform = RateTransform()
    beta_h: RateTransform = RateTransform()
    alpha_n: RateTransform = RateTransform()
    beta_n: RateTransform = RateTransform()

    @classmethod
    def canonical(cls) -> "KineticParameters":
        return cls()

    @classmethod
    def from_vector(cls, values: ArrayLike) -> "KineticParameters":
        vector = np.asarray(values, dtype=float)
        if vector.shape != (len(PARAMETER_NAMES),):
            raise ValueError(
                f"Expected {len(PARAMETER_NAMES)} kinetic parameters, got {vector.shape}"
            )
        transforms = {}
        for index, rate_name in enumerate(RATE_NAMES):
            offset = 3 * index
            transforms[rate_name] = RateTransform(*vector[offset : offset + 3])
        return cls(**transforms)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "KineticParameters":
        return cls.from_vector([values[name] for name in PARAMETER_NAMES])

    def to_vector(self) -> NDArray[np.float64]:
        values = []
        for rate_name in RATE_NAMES:
            transform = getattr(self, rate_name)
            values.extend(
                (
                    transform.log_rate_scale,
                    transform.voltage_shift_mv,
                    transform.log_slope_scale,
                )
            )
        return np.asarray(values, dtype=float)

    def to_mapping(self) -> dict[str, float]:
        return dict(zip(PARAMETER_NAMES, self.to_vector()))

    def rates(self, voltage_mv: ArrayLike) -> dict[str, NDArray[np.float64]]:
        return {
            rate_name: getattr(self, rate_name).apply(rate_name, voltage_mv)
            for rate_name in RATE_NAMES
        }

    def rates_scalar(self, voltage_mv: float) -> tuple[float, float, float, float, float, float]:
        return tuple(
            getattr(self, rate_name).apply_scalar(rate_name, voltage_mv)
            for rate_name in RATE_NAMES
        )

    def steady_state(self, voltage_mv: float) -> tuple[float, float, float]:
        rates = self.rates(voltage_mv)
        m = rates["alpha_m"] / (rates["alpha_m"] + rates["beta_m"])
        h = rates["alpha_h"] / (rates["alpha_h"] + rates["beta_h"])
        n = rates["alpha_n"] / (rates["alpha_n"] + rates["beta_n"])
        return float(m), float(h), float(n)


def default_parameter_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return conservative lower/upper bounds for the 18 transformed parameters."""
    lower = []
    upper = []
    for _ in RATE_NAMES:
        lower.extend((np.log(0.5), -10.0, np.log(0.70)))
        upper.extend((np.log(2.0), 10.0, np.log(1.40)))
    return np.asarray(lower), np.asarray(upper)
