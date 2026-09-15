"""Static membrane parameters varied alongside alpha/beta kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

STATIC_PARAMETER_NAMES = (
    "param__static__log_area_scale",
    "param__static__log_capacitance_scale",
    "param__static__log_gna_scale",
    "param__static__log_gk_scale",
    "param__static__log_gleak_scale",
    "param__static__ena_shift_mv",
    "param__static__ek_shift_mv",
    "param__static__eleak_shift_mv",
    "param__static__q10_m",
    "param__static__q10_h",
    "param__static__q10_n",
)


@dataclass(frozen=True)
class BiophysicalParameters:
    """Physical membrane parameters in classic HH density units."""

    membrane_area_um2: float = 10_000.0
    capacitance_uf_cm2: float = 1.0
    gna_ms_cm2: float = 120.0
    gk_ms_cm2: float = 36.0
    gleak_ms_cm2: float = 0.3
    ena_mv: float = 50.0
    ek_mv: float = -77.0
    eleak_mv: float = -54.387
    q10_m: float = 3.0
    q10_h: float = 3.0
    q10_n: float = 3.0
    reference_temperature_c: float = 6.3

    @property
    def area_cm2(self) -> float:
        return self.membrane_area_um2 * 1e-8

    @property
    def total_capacitance_pf(self) -> float:
        return self.capacitance_uf_cm2 * self.area_cm2 * 1e6

    def current_density_to_pa(self, current_ua_cm2: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(current_ua_cm2, dtype=float) * self.area_cm2 * 1e6

    def current_pa_to_density(self, current_pa: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(current_pa, dtype=float) / (self.area_cm2 * 1e6)

    def temperature_factors(self, temperature_c: float) -> tuple[float, float, float]:
        exponent = (temperature_c - self.reference_temperature_c) / 10.0
        return self.q10_m**exponent, self.q10_h**exponent, self.q10_n**exponent

    def to_physical_mapping(self) -> dict[str, float]:
        return {
            "physical__membrane_area_um2": self.membrane_area_um2,
            "physical__total_capacitance_pf": self.total_capacitance_pf,
            "physical__capacitance_uf_cm2": self.capacitance_uf_cm2,
            "physical__gna_ms_cm2": self.gna_ms_cm2,
            "physical__gk_ms_cm2": self.gk_ms_cm2,
            "physical__gleak_ms_cm2": self.gleak_ms_cm2,
            "physical__ena_mv": self.ena_mv,
            "physical__ek_mv": self.ek_mv,
            "physical__eleak_mv": self.eleak_mv,
            "physical__q10_m": self.q10_m,
            "physical__q10_h": self.q10_h,
            "physical__q10_n": self.q10_n,
        }


@dataclass(frozen=True)
class StaticParameterTransforms:
    """Sampling coordinates around the canonical static HH parameters."""

    log_area_scale: float = 0.0
    log_capacitance_scale: float = 0.0
    log_gna_scale: float = 0.0
    log_gk_scale: float = 0.0
    log_gleak_scale: float = 0.0
    ena_shift_mv: float = 0.0
    ek_shift_mv: float = 0.0
    eleak_shift_mv: float = 0.0
    q10_m: float = 3.0
    q10_h: float = 3.0
    q10_n: float = 3.0

    @classmethod
    def canonical(cls) -> "StaticParameterTransforms":
        return cls()

    @classmethod
    def from_vector(cls, values: ArrayLike) -> "StaticParameterTransforms":
        vector = np.asarray(values, dtype=float)
        if vector.shape != (len(STATIC_PARAMETER_NAMES),):
            raise ValueError(
                f"Expected {len(STATIC_PARAMETER_NAMES)} static parameters, got {vector.shape}"
            )
        return cls(*vector)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "StaticParameterTransforms":
        return cls.from_vector([values[name] for name in STATIC_PARAMETER_NAMES])

    def to_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            (
                self.log_area_scale,
                self.log_capacitance_scale,
                self.log_gna_scale,
                self.log_gk_scale,
                self.log_gleak_scale,
                self.ena_shift_mv,
                self.ek_shift_mv,
                self.eleak_shift_mv,
                self.q10_m,
                self.q10_h,
                self.q10_n,
            ),
            dtype=float,
        )

    def to_mapping(self) -> dict[str, float]:
        return dict(zip(STATIC_PARAMETER_NAMES, self.to_vector()))

    def to_biophysics(self) -> BiophysicalParameters:
        canonical = BiophysicalParameters()
        return BiophysicalParameters(
            membrane_area_um2=canonical.membrane_area_um2 * np.exp(self.log_area_scale),
            capacitance_uf_cm2=(
                canonical.capacitance_uf_cm2 * np.exp(self.log_capacitance_scale)
            ),
            gna_ms_cm2=canonical.gna_ms_cm2 * np.exp(self.log_gna_scale),
            gk_ms_cm2=canonical.gk_ms_cm2 * np.exp(self.log_gk_scale),
            gleak_ms_cm2=canonical.gleak_ms_cm2 * np.exp(self.log_gleak_scale),
            ena_mv=canonical.ena_mv + self.ena_shift_mv,
            ek_mv=canonical.ek_mv + self.ek_shift_mv,
            eleak_mv=canonical.eleak_mv + self.eleak_shift_mv,
            q10_m=self.q10_m,
            q10_h=self.q10_h,
            q10_n=self.q10_n,
        )


def default_static_parameter_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Initial biologically broad, but still bounded, static-parameter prior."""
    lower = np.asarray(
        (
            np.log(0.2),
            np.log(0.6),
            np.log(0.25),
            np.log(0.20),
            np.log(0.10),
            -10.0,
            -20.0,
            -20.0,
            1.5,
            1.5,
            1.5,
        )
    )
    upper = np.asarray(
        (
            np.log(2.0),
            np.log(1.5),
            np.log(2.5),
            np.log(2.5),
            np.log(3.0),
            15.0,
            10.0,
            15.0,
            4.5,
            4.5,
            4.5,
        )
    )
    return lower, upper
