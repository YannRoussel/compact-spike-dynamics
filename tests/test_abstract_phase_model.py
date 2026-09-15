from __future__ import annotations

import unittest

import numpy as np

from inverse_ephys_alpha_beta.abstract_phase_model import (
    AbstractObservedTrace,
    AbstractPhaseConfig,
    fit_abstract_model_ladder,
    fit_abstract_phase_model,
    memory_from_voltage,
    trace_field_metrics,
)


def _periodic_trace(
    name: str,
    input_amplitude: float,
    slow_tau_ms: float | None = None,
) -> AbstractObservedTrace:
    time_ms = np.linspace(0.0, 400.0, 8001)
    omega = 2.0 * np.pi / 40.0
    voltage_mv = -45.0 + 42.0 * np.sin(omega * time_ms)
    velocity_mv_ms = 42.0 * omega * np.cos(omega * time_ms)
    input_value = np.zeros_like(time_ms)
    input_value[(time_ms >= 40.0) & (time_ms < 360.0)] = input_amplitude
    fit_mask = np.ones_like(time_ms, dtype=bool)
    fit_mask[:10] = False
    fit_mask[-10:] = False
    provisional = AbstractObservedTrace(
        name=name,
        role="training",
        time_ms=time_ms,
        voltage_mv=voltage_mv,
        velocity_mv_ms=velocity_mv_ms,
        acceleration_mv_ms2=np.zeros_like(time_ms),
        input_value=input_value,
        fit_mask=fit_mask,
        stimulus_start_ms=40.0,
        stimulus_end_ms=360.0,
    )
    centered_voltage = voltage_mv + 45.0
    acceleration = (
        -0.025 * centered_voltage
        + 0.055
        * (1.0 - (centered_voltage / 34.0) ** 2)
        * velocity_mv_ms
        + (0.008 + 0.00004 * centered_voltage) * input_value
    )
    if slow_tau_ms is not None:
        memory = memory_from_voltage(
            provisional,
            tau_ms=slow_tau_ms,
            vhalf_mv=-20.0,
            slope_mv=5.0,
        )
        acceleration += (
            -0.012
            * (1.0 + 0.15 * centered_voltage / 42.0)
            * memory
        )
    return AbstractObservedTrace(
        name=name,
        role="training",
        time_ms=time_ms,
        voltage_mv=voltage_mv,
        velocity_mv_ms=velocity_mv_ms,
        acceleration_mv_ms2=acceleration,
        input_value=input_value,
        fit_mask=fit_mask,
        stimulus_start_ms=40.0,
        stimulus_end_ms=360.0,
    )


class AbstractPhaseModelTests(unittest.TestCase):
    def setUp(self):
        self.config = AbstractPhaseConfig(
            spline_basis_count=10,
            smoothness_penalty=1e-8,
            ridge_penalty=1e-10,
            rest_anchor_weight=0.0,
            rest_stability_weight=0.0,
            repetitive_spiking_instability_weight=0.0,
            cycle_contraction_weight=0.0,
            maximum_samples_per_trace=3000,
            slow_tau_candidates_ms=(50.0, 100.0, 200.0),
            slow_complexity_penalty=0.002,
        )

    def test_two_state_field_recovers_known_acceleration(self):
        training = [
            _periodic_trace("train_low", 40.0),
            _periodic_trace("train_high", 100.0),
        ]
        validation = _periodic_trace("validation", 70.0)
        model = fit_abstract_phase_model(training, self.config)
        metrics = trace_field_metrics(model, validation)
        self.assertLess(metrics.acceleration_nrmse, 0.03)
        self.assertGreater(metrics.acceleration_correlation, 0.99)

    def test_slow_state_is_selected_for_memory_dependent_field(self):
        training = [
            _periodic_trace("train_low", 40.0, slow_tau_ms=100.0),
            _periodic_trace("train_high", 100.0, slow_tau_ms=100.0),
            _periodic_trace("train_mid", 70.0, slow_tau_ms=100.0),
        ]
        validation = [
            _periodic_trace("validation", 85.0, slow_tau_ms=100.0)
        ]
        result = fit_abstract_model_ladder(
            training,
            validation,
            self.config,
        )
        self.assertTrue(result.selected.has_slow_state)
        self.assertEqual(result.selected.slow_tau_ms, 100.0)
        two_state_loss = result.validation_metrics[
            "two_state__validation"
        ].acceleration_nrmse
        selected_loss = result.validation_metrics[
            "selected__validation"
        ].acceleration_nrmse
        self.assertLess(selected_loss, 0.5 * two_state_loss)


if __name__ == "__main__":
    unittest.main()
