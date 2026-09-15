import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
)
from inverse_ephys_alpha_beta.direct_kinetics import (
    DirectSlowKinetics,
    direct_slow_k_initial,
    direct_slow_k_parameter_bounds,
)
from inverse_ephys_alpha_beta.direct_phase_fit import (
    DirectPhaseFitConfig,
    _baseline_centers,
    decode_direct_model,
)
from inverse_ephys_alpha_beta.direct_slow_k import (
    _mean_spike_count_loss,
    _voltage_rhs,
    decode_direct_slow_k_model,
    direct_slow_k_fit_initial,
    direct_slow_k_initial_state,
    simulate_direct_slow_k,
)
from inverse_ephys_alpha_beta.hh_model import SimulationConfig, Stimulus
from inverse_ephys_alpha_beta.model_ladder import AIS_PARAMETER_NAMES


class DirectSlowKTests(unittest.TestCase):
    def setUp(self):
        passive = PassiveAnchor(
            resting_voltage_mv=-65.0,
            input_resistance_mohm=10.0,
            membrane_tau_ms=1.0,
            total_capacitance_pf=100.0,
            membrane_area_um2=10_000.0,
            capacitance_uf_cm2=1.0,
            gleak_ms_cm2=1.0,
            passive_current_pa=-40.0,
            passive_duration_ms=600.0,
            passive_sweep_number=1,
        )
        self.target = CellOptimizationTarget(
            dataset="test",
            cell_id="cell",
            nwb_path="cell.nwb",
            sampled_rheobase_pa=100.0,
            temperature_c=22.0,
            passive=passive,
            training_protocols=(),
            validation_protocols=(),
        )
        baseline = pd.DataFrame(
            [
                {
                    name: 0.0
                    for name in (
                        *CELL_PARAMETER_NAMES,
                        *AIS_PARAMETER_NAMES,
                    )
                }
            ]
        )
        kinetic, static = _baseline_centers(
            baseline,
            DirectPhaseFitConfig(),
        )
        self.base = decode_direct_model(
            kinetic,
            static,
            self.target,
        )

    def test_slow_gate_is_monotone_positive_and_temperature_scaled(self):
        parameters = direct_slow_k_initial()
        lower, upper = direct_slow_k_parameter_bounds()
        self.assertTrue(np.all(parameters >= lower))
        self.assertTrue(np.all(parameters <= upper))
        kinetics = DirectSlowKinetics.from_parameters(parameters)
        voltage = np.linspace(-100.0, 60.0, 321)
        steady, tau_rt = kinetics.gate_curves(voltage, 22.0)
        _, tau_warm = kinetics.gate_curves(voltage, 34.0)
        self.assertTrue(np.all(np.diff(steady) >= -1e-12))
        self.assertTrue(np.all(tau_rt > 0.0))
        self.assertTrue(np.all(tau_warm < tau_rt))

    def test_slow_k_reanchoring_preserves_coupled_rest(self):
        model = decode_direct_slow_k_model(
            self.base,
            direct_slow_k_fit_initial(),
            self.target,
        )
        state = direct_slow_k_initial_state(model, self.target)
        config = SimulationConfig(
            duration_ms=2.0,
            dt_ms=0.025,
            initial_voltage_mv=-65.0,
            stimulus=Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=0.0,
                start_ms=0.0,
                end_ms=2.0,
            ),
            conductances=model.base.biophysics,
            temperature_c=22.0,
        )
        np.testing.assert_allclose(
            _voltage_rhs(0.0, state, model, config),
            0.0,
            atol=1e-9,
        )
        trace = simulate_direct_slow_k(model, config, state)
        self.assertLess(
            np.max(np.abs(trace.voltage_mv + 65.0)),
            1e-6,
        )

    def test_spike_count_loss_is_explicit(self):
        protocols = (SimpleNamespace(name="step"),)
        biological = {
            "step": {
                "spike_count": 10.0,
                "firing_rate_hz": 20.0,
            }
        }
        self.assertEqual(
            _mean_spike_count_loss(
                biological,
                biological,
                protocols,
            ),
            0.0,
        )
        mismatched = {
            "step": {
                "spike_count": 20.0,
                "firing_rate_hz": 40.0,
            }
        }
        self.assertGreater(
            _mean_spike_count_loss(
                mismatched,
                biological,
                protocols,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
