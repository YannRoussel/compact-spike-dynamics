import math
import unittest
from dataclasses import replace

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
)
from inverse_ephys_alpha_beta.direct_phase_fit import (
    DirectPhaseFitConfig,
    _baseline_centers,
)
from inverse_ephys_alpha_beta.effective_components import (
    CANONICAL_TOPOLOGY,
    EFFECTIVE_PARAMETER_NAMES,
    EffectiveEvaluation,
    EffectiveTopology,
    _SLOW_NA_G_INDEX,
    _SLOW_NA_SLICE,
    _component_currents,
    _joint_is_eligible,
    _slow_na_initial,
    _voltage_rhs,
    decode_effective_model,
    effective_initial_state,
    effective_parameter_bounds,
    fast_topology_candidates,
    simulate_effective_components,
)
from inverse_ephys_alpha_beta.hh_model import SimulationConfig, Stimulus
from inverse_ephys_alpha_beta.model_ladder import AIS_PARAMETER_NAMES


class EffectiveComponentTests(unittest.TestCase):
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
        self.parameters = np.concatenate(
            (
                kinetic,
                static,
                _slow_na_initial(),
                (math.log(0.05),),
                direct_slow_k_initial(),
                (math.log(0.05),),
            )
        )

    def test_discrete_topology_family_and_memory_count(self):
        self.assertEqual(len(fast_topology_candidates()), 32)
        self.assertEqual(CANONICAL_TOPOLOGY.active_state_count, 3)
        topology = EffectiveTopology(
            slow_na_activation=2,
            slow_na_inactivation=1,
            slow_k_activation=3,
        )
        self.assertEqual(topology.active_state_count, 5)
        self.assertEqual(topology.active_component_count, 4)
        with self.assertRaises(ValueError):
            EffectiveTopology(slow_na_activation=1)
        with self.assertRaises(ValueError):
            EffectiveTopology(slow_k_inactivation=1)

    def test_slow_na_availability_decreases_with_voltage(self):
        kinetics = DirectSlowKinetics.from_parameters(
            self.parameters[_SLOW_NA_SLICE]
        )
        voltage = np.linspace(-100.0, 60.0, 321)
        unavailable, tau_rt = kinetics.gate_curves(voltage, 22.0)
        _, tau_warm = kinetics.gate_curves(voltage, 34.0)
        availability = 1.0 - unavailable
        self.assertTrue(np.all(np.diff(availability) <= 1e-12))
        self.assertTrue(np.all(tau_warm < tau_rt))

    def test_four_component_reanchoring_preserves_coupled_rest(self):
        topology = EffectiveTopology(
            slow_na_activation=2,
            slow_na_inactivation=1,
            slow_k_activation=2,
        )
        model = decode_effective_model(
            self.parameters,
            topology,
            self.target,
        )
        state = effective_initial_state(model, self.target)
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
        trace = simulate_effective_components(model, config, state)
        self.assertLess(
            float(np.max(np.abs(trace.voltage_mv + 65.0))),
            1e-6,
        )

    def test_integer_powers_change_current_without_adding_state(self):
        canonical = decode_effective_model(
            self.parameters,
            CANONICAL_TOPOLOGY,
            self.target,
        )
        alternative_topology = EffectiveTopology(
            fast_na_activation=1,
            fast_na_inactivation=1,
            fast_k_activation=1,
        )
        alternative = decode_effective_model(
            self.parameters,
            alternative_topology,
            self.target,
        )
        gates = (0.4, 0.7, 0.3)
        canonical_current = _component_currents(
            -20.0,
            *gates,
            1.0,
            0.0,
            canonical,
            "soma",
        )
        alternative_current = _component_currents(
            -20.0,
            *gates,
            1.0,
            0.0,
            alternative,
            "soma",
        )
        self.assertNotAlmostEqual(
            canonical_current[0],
            alternative_current[0],
        )
        self.assertNotAlmostEqual(
            canonical_current[1],
            alternative_current[1],
        )
        self.assertEqual(
            effective_initial_state(canonical, self.target).shape,
            effective_initial_state(alternative, self.target).shape,
        )

    def test_parameter_vector_matches_bounds(self):
        lower, upper = effective_parameter_bounds()
        self.assertEqual(
            self.parameters.shape,
            (len(EFFECTIVE_PARAMETER_NAMES),),
        )
        self.assertTrue(np.all(self.parameters >= lower))
        self.assertTrue(np.all(self.parameters <= upper))
        self.assertAlmostEqual(
            math.exp(self.parameters[_SLOW_NA_G_INDEX]),
            0.05,
        )

    def test_joint_guard_rejects_train_pattern_collapse(self):
        seed = EffectiveEvaluation(
            valid=True,
            reason="accepted",
            objective_total=500.0,
            selection_score=500.0,
            firing_pattern_loss=5.0,
            spike_count_loss=8.0,
        )
        collapsed = EffectiveEvaluation(
            valid=True,
            reason="accepted",
            objective_total=390.0,
            selection_score=390.0,
            firing_pattern_loss=28.0,
            spike_count_loss=3.0,
        )
        self.assertFalse(_joint_is_eligible(collapsed, seed))
        preserved = replace(
            collapsed,
            firing_pattern_loss=5.2,
            spike_count_loss=7.5,
        )
        self.assertTrue(_joint_is_eligible(preserved, seed))


if __name__ == "__main__":
    unittest.main()
