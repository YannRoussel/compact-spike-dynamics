import math
import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
)
from inverse_ephys_alpha_beta.direct_kinetics import (
    direct_slow_k_initial,
)
from inverse_ephys_alpha_beta.direct_phase_fit import (
    DirectPhaseFitConfig,
    _baseline_centers,
)
from inverse_ephys_alpha_beta.effective_components import (
    CANONICAL_TOPOLOGY,
    EffectiveTopology,
    _slow_na_initial,
    decode_effective_model,
    effective_initial_state,
    simulate_effective_components,
)
from inverse_ephys_alpha_beta.hh_model import SimulationConfig, Stimulus
from inverse_ephys_alpha_beta.model_ladder import AIS_PARAMETER_NAMES
from inverse_ephys_alpha_beta.two_compartment_effective import (
    TWO_COMPARTMENT_PARAMETER_NAMES,
    CompartmentTopologies,
    _AIS_SLOW_K_G_INDEX,
    _AIS_SLOW_NA_G_INDEX,
    _two_compartment_voltage_rhs,
    decode_two_compartment_model,
    initialize_two_compartment_parent,
    simulate_two_compartment_effective,
    two_compartment_initial_state,
    two_compartment_parameter_bounds,
)


class TwoCompartmentEffectiveTests(unittest.TestCase):
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
            baseline, DirectPhaseFitConfig()
        )
        self.soma_parameters = np.concatenate(
            (
                kinetic,
                static,
                _slow_na_initial(),
                (math.log(0.05),),
                direct_slow_k_initial(),
                (math.log(0.05),),
            )
        )
        self.parameters, self.topologies = (
            initialize_two_compartment_parent(
                self.soma_parameters,
                CANONICAL_TOPOLOGY,
                self.target,
            )
        )

    def _config(self, current_pa=0.0):
        model = decode_two_compartment_model(
            self.parameters, self.topologies, self.target
        )
        return SimulationConfig(
            duration_ms=8.0,
            dt_ms=0.025,
            initial_voltage_mv=-65.0,
            stimulus=Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=current_pa,
                start_ms=2.0,
                end_ms=7.0,
            ),
            conductances=model.soma.base.biophysics,
            temperature_c=22.0,
        )

    def test_parent_vector_and_bounds(self):
        lower, upper = two_compartment_parameter_bounds()
        self.assertEqual(
            self.parameters.shape,
            (len(TWO_COMPARTMENT_PARAMETER_NAMES),),
        )
        self.assertEqual(self.parameters.shape, lower.shape)
        self.assertTrue(np.all(self.parameters >= lower))
        self.assertTrue(np.all(self.parameters <= upper))
        self.assertEqual(self.topologies.active_state_count, 6)

    def test_coupled_rest_is_preserved(self):
        model = decode_two_compartment_model(
            self.parameters, self.topologies, self.target
        )
        state = two_compartment_initial_state(model, self.target)
        np.testing.assert_allclose(
            _two_compartment_voltage_rhs(
                0.0, state, model, self._config()
            ),
            0.0,
            atol=1e-9,
        )
        result = simulate_two_compartment_effective(
            model, self._config(), state
        )
        self.assertLess(
            float(np.max(np.abs(result.soma.voltage_mv + 65.0))),
            1e-6,
        )
        self.assertLess(
            float(np.max(np.abs(result.ais_voltage_mv + 65.0))),
            1e-6,
        )

    def test_lifted_parent_exactly_matches_previous_simulator(self):
        model = decode_two_compartment_model(
            self.parameters, self.topologies, self.target
        )
        old_model = decode_effective_model(
            self.soma_parameters,
            CANONICAL_TOPOLOGY,
            self.target,
        )
        config = self._config(current_pa=120.0)
        old_trace = simulate_effective_components(
            old_model,
            config,
            effective_initial_state(old_model, self.target),
        )
        new_result = simulate_two_compartment_effective(
            model,
            config,
            two_compartment_initial_state(model, self.target),
        )
        np.testing.assert_allclose(
            new_result.soma.voltage_mv,
            old_trace.voltage_mv,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            new_result.soma.dvdt_mv_ms,
            old_trace.dvdt_mv_ms,
            atol=1e-12,
        )

    def test_ais_slow_currents_add_independent_memory_states(self):
        topologies = CompartmentTopologies(
            soma=CANONICAL_TOPOLOGY,
            ais=EffectiveTopology(
                slow_na_activation=2,
                slow_na_inactivation=1,
                slow_k_activation=3,
            ),
        )
        values = self.parameters.copy()
        values[_AIS_SLOW_NA_G_INDEX] = math.log(0.5)
        values[_AIS_SLOW_K_G_INDEX] = math.log(0.5)
        model = decode_two_compartment_model(
            values, topologies, self.target
        )
        self.assertEqual(topologies.active_state_count, 8)
        state = two_compartment_initial_state(model, self.target)
        self.assertEqual(state.shape, (12,))
        self.assertTrue(0.0 <= state[10] <= 1.0)
        self.assertTrue(0.0 <= state[11] <= 1.0)


if __name__ == "__main__":
    unittest.main()
