import unittest

import numpy as np

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
)
from inverse_ephys_alpha_beta.hh_model import SimulationConfig, Stimulus
from inverse_ephys_alpha_beta.model_ladder import (
    _initial_state,
    _slow_gate,
    _steady_currents,
    decode_ladder_model,
    extension_parameter_bounds,
    simulate_ladder,
    simulate_ladder_adaptive,
    simulate_ladder_rush_larsen,
)


class ModelLadderTests(unittest.TestCase):
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
        self.base_vector = np.zeros(len(CELL_PARAMETER_NAMES))

    def test_slow_gate_steady_states_have_expected_direction(self):
        na_gate = _slow_gate(
            "na-slow",
            extension_parameter_bounds("na-slow")[2],
        )
        k_gate = _slow_gate(
            "k-slow",
            extension_parameter_bounds("k-slow")[2],
        )

        self.assertGreater(
            na_gate.steady_state(-80.0),
            na_gate.steady_state(0.0),
        )
        self.assertLess(
            k_gate.steady_state(-80.0),
            k_gate.steady_state(0.0),
        )

    def test_all_extensions_preserve_the_anchored_rest(self):
        for variant in ("na-slow", "k-slow", "soma-ais"):
            with self.subTest(variant=variant):
                initial = extension_parameter_bounds(variant)[2]
                model = decode_ladder_model(
                    variant,
                    self.base_vector,
                    initial,
                    self.target,
                )
                soma_current = sum(
                    _steady_currents(-65.0, model, "soma")
                )
                self.assertAlmostEqual(soma_current, 0.0, places=9)
                if variant == "soma-ais":
                    ais_current = sum(
                        _steady_currents(-65.0, model, "ais")
                    )
                    self.assertAlmostEqual(ais_current, 0.0, places=9)

                state = _initial_state(model, self.target)
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
                    conductances=model.biophysics,
                    temperature_c=22.0,
                )
                trace = simulate_ladder(model, config, state)
                self.assertLess(
                    np.max(np.abs(trace.voltage_mv + 65.0)),
                    1e-6,
                )
                adaptive_trace = simulate_ladder_adaptive(
                    model,
                    config,
                    state,
                )
                self.assertLess(
                    np.max(np.abs(adaptive_trace.voltage_mv + 65.0)),
                    1e-6,
                )
                if variant == "soma-ais":
                    rush_larsen_trace = simulate_ladder_rush_larsen(
                        model,
                        config,
                        state,
                    )
                    self.assertLess(
                        np.max(
                            np.abs(
                                rush_larsen_trace.voltage_mv + 65.0
                            )
                        ),
                        1e-6,
                    )


if __name__ == "__main__":
    unittest.main()
