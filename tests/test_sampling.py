import unittest

import numpy as np

from inverse_ephys_alpha_beta.kinetics import default_parameter_bounds
from inverse_ephys_alpha_beta.sampling import (
    ALL_PARAMETER_NAMES,
    latin_hypercube_parameters,
)
from inverse_ephys_alpha_beta.static_parameters import (
    StaticParameterTransforms,
    default_static_parameter_bounds,
)


class SamplingTests(unittest.TestCase):
    def test_canonical_and_bounds(self):
        frame = latin_hypercube_parameters(8, seed=7, include_canonical=True)
        self.assertEqual(frame.shape, (8, len(ALL_PARAMETER_NAMES) + 1))
        canonical = np.concatenate(
            (
                np.zeros(len(default_parameter_bounds()[0])),
                StaticParameterTransforms.canonical().to_vector(),
            )
        )
        np.testing.assert_allclose(frame.loc[0, list(ALL_PARAMETER_NAMES)], canonical)
        lower = np.concatenate(
            (default_parameter_bounds()[0], default_static_parameter_bounds()[0])
        )
        upper = np.concatenate(
            (default_parameter_bounds()[1], default_static_parameter_bounds()[1])
        )
        sampled = frame.loc[:, list(ALL_PARAMETER_NAMES)].to_numpy()
        self.assertTrue(np.all(sampled >= lower))
        self.assertTrue(np.all(sampled <= upper))

    def test_static_only_keeps_canonical_kinetics(self):
        frame = latin_hypercube_parameters(
            8,
            seed=7,
            include_canonical=True,
            include_static=True,
            sample_kinetics=False,
        )
        kinetic_names = ALL_PARAMETER_NAMES[: len(default_parameter_bounds()[0])]
        np.testing.assert_allclose(frame.loc[:, list(kinetic_names)], 0.0)
        static_values = frame.loc[1:, list(ALL_PARAMETER_NAMES[len(kinetic_names) :])]
        self.assertGreater(static_values.to_numpy().std(), 0.0)

    def test_custom_bounds_are_applied(self):
        parameter_name = ALL_PARAMETER_NAMES[-1]
        frame = latin_hypercube_parameters(
            8,
            seed=7,
            include_canonical=False,
            parameter_bounds={parameter_name: (2.0, 2.1)},
        )
        self.assertTrue(frame[parameter_name].between(2.0, 2.1).all())

    def test_gate_correlated_sampling_limits_alpha_beta_differences(self):
        frame = latin_hypercube_parameters(
            32,
            seed=9,
            include_canonical=False,
            kinetic_sampling_mode="gate-correlated",
        )
        for gate_name in ("m", "h", "n"):
            for field_name in (
                "log_rate_scale",
                "voltage_shift_mv",
                "log_slope_scale",
            ):
                alpha = frame[f"param__alpha_{gate_name}__{field_name}"]
                beta = frame[f"param__beta_{gate_name}__{field_name}"]
                default_width = 20.0 if field_name == "voltage_shift_mv" else None
                if default_width is None:
                    kinetic_lower, kinetic_upper = default_parameter_bounds()
                    parameter_index = list(ALL_PARAMETER_NAMES).index(
                        f"param__alpha_{gate_name}__{field_name}"
                    )
                    default_width = (
                        kinetic_upper[parameter_index]
                        - kinetic_lower[parameter_index]
                    )
                self.assertLessEqual(
                    float((alpha - beta).abs().max()),
                    0.25 * default_width + 1e-12,
                )

    def test_mixed_sampling_correlates_second_half(self):
        frame = latin_hypercube_parameters(
            11,
            seed=9,
            include_canonical=True,
            kinetic_sampling_mode="mixed",
        )
        correlated = frame.iloc[6:]
        for gate_name in ("m", "h", "n"):
            for field_name in (
                "log_rate_scale",
                "voltage_shift_mv",
                "log_slope_scale",
            ):
                alpha_name = f"param__alpha_{gate_name}__{field_name}"
                beta_name = f"param__beta_{gate_name}__{field_name}"
                parameter_index = list(ALL_PARAMETER_NAMES).index(alpha_name)
                if field_name == "voltage_shift_mv":
                    default_width = 20.0
                else:
                    kinetic_lower, kinetic_upper = default_parameter_bounds()
                    default_width = (
                        kinetic_upper[parameter_index]
                        - kinetic_lower[parameter_index]
                    )
                self.assertLessEqual(
                    float(
                        (
                            correlated[alpha_name] - correlated[beta_name]
                        ).abs().max()
                    ),
                    0.25 * default_width + 1e-12,
                )


if __name__ == "__main__":
    unittest.main()
