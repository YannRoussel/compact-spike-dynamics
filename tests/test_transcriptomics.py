from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from inverse_ephys_alpha_beta.model_transcriptomics import (
    build_model_target_matrix,
    inverse_model_target_matrix,
    transform_model_target_matrix,
)
from inverse_ephys_alpha_beta.transcriptomics import (
    PatchSeqExpression,
    align_expression_to_parameters,
    broad_transcriptomic_class,
    expression_matrix,
    is_kcn_gene,
    is_scn_gene,
)


class TranscriptomicsTests(unittest.TestCase):
    def test_training_defined_model_target_transform_round_trip(self):
        training = pd.DataFrame(
            {
                "dataset": ("example", "example", "example"),
                "cell_id": ("a", "b", "c"),
                "rheobase_current_pa": (10.0, 20.0, 40.0),
                "recovery_voltage_drive": (-1.0, 0.0, 2.0),
            }
        )
        heldout = pd.DataFrame(
            {
                "dataset": ("example",),
                "cell_id": ("d",),
                "rheobase_current_pa": (30.0,),
                "recovery_voltage_drive": (1.0,),
            }
        )
        target = build_model_target_matrix(training)
        transformed = transform_model_target_matrix(heldout, target)
        recovered = inverse_model_target_matrix(transformed, target)
        expected = heldout.loc[:, list(target.names)].to_numpy(dtype=float)
        np.testing.assert_allclose(recovered, expected)

    def test_channel_family_symbols_exclude_scnn_and_modifiers(self):
        self.assertTrue(is_scn_gene("Scn8a"))
        self.assertTrue(is_scn_gene("Scn10a"))
        self.assertFalse(is_scn_gene("Scnn1a"))
        self.assertFalse(is_scn_gene("Scnm1"))
        self.assertTrue(is_kcn_gene("Kcnc1"))

    def test_alignment_and_cpm_log_transform(self):
        expression = PatchSeqExpression(
            dataset="example",
            cell_ids=np.asarray(("a", "b", "c")),
            genes=np.asarray(("Gene1", "Scn8a")),
            values=sparse.csr_matrix(
                np.asarray(
                    (
                        (10.0, 0.0),
                        (5.0, 5.0),
                        (0.0, 10.0),
                    )
                )
            ),
            donor_ids=np.asarray(("d1", "d1", "d2")),
            transcriptomic_types=np.asarray(("t1", "t1", "t2")),
            already_log_normalized=False,
            library_size=np.asarray((10.0, 10.0, 10.0)),
            source_cell_count=3,
        )
        parameters = pd.DataFrame(
            {
                "dataset": ("example", "example"),
                "cell_id": ("c", "a"),
            }
        )

        aligned, aligned_parameters = align_expression_to_parameters(
            expression,
            parameters,
        )
        values = expression_matrix(aligned)

        self.assertEqual(aligned_parameters["cell_id"].tolist(), ["c", "a"])
        self.assertEqual(aligned.cell_ids.tolist(), ["c", "a"])
        self.assertAlmostEqual(values[0, 1], np.log2(1e6 + 1.0))
        self.assertEqual(aligned.source_cell_count, 3)

    def test_broad_transcriptomic_class_mapping(self):
        self.assertEqual(
            broad_transcriptomic_class("Pvalb Il1rapl2"),
            "Pvalb",
        )
        self.assertEqual(
            broad_transcriptomic_class("L2/3 IT_3"),
            "Glut_IT",
        )
        self.assertEqual(
            broad_transcriptomic_class("L5 PT_2"),
            "Glut_ET",
        )
        self.assertEqual(
            broad_transcriptomic_class("L6 CT Cpa6"),
            "Glut_CT",
        )
        self.assertEqual(
            broad_transcriptomic_class("L6b Kcnip1"),
            "Glut_L6b",
        )


if __name__ == "__main__":
    unittest.main()
