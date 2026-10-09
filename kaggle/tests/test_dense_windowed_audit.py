import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense_windowed_baseline_audit import (
    compute_zero_output_baselines,
    find_ffp_after_healthy,
    reconstruction_improvement_percent,
)


class DenseWindowedAuditTests(unittest.TestCase):
    def test_find_ffp_after_healthy_reports_zero_and_one_based_index(self):
        hi = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 1.4, 1.5])

        result = find_ffp_after_healthy(
            hi,
            threshold=1.0,
            healthy_samples=3,
            consecutive=5,
        )

        self.assertEqual(result["ffp_zero_based"], 3)
        self.assertEqual(result["ffp_one_based"], 4)
        self.assertEqual(result["run_length"], 5)

    def test_find_ffp_after_healthy_ignores_runs_inside_healthy_region(self):
        hi = np.array([2.0, 2.0, 2.0, 0.1, 0.2, 1.1, 1.2, 1.3, 1.4, 1.5])

        result = find_ffp_after_healthy(
            hi,
            threshold=1.0,
            healthy_samples=5,
            consecutive=5,
        )

        self.assertEqual(result["ffp_zero_based"], 5)
        self.assertEqual(result["ffp_one_based"], 6)

    def test_compute_zero_output_baselines_keeps_sample_mean_and_window_p95_separate(self):
        samples = np.array(
            [
                [1.0, 1.0, 3.0, 3.0],
                [2.0, 2.0, 4.0, 4.0],
            ]
        )

        baselines = compute_zero_output_baselines(samples, window_size=2)

        np.testing.assert_allclose(baselines["raw_rms2"], np.array([5.0, 10.0]))
        np.testing.assert_allclose(baselines["zero_window_p95"], np.array([8.6, 15.4]))

    def test_reconstruction_improvement_percent_uses_global_energy_ratio(self):
        zero_mse = np.array([10.0, 20.0, 30.0])
        recon_mse = np.array([9.0, 18.0, 27.0])

        improvement = reconstruction_improvement_percent(recon_mse, zero_mse)

        self.assertAlmostEqual(improvement, 10.0)


if __name__ == "__main__":
    unittest.main()
