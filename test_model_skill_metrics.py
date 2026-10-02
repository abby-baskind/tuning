import math
import unittest

import numpy as np

from model_skill_metrics import calculate_skill, pairwise_finite


class ModelSkillMetricTests(unittest.TestCase):
    def test_perfect_model(self):
        obs = np.array([1.0, 2.0, 4.0, 8.0])
        result = calculate_skill(obs, obs)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["bias_star"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["rmse_star"], 0.0)
        self.assertAlmostEqual(result["rmse_centered_star"], 0.0)
        self.assertAlmostEqual(result["ward_cost"], 0.0)
        self.assertAlmostEqual(result["correlation"], 1.0)

    def test_constant_bias_decomposition(self):
        obs = np.array([0.0, 1.0, 2.0, 3.0])
        model = obs + 2.0
        result = calculate_skill(model, obs)
        self.assertAlmostEqual(result["rmse"], 2.0)
        self.assertAlmostEqual(result["rmse_centered"], 0.0)
        self.assertAlmostEqual(result["rmse_star"], abs(result["bias_star"]))
        self.assertAlmostEqual(result["ward_cost"], result["rmse_star"] ** 2)
        self.assertAlmostEqual(result["identity_error"], 0.0, places=12)

    def test_general_jolliff_identity(self):
        obs = np.array([1.0, 4.0, 2.0, 8.0, 3.0])
        model = np.array([0.5, 5.0, 3.0, 6.0, 6.0])
        result = calculate_skill(model, obs)
        lhs = result["rmse_star"] ** 2
        rhs = result["bias_star"] ** 2 + result["rmse_centered_star"] ** 2
        self.assertAlmostEqual(lhs, rhs, places=12)
        self.assertAlmostEqual(result["ward_cost"], lhs, places=12)

    def test_joint_finite_mask(self):
        model, obs = pairwise_finite(
            np.array([1.0, np.nan, 3.0, 4.0]),
            np.array([1.0, 2.0, np.nan, 5.0]),
        )
        np.testing.assert_array_equal(model, [1.0, 4.0])
        np.testing.assert_array_equal(obs, [1.0, 5.0])
        result = calculate_skill(model, obs)
        self.assertEqual(result["n"], 2)

    def test_target_sign_describes_variance(self):
        obs = np.array([-2.0, -1.0, 1.0, 2.0])
        narrow = obs * 0.5
        wide = obs * 2.0
        self.assertLess(calculate_skill(narrow, obs)["target_x"], 0)
        self.assertGreater(calculate_skill(wide, obs)["target_x"], 0)

    def test_zero_observation_variance_is_explicit(self):
        result = calculate_skill(np.array([1.0, 2.0]), np.array([3.0, 3.0]))
        self.assertEqual(result["status"], "nonpositive_observation_std")
        self.assertTrue(math.isnan(result["ward_cost"]))


if __name__ == "__main__":
    unittest.main()
