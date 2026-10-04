import tempfile
from pathlib import Path
import unittest

import numpy as np
import candidate_parameter_utils as cpu

from canonical_mo_bio import (
    CanonicalOutputPaths,
    ComponentSpec,
    OBJECTIVE_VERSION,
    calculate_component,
    canonical_search_space,
    parse_tracer_advection,
    evaluate_cost_file,
    load_component_specs,
)


def spec(**overrides):
    values = dict(
        component="test",
        label="Test",
        source="Synthetic",
        model_variable="model",
        observation_variable="observation",
        depth=None,
        weight=2.0,
        error_mode="absolute",
        error_value=1.0,
        include_in_total=True,
        bottom_ph_objective=False,
        required=True,
    )
    values.update(overrides)
    return ComponentSpec(**values)


class CanonicalMoBioTests(unittest.TestCase):
    def test_total_rmse_includes_bias_and_uses_observation_error(self):
        observation = np.array([0.0, 2.0])
        model = observation + 2.0
        result = calculate_component(model, observation, spec())
        # obs variance=1, error variance=1, MSE=4
        self.assertAlmostEqual(result["normalized_mse"], 2.0)
        self.assertAlmostEqual(result["normalized_rmse"], np.sqrt(2.0))
        self.assertAlmostEqual(result["weighted_contribution"], 8.0)
        self.assertAlmostEqual(result["bias"], 2.0)

    def test_population_variance_is_used(self):
        observation = np.array([0.0, 2.0])
        result = calculate_component(observation, observation, spec(error_value=0.0))
        self.assertAlmostEqual(result["observation_std_ddof0"], 1.0)
        self.assertEqual(result["normalized_mse"], 0.0)

    def test_incomplete_model_coverage_invalidates_component(self):
        result = calculate_component(
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, 2.0, np.nan]),
            spec(),
        )
        self.assertEqual(result["status"], "incomplete_model_coverage")
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["paired_count"], 1)
        self.assertAlmostEqual(result["model_coverage"], 0.5)
        self.assertTrue(np.isnan(result["weighted_contribution"]))

    def test_nonfinite_observations_do_not_reduce_coverage(self):
        result = calculate_component(
            np.array([1.0, 999.0, 3.0]),
            np.array([1.0, np.nan, 3.0]),
            spec(),
        )
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["paired_count"], 2)

    def test_mean_fraction_error_matches_legacy_sediment_rule(self):
        result = calculate_component(
            np.array([-1.0, -3.0]),
            np.array([-2.0, -2.0]),
            spec(error_mode="mean_fraction", error_value=1.0),
        )
        self.assertAlmostEqual(result["observational_error"], 2.0)
        self.assertAlmostEqual(result["normalization_variance"], 4.0)

    def test_component_configuration_keeps_chrp_diagnostic_only(self):
        specs = load_component_specs()
        chrp = [item for item in specs if item.source == "CHRP"]
        self.assertTrue(chrp)
        self.assertTrue(all(not item.include_in_total for item in chrp))
        self.assertTrue(all(not item.required for item in chrp))
        self.assertEqual(sum(item.bottom_ph_objective for item in specs), 1)

    def test_cost_file_objectives_use_squared_weights_and_bottom_rmse(self):
        from netCDF4 import Dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.nc"
            with Dataset(path, mode="w") as dataset:
                dataset.createDimension("Depth", 2)
                dataset.createDimension("Sample", 2)
                observation = dataset.createVariable(
                    "pH_obs", "f8", ("Depth", "Sample")
                )
                model = dataset.createVariable(
                    "pH_mod", "f8", ("Depth", "Sample")
                )
                observation[:] = [[0.0, 2.0], [1.0, 3.0]]
                model[:] = [[1.0, 3.0], [3.0, 5.0]]
            specs = (
                spec(
                    component="surface",
                    depth="Surface",
                    weight=1.0,
                    model_variable="pH_mod",
                    observation_variable="pH_obs",
                    bottom_ph_objective=False,
                ),
                spec(
                    component="bottom",
                    depth="Bottom",
                    weight=2.0,
                    model_variable="pH_mod",
                    observation_variable="pH_obs",
                    bottom_ph_objective=True,
                ),
            )
            _, summary = evaluate_cost_file(path, specs, run_name="synthetic")
        # Surface: 1 / (1 + 1) * 1^2 = 0.5.
        # Bottom: 4 / (1 + 1) * 2^2 = 8; objective two=sqrt(2).
        self.assertAlmostEqual(summary["Canonical Cost"], 8.5)
        self.assertAlmostEqual(
            summary["Canonical Bottom pH Objective"], np.sqrt(2.0)
        )
        self.assertEqual(summary["Objective Status"], "valid")

    def test_output_paths_are_isolated(self):
        paths = CanonicalOutputPaths(Path("/tmp/MO-BIO-CANONICAL"))
        self.assertEqual(
            paths.study_database,
            Path("/tmp/MO-BIO-CANONICAL/optuna/model_calibration_bio_canonical_v1.db"),
        )
        self.assertNotIn("/MO-BIO/", str(paths.study_database))
        self.assertIn("canonical", OBJECTIVE_VERSION)
        self.assertEqual(
            paths.continuation_manifest_file(2006),
            Path("/tmp/MO-BIO-CANONICAL/candidates/continuations/2006/continuation_manifest.csv"),
        )

    def test_current_prior_file_defines_a_valid_search_space(self):
        priors, numeric, categorical = canonical_search_space(
            Path("canonical_parameter_priors.csv")
        )
        self.assertEqual(len(priors), 38)
        self.assertTrue(numeric)
        self.assertTrue(all(low < high for low, high in numeric.values()))
        self.assertFalse(priors["decision"].eq("review").any())
        self.assertEqual(categorical, {})

    def test_biological_advection_validation_requires_every_u3_c4_tracer(self):
        def block(keyword, code):
            lines = [f"{keyword} == {code} ! idbio({1:2d})"]
            lines.extend(
                f"             {code} ! idbio({index:2d})"
                for index in range(2, 16)
            )
            return "\n".join(lines)

        valid = block("Hadvection", "U3") + "\n" + block("Vadvection", "C4")
        result = cpu.validate_biological_advection_texts(
            {"candidate": valid},
            horizontal_code="U3",
            vertical_code="C4",
            biological_tracer_count=15,
        )
        self.assertEqual(result.iloc[0]["Horizontal Tracers Validated"], 15)
        self.assertEqual(result.iloc[0]["Vertical Tracers Validated"], 15)

        invalid = valid.replace("C4 ! idbio(15)", "A4 ! idbio(15)")
        with self.assertRaisesRegex(ValueError, "Vadvection tracer 15"):
            cpu.validate_biological_advection_texts(
                {"candidate": invalid},
                horizontal_code="U3",
                vertical_code="C4",
                biological_tracer_count=15,
            )

    def test_all_tracer_advection_text_is_parsed(self):
        parsed = parse_tracer_advection(
            "ADVECTION: HORIZONTAL VERTICAL\n"
            "temp: Upstream3 Centered4\n"
            "salt: Akima4 Akima4\n"
            "NO3: Upstream3 Centered4\n"
        )
        self.assertEqual(parsed["temp"], ("Upstream3", "Centered4"))
        self.assertEqual(parsed["salt"], ("Akima4", "Akima4"))
        self.assertEqual(parsed["NO3"], ("Upstream3", "Centered4"))
        self.assertNotIn("ADVECTION", parsed)


if __name__ == "__main__":
    unittest.main()
