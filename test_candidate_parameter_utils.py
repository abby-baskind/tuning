"""Fast standard-library regression tests for stable workflow mechanics."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import xarray as xr

import candidate_parameter_utils as cpu


class CandidateParameterUtilityTests(unittest.TestCase):

  def test_mo_bio_output_layout_is_rooted_consistently(self):
    paths = cpu.MoBioOutputPaths(Path("/tmp/example_mo_bio"))
    self.assertEqual(
        paths.candidate_summary_file,
        Path("/tmp/example_mo_bio/candidates/mo_bio_candidate_summary.csv"),
    )
    self.assertEqual(
        paths.study_database,
        Path("/tmp/example_mo_bio/optuna/model_calibration_bio_cost_bottom_ph.db"),
    )


  def test_complete_candidate_summary_includes_fixed_values_and_provenance(self):
    candidates = pd.DataFrame(
        [{"Run Name": "OPTUNA_BIO_MO_7", "Trial Number": 7, "active_x": 1.5}]
    )
    priors = pd.DataFrame(
        [
            {
                "parameter": "child_x",
                "parameter_type": "float",
                "decision": "exclude",
                "parent_option": "parent_on",
            }
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
      prior_file = Path(directory) / "priors.csv"
      prior_file.write_text("parameter\nactive_x\n")
      summary = cpu.build_complete_candidate_summary(
          candidates,
          study_name="study",
          objective_version="objective",
          search_space_fingerprint="fingerprint",
          parameter_priors_file=prior_file,
          baseline_run_name="baseline",
          sampler_name="TPESampler",
          fixed_values={"fixed_x": 2.0, "parent_on": False},
          priors=priors,
      )
    self.assertEqual(summary.at[0, "fixed_x"], 2.0)
    self.assertEqual(summary.at[0, "Study Name"], "study")
    self.assertEqual(summary.at[0, "Trial State"], "RUNNING")
    self.assertIn("child_x", summary.at[0, "Inactive Conditional Parameters"])
    self.assertEqual(len(summary.at[0, "Parameter Priors SHA256"]), 64)


  def test_candidate_summary_append_preserves_existing_trials(self):
    columns = ["Run Name", "Trial Number", "Study Name", "value"]
    first = pd.DataFrame([["run_1", 1, "study", 0.5]], columns=columns)
    second = pd.DataFrame([["run_2", 2, "study", 0.7]], columns=columns)
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "summary.csv"
      cpu.append_candidate_summary(first, output)
      combined = cpu.append_candidate_summary(second, output)
      self.assertEqual(len(combined), 2)
      with self.assertRaises(ValueError):
        cpu.append_candidate_summary(
            pd.DataFrame([["run_changed", 1, "study", 9.9]], columns=columns),
            output,
        )

  def test_cost_breakdown_preserves_reference_weighting(self):
    dataset = xr.Dataset(
        {
            "pH_cost": ("Depth", [1.0, 2.0]),
            "oxygen_cost": ("Depth", [3.0, 4.0]),
            "SD_cost": xr.DataArray(5.0),
        }
    )
    result = cpu.calculate_cost_breakdown(
        dataset,
        {
            "Surface pH": "pH_cost",
            "Bottom pH": "pH_cost",
            "Surface Oxygen": "oxygen_cost",
            "Bottom Oxygen": "oxygen_cost",
            "Secchi Depth": "SD_cost",
        },
        {
            "Surface pH": 2,
            "Bottom pH": 2,
            "Surface Oxygen": 2,
            "Bottom Oxygen": 2,
            "Secchi Depth": 1,
        },
    )
    self.assertEqual(result["Cost"], 45.0)
    self.assertEqual(result["Bottom pH Cost Contribution"], 8.0)
    self.assertEqual(result["Non-bottom-pH Cost"], 37.0)


  def test_pareto_ranks_for_two_minimization_objectives(self):
    values = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]], dtype=float)
    ranks = cpu.pareto_ranks(values)
    self.assertEqual(ranks.tolist(), [0, 0, 0, 1, 2])


  def test_unresolved_suspicious_value_is_removed_from_analysis(self):
    raw = pd.DataFrame({"Run Name": ["run_a"], "parameter_x": [1e-310]})
    priors = pd.DataFrame(
        [
            {
                "parameter": "parameter_x",
                "parameter_type": "float",
                "decision": "review",
                "bottom_ph_relevance": "unknown",
                "holistic_relevance": "unknown",
                "confidence_in_current_value": "unknown",
                "scale": "linear",
                "allowed_choices": "",
                "proposed_low": np.nan,
                "proposed_high": np.nan,
                "fixed_value": np.nan,
                "parent_option": "",
                "scientific_rationale": "",
                "future_question": "",
                "notes": "",
            }
        ]
    )
    issues = cpu.detect_data_quality_issues(raw, priors)
    corrected, issue_table, audit = cpu.apply_corrections(
        raw, issues, pd.DataFrame(columns=cpu.CORRECTION_COLUMNS)
    )
    self.assertEqual(len(issues), 1)
    self.assertEqual(issue_table.iloc[0]["resolution_status"], "unresolved")
    self.assertTrue(pd.isna(corrected.iloc[0]["parameter_x"]))
    self.assertTrue(audit.empty)


  def test_historical_values_outside_proposed_bounds_are_not_quality_issues(self):
    raw = pd.DataFrame(
        {
            "Run Name": ["below", "inside", "above"],
            "parameter_x": [0.25, 1.5, 4.0],
        }
    )
    priors = pd.DataFrame(
        [
            {
                "parameter": "parameter_x",
                "parameter_type": "float",
                "decision": "active",
                "bottom_ph_relevance": "unknown",
                "holistic_relevance": "unknown",
                "confidence_in_current_value": "unknown",
                "scale": "linear",
                "allowed_choices": "",
                "proposed_low": 1.0,
                "proposed_high": 2.0,
                "fixed_value": np.nan,
                "parent_option": "",
                "scientific_rationale": "",
                "future_question": "",
                "notes": "",
            }
        ]
    )

    issues = cpu.detect_data_quality_issues(raw, priors)

    self.assertTrue(issues.empty)


  @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna not installed")
  def test_observed_support_imports_history_but_candidates_stay_in_bounds(self):
    import optuna

    frame = pd.DataFrame(
        {
            "Run Name": ["low", "inside", "high"],
            "parameter_x": [0.25, 1.5, 4.0],
            "Cost": [3.0, 2.0, 1.0],
            "Bottom pH Objective": [1.0, 0.8, 0.6],
        }
    )
    bounds = {"parameter_x": (1.0, 2.0)}
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=1),
    )

    imported = cpu.import_historical_multiobjective_trials(
        study,
        frame,
        numeric_bounds=bounds,
        categorical_choices={},
        objective_version="test",
        numeric_import_policy="observed_support",
    )

    self.assertEqual(imported["Import Status"].tolist(), ["added"] * 3)
    self.assertEqual(imported["Within Candidate Bounds"].tolist(), [False, True, False])
    self.assertEqual(len(study.trials), 3)
    for trial in study.trials:
      distribution = trial.distributions["parameter_x"]
      self.assertEqual((distribution.low, distribution.high), (0.25, 4.0))

    candidates = cpu.ask_candidate_trials(
        study,
        numeric_bounds=bounds,
        categorical_choices={},
        count=8,
        run_name_prefix="TEST_",
        objective_version="test",
    )
    self.assertTrue(candidates["parameter_x"].between(1.0, 2.0).all())


  @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna not installed")
  def test_candidate_bounds_policy_still_excludes_outside_history(self):
    import optuna

    study = optuna.create_study(directions=["minimize", "minimize"])
    imported = cpu.import_historical_multiobjective_trials(
        study,
        pd.DataFrame(
            {
                "Run Name": ["outside"],
                "parameter_x": [4.0],
                "Cost": [1.0],
                "Bottom pH Objective": [1.0],
            }
        ),
        numeric_bounds={"parameter_x": (1.0, 2.0)},
        categorical_choices={},
        objective_version="test",
        numeric_import_policy="candidate_bounds",
    )

    self.assertEqual(imported.at[0, "Import Status"], "outside configured bounds")
    self.assertEqual(len(study.trials), 0)


  @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna not installed")
  def test_study_records_and_validates_historical_import_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      storage = f"sqlite:///{Path(directory) / 'study.db'}"
      study = cpu.create_or_load_multiobjective_study(
          study_name="test_policy",
          storage=storage,
          objective_version="test",
          fingerprint="fingerprint",
          seed=7,
          n_startup_trials=1,
          multivariate=False,
          historical_numeric_import_policy="observed_support",
      )
      self.assertEqual(
          study.user_attrs["Historical Numeric Import Policy"], "observed_support"
      )
      with self.assertRaises(ValueError):
        cpu.create_or_load_multiobjective_study(
            study_name="test_policy",
            storage=storage,
            objective_version="test",
            fingerprint="fingerprint",
            seed=7,
            n_startup_trials=1,
            multivariate=False,
            historical_numeric_import_policy="candidate_bounds",
        )


  def test_readiness_respects_decision_for_confounded_parameters(self):
    priors = pd.DataFrame(
        [
            {
                "parameter": parameter,
                "parameter_type": "float",
                "decision": decision,
                "bottom_ph_relevance": "unknown",
                "holistic_relevance": "unknown",
            }
            for parameter, decision in [
                ("active_x", "active"),
                ("fixed_x", "fixed"),
                ("reserve_x", "reserve"),
            ]
        ]
    )
    numeric_summary = pd.DataFrame(
        [
            {"Parameter": parameter, "Unique Values": 10, "Promising Count": 5}
            for parameter in priors["parameter"]
        ]
    )
    confounding = pd.DataFrame(
        [
            {"Parameter 1": "active_x", "Parameter 2": "fixed_x"},
            {"Parameter 1": "reserve_x", "Parameter 2": "fixed_x"},
        ]
    )

    readiness = cpu.readiness_summary(
        priors, numeric_summary, pd.DataFrame(), confounding
    ).set_index("Parameter")

    self.assertEqual(readiness.at["active_x", "Readiness"], "Optimize cautiously")
    self.assertEqual(readiness.at["fixed_x", "Readiness"], "Fix cautiously")
    self.assertEqual(readiness.at["reserve_x", "Readiness"], "Held in reserve")


  @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
  def test_nonlinear_screening_accepts_boolean_categories(self):
    numeric = np.linspace(0.0, 1.0, 20)
    frame = pd.DataFrame(
        {
            "numeric_x": numeric,
            "boolean_x": np.arange(20) % 2 == 0,
            "Cost": 2.0 + numeric**2,
            "Bottom pH Objective": 1.0 - 0.5 * numeric,
        }
    )

    performance, importance = cpu.cross_validated_mixed_importance(
        frame,
        ["numeric_x"],
        ["boolean_x"],
        ["Cost", "Bottom pH Objective"],
        random_state=7,
        n_estimators=10,
        folds=2,
        permutation_repeats=1,
    )

    self.assertFalse(performance.empty)
    self.assertIn("boolean_x", set(importance["Parameter"]))


  def test_roms_number_formatting(self):
    self.assertEqual(cpu.format_roms_value(2), "2.0d0")
    self.assertEqual(cpu.format_roms_value(0.125), "0.125d0")
    self.assertEqual(cpu.format_roms_value(1e-5), "1d-5")


if __name__ == "__main__":
    unittest.main()
