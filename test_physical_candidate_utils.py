from pathlib import Path
import tempfile
import unittest

import pandas as pd

import physical_candidate_utils as pcu


SPEC_PATH = Path(__file__).with_name("physical_tuning_spec.toml")


def approved_spec():
    spec = pcu.load_physical_spec(SPEC_PATH)
    spec["mixing_orientation"]["review_status"] = "approved"
    spec["optional_cpp_options"]["review_status"] = "approved"
    return spec


def historical_row(**overrides):
    row = {
        "Run Name": "example",
        "Run Date": pd.Timestamp("2026-09-18 20:46:28"),
        "Cost": 0.2,
        "advection_scheme": "U3_C4",
        "nl_tnu2_shared": 0.5,
        "nl_tnu2_tracer0": 0.5,
        "nl_tnu2_tracer1": 0.5,
        "UV orientation": "MIX_S_UV",
        "TS orientation": "MIX_GEO_TS",
        "GLS_MIXING": True,
        "RI_SPLINES": True,
        "N2S2_HORAVG": True,
        "KANTHA_CLAYSON": False,
        "CRAIG_BANNER": False,
        "CANUTO_A": False,
        "temp_horizontal_advection": "Upstream3",
        "temp_vertical_advection": "Centered4",
        "salt_horizontal_advection": "Upstream3",
        "salt_vertical_advection": "Centered4",
    }
    row.update(overrides)
    return row


class PhysicalCandidateUtilsTests(unittest.TestCase):
    def test_toml_review_blockers_match_current_review_statuses(self):
        spec = pcu.load_physical_spec(SPEC_PATH)
        blockers = pcu.candidate_generation_blockers(spec)
        expected = sum(
            spec[section]["review_status"] != "approved"
            for section in ("mixing_orientation", "optional_cpp_options")
        )
        self.assertEqual(len(blockers), expected)

    def test_cpp_parser_handles_disabled_tokens_and_missing_commas(self):
        enabled, disabled = pcu.parse_cpp_options(
            "GLS_MIXING, !CANUTO_A, PCO2AIR_SEA RI_SPLINES, N2S2_HORAVG"
        )
        self.assertTrue({"GLS_MIXING", "RI_SPLINES", "N2S2_HORAVG"}.issubset(enabled))
        self.assertNotIn("CANUTO_A", enabled)
        self.assertIn("CANUTO_A", disabled)

    def test_history_parser_uses_date_and_time(self):
        parsed = pcu.parse_history_datetime(
            "ROMS, Version 4.3, Friday - September 18, 2026 -  8:46:28 PM"
        )
        self.assertEqual(parsed, pd.Timestamp("2026-09-18 20:46:28"))

    def test_advection_identification_uses_exact_station_spelling(self):
        spec = approved_spec()
        values = pcu.extract_advection("temp: HSIMT HSIMT\nsalt: HSIMT HSIMT\n")
        self.assertEqual(pcu.identify_advection_scheme(values, spec), "HSIMT_HSIMT")
        typo = {column: "HSMIT" for column in pcu.ADVECTION_COLUMNS}
        self.assertTrue(pd.isna(pcu.identify_advection_scheme(typo, spec)))

    def test_component_factorial_resolves_all_constraints(self):
        spec = approved_spec()
        space = pcu.build_candidate_space(pd.DataFrame([historical_row()]), spec)
        expected = (
            sum(item["candidate_allowed"] for item in spec["advection"]["schemes"])
            * len(spec["nl_tnu2"]["candidate_values"])
            * len(spec["mixing_orientation"]["approved_pairs"])
            * len(spec["optional_cpp_options"]["approved_combinations"])
        )
        self.assertEqual(len(space), expected)
        self.assertTrue(space["GLS_MIXING"].all())
        self.assertTrue(space["RI_SPLINES"].all())
        self.assertTrue(space["N2S2_HORAVG"].all())
        self.assertTrue((space["nl_tnu2_tracer0"] == space["nl_tnu2_tracer1"]).all())
        self.assertEqual(int(space["observed_exactly_in_archive"].sum()), 1)

    def test_observed_only_mode_keeps_exact_historical_tuple(self):
        spec = approved_spec()
        spec["workflow"]["candidate_space_mode"] = "observed_configurations_only"
        space = pcu.build_candidate_space(pd.DataFrame([historical_row()]), spec)
        self.assertEqual(len(space), 1)
        self.assertTrue(bool(space.iloc[0]["observed_exactly_in_archive"]))

    def test_prohibited_historical_value_is_retained_but_not_compatible(self):
        spec = approved_spec()
        frame = pd.DataFrame(
            [historical_row(), historical_row(**{"Run Name": "old-high", "nl_tnu2_shared": 2.0})]
        )
        output = pcu.add_configuration_columns(frame, spec)
        self.assertEqual(len(output), 2)
        self.assertTrue(bool(output.loc[output["Run Name"].eq("example"), "candidate_compatible"].iloc[0]))
        self.assertFalse(bool(output.loc[output["Run Name"].eq("old-high"), "candidate_compatible"].iloc[0]))

    def test_run_map_preserves_user_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            run_map_file = Path(directory) / "map.csv"
            pd.DataFrame(
                [{
                    "candidate_id": "PHYS_000",
                    "optuna_trial_number": "0",
                    "actual_run_name": "REAL_A",
                    "status": "running",
                    "station_file": "ocean_sta_REAL_A.nc",
                    "completion_date": pd.NA,
                    "notes": "keep me",
                }]
            ).to_csv(run_map_file, index=False)
            candidates = pd.DataFrame(
                [
                    {"candidate_id": "PHYS_000", "optuna_trial_number": 0},
                    {"candidate_id": "PHYS_001", "optuna_trial_number": 1},
                ]
            )
            result = pcu.update_run_map(run_map_file, candidates)
            first = result.loc[result["candidate_id"].eq("PHYS_000")].iloc[0]
            self.assertEqual(first["actual_run_name"], "REAL_A")
            self.assertEqual(first["notes"], "keep me")
            self.assertIn("mo_bio_study_name", result.columns)
            self.assertIn("mo_bio_trial_number", result.columns)
            new_status = result.loc[
                result["candidate_id"].eq("PHYS_001"), "status"
            ].iloc[0]
            self.assertEqual(new_status, "planned")


if __name__ == "__main__":
    unittest.main()
