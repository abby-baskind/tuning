#!/usr/bin/env python3
"""Isolated canonical audit and Optuna-study preparation for MO-BIO.

This module never reads or writes the legacy Optuna database.  The default
``audit`` command recalculates annual objectives directly from cost-summary
NetCDF files.  ``prepare-study`` creates/updates a separate study and imports
only completed runs with valid canonical objectives.  Re-running it later is
the intended way to add trials that are still running in the legacy study.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import candidate_parameter_utils as cpu


OBJECTIVE_VERSION = "canonical_mo_bio_ward_obs_error_v1"
OBJECTIVE_COLUMNS = ("Canonical Cost", "Canonical Bottom pH Objective")
DEFAULT_STUDY_NAME = "model_calibration_bio_canonical_v1"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPONENTS_FILE = DEFAULT_PROJECT_ROOT / "canonical_mo_bio_components.csv"
DEFAULT_OPTIMIZATION_ROOT = Path.home() / "Desktop" / "Optimization"
DEFAULT_COST_ROOT = Path.home() / "Desktop" / "COST_FILES"
DEFAULT_OUTPUT_ROOT = DEFAULT_OPTIMIZATION_ROOT / "MO-BIO-CANONICAL"
CPP_OPTION_NAMES = ("CACO3", "RIVER_SEDIMENT", "USE_MOCSY")
CPP_CONDITIONAL_VARIABLES = {
    "CACO3": {
        "cacopf": ("cacopf",),
        "cacodr": ("cacodr",),
        "omega_thresh": ("omega_thresh",),
        "wsPCa": ("wsPCa",),
        "bUmaxCa": ("bUmaxCa_nspc0", "bUmaxCa_nspc1", "bUmaxCa_nspc2"),
    }
}
EXPECTED_PHYSICAL_TRACERS = ("temp", "salt")
EXPECTED_BIOLOGICAL_TRACERS = (
    "NO3", "NH4", "SiOH4", "smallphytoplankton", "diatom",
    "microzooplankton", "mesozooplankton", "detritus", "opal", "PO4",
    "chlorophyll1", "chlorophyll2", "oxygen", "TIC", "alkalinity",
)
CANONICAL_ADVECTION_PAIR = ("Upstream3", "Centered4")


@dataclass(frozen=True)
class CanonicalOutputPaths:
    """All canonical artifacts live below one root, separate from MO-BIO."""

    root: Path

    @property
    def audit_directory(self) -> Path:
        return self.root / "audit"

    @property
    def cache_directory(self) -> Path:
        return self.root / "cache"

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidates"

    @property
    def candidate_input_directory(self) -> Path:
        return self.candidate_directory / "input_files"

    @property
    def candidate_summary_file(self) -> Path:
        return self.candidate_directory / "canonical_candidate_summary.csv"

    def continuation_input_directory(self, target_year: int) -> Path:
        return self.candidate_directory / "continuations" / str(int(target_year)) / "input_files"

    def continuation_manifest_file(self, target_year: int) -> Path:
        return self.continuation_input_directory(target_year).parent / "continuation_manifest.csv"

    @property
    def figure_directory(self) -> Path:
        return self.root / "figures"

    @property
    def optuna_directory(self) -> Path:
        return self.root / "optuna"

    @property
    def provenance_directory(self) -> Path:
        return self.root / "provenance"

    @property
    def quality_directory(self) -> Path:
        return self.root / "quality_control"

    @property
    def run_map_directory(self) -> Path:
        return self.root / "run_maps"

    @property
    def component_metrics_file(self) -> Path:
        return self.audit_directory / "canonical_component_metrics.csv"

    @property
    def objective_file(self) -> Path:
        return self.audit_directory / "canonical_run_objectives.csv"

    @property
    def linked_year_components_file(self) -> Path:
        return self.audit_directory / "canonical_linked_year_pooled_components.csv"

    @property
    def linked_year_comparisons_file(self) -> Path:
        return self.audit_directory / "canonical_linked_year_comparisons.csv"

    @property
    def archive_file(self) -> Path:
        return self.cache_directory / "Model_Run_Summary_Bio_Canonical.csv"

    @property
    def failures_file(self) -> Path:
        return self.quality_directory / "canonical_audit_failures.csv"

    @property
    def quality_issues_file(self) -> Path:
        return self.quality_directory / "canonical_data_quality_issues.csv"

    @property
    def correction_audit_file(self) -> Path:
        return self.quality_directory / "canonical_correction_audit.csv"

    @property
    def study_database(self) -> Path:
        return self.optuna_directory / "model_calibration_bio_canonical_v1.db"

    @property
    def historical_import_file(self) -> Path:
        return self.provenance_directory / "canonical_historical_import.csv"

    @property
    def numeric_support_file(self) -> Path:
        return self.provenance_directory / "canonical_historical_numeric_support.csv"

    @property
    def manifest_file(self) -> Path:
        return self.provenance_directory / "canonical_objective_manifest.json"

    @property
    def trial_provenance_file(self) -> Path:
        return self.provenance_directory / "canonical_trial_provenance.csv"


def ensure_output_layout(paths: CanonicalOutputPaths) -> None:
    for directory in (
        paths.audit_directory,
        paths.cache_directory,
        paths.candidate_directory,
        paths.candidate_input_directory,
        paths.figure_directory,
        paths.optuna_directory,
        paths.provenance_directory,
        paths.quality_directory,
        paths.run_map_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ComponentSpec:
    component: str
    label: str
    source: str
    model_variable: str
    observation_variable: str
    depth: str | None
    weight: float
    error_mode: str
    error_value: float
    include_in_total: bool
    bottom_ph_objective: bool
    required: bool


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"Expected true/false, found {value!r}")
    return text == "true"


def load_component_specs(path: Path | str = DEFAULT_COMPONENTS_FILE) -> tuple[ComponentSpec, ...]:
    table = pd.read_csv(path, keep_default_na=False)
    required_columns = set(ComponentSpec.__dataclass_fields__)
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"Component specification is missing columns: {missing}")
    if table["component"].duplicated().any():
        raise ValueError("Canonical component names must be unique")

    specs: list[ComponentSpec] = []
    for row in table.to_dict("records"):
        depth = str(row["depth"]).strip() or None
        if depth not in {None, "Surface", "Bottom"}:
            raise ValueError(f"Unknown depth {depth!r} for {row['component']}")
        error_mode = str(row["error_mode"]).strip()
        if error_mode not in {"absolute", "mean_fraction"}:
            raise ValueError(f"Unknown observational-error mode {error_mode!r}")
        spec = ComponentSpec(
            component=str(row["component"]).strip(),
            label=str(row["label"]).strip(),
            source=str(row["source"]).strip(),
            model_variable=str(row["model_variable"]).strip(),
            observation_variable=str(row["observation_variable"]).strip(),
            depth=depth,
            weight=float(row["weight"]),
            error_mode=error_mode,
            error_value=float(row["error_value"]),
            include_in_total=_as_bool(row["include_in_total"]),
            bottom_ph_objective=_as_bool(row["bottom_ph_objective"]),
            required=_as_bool(row["required"]),
        )
        if not np.isfinite(spec.weight) or spec.weight < 0:
            raise ValueError(f"Invalid weight for {spec.component}: {spec.weight}")
        if not np.isfinite(spec.error_value) or spec.error_value < 0:
            raise ValueError(
                f"Invalid observational error for {spec.component}: {spec.error_value}"
            )
        specs.append(spec)
    bottom = [spec for spec in specs if spec.bottom_ph_objective]
    if len(bottom) != 1 or not bottom[0].include_in_total:
        raise ValueError("Exactly one total-cost component must define bottom-pH objective two")
    if any(spec.required and not spec.include_in_total for spec in specs):
        raise ValueError("Diagnostic-only components cannot be required objective components")
    return tuple(specs)


def observational_error(spec: ComponentSpec, observation_mean: float) -> float:
    """Return the configured observational-error scale in dimensional units."""

    if spec.error_mode == "absolute":
        return spec.error_value
    return abs(observation_mean) * spec.error_value


def parse_tracer_advection(text: Any) -> dict[str, tuple[str, str]]:
    """Parse ROMS NLM_TADV text into tracer-specific horizontal/vertical pairs."""

    pairs: dict[str, tuple[str, str]] = {}
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*([^:\s]+):\s+(\S+)\s+(\S+)\s*$", line)
        if match and match.group(1).upper() != "ADVECTION":
            pairs[match.group(1)] = (match.group(2), match.group(3))
    return pairs


def advection_metadata(dataset: Any) -> dict[str, Any]:
    """Return actual tracer schemes and canonical-policy compliance flags."""

    pairs = parse_tracer_advection(getattr(dataset, "NLM_TADV", ""))
    result: dict[str, Any] = {
        f"Advection {tracer}": "_".join(pairs[tracer]) if tracer in pairs else np.nan
        for tracer in (*EXPECTED_PHYSICAL_TRACERS, *EXPECTED_BIOLOGICAL_TRACERS)
    }
    physical_complete = all(tracer in pairs for tracer in EXPECTED_PHYSICAL_TRACERS)
    biological_complete = all(tracer in pairs for tracer in EXPECTED_BIOLOGICAL_TRACERS)
    physical_ok = physical_complete and all(
        pairs[tracer] == CANONICAL_ADVECTION_PAIR for tracer in EXPECTED_PHYSICAL_TRACERS
    )
    biological_ok = biological_complete and all(
        pairs[tracer] == CANONICAL_ADVECTION_PAIR for tracer in EXPECTED_BIOLOGICAL_TRACERS
    )
    result.update(
        {
            "Physical Advection Metadata Complete": physical_complete,
            "Biological Advection Metadata Complete": biological_complete,
            "Physical Advection Policy Compliant": physical_ok,
            "Biological Advection Policy Compliant": biological_ok,
            "All Tracer Advection Policy Compliant": physical_ok and biological_ok,
            "Advection Configuration": json.dumps(pairs, sort_keys=True),
        }
    )
    return result


def cpp_metadata(dataset: Any) -> dict[str, Any]:
    """Extract CPP states and conditional CACO3 values without xarray."""

    states = cpu.get_cpp_option_states(dataset, CPP_OPTION_NAMES)
    result: dict[str, Any] = dict(states)
    for parent, variables in CPP_CONDITIONAL_VARIABLES.items():
        for variable_name, output_names in variables.items():
            if not states[parent]:
                result.update({name: np.nan for name in output_names})
                continue
            if variable_name not in dataset.variables:
                raise ValueError(f"{parent} is enabled but {variable_name} is missing")
            values = np.ma.asarray(dataset.variables[variable_name][...], dtype=float).filled(np.nan).reshape(-1)
            if len(values) != len(output_names) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {parent} values for {variable_name}")
            result.update({name: float(value) for name, value in zip(output_names, values)})
    result["CPP Policy Valid"] = not bool(states["USE_MOCSY"]) or bool(states["CACO3"])
    return result


def calculate_component(
    model: np.ndarray,
    observation: np.ndarray,
    spec: ComponentSpec,
) -> dict[str, Any]:
    """Calculate one Ward normalized-MSE component using ddof=0.

    Every finite observation must have a finite model value.  This prevents a
    model from improving an objective merely by dropping difficult observed
    points.  Non-finite observations are outside the comparison population.
    """

    model_values = np.ma.asarray(model, dtype=float).filled(np.nan)
    observation_values = np.ma.asarray(observation, dtype=float).filled(np.nan)
    base: dict[str, Any] = {
        "status": "calculated",
        "observation_count": 0,
        "paired_count": 0,
        "model_coverage": np.nan,
        "observation_mean": np.nan,
        "observation_std_ddof0": np.nan,
        "observational_error": np.nan,
        "normalization_variance": np.nan,
        "bias": np.nan,
        "rmse": np.nan,
        "normalized_rmse": np.nan,
        "normalized_mse": np.nan,
        "weight": spec.weight,
        "squared_weight": spec.weight**2,
        "weighted_contribution": np.nan,
    }
    if model_values.shape != observation_values.shape:
        base["status"] = "shape_mismatch"
        return base

    observed = np.isfinite(observation_values)
    observation_count = int(observed.sum())
    paired = observed & np.isfinite(model_values)
    paired_count = int(paired.sum())
    coverage = paired_count / observation_count if observation_count else np.nan
    base.update(
        observation_count=observation_count,
        paired_count=paired_count,
        model_coverage=coverage,
    )
    if observation_count == 0:
        base["status"] = "no_finite_observations"
        return base
    if paired_count != observation_count:
        base["status"] = "incomplete_model_coverage"
        return base

    paired_model = model_values[paired]
    paired_observation = observation_values[paired]
    mean_observation = float(np.mean(paired_observation))
    std_observation = float(np.std(paired_observation, ddof=0))
    error = float(observational_error(spec, mean_observation))
    denominator = std_observation**2 + error**2
    residual = paired_model - paired_observation
    bias = float(np.mean(residual))
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    base.update(
        observation_mean=mean_observation,
        observation_std_ddof0=std_observation,
        observational_error=error,
        normalization_variance=denominator,
        bias=bias,
        rmse=rmse,
    )
    if not np.isfinite(denominator) or denominator <= 0:
        base["status"] = "nonpositive_normalization_variance"
        return base

    normalized_mse = mse / denominator
    base.update(
        normalized_rmse=float(np.sqrt(normalized_mse)),
        normalized_mse=float(normalized_mse),
        weighted_contribution=float(normalized_mse * spec.weight**2),
    )
    return base


def _variable_array(dataset: Any, variable_name: str, depth: str | None) -> np.ndarray:
    if variable_name not in dataset.variables:
        raise KeyError(variable_name)
    variable = dataset.variables[variable_name]
    values = np.ma.asarray(variable[...], dtype=float).filled(np.nan)
    if depth is None:
        return values
    if "Depth" not in variable.dimensions:
        raise ValueError(f"{variable_name} has no Depth dimension")
    depth_axis = variable.dimensions.index("Depth")
    depth_index = 0 if depth == "Surface" else 1
    if values.shape[depth_axis] <= depth_index:
        raise ValueError(f"{variable_name} has no {depth} level")
    return np.take(values, depth_index, axis=depth_axis)


def evaluate_cost_file(
    cost_file: Path | str,
    specs: Sequence[ComponentSpec],
    *,
    run_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one annual cost-summary file and return components/objectives."""

    from netCDF4 import Dataset

    records: list[dict[str, Any]] = []
    with Dataset(Path(cost_file), mode="r") as dataset:
        for spec in specs:
            try:
                model = _variable_array(dataset, spec.model_variable, spec.depth)
                observation = _variable_array(
                    dataset, spec.observation_variable, spec.depth
                )
                result = calculate_component(model, observation, spec)
            except (KeyError, ValueError, IndexError) as error:
                result = calculate_component(np.array([]), np.array([]), spec)
                result["status"] = f"unavailable: {error}"
            records.append(
                {
                    "Run Name": run_name,
                    **asdict(spec),
                    **result,
                }
            )

    components = pd.DataFrame(records)
    required = components.loc[components["required"]]
    required_valid = required["status"].eq("calculated").all()
    included = components.loc[components["include_in_total"]]
    total = (
        float(included["weighted_contribution"].sum())
        if required_valid and included["weighted_contribution"].notna().all()
        else np.nan
    )
    weight_sum = float(included["squared_weight"].sum())
    bottom = components.loc[components["bottom_ph_objective"]]
    bottom_objective = (
        float(bottom.iloc[0]["normalized_rmse"])
        if len(bottom) == 1 and bottom.iloc[0]["status"] == "calculated"
        else np.nan
    )
    summary = {
        "Run Name": run_name,
        "Cost File": str(cost_file),
        OBJECTIVE_COLUMNS[0]: total,
        OBJECTIVE_COLUMNS[1]: bottom_objective,
        "Canonical Weighted Mean Diagnostic": total / weight_sum
        if np.isfinite(total) and weight_sum > 0
        else np.nan,
        "Required Components": int(len(required)),
        "Valid Required Components": int(required["status"].eq("calculated").sum()),
        "Objective Status": "valid" if np.isfinite(total) and np.isfinite(bottom_objective) else "invalid",
        "Objective Version": OBJECTIVE_VERSION,
    }
    return components, summary


def audit_inventory(
    inventory: pd.DataFrame,
    *,
    cost_root: Path | str,
    specs: Sequence[ComponentSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit every inventory row with a completed, readable cost file."""

    required = {"Run Name", "Cost File"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Run inventory is missing columns: {missing}")
    root = Path(cost_root)
    component_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    normalized = (
        cpu.normalize_run_inventory_names(inventory.copy())
        if "Station File" in inventory.columns
        else inventory.copy()
    )
    for _, row in normalized.dropna(subset=["Run Name", "Cost File"]).iterrows():
        run_name = str(row["Run Name"])
        filename = str(row["Cost File"])
        path = root / filename
        if not path.is_file():
            failures.append(
                {"Run Name": run_name, "Cost File": filename, "Status": "not complete", "Error": "file not found"}
            )
            continue
        try:
            components, summary = evaluate_cost_file(path, specs, run_name=run_name)
            component_tables.append(components)
            summaries.append(summary)
        except Exception as error:
            failures.append(
                {"Run Name": run_name, "Cost File": filename, "Status": "unreadable", "Error": str(error)}
            )
    component_table = (
        pd.concat(component_tables, ignore_index=True)
        if component_tables
        else pd.DataFrame()
    )
    return component_table, pd.DataFrame(summaries), pd.DataFrame(failures)


def refresh_canonical_archive(
    inventory: pd.DataFrame,
    *,
    cost_root: Path | str,
    parameter_names: Sequence[str],
    specs: Sequence[ComponentSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a self-contained canonical archive directly from source NetCDF files."""

    from netCDF4 import Dataset

    required = {"Run Name", "Cost File", "Station File"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Run inventory is missing columns: {missing}")
    root = Path(cost_root)
    normalized = cpu.normalize_run_inventory_names(inventory.copy())
    records: list[dict[str, Any]] = []
    component_tables: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    conditional_input_names = {
        variable_name
        for definitions in CPP_CONDITIONAL_VARIABLES.values()
        for variable_name in definitions
    }
    for _, row in normalized.dropna(subset=["Run Name", "Cost File", "Station File"]).iterrows():
        run_name = str(row["Run Name"])
        cost_name = str(row["Cost File"])
        station_name = str(row["Station File"])
        result: dict[str, Any] = {
            "Run Name": run_name,
            "Cost File": cost_name,
            "Station File": station_name,
            "Objective Version": OBJECTIVE_VERSION,
        }
        cost_path = root / cost_name
        if cost_path.is_file():
            try:
                components, summary = evaluate_cost_file(cost_path, specs, run_name=run_name)
                component_tables.append(components)
                result.update({key: value for key, value in summary.items() if key not in {"Run Name", "Cost File"}})
                for component in components.itertuples(index=False):
                    result[f"Canonical Component: {component.component}"] = component.normalized_mse
                result["Cost Status"] = "Success"
            except Exception as error:
                result["Cost Status"] = f"ERROR: {error}"
                result.setdefault(OBJECTIVE_COLUMNS[0], np.nan)
                result.setdefault(OBJECTIVE_COLUMNS[1], np.nan)
                result["Objective Status"] = "invalid"
                failures.append({"Run Name": run_name, "File Type": "Cost", "File": str(cost_path), "Error": str(error)})
        else:
            result["Cost Status"] = "ERROR: file not found"
            result[OBJECTIVE_COLUMNS[0]] = np.nan
            result[OBJECTIVE_COLUMNS[1]] = np.nan
            result["Objective Status"] = "incomplete"
            failures.append({"Run Name": run_name, "File Type": "Cost", "File": str(cost_path), "Error": "file not found"})

        station_path = root / station_name
        if station_path.is_file():
            try:
                with Dataset(station_path, mode="r") as dataset:
                    result.update(cpp_metadata(dataset))
                    for parameter in parameter_names:
                        if parameter in conditional_input_names:
                            continue
                        result.update(cpu.get_parameter_values(dataset, parameter))
                    result["dstart_year"] = cpu.get_model_year(dataset)
                    run_date, fallback, history = cpu.get_run_date_from_history(dataset)
                    result["Station Run Date"] = run_date
                    result["Station Date Is Fallback"] = fallback
                    result["Station History"] = history
                    nl_tnu2 = cpu.get_parameter_values(dataset, "nl_tnu2")
                    result["nl_tnu2_tracer9"] = nl_tnu2.get("nl_tnu2_tracer9", np.nan)
                    result.update(advection_metadata(dataset))
                result["Parameter Status"] = "Success"
            except Exception as error:
                result["Parameter Status"] = f"ERROR: {error}"
                failures.append({"Run Name": run_name, "File Type": "Station", "File": str(station_path), "Error": str(error)})
        else:
            result["Parameter Status"] = "ERROR: file not found"
            failures.append({"Run Name": run_name, "File Type": "Station", "File": str(station_path), "Error": "file not found"})
        records.append(result)
    archive = pd.DataFrame(records).dropna(axis=1, how="all")
    components = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    return archive, components, pd.DataFrame(failures, columns=["Run Name", "File Type", "File", "Error"])


def load_or_refresh_canonical_cache(
    *,
    mode: str,
    inventory_file: Path,
    inventory_sheet: str,
    parameter_list_file: Path,
    cost_root: Path,
    components_file: Path,
    output_paths: CanonicalOutputPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load or rebuild the independent canonical cache under an explicit mode."""

    valid_modes = {"cache", "refresh", "refresh_and_save"}
    if mode not in valid_modes:
        raise ValueError(f"DATA_LOAD_MODE must be one of {sorted(valid_modes)}")
    if mode == "cache":
        if not output_paths.archive_file.is_file():
            raise FileNotFoundError(
                f"Canonical cache not found: {output_paths.archive_file}. "
                "Use DATA_LOAD_MODE='refresh' or 'refresh_and_save'."
            )
        return pd.read_csv(output_paths.archive_file), pd.DataFrame(), pd.DataFrame(), "canonical cache"

    inventory = pd.read_excel(inventory_file, sheet_name=inventory_sheet)
    parameter_table = pd.read_csv(parameter_list_file)
    if "Parameter Name" not in parameter_table:
        raise ValueError("Parameter list must contain 'Parameter Name'")
    parameter_names = parameter_table["Parameter Name"].dropna().astype(str).str.strip().tolist()
    archive, components, failures = refresh_canonical_archive(
        inventory,
        cost_root=cost_root,
        parameter_names=parameter_names,
        specs=load_component_specs(components_file),
    )
    if mode == "refresh_and_save":
        ensure_output_layout(output_paths)
        cpu.atomic_write_csv(archive, output_paths.archive_file, index=False)
        cpu.atomic_write_csv(components, output_paths.component_metrics_file, index=False)
        cpu.atomic_write_csv(failures, output_paths.failures_file, index=False)
        objectives = archive[[column for column in ["Run Name", "Cost File", *OBJECTIVE_COLUMNS, "Objective Status", "Objective Version"] if column in archive]]
        cpu.atomic_write_csv(objectives, output_paths.objective_file, index=False)
        _atomic_write_json(
            {
                "objective_version": OBJECTIVE_VERSION,
                "objective_columns": list(OBJECTIVE_COLUMNS),
                "cache_is_independent_of_legacy_mo_bio": True,
                "cache_file": str(output_paths.archive_file),
                "ddof": 0,
                "model_coverage_at_finite_observations": 1.0,
                "component_weighting": "normalized_mse_times_weight_squared",
                "bottom_ph_objective": "normalized_rmse_including_bias",
                "chrp_policy": "diagnostic_only",
                "historical_policy": "retain all objective-valid runs regardless of future candidate policy",
                "future_advection_policy": "Upstream3 horizontal and Centered4 vertical for every tracer",
                "components_file": str(components_file),
                "components_sha256": cpu.file_sha256(components_file),
                "inventory_file": str(inventory_file),
                "parameter_list_file": str(parameter_list_file),
                "cost_root": str(cost_root),
                "output_root": str(output_paths.root),
                "generated_utc": cpu.utc_now_text(),
            },
            output_paths.manifest_file,
        )
        description = "fresh source files; canonical cache replaced"
    else:
        description = "fresh source files; canonical cache left unchanged"
    return archive, components, failures, description


def pooled_linked_year_components(
    matched_archive: pd.DataFrame,
    specs: Sequence[ComponentSpec],
    *,
    cost_root: Path | str | None = None,
) -> pd.DataFrame:
    """Pool raw pairs only within complete cross-year parameter matches.

    These rows are diagnostics.  They never replace the annual objectives used
    as Optuna trial values.
    """

    from netCDF4 import Dataset

    required_columns = {
        "Run Name",
        "Cost File",
        "dstart_year",
        "Parameter Match Group",
        "Parameter Configuration Complete",
        "Cross-Year Parameter Match",
    }
    missing = sorted(required_columns - set(matched_archive.columns))
    if missing:
        return pd.DataFrame()
    selected = matched_archive.loc[
        matched_archive["Parameter Configuration Complete"]
        & matched_archive["Cross-Year Parameter Match"]
    ]
    records: list[dict[str, Any]] = []
    for group_id, members in selected.groupby("Parameter Match Group", sort=True):
        arrays: dict[str, dict[str, list[np.ndarray]]] = {
            spec.component: {"model": [], "observation": []} for spec in specs
        }
        errors: dict[str, list[str]] = {spec.component: [] for spec in specs}
        for _, member in members.iterrows():
            path = Path(str(member["Cost File"]))
            if cost_root is not None and not path.is_absolute():
                path = Path(cost_root) / path
            try:
                with Dataset(path, mode="r") as dataset:
                    for spec in specs:
                        try:
                            arrays[spec.component]["model"].append(
                                _variable_array(dataset, spec.model_variable, spec.depth).ravel()
                            )
                            arrays[spec.component]["observation"].append(
                                _variable_array(
                                    dataset, spec.observation_variable, spec.depth
                                ).ravel()
                            )
                        except (KeyError, ValueError, IndexError) as error:
                            errors[spec.component].append(str(error))
            except OSError as error:
                for spec in specs:
                    errors[spec.component].append(str(error))
        run_names = "|".join(members["Run Name"].astype(str))
        years = "|".join(
            map(
                str,
                sorted(
                    pd.to_numeric(members["dstart_year"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                ),
            )
        )
        for spec in specs:
            values = arrays[spec.component]
            if errors[spec.component] or not values["model"]:
                result = calculate_component(np.array([]), np.array([]), spec)
                result["status"] = "unavailable: " + "; ".join(errors[spec.component])
            else:
                result = calculate_component(
                    np.concatenate(values["model"]),
                    np.concatenate(values["observation"]),
                    spec,
                )
            records.append(
                {
                    "Parameter Match Group": group_id,
                    "Runs": run_names,
                    "Years": years,
                    "Annual Run Count": len(members),
                    **asdict(spec),
                    **result,
                }
            )
    return pd.DataFrame(records)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_audit(
    *,
    inventory_file: Path,
    inventory_sheet: str,
    cost_root: Path,
    legacy_cache: Path,
    priors_file: Path,
    components_file: Path,
    output_paths: CanonicalOutputPaths,
) -> pd.DataFrame:
    """Run and persist the read-only canonical audit products."""

    ensure_output_layout(output_paths)
    specs = load_component_specs(components_file)
    inventory = pd.read_excel(inventory_file, sheet_name=inventory_sheet)
    components, objectives, failures = audit_inventory(
        inventory, cost_root=cost_root, specs=specs
    )
    archive = objectives.copy()
    if legacy_cache.is_file():
        parameters = pd.read_csv(legacy_cache)
        parameters = cpu.normalize_run_inventory_names(parameters)
        archive = objectives.merge(
            parameters.drop_duplicates("Run Name", keep="last"),
            on="Run Name",
            how="left",
            suffixes=("", " Legacy"),
            validate="one_to_one",
        )
    linked_components = pd.DataFrame()
    linked_comparisons = pd.DataFrame()
    if priors_file.is_file() and "dstart_year" in archive.columns:
        priors = cpu.load_parameter_priors(priors_file)
        archive = cpu.add_biological_parameter_match_groups(archive, priors)
        linked_components = pooled_linked_year_components(archive, specs)
        linked_comparisons = cpu.matched_year_comparisons(
            archive.loc[archive["Objective Status"].eq("valid")],
            objective_columns=OBJECTIVE_COLUMNS,
        )
    cpu.atomic_write_csv(components, output_paths.component_metrics_file, index=False)
    cpu.atomic_write_csv(objectives, output_paths.objective_file, index=False)
    cpu.atomic_write_csv(archive, output_paths.archive_file, index=False)
    cpu.atomic_write_csv(
        linked_components, output_paths.linked_year_components_file, index=False
    )
    cpu.atomic_write_csv(
        linked_comparisons, output_paths.linked_year_comparisons_file, index=False
    )
    cpu.atomic_write_csv(failures, output_paths.failures_file, index=False)
    manifest = {
        "objective_version": OBJECTIVE_VERSION,
        "objective_columns": list(OBJECTIVE_COLUMNS),
        "annual_objectives": True,
        "ddof": 0,
        "model_coverage_at_finite_observations": 1.0,
        "component_weighting": "normalized_mse_times_weight_squared",
        "bottom_ph_objective": "normalized_rmse_including_bias",
        "chrp_policy": "diagnostic_only",
        "year_policy": "annual objectives; pooled pairs only for linked-year diagnostics",
        "components_file": str(components_file),
        "components_sha256": cpu.file_sha256(components_file),
        "inventory_file": str(inventory_file),
        "cost_root": str(cost_root),
        "legacy_cache_read_only": str(legacy_cache),
        "output_root": str(output_paths.root),
        "generated_utc": cpu.utc_now_text(),
    }
    _atomic_write_json(manifest, output_paths.manifest_file)
    return archive


def canonical_search_space(
    priors_file: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], dict[str, Sequence[Any]]]:
    """Return the shared active parameter decisions used by canonical modes."""

    priors = cpu.load_parameter_priors(priors_file)
    active = priors.loc[priors["decision"].eq("active")]
    numeric = active.loc[active["parameter_type"].isin(["float", "integer"])]
    categorical = active.loc[active["parameter_type"].isin(["categorical", "boolean"])]
    numeric_bounds = {
        row.parameter: (float(row.proposed_low), float(row.proposed_high))
        for row in numeric.itertuples(index=False)
    }
    categorical_choices = {
        row.parameter: cpu.parse_prior_choices(row)
        for row in categorical.itertuples(index=False)
    }
    return priors, numeric_bounds, categorical_choices


def canonical_study_fingerprint(
    numeric_bounds: Mapping[str, Sequence[float]],
    categorical_choices: Mapping[str, Sequence[Any]],
    *,
    components_file: Path,
    seed: int,
) -> str:
    sampler_settings = {
        "seed": seed,
        "n_startup_trials": 10,
        "multivariate": True,
        "components_sha256": cpu.file_sha256(components_file),
    }
    return cpu.search_space_fingerprint(
        OBJECTIVE_COLUMNS,
        numeric_bounds,
        categorical_choices,
        sampler_settings=sampler_settings,
    )


def create_or_load_canonical_study(
    *,
    output_paths: CanonicalOutputPaths,
    priors_file: Path,
    components_file: Path,
    study_name: str = DEFAULT_STUDY_NAME,
    seed: int = 42,
) -> tuple[Any, pd.DataFrame, dict[str, tuple[float, float]], dict[str, Sequence[Any]], str]:
    """Load the isolated canonical study with full compatibility checks."""

    try:
        import optuna  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Canonical study modes require the project's Python 3.11+ "
            "environment with Optuna installed."
        ) from error
    priors, numeric_bounds, categorical_choices = canonical_search_space(priors_file)
    if not numeric_bounds and not categorical_choices:
        raise ValueError(
            "No canonical parameters are active. Complete the review decisions "
            "before creating or loading a study."
        )
    fingerprint = canonical_study_fingerprint(
        numeric_bounds,
        categorical_choices,
        components_file=components_file,
        seed=seed,
    )
    study = cpu.create_or_load_multiobjective_study(
        study_name=study_name,
        storage=f"sqlite:///{output_paths.study_database}",
        objective_version=OBJECTIVE_VERSION,
        fingerprint=fingerprint,
        seed=seed,
        n_startup_trials=10,
        multivariate=True,
        historical_numeric_import_policy="observed_support",
    )
    return study, priors, numeric_bounds, categorical_choices, fingerprint


def prepare_study(
    archive: pd.DataFrame,
    *,
    output_paths: CanonicalOutputPaths,
    priors_file: Path,
    components_file: Path,
    corrections_file: Path | None = None,
    study_name: str = DEFAULT_STUDY_NAME,
    seed: int = 42,
) -> pd.DataFrame:
    """Create/update the isolated study with completed canonical trials only."""

    study, priors, numeric_bounds, categorical_choices, _ = (
        create_or_load_canonical_study(
            output_paths=output_paths,
            priors_file=priors_file,
            components_file=components_file,
            study_name=study_name,
            seed=seed,
        )
    )
    corrected = archive.copy()
    if corrections_file is not None:
        issues = cpu.detect_data_quality_issues(corrected, priors)
        corrections = (
            cpu.load_corrections(corrections_file)
            if corrections_file.is_file()
            else pd.DataFrame(columns=cpu.CORRECTION_COLUMNS)
        )
        corrected, issue_table, correction_audit = cpu.apply_corrections(
            corrected, issues, corrections
        )
        cpu.atomic_write_csv(
            issue_table, output_paths.quality_issues_file, index=False
        )
        cpu.atomic_write_csv(
            correction_audit, output_paths.correction_audit_file, index=False
        )
    valid = corrected.loc[corrected["Objective Status"].eq("valid")].copy()
    support = cpu.historical_numeric_support_table(valid, numeric_bounds)
    imported = cpu.import_historical_multiobjective_trials(
        study,
        valid,
        numeric_bounds=numeric_bounds,
        categorical_choices=categorical_choices,
        objective_columns=OBJECTIVE_COLUMNS,
        objective_version=OBJECTIVE_VERSION,
        numeric_import_policy="observed_support",
    )
    study.set_user_attr("Historical Import Completed", True)
    study.set_user_attr("Historical Import Last Updated UTC", cpu.utc_now_text())
    study.set_user_attr("Canonical Output Root", str(output_paths.root))
    cpu.atomic_write_csv(imported, output_paths.historical_import_file, index=False)
    cpu.atomic_write_csv(support, output_paths.numeric_support_file, index=False)
    return imported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "prepare-study"), nargs="?", default="audit"
    )
    parser.add_argument(
        "--inventory-file",
        type=Path,
        default=DEFAULT_OPTIMIZATION_ROOT / "FileNames.xlsx",
    )
    parser.add_argument("--inventory-sheet", default="Master")
    parser.add_argument("--cost-root", type=Path, default=DEFAULT_COST_ROOT)
    parser.add_argument(
        "--parameter-list",
        type=Path,
        default=DEFAULT_OPTIMIZATION_ROOT / "ParameterList copy.csv",
    )
    parser.add_argument(
        "--data-load-mode", choices=("cache", "refresh", "refresh_and_save"), default=None
    )
    parser.add_argument("--components-file", type=Path, default=DEFAULT_COMPONENTS_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--priors-file",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "canonical_parameter_priors.csv",
    )
    parser.add_argument(
        "--match-priors-file",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "parameter_priors.csv",
    )
    parser.add_argument(
        "--corrections-file",
        type=Path,
        default=DEFAULT_OPTIMIZATION_ROOT / "MO-BIO" / "quality_control" / "data_corrections.csv",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = CanonicalOutputPaths(args.output_root)
    mode = args.data_load_mode or ("refresh_and_save" if args.command == "audit" else "cache")
    archive, components, failures, description = load_or_refresh_canonical_cache(
        mode=mode,
        inventory_file=args.inventory_file,
        inventory_sheet=args.inventory_sheet,
        parameter_list_file=args.parameter_list,
        cost_root=args.cost_root,
        components_file=args.components_file,
        output_paths=paths,
    )
    if args.match_priors_file.is_file():
        archive = cpu.add_biological_parameter_match_groups(
            archive, cpu.load_parameter_priors(args.match_priors_file)
        )
        linked = cpu.matched_year_comparisons(
            archive.loc[archive["Objective Status"].eq("valid")],
            objective_columns=OBJECTIVE_COLUMNS,
        )
        if mode == "cache" and paths.linked_year_components_file.is_file():
            pooled = pd.read_csv(paths.linked_year_components_file)
        else:
            pooled = pooled_linked_year_components(
                archive, load_component_specs(args.components_file), cost_root=args.cost_root
            )
        if mode == "refresh_and_save":
            cpu.atomic_write_csv(archive, paths.archive_file, index=False)
            cpu.atomic_write_csv(linked, paths.linked_year_comparisons_file, index=False)
            cpu.atomic_write_csv(pooled, paths.linked_year_components_file, index=False)
    print(f"Canonical audit: {paths.root}")
    print(f"Data source: {description}")
    print(
        f"Valid objectives: {int(archive['Objective Status'].eq('valid').sum())} "
        f"of {len(archive)} inventory rows"
    )
    if args.command == "prepare-study":
        imported = prepare_study(
            archive,
            output_paths=paths,
            priors_file=args.priors_file,
            components_file=args.components_file,
            corrections_file=args.corrections_file,
            study_name=args.study_name,
            seed=args.seed,
        )
        counts = imported["Import Status"].value_counts().to_dict()
        print(f"Canonical study: {paths.study_database}")
        print(f"Historical import: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
