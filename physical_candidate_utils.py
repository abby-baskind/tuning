"""Stable helpers for the PHYS constrained physical-tuning workflow.

The notebook is intentionally kept focused on scientific decisions.  Parsing,
validation, provenance, candidate construction, and output-table mechanics live
here because they should change less often than the editable scientific rules in
``physical_tuning_spec.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset


RUN_DATE_COLUMN = "Run Date"
ADVECTION_COLUMNS = [
    "temp_horizontal_advection",
    "temp_vertical_advection",
    "salt_horizontal_advection",
    "salt_vertical_advection",
]
COST_COMPONENT_COLUMNS = [
    "Surface Temperature Cost",
    "Bottom Temperature Cost",
    "Surface Salinity Cost",
    "Bottom Salinity Cost",
]


@dataclass(frozen=True)
class WorkflowPaths:
    """Resolved PHYS input and output paths."""

    cost_directory: Path
    file_list: Path
    parameter_list: Path
    output_root: Path

    @property
    def audit_directory(self) -> Path:
        return self.output_root / "audit"

    @property
    def candidate_directory(self) -> Path:
        return self.output_root / "candidates"

    @property
    def provenance_directory(self) -> Path:
        return self.output_root / "provenance"

    @property
    def quality_directory(self) -> Path:
        return self.output_root / "quality_control"

    @property
    def figure_directory(self) -> Path:
        return self.output_root / "figures"

    @property
    def cache_directory(self) -> Path:
        return self.output_root / "cache"

    @property
    def optuna_directory(self) -> Path:
        return self.output_root / "optuna"

    @property
    def corrections_file(self) -> Path:
        return self.quality_directory / "physical_data_corrections.csv"

    @property
    def issues_file(self) -> Path:
        return self.quality_directory / "physical_data_quality_issues.csv"

    @property
    def run_map_file(self) -> Path:
        return self.provenance_directory / "physical_candidate_run_map.csv"

    @property
    def cache_file(self) -> Path:
        return self.cache_directory / "physical_archive_cache.pkl"

    @property
    def cache_metadata_file(self) -> Path:
        return self.cache_directory / "physical_archive_cache_metadata.json"

    @property
    def study_database(self) -> Path:
        return self.optuna_directory / "physical_tuning.db"


def ensure_output_layout(paths: WorkflowPaths) -> None:
    """Create the configured output tree and safe, empty user-editable files."""

    for directory in (
        paths.audit_directory,
        paths.candidate_directory,
        paths.provenance_directory,
        paths.quality_directory,
        paths.figure_directory,
        paths.cache_directory,
        paths.optuna_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not paths.corrections_file.exists():
        pd.DataFrame(
            columns=[
                "run_name",
                "field",
                "corrected_value",
                "reason",
                "verified_by",
                "verified_date",
            ]
        ).to_csv(paths.corrections_file, index=False)

    if not paths.run_map_file.exists():
        pd.DataFrame(
            columns=[
                "candidate_id",
                "optuna_trial_number",
                "actual_run_name",
                "mo_bio_study_name",
                "mo_bio_trial_number",
                "status",
                "station_file",
                "completion_date",
                "notes",
            ]
        ).to_csv(paths.run_map_file, index=False)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_physical_spec(path: Path) -> dict[str, Any]:
    """Load and strictly validate the human-edited PHYS TOML specification."""

    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    validate_physical_spec(spec)
    spec["_spec_path"] = str(path.resolve())
    spec["_spec_sha256"] = file_sha256(path)
    return spec


def validate_physical_spec(spec: Mapping[str, Any]) -> None:
    required_sections = {
        "workflow",
        "numeric_policy",
        "nl_tnu2",
        "advection",
        "mixing_orientation",
        "mass_conservation_bundle",
        "optional_cpp_options",
    }
    missing = sorted(required_sections.difference(spec))
    if missing:
        raise ValueError(f"Specification is missing sections: {missing}")

    mode = spec["workflow"].get("candidate_space_mode")
    if mode not in {"component_factorial", "observed_configurations_only"}:
        raise ValueError(
            "workflow.candidate_space_mode must be 'component_factorial' or "
            "'observed_configurations_only'."
        )
    strategy = spec["workflow"].get("candidate_strategy")
    if strategy not in {"factorial", "tpe"}:
        raise ValueError("workflow.candidate_strategy must be 'factorial' or 'tpe'.")

    members = spec["nl_tnu2"].get("members", [])
    if len(members) != 2 or len(set(members)) != 2:
        raise ValueError("nl_tnu2.members must contain the two distinct tracer fields.")
    values = spec["nl_tnu2"].get("candidate_values", [])
    if not values or len(values) != len(set(float(value) for value in values)):
        raise ValueError("nl_tnu2.candidate_values must be nonempty and unique.")

    schemes = spec["advection"].get("schemes", [])
    labels = [item.get("name") for item in schemes]
    if not schemes or any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("Every advection scheme needs a unique nonempty name.")
    for item in schemes:
        if not item.get("horizontal") or not item.get("vertical"):
            raise ValueError(f"Incomplete advection scheme: {item}")

    orientation = spec["mixing_orientation"]
    uv_group = orientation.get("uv_group", [])
    ts_group = orientation.get("ts_group", [])
    if len(uv_group) != 3 or len(ts_group) != 3:
        raise ValueError("The UV and TS orientation groups must each contain 3 options.")
    for pair in orientation.get("approved_pairs", []):
        if pair.get("uv") not in uv_group or pair.get("ts") not in ts_group:
            raise ValueError(f"Approved orientation pair uses an unknown option: {pair}")

    required = spec["mass_conservation_bundle"].get("required_true", [])
    if len(required) != len(set(required)) or not required:
        raise ValueError("mass_conservation_bundle.required_true must be nonempty and unique.")

    tracked = spec["optional_cpp_options"].get("tracked", [])
    if len(tracked) != len(set(tracked)):
        raise ValueError("optional_cpp_options.tracked contains duplicates.")
    combination_names: list[str] = []
    for combination in spec["optional_cpp_options"].get("approved_combinations", []):
        name = combination.get("name")
        enabled = combination.get("enabled", [])
        if not name or name in combination_names:
            raise ValueError("Each optional CPP combination needs a unique name.")
        if not set(enabled).issubset(tracked):
            raise ValueError(f"Optional combination {name!r} contains an untracked option.")
        combination_names.append(name)


def candidate_generation_blockers(spec: Mapping[str, Any]) -> list[str]:
    """Return unresolved review items that must block candidate creation."""

    blockers = []
    for section in ("mixing_orientation", "optional_cpp_options"):
        status = str(spec[section].get("review_status", "pending")).lower()
        if status != "approved":
            blockers.append(f"{section}.review_status is {status!r}, not 'approved'")
    if not spec["mixing_orientation"].get("approved_pairs"):
        blockers.append("no approved mixing-orientation pairs")
    if not spec["optional_cpp_options"].get("approved_combinations"):
        blockers.append("no approved optional CPP combinations")
    if not [item for item in spec["advection"]["schemes"] if item.get("candidate_allowed")]:
        blockers.append("no candidate-allowed advection schemes")
    return blockers


def scientific_spec_fingerprint(spec: Mapping[str, Any]) -> str:
    """Hash parsed scientific content; comments and file location do not matter."""

    content = {key: value for key, value in spec.items() if not key.startswith("_")}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def parse_cpp_options(text: Any) -> tuple[set[str], set[str]]:
    """Return enabled and explicitly disabled CPP tokens.

    Tokenization tolerates missing commas in old provenance strings.  A leading
    exclamation mark means explicitly disabled and is never counted as enabled.
    """

    tokens = re.findall(r"!?[A-Za-z][A-Za-z0-9_]*", str(text or ""))
    enabled = {token for token in tokens if not token.startswith("!")}
    disabled = {token[1:] for token in tokens if token.startswith("!")}
    return enabled, disabled


def extract_advection(text: Any) -> dict[str, Any]:
    output = {column: np.nan for column in ADVECTION_COLUMNS}
    for tracer, prefix in (("temp", "temp"), ("salt", "salt")):
        match = re.search(
            rf"^\s*{tracer}:\s+(\S+)\s+(\S+)", str(text or ""), re.MULTILINE
        )
        if match:
            output[f"{prefix}_horizontal_advection"] = match.group(1)
            output[f"{prefix}_vertical_advection"] = match.group(2)
    return output


def identify_advection_scheme(
    values: Mapping[str, Any], spec: Mapping[str, Any]
) -> Any:
    if any(pd.isna(values.get(column)) for column in ADVECTION_COLUMNS):
        return np.nan
    for scheme in spec["advection"]["schemes"]:
        if (
            values["temp_horizontal_advection"] == scheme["horizontal"]
            and values["salt_horizontal_advection"] == scheme["horizontal"]
            and values["temp_vertical_advection"] == scheme["vertical"]
            and values["salt_vertical_advection"] == scheme["vertical"]
        ):
            return scheme["name"]
    return np.nan


def parse_history_datetime(text: Any) -> pd.Timestamp:
    """Parse the completion timestamp recorded in a station-file history string."""

    match = re.search(
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},\s+\d{4})"
        r"(?:\s*-\s*(\d{1,2}:\d{2}:\d{2}\s*[AP]M))?",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return pd.NaT
    value = match.group(1)
    if match.group(2):
        value += " " + match.group(2)
    return pd.to_datetime(value, errors="coerce")


def _as_python_scalar(value: Any) -> Any:
    if np.ma.is_masked(value):
        return np.nan
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value


def extract_parameter_values(ds: Dataset, parameter_name: str) -> dict[str, Any]:
    if parameter_name not in ds.variables:
        return {}
    variable = ds.variables[parameter_name]
    values = np.asarray(np.ma.filled(variable[...], np.nan)).squeeze()
    if values.ndim == 0:
        return {parameter_name: values.item()}
    if values.ndim == 1:
        dimension = variable.dimensions[0] if variable.dimensions else "index"
        return {
            f"{parameter_name}_{dimension}{index}": _as_python_scalar(value)
            for index, value in enumerate(values)
        }
    return {
        f"{parameter_name}_{index}": _as_python_scalar(value)
        for index, value in enumerate(values.ravel())
    }


def _issue(
    records: list[dict[str, Any]],
    run_name: str,
    category: str,
    field: str,
    observed_value: Any,
    details: str,
) -> None:
    records.append(
        {
            "run_name": run_name,
            "category": category,
            "field": field,
            "observed_value": observed_value,
            "details": details,
        }
    )


def _cost_values(cost_path: Path) -> dict[str, float]:
    with Dataset(cost_path) as ds:
        temperature = np.asarray(ds.variables["temperature_cost"][:]).squeeze()
        salinity = np.asarray(ds.variables["salt_cost"][:]).squeeze()
        if temperature.size < 2 or salinity.size < 2:
            raise ValueError("temperature_cost and salt_cost must each contain two depths")
        values = {
            "Surface Temperature Cost": float(temperature[0]),
            "Bottom Temperature Cost": float(temperature[1]),
            "Surface Salinity Cost": float(salinity[0]),
            "Bottom Salinity Cost": float(salinity[1]),
        }
    values["Cost"] = float(sum(values.values()))
    return values


def archive_fingerprint(
    file_list: Path, parameter_list: Path, cost_directory: Path
) -> str:
    """Create a cheap cache fingerprint from manifests and referenced file metadata."""

    digest = sha256()
    for path in (file_list, parameter_list):
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    runs = pd.read_excel(file_list)
    for column in ("Cost File", "Station File"):
        if column not in runs:
            continue
        for name in runs[column].dropna().astype(str):
            path = cost_directory / name
            if path.exists():
                stat = path.stat()
                digest.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
            else:
                digest.update(f"{name}|MISSING".encode())
    return digest.hexdigest()


def read_physical_archive(
    paths: WorkflowPaths,
    spec: Mapping[str, Any],
    cost_threshold: float = 200.0,
    suspicious_tiny_abs: float = 1e-250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all historical runs while retaining failures in a separate issue table."""

    runs = pd.read_excel(paths.file_list)
    required = ["Run Name", "Cost File", "Station File"]
    missing_columns = sorted(set(required).difference(runs.columns))
    if missing_columns:
        raise ValueError(f"File list is missing columns: {missing_columns}")
    runs = runs[required].dropna(subset=required).copy()
    parameter_names = (
        pd.read_csv(paths.parameter_list)["Variable"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    orientation = spec["mixing_orientation"]
    required_bundle = spec["mass_conservation_bundle"]["required_true"]
    optional = spec["optional_cpp_options"]["tracked"]
    tracked_cpp = list(dict.fromkeys(
        orientation["uv_group"] + orientation["ts_group"] + required_bundle + optional
    ))
    members = spec["nl_tnu2"]["members"]

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for _, manifest_row in runs.iterrows():
        run_name = str(manifest_row["Run Name"]).strip()
        station_path = paths.cost_directory / str(manifest_row["Station File"])
        cost_path = paths.cost_directory / str(manifest_row["Cost File"])
        row: dict[str, Any] = {
            "Run Name": run_name,
            "Cost File": str(manifest_row["Cost File"]),
            "Station File": str(manifest_row["Station File"]),
        }

        try:
            row.update(_cost_values(cost_path))
            row["Cost Status"] = "Success"
            if row["Cost"] >= cost_threshold:
                _issue(
                    issues, run_name, "unreasonably high cost", "Cost", row["Cost"],
                    f"Cost is at or above the configured threshold {cost_threshold}.",
                )
        except Exception as error:
            row["Cost Status"] = "Error"
            row["Cost"] = np.nan
            _issue(issues, run_name, "file read error", "Cost File", cost_path, str(error))

        try:
            with Dataset(station_path) as ds:
                for parameter in parameter_names:
                    try:
                        row.update(extract_parameter_values(ds, parameter))
                    except Exception as error:
                        _issue(
                            issues, run_name, "parameter read error", parameter, np.nan,
                            str(error),
                        )
                row[RUN_DATE_COLUMN] = parse_history_datetime(getattr(ds, "history", ""))
                if pd.isna(row[RUN_DATE_COLUMN]):
                    _issue(
                        issues, run_name, "missing provenance", RUN_DATE_COLUMN, np.nan,
                        "No parseable completion timestamp in station-file history.",
                    )

                raw_cpp = str(getattr(ds, "CPP_options", ""))
                row["CPP_options_raw"] = raw_cpp
                enabled, _ = parse_cpp_options(raw_cpp)
                for option in tracked_cpp:
                    row[option] = option in enabled

                uv_enabled = [name for name in orientation["uv_group"] if name in enabled]
                ts_enabled = [name for name in orientation["ts_group"] if name in enabled]
                row["UV orientation"] = uv_enabled[0] if len(uv_enabled) == 1 else np.nan
                row["TS orientation"] = ts_enabled[0] if len(ts_enabled) == 1 else np.nan
                if len(uv_enabled) != 1:
                    _issue(
                        issues, run_name, "exclusive CPP violation", "UV orientation",
                        "+".join(uv_enabled) if uv_enabled else "<none>",
                        "Expected exactly one enabled UV mixing orientation.",
                    )
                if len(ts_enabled) != 1:
                    _issue(
                        issues, run_name, "exclusive CPP violation", "TS orientation",
                        "+".join(ts_enabled) if ts_enabled else "<none>",
                        "Expected exactly one enabled TS mixing orientation.",
                    )

                advection = extract_advection(getattr(ds, "NLM_TADV", ""))
                row.update(advection)
                row["advection_scheme"] = identify_advection_scheme(advection, spec)
                if any(pd.isna(advection[column]) for column in ADVECTION_COLUMNS):
                    _issue(
                        issues, run_name, "missing parameter", "NLM_TADV", np.nan,
                        "Temperature/salinity advection fields are incomplete.",
                    )
                elif (
                    advection["temp_horizontal_advection"]
                    != advection["salt_horizontal_advection"]
                    or advection["temp_vertical_advection"]
                    != advection["salt_vertical_advection"]
                ):
                    _issue(
                        issues, run_name, "coupling violation", "NLM_TADV",
                        " | ".join(str(advection[column]) for column in ADVECTION_COLUMNS),
                        "Temperature and salinity advection schemes must match.",
                    )
                elif pd.isna(row["advection_scheme"]):
                    _issue(
                        issues, run_name, "unrecognized historical value", "NLM_TADV",
                        f"{advection['temp_horizontal_advection']} / "
                        f"{advection['temp_vertical_advection']}",
                        "Complete pair is not registered in the TOML specification.",
                    )
            row["Parameter Status"] = "Success"
        except Exception as error:
            row["Parameter Status"] = "Error"
            row[RUN_DATE_COLUMN] = pd.NaT
            _issue(
                issues, run_name, "file read error", "Station File", station_path,
                str(error),
            )

        if all(member in row and pd.notna(row[member]) for member in members):
            numeric_members = [float(row[member]) for member in members]
            if spec["nl_tnu2"].get("require_equal", True) and not np.isclose(
                numeric_members[0], numeric_members[1], rtol=0.0, atol=0.0
            ):
                _issue(
                    issues, run_name, "coupling violation", "nl_tnu2",
                    repr(numeric_members), "The two physical tracer values must match.",
                )
                row["nl_tnu2_shared"] = np.nan
            else:
                row["nl_tnu2_shared"] = numeric_members[0]
        else:
            row["nl_tnu2_shared"] = np.nan
            _issue(
                issues, run_name, "missing parameter", "nl_tnu2", np.nan,
                f"One or both required members are missing: {members}.",
            )

        for field, value in list(row.items()):
            if field in {"Cost", *COST_COMPONENT_COLUMNS}:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric = float(value)
                if 0 < abs(numeric) < suspicious_tiny_abs:
                    _issue(
                        issues, run_name, "suspicious tiny value", field, numeric,
                        f"Absolute nonzero value is below {suspicious_tiny_abs:g}.",
                    )
        records.append(row)

    frame = pd.DataFrame(records).dropna(axis=1, how="all")
    issue_frame = pd.DataFrame(
        issues,
        columns=["run_name", "category", "field", "observed_value", "details"],
    )
    return frame, issue_frame


def read_archive_with_cache(
    paths: WorkflowPaths,
    spec: Mapping[str, Any],
    force_refresh: bool = False,
    cost_threshold: float = 200.0,
    suspicious_tiny_abs: float = 1e-250,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read cached archive data when inputs are unchanged; otherwise rebuild it."""

    fingerprint = archive_fingerprint(
        paths.file_list, paths.parameter_list, paths.cost_directory
    )
    if (
        not force_refresh
        and paths.cache_file.exists()
        and paths.cache_metadata_file.exists()
    ):
        try:
            metadata = json.loads(paths.cache_metadata_file.read_text())
            if (
                metadata.get("archive_fingerprint") == fingerprint
                and metadata.get("spec_sha256") == spec["_spec_sha256"]
            ):
                cached = pd.read_pickle(paths.cache_file)
                return cached["archive"], cached["issues"], "cache"
        except Exception:
            pass

    archive, issues = read_physical_archive(
        paths,
        spec,
        cost_threshold=cost_threshold,
        suspicious_tiny_abs=suspicious_tiny_abs,
    )
    pd.to_pickle({"archive": archive, "issues": issues}, paths.cache_file)
    paths.cache_metadata_file.write_text(
        json.dumps(
            {
                "archive_fingerprint": fingerprint,
                "spec_sha256": spec["_spec_sha256"],
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    return archive, issues, "fresh read"


def apply_verified_corrections(
    frame: pd.DataFrame, corrections_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply explicit corrections without changing source station files."""

    corrected = frame.copy()
    if not corrections_file.exists():
        return corrected, pd.DataFrame()
    corrections = pd.read_csv(corrections_file, dtype="string").dropna(
        subset=["run_name", "field", "corrected_value"]
    )
    applied = []
    for index, correction in corrections.iterrows():
        run_name = correction["run_name"].strip()
        field = correction["field"].strip()
        matches = corrected.index[corrected["Run Name"].astype(str).eq(run_name)]
        if len(matches) != 1:
            raise ValueError(
                f"Correction row {index + 2}: run {run_name!r} matches {len(matches)} rows."
            )
        if field not in corrected.columns:
            raise ValueError(f"Correction row {index + 2}: unknown field {field!r}.")
        original = corrected.at[matches[0], field]
        raw = correction["corrected_value"]
        if pd.api.types.is_bool_dtype(corrected[field]):
            lowered = raw.strip().lower()
            if lowered not in {"true", "false"}:
                raise ValueError(
                    f"Correction row {index + 2}: Boolean value must be true or false."
                )
            value: Any = lowered == "true"
        elif pd.api.types.is_numeric_dtype(corrected[field]):
            value = float(raw)
        elif pd.api.types.is_datetime64_any_dtype(corrected[field]):
            value = pd.to_datetime(raw, errors="raise")
        else:
            value = raw
        corrected.at[matches[0], field] = value
        applied.append(
            {
                "correction_row": index + 2,
                "run_name": run_name,
                "field": field,
                "original_value": original,
                "corrected_value": value,
                "reason": correction.get("reason", pd.NA),
            }
        )
    return corrected, pd.DataFrame(applied)


def add_configuration_columns(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    output = frame.copy()
    required = spec["mass_conservation_bundle"]["required_true"]
    output["mass_bundle_complete"] = output.reindex(columns=required).eq(True).all(axis=1)
    allowed_advection = {
        item["name"] for item in spec["advection"]["schemes"]
        if item.get("candidate_allowed", False)
    }
    allowed_tnu2 = {float(value) for value in spec["nl_tnu2"]["candidate_values"]}
    approved_pairs = {
        (item["uv"], item["ts"])
        for item in spec["mixing_orientation"].get("approved_pairs", [])
    }
    optional_tracked = spec["optional_cpp_options"]["tracked"]
    approved_optional = {
        tuple(option in set(item.get("enabled", [])) for option in optional_tracked)
        for item in spec["optional_cpp_options"].get("approved_combinations", [])
    }
    observed_optional = output.reindex(columns=optional_tracked).eq(True).apply(tuple, axis=1)
    output["candidate_compatible"] = (
        output["mass_bundle_complete"]
        & output["advection_scheme"].isin(allowed_advection)
        & output["nl_tnu2_shared"].isin(allowed_tnu2)
        & pd.Series(
            list(zip(output["UV orientation"], output["TS orientation"])),
            index=output.index,
        ).isin(approved_pairs)
        & observed_optional.isin(approved_optional)
    )
    return output


def refresh_derived_configuration_fields(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    """Recompute conceptual fields after verified raw-field corrections."""

    output = frame.copy()
    members = spec["nl_tnu2"]["members"]
    shared = []
    for _, row in output.iterrows():
        values = [pd.to_numeric(row.get(member), errors="coerce") for member in members]
        if all(pd.notna(value) for value in values) and np.isclose(
            float(values[0]), float(values[1]), rtol=0.0, atol=0.0
        ):
            shared.append(float(values[0]))
        else:
            shared.append(np.nan)
    output["nl_tnu2_shared"] = shared

    uv_group = spec["mixing_orientation"]["uv_group"]
    ts_group = spec["mixing_orientation"]["ts_group"]
    output["UV orientation"] = [
        enabled[0] if len(enabled) == 1 else np.nan
        for enabled in (
            [name for name in uv_group if bool(row.get(name, False))]
            for _, row in output.iterrows()
        )
    ]
    output["TS orientation"] = [
        enabled[0] if len(enabled) == 1 else np.nan
        for enabled in (
            [name for name in ts_group if bool(row.get(name, False))]
            for _, row in output.iterrows()
        )
    ]
    output["advection_scheme"] = [
        identify_advection_scheme(row, spec) for _, row in output.iterrows()
    ]
    return add_configuration_columns(output, spec)


def observed_configuration_tables(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    orientation = (
        frame.groupby(["UV orientation", "TS orientation"], dropna=False)
        .agg(Runs=("Run Name", "size"), Median_Cost=("Cost", "median"), Best_Cost=("Cost", "min"))
        .reset_index()
        .sort_values(["Runs", "Median_Cost"], ascending=[False, True])
    )
    optional = spec["optional_cpp_options"]["tracked"]
    optional_table = (
        frame.groupby(optional, dropna=False)
        .agg(Runs=("Run Name", "size"), Median_Cost=("Cost", "median"), Best_Cost=("Cost", "min"))
        .reset_index()
        .sort_values(["Runs", "Median_Cost"], ascending=[False, True])
    )
    advection = (
        frame.groupby(["advection_scheme", *ADVECTION_COLUMNS], dropna=False)
        .agg(Runs=("Run Name", "size"), Median_Cost=("Cost", "median"), Best_Cost=("Cost", "min"))
        .reset_index()
        .sort_values(["Runs", "Median_Cost"], ascending=[False, True])
    )
    return {"orientation": orientation, "optional_cpp": optional_table, "advection": advection}


def suggested_toml_blocks(
    observed: Mapping[str, pd.DataFrame], spec: Mapping[str, Any]
) -> dict[str, str]:
    """Format observed states for copy/review without rewriting the TOML."""

    orientation_lines = []
    for _, row in observed["orientation"].dropna(
        subset=["UV orientation", "TS orientation"]
    ).iterrows():
        uv = row["UV orientation"]
        ts = row["TS orientation"]
        orientation_lines.extend(
            [
                "[[mixing_orientation.approved_pairs]]",
                f'name = "{uv.replace("MIX_", "")}__{ts.replace("MIX_", "")}"',
                f'uv = "{uv}"',
                f'ts = "{ts}"',
                "",
            ]
        )

    tracked = spec["optional_cpp_options"]["tracked"]
    optional_lines = []
    for index, row in observed["optional_cpp"].iterrows():
        if any(pd.isna(row.get(option)) for option in tracked):
            continue
        enabled = [option for option in tracked if bool(row[option])]
        label = "none" if not enabled else "_and_".join(enabled)
        serialized = ", ".join(f'"{option}"' for option in enabled)
        optional_lines.extend(
            [
                "[[optional_cpp_options.approved_combinations]]",
                f'name = "{label}"',
                f"enabled = [{serialized}]",
                "",
            ]
        )
    return {
        "mixing_orientation": "\n".join(orientation_lines).rstrip(),
        "optional_cpp_options": "\n".join(optional_lines).rstrip(),
    }


def varied_numeric_inventory(
    frame: pd.DataFrame, metadata_columns: Iterable[str] = ()
) -> pd.DataFrame:
    excluded = set(metadata_columns) | {"Cost", *COST_COMPONENT_COLUMNS}
    rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        unique = np.sort(values.unique())
        if len(unique) <= 1:
            continue
        rows.append(
            {
                "parameter": column,
                "historical_unique_count": len(unique),
                "historical_values": ", ".join(f"{value:g}" for value in unique),
                "minimum": float(unique.min()),
                "maximum": float(unique.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)


def latest_reference_run(frame: pd.DataFrame) -> pd.Series:
    usable = frame.dropna(subset=[RUN_DATE_COLUMN]).copy()
    if usable.empty:
        raise ValueError("No station file has a parseable completion timestamp.")
    usable["_archive_order"] = np.arange(len(usable))
    return usable.sort_values([RUN_DATE_COLUMN, "_archive_order"]).iloc[-1]


def numeric_differences_from_latest(
    frame: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = latest_reference_run(frame)
    parameters = inventory["parameter"].tolist()
    reference = pd.DataFrame(
        {
            "parameter": parameters,
            "latest_run": latest["Run Name"],
            "latest_completion": latest[RUN_DATE_COLUMN],
            "latest_value": [latest.get(parameter, np.nan) for parameter in parameters],
        }
    )
    rows = []
    for _, run in frame.iterrows():
        differences = []
        for parameter in parameters:
            current = run.get(parameter, np.nan)
            baseline = latest.get(parameter, np.nan)
            equal = (
                pd.isna(current) and pd.isna(baseline)
            ) or (
                pd.notna(current) and pd.notna(baseline)
                and np.isclose(float(current), float(baseline), equal_nan=True)
            )
            if not equal:
                differences.append(
                    {
                        "Run Name": run["Run Name"],
                        RUN_DATE_COLUMN: run.get(RUN_DATE_COLUMN),
                        "Cost": run.get("Cost"),
                        "parameter": parameter,
                        "run_value": current,
                        "latest_value": baseline,
                        "latest_run": latest["Run Name"],
                    }
                )
        rows.extend(differences)
    return reference, pd.DataFrame(rows)


def factor_performance_table(
    frame: pd.DataFrame, factors: Sequence[str], bundle_only: bool = False
) -> pd.DataFrame:
    data = frame.loc[frame["Cost"].notna()].copy()
    if bundle_only:
        data = data.loc[data["mass_bundle_complete"]]
    rows = []
    for factor in factors:
        if factor not in data:
            continue
        grouped = data.groupby(factor, dropna=False)["Cost"]
        for level, costs in grouped:
            rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "scope": "required bundle only" if bundle_only else "all readable runs",
                    "runs": int(costs.count()),
                    "median_cost": costs.median(),
                    "mean_cost": costs.mean(),
                    "best_cost": costs.min(),
                    "worst_cost": costs.max(),
                }
            )
    return pd.DataFrame(rows)


def _approved_factor_levels(spec: Mapping[str, Any]) -> dict[str, list[Any]]:
    schemes = [
        item["name"] for item in spec["advection"]["schemes"]
        if item.get("candidate_allowed", False)
    ]
    orientations = [item["name"] for item in spec["mixing_orientation"]["approved_pairs"]]
    optional = [item["name"] for item in spec["optional_cpp_options"]["approved_combinations"]]
    return {
        "advection_scheme": schemes,
        "nl_tnu2_shared": [float(value) for value in spec["nl_tnu2"]["candidate_values"]],
        "mixing_orientation_pair": orientations,
        "optional_cpp_combination": optional,
    }


def _configuration_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("advection_scheme"),
        float(row.get("nl_tnu2_shared")) if pd.notna(row.get("nl_tnu2_shared")) else np.nan,
        row.get("mixing_orientation_pair"),
        row.get("optional_cpp_combination"),
    )


def label_historical_factors(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    output = frame.copy()
    orientation_lookup = {
        (item["uv"], item["ts"]): item["name"]
        for item in spec["mixing_orientation"]["approved_pairs"]
    }
    output["mixing_orientation_pair"] = [
        orientation_lookup.get((uv, ts), np.nan)
        for uv, ts in zip(output["UV orientation"], output["TS orientation"])
    ]
    tracked = spec["optional_cpp_options"]["tracked"]
    optional_lookup = {
        tuple(option in set(item.get("enabled", [])) for option in tracked): item["name"]
        for item in spec["optional_cpp_options"]["approved_combinations"]
    }
    output["optional_cpp_combination"] = [
        optional_lookup.get(
            tuple(
                pd.notna(row.get(option, np.nan)) and bool(row.get(option))
                for option in tracked
            ),
            np.nan,
        )
        for _, row in output.iterrows()
    ]
    return output


def build_candidate_space(
    historical: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    """Build the constrained component factorial or exact observed candidate space."""

    levels = _approved_factor_levels(spec)
    columns = list(levels)
    factorial = pd.DataFrame(
        [dict(zip(columns, values)) for values in product(*(levels[c] for c in columns))]
    )
    if factorial.empty:
        return factorial

    history = label_historical_factors(
        add_configuration_columns(historical, spec), spec
    )
    observed_keys = {
        _configuration_key(row)
        for _, row in history.loc[history["mass_bundle_complete"]]
        .dropna(subset=columns)
        .iterrows()
    }
    factorial["observed_exactly_in_archive"] = [
        _configuration_key(row) in observed_keys for _, row in factorial.iterrows()
    ]
    if spec["workflow"]["candidate_space_mode"] == "observed_configurations_only":
        factorial = factorial.loc[factorial["observed_exactly_in_archive"]].copy()

    orientation_lookup = {
        item["name"]: item for item in spec["mixing_orientation"]["approved_pairs"]
    }
    optional_lookup = {
        item["name"]: item for item in spec["optional_cpp_options"]["approved_combinations"]
    }
    advection_lookup = {item["name"]: item for item in spec["advection"]["schemes"]}
    required = spec["mass_conservation_bundle"]["required_true"]
    optional_tracked = spec["optional_cpp_options"]["tracked"]
    members = spec["nl_tnu2"]["members"]

    resolved = []
    for index, row in factorial.reset_index(drop=True).iterrows():
        values = row.to_dict()
        adv = advection_lookup[values["advection_scheme"]]
        orientation = orientation_lookup[values["mixing_orientation_pair"]]
        optional = set(optional_lookup[values["optional_cpp_combination"]].get("enabled", []))
        values.update(
            {
                "configuration_index": index + 1,
                "temp_horizontal_advection": adv["horizontal"],
                "temp_vertical_advection": adv["vertical"],
                "salt_horizontal_advection": adv["horizontal"],
                "salt_vertical_advection": adv["vertical"],
                "UV orientation": orientation["uv"],
                "TS orientation": orientation["ts"],
            }
        )
        for member in members:
            values[member] = float(values["nl_tnu2_shared"])
        for option in required:
            values[option] = True
        for option in optional_tracked:
            values[option] = option in optional
        resolved.append(values)
    return pd.DataFrame(resolved)


def candidate_space_summary(candidate_space: pd.DataFrame) -> pd.DataFrame:
    if candidate_space.empty:
        return pd.DataFrame(
            [{"candidate_cells": 0, "observed_cells": 0, "new_cells": 0}]
        )
    observed = int(candidate_space["observed_exactly_in_archive"].sum())
    return pd.DataFrame(
        [{
            "candidate_cells": len(candidate_space),
            "observed_cells": observed,
            "new_cells": len(candidate_space) - observed,
        }]
    )


def attach_candidate_identity(
    candidates: pd.DataFrame,
    trial_numbers: Sequence[int],
    study_name: str,
    strategy: str,
    spec_sha256: str,
    latest_run: str,
) -> pd.DataFrame:
    if len(candidates) != len(trial_numbers):
        raise ValueError("Candidate rows and trial numbers have different lengths.")
    output = candidates.copy().reset_index(drop=True)
    output.insert(0, "candidate_id", [f"PHYS_{number:03d}" for number in trial_numbers])
    output.insert(1, "optuna_trial_number", list(trial_numbers))
    output.insert(2, "study_name", study_name)
    output.insert(3, "strategy", strategy)
    output["latest_reference_run"] = latest_run
    output["spec_sha256"] = spec_sha256
    output["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return output


def build_change_manifest(
    candidates: pd.DataFrame,
    latest: pd.Series,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Describe manual changes; never edits ROMS files."""

    rows = []
    members = spec["nl_tnu2"]["members"]
    required = spec["mass_conservation_bundle"]["required_true"]
    optional = spec["optional_cpp_options"]["tracked"]
    for _, candidate in candidates.iterrows():
        identity = {
            "candidate_id": candidate["candidate_id"],
            "optuna_trial_number": candidate["optuna_trial_number"],
        }
        rows.append(
            {
                **identity,
                "setting_group": "coupled numeric",
                "target_fields": " + ".join(members),
                "reference_value": " + ".join(str(latest.get(field, "")) for field in members),
                "candidate_value": candidate["nl_tnu2_shared"],
                "action": "Set both identically",
            }
        )
        rows.append(
            {
                **identity,
                "setting_group": "coupled advection",
                "target_fields": " + ".join(ADVECTION_COLUMNS),
                "reference_value": latest.get("advection_scheme", ""),
                "candidate_value": (
                    f"{candidate['temp_horizontal_advection']} / "
                    f"{candidate['temp_vertical_advection']}"
                ),
                "action": "Set both tracers identically",
            }
        )
        for option in (
            spec["mixing_orientation"]["uv_group"]
            + spec["mixing_orientation"]["ts_group"]
            + required
            + optional
        ):
            desired = bool(candidate.get(option, False))
            rows.append(
                {
                    **identity,
                    "setting_group": "CPP option",
                    "target_fields": option,
                    "reference_value": latest.get(option, np.nan),
                    "candidate_value": desired,
                    "action": "Enable" if desired else "Disable",
                }
            )
    return pd.DataFrame(rows)


def update_run_map(run_map_file: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add new candidates while preserving all user-maintained mapping fields."""

    columns = [
        "candidate_id",
        "optuna_trial_number",
        "actual_run_name",
        "mo_bio_study_name",
        "mo_bio_trial_number",
        "status",
        "station_file",
        "completion_date",
        "notes",
    ]
    if run_map_file.exists():
        run_map = pd.read_csv(run_map_file, dtype="string")
    else:
        run_map = pd.DataFrame(columns=columns)
    upgraded = False
    upgrade_columns = ["mo_bio_study_name", "mo_bio_trial_number"]
    for column in upgrade_columns:
        if column not in run_map:
            run_map[column] = pd.NA
            upgraded = True
    missing = sorted(set(columns).difference(run_map.columns))
    if missing:
        raise ValueError(f"Run map is missing columns: {missing}")

    for key in ("candidate_id", "optuna_trial_number"):
        duplicates = run_map[key].dropna().duplicated(keep=False)
        if duplicates.any():
            values = run_map.loc[run_map[key].dropna().index[duplicates], key].tolist()
            raise ValueError(f"Run map contains duplicate {key} values: {values}")
    actual = run_map["actual_run_name"].dropna()
    actual = actual[actual.str.strip().ne("")]
    if actual.duplicated(keep=False).any():
        raise ValueError("Run map contains duplicate nonblank actual_run_name values.")

    existing_ids = set(run_map["candidate_id"].dropna())
    additions = []
    for _, candidate in candidates.iterrows():
        candidate_id = str(candidate["candidate_id"])
        trial_number = str(int(candidate["optuna_trial_number"]))
        if candidate_id in existing_ids:
            existing = run_map.loc[run_map["candidate_id"].eq(candidate_id)].iloc[0]
            if str(existing["optuna_trial_number"]) != trial_number:
                raise ValueError(
                    f"{candidate_id} already maps to trial "
                    f"{existing['optuna_trial_number']}, not {trial_number}."
                )
            continue
        additions.append(
            {
                "candidate_id": candidate_id,
                "optuna_trial_number": trial_number,
                "actual_run_name": pd.NA,
                "mo_bio_study_name": pd.NA,
                "mo_bio_trial_number": pd.NA,
                "status": "planned",
                "station_file": pd.NA,
                "completion_date": pd.NA,
                "notes": pd.NA,
            }
        )
    if additions:
        for addition in additions:
            run_map.loc[len(run_map), columns] = [addition[column] for column in columns]
        run_map.to_csv(run_map_file, index=False)
    elif upgraded:
        run_map[columns].to_csv(run_map_file, index=False)
    return run_map[columns]


def save_audit_outputs(
    paths: WorkflowPaths,
    archive: pd.DataFrame,
    issues: pd.DataFrame,
    observed: Mapping[str, pd.DataFrame],
    numeric_inventory: pd.DataFrame,
    numeric_differences: pd.DataFrame,
    candidate_space: pd.DataFrame,
) -> None:
    archive.to_csv(paths.audit_directory / "physical_archive_inventory.csv", index=False)
    issues.to_csv(paths.issues_file, index=False)
    observed["orientation"].to_csv(
        paths.audit_directory / "physical_observed_orientation_pairs.csv", index=False
    )
    observed["optional_cpp"].to_csv(
        paths.audit_directory / "physical_observed_optional_cpp_combinations.csv",
        index=False,
    )
    observed["advection"].to_csv(
        paths.audit_directory / "physical_observed_advection_schemes.csv", index=False
    )
    numeric_inventory.to_csv(
        paths.audit_directory / "physical_varied_numeric_parameters.csv", index=False
    )
    numeric_differences.to_csv(
        paths.audit_directory / "physical_numeric_parameter_differences.csv", index=False
    )
    candidate_space.to_csv(
        paths.audit_directory / "physical_factorial_audit.csv", index=False
    )
