"""Reusable utilities for the ROMS-COSiNE multi-objective tuning workflow.

The notebook owns scientific choices and user-facing explanations.  This module
owns stable mechanics: data validation, objective calculations, parameter-file
parsing, Pareto summaries, quality-control ledgers, and Optuna bookkeeping.

Functions in this module should not print during normal operation.  They return
dataframes, dictionaries, or explicit exceptions so the notebook can decide
what deserves the user's attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PRIOR_DECISIONS = {"active", "fixed", "reserve", "exclude", "review"}
RELEVANCE_LEVELS = {"high", "medium", "low", "unknown"}
CONFIDENCE_LEVELS = {"high", "medium", "low", "unknown"}
PARAMETER_TYPES = {"float", "integer", "categorical", "boolean"}
SCALES = {"linear", "log", "categorical"}
CORRECTION_ACTIONS = {"replace", "accept", "set_missing", "exclude_run", "unresolved"}

PRIOR_COLUMNS = [
    "parameter",
    "parameter_type",
    "decision",
    "bottom_ph_relevance",
    "holistic_relevance",
    "confidence_in_current_value",
    "scale",
    "allowed_choices",
    "proposed_low",
    "proposed_high",
    "fixed_value",
    "parent_option",
    "scientific_rationale",
    "future_question",
    "notes",
]

ISSUE_COLUMNS = [
    "issue_id",
    "run_name",
    "source_file",
    "field",
    "observed_value",
    "issue_type",
    "reason",
    "expected_low",
    "expected_high",
    "detected_utc",
    "resolution_status",
]

CORRECTION_COLUMNS = [
    "issue_id",
    "run_name",
    "field",
    "original_value",
    "action",
    "replacement_value",
    "evidence",
    "notes",
    "reviewed_by",
    "reviewed_date",
]

# Stable inventories used to route candidate values to ROMS input templates.
OPTICS_PARAMETERS = [
    "reg1", "reg2", "gmaxs1", "gmaxs2", "rrb1", "rrb2", "rrg1", "rrg2",
    "beta1", "beta2", "akz1", "akz2", "PARfrac", "amaxs1", "amaxs2",
    "parsats1", "parsats2", "pis1", "pis2", "akno3s1", "akno3s2",
    "aknh4s1", "aknh4s2", "akpo4s1", "akpo4s2", "akco2s1", "akco2s2",
    "aksio4s2", "ak1", "ak2", "bgamma0", "bgamma1", "bgamma2", "bgamma3",
    "bgamma4", "bgamma5", "bgamma5s", "bgamma6", "bgamma7", "wsd", "wsdsi",
    "wsp", "pco2a", "si2n", "p2n", "o2no", "o2nh", "c2n", "ro5", "ro6",
    "ro7", "akox", "Chl2cs1_m", "Chl2cs2_m",
]

SEDIMENT_PARAMETERS = [
    "bUmax_nspc0", "bUmax_nspc1", "bUmax_nspc2", "bUmaxSi_nspc0",
    "bUmaxSi_nspc1", "bUmaxSi_nspc2", "bdep", "balpha", "bw", "btheta_diag",
    "bnit", "btheta_nit", "bdo_nit", "bdenit", "btheta_denit", "bpsi_n",
    "bdo_c", "bao2", "bpi", "bpsi_p", "bfc_nspc0", "bfc_nspc1", "bfc_nspc2",
    "bfn_nspc0", "bfn_nspc1", "bfn_nspc2", "bfp_nspc0", "bfp_nspc1",
    "bfp_nspc2", "bfs_nspc0", "bfs_nspc1", "bfs_nspc2",
]


@dataclass(frozen=True)
class CacheStatus:
    """Concise description of a cache and its known source files."""

    cache_exists: bool
    cache_modified_utc: str | None
    source_files_checked: int
    newer_source_files: tuple[str, ...]
    missing_source_files: tuple[str, ...]


@dataclass(frozen=True)
class MoBioOutputPaths:
    """Authoritative organized output paths for the MO-BIO workflow."""

    root: Path

    @property
    def audit_directory(self) -> Path:
        return self.root / "audit"

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidates"

    @property
    def candidate_input_directory(self) -> Path:
        return self.candidate_directory / "input_files"

    @property
    def provenance_directory(self) -> Path:
        return self.root / "provenance"

    @property
    def quality_directory(self) -> Path:
        return self.root / "quality_control"

    @property
    def figure_directory(self) -> Path:
        return self.root / "figures"

    @property
    def cache_directory(self) -> Path:
        return self.root / "cache"

    @property
    def optuna_directory(self) -> Path:
        return self.root / "optuna"

    @property
    def migration_log(self) -> Path:
        return self.audit_directory / "OUTPUT_MIGRATION_LOG.txt"

    @property
    def run_cache_file(self) -> Path:
        return self.cache_directory / "Model_Run_Summary_Bio.csv"

    @property
    def quality_issues_file(self) -> Path:
        return self.quality_directory / "data_quality_issues.csv"

    @property
    def corrections_file(self) -> Path:
        return self.quality_directory / "data_corrections.csv"

    @property
    def candidate_summary_file(self) -> Path:
        return self.candidate_directory / "mo_bio_candidate_summary.csv"

    @property
    def trial_provenance_file(self) -> Path:
        return self.provenance_directory / "mo_bio_trial_provenance.csv"

    @property
    def study_database(self) -> Path:
        return self.optuna_directory / "model_calibration_bio_cost_bottom_ph.db"


def ensure_mo_bio_output_layout(paths: MoBioOutputPaths) -> None:
    """Create the MO-BIO output directories without replacing any file."""

    for directory in (
        paths.audit_directory,
        paths.candidate_directory,
        paths.candidate_input_directory,
        paths.provenance_directory,
        paths.quality_directory,
        paths.figure_directory,
        paths.cache_directory,
        paths.optuna_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for provenance and migration checks."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_text() -> str:
    """Return a stable UTC timestamp for logs and generated metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _stable_issue_id(run_name: Any, field: Any, issue_type: Any, observed_value: Any) -> str:
    payload = "|".join(
        [_clean_text(run_name), _clean_text(field), _clean_text(issue_type), repr(observed_value)]
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def atomic_write_csv(frame: pd.DataFrame, path: Path | str, *, index: bool = False) -> None:
    """Write a CSV atomically so a failed write cannot destroy a valid file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp.csv",
        prefix=f".{output.stem}_",
        dir=output.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=index)
    try:
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def cache_status(cache_file: Path | str, source_files: Iterable[Path | str]) -> CacheStatus:
    """Compare cache modification time with source-file modification times."""

    cache = Path(cache_file)
    sources = [Path(path) for path in source_files]
    missing = tuple(str(path) for path in sources if not path.is_file())
    if not cache.is_file():
        return CacheStatus(False, None, len(sources), (), missing)

    cache_mtime = cache.stat().st_mtime
    newer = tuple(
        str(path)
        for path in sources
        if path.is_file() and path.stat().st_mtime > cache_mtime
    )
    timestamp = datetime.fromtimestamp(cache_mtime, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()
    return CacheStatus(True, timestamp, len(sources), newer, missing)


def load_cached_or_refresh(
    cache_file: Path | str,
    mode: str,
    refresh_function: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    """Load a cache or call a supplied source-reader according to an explicit mode."""

    valid_modes = {"cache", "refresh", "refresh_and_save"}
    if mode not in valid_modes:
        raise ValueError(f"DATA_LOAD_MODE must be one of {sorted(valid_modes)}; got {mode!r}.")

    cache = Path(cache_file)
    if mode == "cache":
        if not cache.is_file():
            raise FileNotFoundError(
                f"Validated run cache not found: {cache}. Use DATA_LOAD_MODE='refresh' "
                "or 'refresh_and_save'."
            )
        return pd.read_csv(cache), "validated cache"

    refreshed = refresh_function()
    if not isinstance(refreshed, pd.DataFrame) or refreshed.empty:
        raise ValueError("Fresh source read did not produce a non-empty dataframe.")
    if mode == "refresh_and_save":
        atomic_write_csv(refreshed, cache, index=False)
        return refreshed, "fresh source files; validated cache replaced"
    return refreshed, "fresh source files; cache left unchanged"


# ---------------------------------------------------------------------------
# Objective and data-extraction mechanics
# ---------------------------------------------------------------------------


def get_observational_error(variable_name: str, observation_mean: float) -> float:
    """Return the legacy observational-error term used by normalized RMSD."""

    if variable_name.startswith("sed"):
        return observation_mean
    if variable_name.startswith("pH"):
        return 0.1
    if variable_name.startswith("temp"):
        return 0.01
    if variable_name.startswith("salt"):
        return 0.005 * observation_mean
    if variable_name.startswith("oxy"):
        return 3.125
    if variable_name.startswith("N") or variable_name.startswith("Si"):
        return 0.1
    if variable_name.startswith("SD"):
        return 0.2
    return 0.0


def _coordinate_examples(data_array: Any, mask: np.ndarray, limit: int) -> list[str]:
    examples: list[str] = []
    for index_values in np.argwhere(mask)[:limit]:
        labels: list[str] = []
        for dimension, index in zip(data_array.dims, index_values):
            if dimension in data_array.coords and data_array.coords[dimension].ndim == 1:
                labels.append(f"{dimension}={data_array.coords[dimension].values[index]}")
            else:
                labels.append(f"{dimension}[{index}]")
        examples.append(", ".join(labels) if labels else "scalar")
    return examples


def append_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    run_name: str,
    metric: str,
    issue: str,
    details: str,
    severity: str = "detail",
) -> None:
    """Append one structured diagnostic without printing it."""

    diagnostics.append(
        {
            "Run Name": run_name,
            "Metric": metric,
            "Issue": issue,
            "Details": details,
            "Severity": severity,
        }
    )


def calculate_rmse_star(
    dataset: Any,
    metric_name: str,
    model_variable: str,
    observation_variable: str,
    run_name: str,
    diagnostics: list[dict[str, Any]],
    max_mismatch_locations: int = 10,
) -> float:
    """Calculate the existing signed normalized RMSD from paired finite values."""

    missing = [
        variable
        for variable in (model_variable, observation_variable)
        if variable not in dataset.variables
    ]
    if missing:
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Missing variable",
            details=f"not found: {missing}",
            severity="actionable",
        )
        return np.nan

    model_data = dataset[model_variable]
    observation_data = dataset[observation_variable]
    if metric_name.startswith("Surface"):
        model_data = model_data.isel(Depth=0)
        observation_data = observation_data.isel(Depth=0)
    elif metric_name.startswith("Bottom"):
        model_data = model_data.isel(Depth=1)
        observation_data = observation_data.isel(Depth=1)

    if model_data.dims != observation_data.dims or model_data.shape != observation_data.shape:
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Alignment mismatch",
            details=(
                f"model dims/shape={model_data.dims}/{model_data.shape}; "
                f"observation dims/shape={observation_data.dims}/{observation_data.shape}"
            ),
            severity="actionable",
        )
        return np.nan

    model_values = np.asarray(model_data.values)
    observation_values = np.asarray(observation_data.values)
    model_finite = np.isfinite(model_values)
    observation_finite = np.isfinite(observation_values)
    model_only = model_finite & ~observation_finite
    observation_only = ~model_finite & observation_finite
    if model_only.any() or observation_only.any():
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Missing-data mismatch",
            details=(
                f"model-only={int(model_only.sum())} at "
                f"{_coordinate_examples(model_data, model_only, max_mismatch_locations)}; "
                f"observation-only={int(observation_only.sum())} at "
                f"{_coordinate_examples(observation_data, observation_only, max_mismatch_locations)}"
            ),
            severity="detail",
        )

    paired = model_finite & observation_finite
    model = model_values[paired].astype(float)
    observation = observation_values[paired].astype(float)
    if model.size < 2:
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Insufficient paired data",
            details=f"found {model.size} paired finite value(s); at least 2 required",
            severity="actionable",
        )
        return np.nan

    model_mean = model.mean()
    observation_mean = observation.mean()
    model_std = model.std()
    observed_sample_std = observation.std()
    error = get_observational_error(observation_variable, observation_mean)
    observation_std = np.sqrt(observed_sample_std**2 + error**2)
    if model_std == 0 or not np.isfinite(model_std):
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Invalid model variance",
            details=f"model standard deviation={model_std}",
            severity="actionable",
        )
        return np.nan
    if observation_std == 0 or not np.isfinite(observation_std):
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Invalid observation variance",
            details=f"adjusted observation standard deviation={observation_std}",
            severity="actionable",
        )
        return np.nan

    covariance = np.mean((model - model_mean) * (observation - observation_mean))
    correlation = covariance / (model_std * observation_std)
    normalized_std = model_std / observation_std
    inside = 1 + normalized_std**2 - 2 * normalized_std * correlation
    if inside < -1e-12 or not np.isfinite(inside):
        append_diagnostic(
            diagnostics,
            run_name=run_name,
            metric=metric_name,
            issue="Invalid RMSD expression",
            details=f"value under square root={inside}",
            severity="actionable",
        )
        return np.nan
    return float(np.sqrt(max(inside, 0.0)) * np.sign(model_std - observation_std))


def calculate_cost_breakdown(
    dataset: Any,
    cost_components: Mapping[str, str],
    cost_weights: Mapping[str, float],
    *,
    bottom_ph_target: str = "Bottom pH",
) -> dict[str, Any]:
    """Calculate total cost and labeled weighted contributions exactly as the reference."""

    contributions: dict[str, float] = {}
    for target, variable_name in cost_components.items():
        weight = float(cost_weights[target])
        if target.startswith("Surface"):
            component = float(dataset[variable_name][0])
        elif target.startswith("Bottom"):
            component = float(dataset[variable_name][1])
        else:
            component = float(dataset[variable_name])
        contributions[target] = component * weight**2

    total = float(sum(contributions.values()))
    bottom = float(contributions.get(bottom_ph_target, np.nan))
    non_bottom = total - bottom if np.isfinite(bottom) else np.nan
    return {
        "Cost": total,
        "Bottom pH Cost Contribution": bottom,
        "Non-bottom-pH Cost": non_bottom,
        "Cost Contributions": contributions,
    }


def get_parameter_values(dataset: Any, parameter_name: str) -> dict[str, Any]:
    """Extract a scalar or flattened NetCDF parameter into labeled columns."""

    if parameter_name not in dataset.variables:
        return {}
    variable = dataset.variables[parameter_name]
    if variable.ndim == 0:
        value = variable[...]
        if np.ma.is_masked(value):
            value = np.nan
        return {parameter_name: np.asarray(value).item()}

    values = np.array(variable[:], copy=True).squeeze()
    if values.ndim == 0:
        return {parameter_name: values.item()}
    if values.ndim == 1:
        dimension = variable.dimensions[0]
        return {
            f"{parameter_name}_{dimension}{index}": value
            for index, value in enumerate(values)
        }
    return {
        f"{parameter_name}_{index}": value
        for index, value in enumerate(values.ravel())
    }


def get_run_date_from_history(
    dataset: Any,
    *,
    attribute_name: str = "history",
) -> tuple[pd.Timestamp, bool, str]:
    """Return station-history date, fallback flag, and preserved raw history text."""

    history = getattr(dataset, attribute_name, None)
    raw_history = "" if history is None else str(history)
    match = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},\s+\d{4}\b",
        raw_history,
    )
    if match is None:
        return pd.NaT, True, raw_history
    parsed = pd.to_datetime(match.group(0), format="%B %d, %Y", errors="coerce")
    if pd.isna(parsed):
        return pd.NaT, True, raw_history
    return parsed.normalize(), False, raw_history


def get_dstart_year(dataset: Any, variable_name: str = "dstart") -> float | int:
    """Decode a scalar NetCDF time variable and return its year."""

    if variable_name not in dataset.variables:
        return np.nan
    variable = dataset.variables[variable_name]
    raw = variable[...]
    if np.ma.is_masked(raw):
        return np.nan
    value = np.asarray(raw).squeeze()
    if value.size != 1:
        return np.nan
    if np.issubdtype(value.dtype, np.datetime64):
        return int(pd.to_datetime(value.reshape(1))[0].year)
    units = getattr(variable, "units", None)
    if units is None:
        raise ValueError(f"{variable_name} has no NetCDF time units")
    from netCDF4 import num2date

    timestamp = num2date(
        value.item(),
        units=units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return int(timestamp.year)


def get_cpp_option_states(
    dataset: Any,
    option_names: Sequence[str],
    *,
    attribute_name: str = "CPP_options",
) -> dict[str, bool]:
    """Return configured CPP options as explicit Boolean states."""

    text = str(dataset.getncattr(attribute_name)) if attribute_name in dataset.ncattrs() else ""
    tokens = {
        token.upper()
        for token in re.findall(r"!?[A-Za-z_][A-Za-z0-9_.]*", text)
    }
    return {
        option: option.upper() in tokens and f"!{option.upper()}" not in tokens
        for option in option_names
    }


def get_cpp_parameter_values(
    dataset: Any,
    option_names: Sequence[str],
    conditional_variables: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    attribute_name: str = "CPP_options",
) -> dict[str, Any]:
    """Extract CPP states and conditional values; disabled children are not applicable."""

    states = get_cpp_option_states(dataset, option_names, attribute_name=attribute_name)
    results: dict[str, Any] = dict(states)
    enabled: dict[str, Mapping[str, Sequence[str]]] = {}
    for option in option_names:
        specifications = conditional_variables.get(option, {})
        if states[option]:
            enabled[option] = specifications
        else:
            for output_names in specifications.values():
                for output_name in output_names:
                    results[output_name] = np.nan
    if not enabled:
        return results

    import xarray as xr

    with xr.open_dataset(dataset.filepath(), engine="h5netcdf", decode_cf=False) as cpp_data:
        for option, specifications in enabled.items():
            for variable_name, output_names in specifications.items():
                if variable_name not in cpp_data.variables:
                    raise ValueError(
                        f"CPP option {option} is enabled, but {variable_name} is missing"
                    )
                values = np.asarray(cpp_data[variable_name].values, dtype=float).reshape(-1)
                if values.size != len(output_names):
                    raise ValueError(
                        f"{variable_name} has {values.size} values; expected {len(output_names)}"
                    )
                for output_name, value in zip(output_names, values):
                    if not np.isfinite(value):
                        raise ValueError(f"{output_name} is non-finite while {option} is enabled")
                    results[output_name] = float(value)
    return results


def get_biological_advection_scheme(
    dataset: Any,
    schemes: Mapping[str, Sequence[str]],
    *,
    tracer_name: str = "detritus",
    attribute_name: str = "NLM_TADV",
) -> Any:
    """Return the configured category matching one tracer's H/V advection pair."""

    if attribute_name not in dataset.ncattrs():
        return np.nan
    match = re.search(
        rf"^\s*{re.escape(tracer_name)}:\s+(\S+)\s+(\S+)",
        str(dataset.getncattr(attribute_name)),
        flags=re.MULTILINE,
    )
    if match is None:
        return np.nan
    pair = match.groups()
    for category, expected in schemes.items():
        if pair == tuple(expected):
            return category
    return np.nan


def refresh_run_archive(
    runs: pd.DataFrame,
    *,
    cost_directory: Path | str,
    parameter_names: Sequence[str],
    metric_info: Mapping[str, Sequence[str]],
    cost_components: Mapping[str, str],
    cost_weights: Mapping[str, float],
    cpp_option_names: Sequence[str],
    cpp_conditional_variables: Mapping[str, Mapping[str, Sequence[str]]],
    biological_nl_tnu2_parameter: str,
    biological_advection_parameter: str,
    biological_advection_schemes: Mapping[str, Sequence[str]],
    max_mismatch_locations: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all listed cost/station files once and return quiet structured results."""

    import gc
    import xarray as xr
    from netCDF4 import Dataset

    required = {"Run Name", "Cost File", "Station File"}
    missing_columns = sorted(required.difference(runs.columns))
    if missing_columns:
        raise ValueError(f"Run inventory is missing columns: {missing_columns}")
    root = Path(cost_directory)
    all_results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    conditional_station_variables = {
        variable
        for specifications in cpp_conditional_variables.values()
        for variable in specifications
    }

    for _, row_series in runs.iterrows():
        run_name = str(row_series["Run Name"])
        cost_file = str(row_series["Cost File"])
        station_file = str(row_series["Station File"])
        result: dict[str, Any] = {
            "Run Name": run_name,
            "Cost File": cost_file,
            "Station File": station_file,
        }
        cost_path = root / cost_file
        station_path = root / station_file
        try:
            with xr.open_dataset(cost_path) as dataset:
                for metric, variables in metric_info.items():
                    result[metric] = calculate_rmse_star(
                        dataset,
                        metric,
                        str(variables[0]),
                        str(variables[1]),
                        run_name,
                        diagnostics,
                        max_mismatch_locations,
                    )
                breakdown = calculate_cost_breakdown(dataset, cost_components, cost_weights)
                result.update({key: value for key, value in breakdown.items() if key != "Cost Contributions"})
                for target, value in breakdown["Cost Contributions"].items():
                    result[f"Cost Component: {target}"] = value
            result["Cost Status"] = "Success"
        except Exception as error:
            result["Cost Status"] = f"ERROR: {error}"
            result.setdefault("Cost", np.nan)
            result.setdefault("Bottom pH Cost Contribution", np.nan)
            result.setdefault("Non-bottom-pH Cost", np.nan)
            for metric in metric_info:
                result.setdefault(metric, np.nan)
            failures.append(
                {"Run Name": run_name, "File Type": "Cost", "File": str(cost_path), "Error": str(error)}
            )

        try:
            with Dataset(station_path, mode="r") as dataset:
                result.update(
                    get_cpp_parameter_values(
                        dataset,
                        cpp_option_names,
                        cpp_conditional_variables,
                    )
                )
                for parameter in parameter_names:
                    if parameter in conditional_station_variables:
                        continue
                    result.update(get_parameter_values(dataset, parameter))
                result["dstart_year"] = get_dstart_year(dataset)
                run_date, fallback, history = get_run_date_from_history(dataset)
                result["Station Run Date"] = run_date
                result["Station Date Is Fallback"] = fallback
                result["Station History"] = history
                nl_tnu2 = get_parameter_values(dataset, "nl_tnu2")
                result[biological_nl_tnu2_parameter] = nl_tnu2.get(
                    biological_nl_tnu2_parameter, np.nan
                )
                result[biological_advection_parameter] = get_biological_advection_scheme(
                    dataset,
                    biological_advection_schemes,
                )
            result["Parameter Status"] = "Success"
        except Exception as error:
            result["Parameter Status"] = f"ERROR: {error}"
            result.setdefault("Station Run Date", pd.NaT)
            result.setdefault("Station Date Is Fallback", True)
            result.setdefault("Station History", "")
            result.setdefault(biological_nl_tnu2_parameter, np.nan)
            result.setdefault(biological_advection_parameter, np.nan)
            failures.append(
                {"Run Name": run_name, "File Type": "Station", "File": str(station_path), "Error": str(error)}
            )
        all_results.append(result)
        gc.collect()

    archive = pd.DataFrame(all_results).dropna(axis=1, how="all")
    for metric in metric_info:
        archive[f"{metric} Objective"] = pd.to_numeric(archive[metric], errors="coerce").abs()
    return (
        archive,
        pd.DataFrame(diagnostics, columns=["Run Name", "Metric", "Issue", "Details", "Severity"]),
        pd.DataFrame(failures, columns=["Run Name", "File Type", "File", "Error"]),
    )


# ---------------------------------------------------------------------------
# ROMS input-file mechanics
# ---------------------------------------------------------------------------


def format_roms_value(value: float) -> str:
    """Format a finite numeric ROMS input value using D exponent notation."""

    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Cannot format non-finite ROMS value {value!r}")
    text = f"{numeric:.16g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    if "e" in text.lower():
        mantissa, exponent = re.split("[eE]", text)
        return f"{mantissa}d{int(exponent):+d}"
    return f"{text}d0"


def read_template_parameter(text: str, parameter: str) -> float:
    """Read a scalar or indexed nspc parameter from ROMS input text."""

    if "_nspc" in parameter:
        base, index_text = parameter.rsplit("_nspc", 1)
        value_index = int(index_text)
    else:
        base, value_index = parameter, 0
    match = re.search(
        rf"^\s*{re.escape(base)}\s*==\s*([^!\n]+)", text, flags=re.MULTILINE
    )
    if match is None:
        return np.nan
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[dDeE][-+]?\d+)?"
    values = re.findall(number_pattern, match.group(1))
    if value_index >= len(values):
        return np.nan
    return float(values[value_index].replace("D", "e").replace("d", "e"))


def replace_scalar_parameter(text: str, parameter: str, value: float) -> str:
    """Replace exactly one scalar ROMS input parameter."""

    pattern = rf"^(\s*{re.escape(parameter)}\s*==\s*)([^\s!]+)"
    updated, count = re.subn(
        pattern,
        rf"\g<1>{format_roms_value(value)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"{parameter} had {count} active-line replacements")
    return updated


def replace_indexed_or_scalar_parameter(text: str, parameter: str, value: float) -> str:
    """Replace one scalar or one indexed value from a multi-value ROMS line."""

    if "_nspc" not in parameter:
        return replace_scalar_parameter(text, parameter, value)
    base, index_text = parameter.rsplit("_nspc", 1)
    index = int(index_text)
    pattern = rf"^(\s*{re.escape(base)}\s*==\s*)([^!\n]+)(.*)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find {base}")
    values = match.group(2).split()
    if index >= len(values):
        raise ValueError(f"{parameter} requests index {index}; {base} has {len(values)} values")
    values[index] = format_roms_value(value)
    replacement = match.group(1) + " ".join(values) + match.group(3)
    return text[: match.start()] + replacement + text[match.end() :]


def replace_repeated_tracer_value(
    text: str,
    keyword: str,
    value: float,
    tracer_count: int,
) -> str:
    """Replace one ROMS ``count * value`` tracer line."""

    pattern = rf"^(\s*{re.escape(keyword)}\s*==\s*{tracer_count}\s*\*\s*)([^\s!]+)"
    updated, count = re.subn(
        pattern,
        rf"\g<1>{format_roms_value(value)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"{keyword} had {count} active-line replacements")
    return updated


def replace_advection_block(
    text: str,
    keyword: str,
    scheme_code: str,
    expected_values: int,
) -> str:
    """Replace every value in one validated ROMS advection block."""

    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(rf"^\s*{re.escape(keyword)}\s*==", line)
    ]
    if len(starts) != 1:
        raise ValueError(f"{keyword} had {len(starts)} active blocks")
    start = starts[0]
    for offset in range(expected_values):
        index = start + offset
        if index >= len(lines):
            raise ValueError(f"{keyword} ended before value {offset + 1}")
        pattern = (
            rf"^(\s*{re.escape(keyword)}\s*==\s*)(\S+)(.*)$"
            if offset == 0
            else r"^(\s*)(\S+)(\s+.*)$"
        )
        updated, count = re.subn(pattern, rf"\g<1>{scheme_code}\g<3>", lines[index], count=1)
        if count != 1 or f"idbio({offset + 1:2d})" not in updated:
            raise ValueError(f"{keyword} value {offset + 1} did not match expected template")
        lines[index] = updated
    return "".join(lines)


def build_candidate_input_texts(
    candidates: pd.DataFrame,
    *,
    optics_template: Path | str,
    sediment_template: Path | str,
    optics_parameters: Sequence[str],
    sediment_parameters: Sequence[str],
    fixed_values: Mapping[str, Any],
    biological_tnu2_parameter: str,
    biological_advection_parameter: str,
    biological_advection_schemes: Mapping[str, Sequence[str]],
    advection_input_codes: Mapping[str, str],
    biological_tracer_count: int,
    remote_sediment_directory: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build validated candidate input text in memory without writing files."""

    optics_source = Path(optics_template).read_text()
    sediment_source = Path(sediment_template).read_text()
    optics_texts: dict[str, str] = {}
    sediment_texts: dict[str, str] = {}
    metadata_columns = {"Run Name", "Trial Number"}
    candidate_parameters = [column for column in candidates.columns if column not in metadata_columns]
    active_optics = [parameter for parameter in candidate_parameters if parameter in optics_parameters]
    active_sediment = [parameter for parameter in candidate_parameters if parameter in sediment_parameters]

    for _, candidate in candidates.iterrows():
        run_name = str(candidate["Run Name"])
        optics = optics_source
        sediment = sediment_source
        for parameter, value in fixed_values.items():
            if parameter in optics_parameters and parameter not in active_optics:
                optics = replace_scalar_parameter(optics, parameter, float(value))
            elif parameter in sediment_parameters and parameter not in active_sediment:
                sediment = replace_indexed_or_scalar_parameter(sediment, parameter, float(value))
        for parameter in active_optics:
            optics = replace_scalar_parameter(optics, parameter, float(candidate[parameter]))
        for parameter in active_sediment:
            sediment = replace_indexed_or_scalar_parameter(
                sediment, parameter, float(candidate[parameter])
            )

        if biological_tnu2_parameter in candidate or biological_tnu2_parameter in fixed_values:
            tnu2_value = candidate.get(
                biological_tnu2_parameter, fixed_values.get(biological_tnu2_parameter)
            )
            optics = replace_repeated_tracer_value(
                optics,
                "TNU2",
                float(tnu2_value),
                biological_tracer_count,
            )
        if biological_advection_parameter in candidate or biological_advection_parameter in fixed_values:
            category = candidate.get(
                biological_advection_parameter, fixed_values.get(biological_advection_parameter)
            )
            if category not in biological_advection_schemes:
                raise ValueError(f"{run_name}: unknown advection category {category!r}")
            horizontal, vertical = biological_advection_schemes[category]
            optics = replace_advection_block(
                optics,
                "Hadvection",
                advection_input_codes[horizontal],
                biological_tracer_count,
            )
            optics = replace_advection_block(
                optics,
                "Vadvection",
                advection_input_codes[vertical],
                biological_tracer_count,
            )

        sediment_filename = f"bio_UMAINE15_sedmodel_{run_name}.in"
        remote_path = f"{remote_sediment_directory.rstrip('/')}/{sediment_filename}"
        pattern = r"^(\s*BSEDPARNAM\s*==\s*)(\S+)"
        optics, count = re.subn(
            pattern, rf"\g<1>{remote_path}", optics, count=1, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"{run_name}: BSEDPARNAM had {count} replacements")
        optics_texts[run_name] = optics
        sediment_texts[run_name] = sediment
    return optics_texts, sediment_texts


def write_candidate_input_files(
    optics_texts: Mapping[str, str],
    sediment_texts: Mapping[str, str],
    output_directory: Path | str,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write validated candidate texts only after checking every destination."""

    if set(optics_texts) != set(sediment_texts):
        raise ValueError("Optics and sediment candidate names do not match")
    output = Path(output_directory)
    destinations: list[tuple[str, Path, Path]] = []
    for run_name in optics_texts:
        optics_path = output / f"bio_UMAINE15_sediment_optics_{run_name}.in"
        sediment_path = output / f"bio_UMAINE15_sedmodel_{run_name}.in"
        destinations.append((run_name, optics_path, sediment_path))
    existing = [str(path) for _, first, second in destinations for path in (first, second) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Candidate files already exist and overwrite is disabled: " + str(existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for run_name, optics_path, sediment_path in destinations:
        optics_path.write_text(optics_texts[run_name])
        sediment_path.write_text(sediment_texts[run_name])
        records.append(
            {
                "Run Name": run_name,
                "Optics File": str(optics_path),
                "Sediment File": str(sediment_path),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# User-maintained priors and quality-control ledgers
# ---------------------------------------------------------------------------


def load_parameter_priors(path: Path | str) -> pd.DataFrame:
    """Load and strictly validate the editable parameter-prior table."""

    priors = pd.read_csv(path, dtype={"parameter": "string"})
    missing = [column for column in PRIOR_COLUMNS if column not in priors.columns]
    if missing:
        raise ValueError(f"Parameter-prior file is missing columns: {missing}")
    priors = priors[PRIOR_COLUMNS].copy()
    priors["parameter"] = priors["parameter"].astype("string").str.strip()
    empty_names = priors["parameter"].isna() | priors["parameter"].eq("")
    if empty_names.any():
        raise ValueError(f"Parameter-prior file has blank names in rows {(priors.index[empty_names] + 2).tolist()}")
    duplicate = priors.loc[priors["parameter"].duplicated(keep=False), "parameter"].tolist()
    if duplicate:
        raise ValueError(f"Parameter-prior file has duplicate names: {sorted(set(duplicate))}")

    validation_sets = {
        "parameter_type": PARAMETER_TYPES,
        "decision": PRIOR_DECISIONS,
        "bottom_ph_relevance": RELEVANCE_LEVELS,
        "holistic_relevance": RELEVANCE_LEVELS,
        "confidence_in_current_value": CONFIDENCE_LEVELS,
        "scale": SCALES,
    }
    errors: list[str] = []
    for column, allowed in validation_sets.items():
        normalized = priors[column].fillna("").astype(str).str.strip().str.lower()
        invalid = ~normalized.isin(allowed)
        if invalid.any():
            details = priors.loc[invalid, ["parameter", column]].to_dict("records")
            errors.append(f"{column}: {details}; allowed={sorted(allowed)}")
        priors[column] = normalized

    for column in ["proposed_low", "proposed_high"]:
        priors[column] = pd.to_numeric(priors[column], errors="coerce")
    active_numeric = priors["decision"].eq("active") & priors["parameter_type"].isin(
        ["float", "integer"]
    )
    invalid_bounds = active_numeric & (
        priors["proposed_low"].isna()
        | priors["proposed_high"].isna()
        | priors["proposed_low"].ge(priors["proposed_high"])
    )
    if invalid_bounds.any():
        errors.append(
            "active numeric parameters need finite increasing bounds: "
            + str(priors.loc[invalid_bounds, "parameter"].tolist())
        )
    active_categorical = priors["decision"].eq("active") & priors["parameter_type"].isin(
        ["categorical", "boolean"]
    )
    missing_choices = active_categorical & priors["allowed_choices"].fillna("").astype(str).str.strip().eq("")
    if missing_choices.any():
        errors.append(
            "active categorical parameters need pipe-separated allowed_choices: "
            + str(priors.loc[missing_choices, "parameter"].tolist())
        )
    fixed_numeric = priors["decision"].eq("fixed") & priors["parameter_type"].isin(
        ["float", "integer"]
    )
    numeric_fixed_values = pd.to_numeric(priors["fixed_value"], errors="coerce")
    fixed_numeric_without_value = fixed_numeric & numeric_fixed_values.isna()
    if fixed_numeric_without_value.any():
        errors.append(
            "fixed numeric parameters need numeric fixed_value: "
            + str(priors.loc[fixed_numeric_without_value, "parameter"].tolist())
        )
    priors.loc[fixed_numeric, "fixed_value"] = numeric_fixed_values.loc[fixed_numeric]
    fixed_categories = priors["decision"].eq("fixed") & priors["parameter_type"].isin(
        ["categorical", "boolean"]
    )
    for index in priors.index[fixed_categories]:
        choices = [choice.strip() for choice in str(priors.at[index, "allowed_choices"]).split("|")]
        value = str(priors.at[index, "fixed_value"]).strip()
        if value not in choices:
            errors.append(
                f"fixed categorical parameter {priors.at[index, 'parameter']!r} needs "
                f"fixed_value in {choices}; got {value!r}"
            )
    if errors:
        raise ValueError("Invalid parameter-prior entries:\n- " + "\n- ".join(errors))
    return priors


def parse_prior_choices(row: Any) -> tuple[Any, ...]:
    """Convert a prior row's pipe-separated choices to their declared type."""

    parameter_type = row["parameter_type"] if isinstance(row, Mapping) else row.parameter_type
    raw = row["allowed_choices"] if isinstance(row, Mapping) else row.allowed_choices
    choices = [choice.strip() for choice in str(raw).split("|") if choice.strip()]
    if parameter_type == "boolean":
        mapping = {"true": True, "false": False}
        invalid = [choice for choice in choices if choice.lower() not in mapping]
        if invalid:
            raise ValueError(f"Boolean choices must be True/False; got {invalid}")
        return tuple(mapping[choice.lower()] for choice in choices)
    return tuple(choices)


def parse_prior_fixed_value(row: Any) -> Any:
    """Convert one maintained fixed value according to its parameter type."""

    parameter_type = row["parameter_type"] if isinstance(row, Mapping) else row.parameter_type
    raw = row["fixed_value"] if isinstance(row, Mapping) else row.fixed_value
    if parameter_type == "float":
        return float(raw)
    if parameter_type == "integer":
        return int(float(raw))
    if parameter_type == "boolean":
        value = str(raw).strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(f"Boolean fixed value must be True/False; got {raw!r}")
        return value == "true"
    return str(raw).strip()


def prior_coverage(priors: pd.DataFrame, discovered_parameters: Sequence[str]) -> pd.DataFrame:
    """Report whether discovered parameters have a maintained prior row."""

    maintained = set(priors["parameter"])
    return pd.DataFrame(
        {
            "Parameter": list(discovered_parameters),
            "In prior file": [parameter in maintained for parameter in discovered_parameters],
        }
    )


def detect_data_quality_issues(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    run_column: str = "Run Name",
    source_column: str = "Station File",
    tiny_nonzero_threshold: float = 1e-250,
) -> pd.DataFrame:
    """Create a durable issue table for missing or suspiciously tiny values.

    Proposed parameter bounds define the future Optuna search space.  Historical
    observations outside those bounds remain valid evidence and are not quality
    issues.
    """

    now = utc_now_text()
    prior_lookup = priors.set_index("parameter")
    records: list[dict[str, Any]] = []
    for parameter in priors["parameter"]:
        if parameter not in frame.columns:
            records.append(
                {
                    "run_name": "<archive>",
                    "source_file": "",
                    "field": parameter,
                    "observed_value": np.nan,
                    "issue_type": "Completely missing parameter",
                    "reason": "No column exists in the run archive",
                    "expected_low": np.nan,
                    "expected_high": np.nan,
                }
            )
            continue
        parameter_type = prior_lookup.at[parameter, "parameter_type"]
        if parameter_type in {"categorical", "boolean"}:
            if frame[parameter].notna().sum() == 0:
                records.append(
                    {
                        "run_name": "<archive>",
                        "source_file": "",
                        "field": parameter,
                        "observed_value": np.nan,
                        "issue_type": "Completely missing parameter",
                        "reason": "Column exists but contains no usable categories",
                        "expected_low": np.nan,
                        "expected_high": np.nan,
                    }
                )
            continue
        series = pd.to_numeric(frame[parameter], errors="coerce")
        if series.notna().sum() == 0:
            records.append(
                {
                    "run_name": "<archive>",
                    "source_file": "",
                    "field": parameter,
                    "observed_value": np.nan,
                    "issue_type": "Completely missing parameter",
                    "reason": "Column exists but contains no usable values",
                    "expected_low": np.nan,
                    "expected_high": np.nan,
                }
            )
            continue

        values = series.to_numpy(float)
        suspicious_tiny = np.isfinite(values) & (values != 0) & (np.abs(values) < tiny_nonzero_threshold)
        for index in np.flatnonzero(suspicious_tiny):
            issue_type = "Suspiciously small nonzero value"
            reason = f"absolute value is below {tiny_nonzero_threshold:g}"
            records.append(
                {
                    "run_name": frame.iloc[index].get(run_column, index),
                    "source_file": frame.iloc[index].get(source_column, ""),
                    "field": parameter,
                    "observed_value": values[index],
                    "issue_type": issue_type,
                    "reason": reason,
                    "expected_low": np.nan,
                    "expected_high": np.nan,
                }
            )

    objective_columns = [column for column in frame.columns if column.endswith(" Objective")]
    for objective in objective_columns:
        usable = pd.to_numeric(frame[objective], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if usable.notna().sum() == 0:
            records.append(
                {
                    "run_name": "<archive>",
                    "source_file": "",
                    "field": objective,
                    "observed_value": np.nan,
                    "issue_type": "Completely missing objective",
                    "reason": "Objective has no finite values",
                    "expected_low": np.nan,
                    "expected_high": np.nan,
                }
            )

    for record in records:
        record["issue_id"] = _stable_issue_id(
            record["run_name"], record["field"], record["issue_type"], record["observed_value"]
        )
        record["detected_utc"] = now
        record["resolution_status"] = "unresolved"
    return pd.DataFrame(records, columns=ISSUE_COLUMNS)


def load_corrections(path: Path | str) -> pd.DataFrame:
    """Load a user-maintained correction ledger or return an empty valid table."""

    correction_path = Path(path)
    if not correction_path.is_file():
        return pd.DataFrame(columns=CORRECTION_COLUMNS)
    corrections = pd.read_csv(correction_path, dtype=str).fillna("")
    missing = [column for column in CORRECTION_COLUMNS if column not in corrections.columns]
    if missing:
        raise ValueError(f"Correction ledger is missing columns: {missing}")
    corrections = corrections[CORRECTION_COLUMNS].copy()
    corrections["action"] = corrections["action"].str.strip().str.lower()
    invalid = ~corrections["action"].isin(CORRECTION_ACTIONS)
    if invalid.any():
        raise ValueError(
            "Correction ledger contains invalid actions: "
            + str(corrections.loc[invalid, ["issue_id", "action"]].to_dict("records"))
        )
    duplicate = corrections["issue_id"].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(
            "Correction ledger contains duplicate issue IDs: "
            + str(corrections.loc[duplicate, "issue_id"].tolist())
        )
    return corrections


def apply_corrections(
    raw_frame: pd.DataFrame,
    issues: pd.DataFrame,
    corrections: pd.DataFrame,
    *,
    run_column: str = "Run Name",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply verified corrections without modifying raw data or source files."""

    analysis = raw_frame.copy(deep=True)
    issue_table = issues.copy()
    audit_records: list[dict[str, Any]] = []
    if issue_table.empty:
        return analysis, issue_table, pd.DataFrame(audit_records)
    issue_lookup = issue_table.set_index("issue_id", drop=False)

    for correction in corrections.itertuples(index=False):
        if correction.issue_id not in issue_lookup.index:
            audit_records.append(
                {"issue_id": correction.issue_id, "status": "stale", "details": "issue no longer detected"}
            )
            continue
        issue = issue_lookup.loc[correction.issue_id]
        if str(issue["run_name"]) != str(correction.run_name) or str(issue["field"]) != str(correction.field):
            audit_records.append(
                {"issue_id": correction.issue_id, "status": "rejected", "details": "run or field mismatch"}
            )
            continue
        original_expected = pd.to_numeric(pd.Series([correction.original_value]), errors="coerce").iloc[0]
        original_observed = pd.to_numeric(pd.Series([issue["observed_value"]]), errors="coerce").iloc[0]
        if pd.notna(original_expected) and not np.isclose(
            original_expected, original_observed, equal_nan=True, rtol=1e-12, atol=0.0
        ):
            audit_records.append(
                {"issue_id": correction.issue_id, "status": "rejected", "details": "original value mismatch"}
            )
            continue

        rows = analysis[run_column].astype(str).eq(str(correction.run_name))
        action = correction.action
        if action == "replace":
            replacement = pd.to_numeric(pd.Series([correction.replacement_value]), errors="coerce").iloc[0]
            if not np.isfinite(replacement):
                audit_records.append(
                    {"issue_id": correction.issue_id, "status": "rejected", "details": "replacement is not finite"}
                )
                continue
            analysis.loc[rows, correction.field] = replacement
        elif action == "set_missing":
            analysis.loc[rows, correction.field] = np.nan
        elif action == "exclude_run":
            analysis = analysis.loc[~rows].copy()
        elif action == "unresolved":
            analysis.loc[rows, correction.field] = np.nan
        elif action == "accept":
            pass
        issue_table.loc[issue_table["issue_id"].eq(correction.issue_id), "resolution_status"] = action
        audit_records.append(
            {"issue_id": correction.issue_id, "status": "applied", "details": action}
        )

    unresolved = issue_table["resolution_status"].eq("unresolved")
    for issue in issue_table.loc[unresolved].itertuples(index=False):
        if issue.run_name == "<archive>" or issue.field not in analysis.columns:
            continue
        rows = analysis[run_column].astype(str).eq(str(issue.run_name))
        analysis.loc[rows, issue.field] = np.nan
    return analysis, issue_table, pd.DataFrame(audit_records)


# ---------------------------------------------------------------------------
# Pareto, screening, and decision-support calculations
# ---------------------------------------------------------------------------


def pareto_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based non-domination ranks for finite minimization objectives."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Pareto values must be a two-dimensional array")
    ranks = np.full(len(array), -1, dtype=int)
    valid_indices = np.flatnonzero(np.isfinite(array).all(axis=1))
    remaining = valid_indices.tolist()
    rank = 0
    while remaining:
        current: list[int] = []
        for index in remaining:
            candidate = array[index]
            dominated = any(
                np.all(array[other] <= candidate) and np.any(array[other] < candidate)
                for other in remaining
                if other != index
            )
            if not dominated:
                current.append(index)
        if not current:
            raise RuntimeError("Could not resolve Pareto ranks")
        ranks[current] = rank
        selected = set(current)
        remaining = [index for index in remaining if index not in selected]
        rank += 1
    return ranks


def add_pareto_columns(
    frame: pd.DataFrame,
    objective_columns: Sequence[str],
    *,
    near_rank: int = 1,
) -> pd.DataFrame:
    """Add Pareto rank, front membership, near-front status, and ideal distance."""

    result = frame.copy()
    objective_values = result[list(objective_columns)].apply(pd.to_numeric, errors="coerce")
    ranks = pareto_ranks(objective_values.to_numpy())
    result["Pareto Rank"] = pd.Series(ranks, index=result.index).replace(-1, pd.NA).astype("Int64")
    result["Pareto Optimal"] = result["Pareto Rank"].eq(0)
    result["Near Pareto"] = result["Pareto Rank"].le(near_rank)

    normalized = objective_values.copy()
    for column in objective_columns:
        low, high = objective_values[column].min(), objective_values[column].max()
        normalized[column] = (
            (objective_values[column] - low) / (high - low)
            if np.isfinite(high - low) and high > low
            else np.nan
        )
    result["Normalized Ideal Distance"] = np.sqrt((normalized**2).sum(axis=1))
    return result


def select_representative_candidates(
    frame: pd.DataFrame,
    cost_column: str = "Cost",
    bottom_ph_column: str = "Bottom pH Objective",
) -> pd.DataFrame:
    """Select endpoint and normalized compromise candidates without hiding duplicates."""

    usable = frame.dropna(subset=[cost_column, bottom_ph_column]).copy()
    if usable.empty:
        return pd.DataFrame()
    if "Pareto Rank" not in usable:
        usable = add_pareto_columns(usable, [cost_column, bottom_ph_column])
    front = usable.loc[usable["Pareto Rank"].eq(0)].copy()
    selections = {
        "Best holistic cost": usable[cost_column].idxmin(),
        "Best Bottom pH": usable[bottom_ph_column].idxmin(),
        "Balanced normalized compromise": front["Normalized Ideal Distance"].idxmin(),
    }
    rows: list[pd.Series] = []
    for role, index in selections.items():
        row = usable.loc[index].copy()
        row["Selection Role"] = role
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(subset=["Selection Role"]).reset_index(drop=True)


def numeric_screening_summary(
    frame: pd.DataFrame,
    parameters: Sequence[str],
    objectives: Sequence[str],
    *,
    good_column: str = "Near Pareto",
) -> pd.DataFrame:
    """Summarize numeric coverage, objective association, and promising intervals."""

    rows: list[dict[str, Any]] = []
    for parameter in parameters:
        if parameter not in frame.columns:
            continue
        values = pd.to_numeric(frame[parameter], errors="coerce")
        valid_values = values.dropna()
        if valid_values.empty:
            continue
        good = frame[good_column].fillna(False).astype(bool) if good_column in frame else pd.Series(False, index=frame.index)
        good_values = values[good & values.notna()]
        record: dict[str, Any] = {
            "Parameter": parameter,
            "Available": int(valid_values.size),
            "Coverage": float(values.notna().mean()),
            "Unique Values": int(valid_values.nunique()),
            "Tested Low": float(valid_values.min()),
            "Tested High": float(valid_values.max()),
            "Promising Count": int(good_values.size),
            "Promising 10%": float(good_values.quantile(0.10)) if len(good_values) else np.nan,
            "Promising Median": float(good_values.median()) if len(good_values) else np.nan,
            "Promising 90%": float(good_values.quantile(0.90)) if len(good_values) else np.nan,
        }
        for objective in objectives:
            paired = pd.concat([values, pd.to_numeric(frame[objective], errors="coerce")], axis=1).dropna()
            record[f"Spearman: {objective}"] = (
                float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"))
                if len(paired) >= 3 and paired.iloc[:, 0].nunique() >= 2
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def categorical_screening_summary(
    frame: pd.DataFrame,
    parameters: Sequence[str],
    *,
    good_column: str = "Near Pareto",
) -> pd.DataFrame:
    """Summarize categorical good-run probabilities and sampling-adjusted enrichment."""

    rows: list[dict[str, Any]] = []
    good = frame[good_column].fillna(False).astype(bool)
    overall_good_rate = float(good.mean()) if len(frame) else np.nan
    for parameter in parameters:
        if parameter not in frame.columns:
            continue
        available = frame[parameter].notna()
        for level, group in frame.loc[available].groupby(parameter, dropna=False):
            indices = group.index
            successes = int(good.loc[indices].sum())
            total = len(indices)
            probability = successes / total if total else np.nan
            low, high = _wilson_interval(successes, total)
            rows.append(
                {
                    "Parameter": parameter,
                    "Category": level,
                    "Runs": total,
                    "Promising Runs": successes,
                    "P(Promising | Category)": probability,
                    "95% Low": low,
                    "95% High": high,
                    "Enrichment": probability / overall_good_rate if overall_good_rate > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def parameter_confounding(
    frame: pd.DataFrame,
    parameters: Sequence[str],
    *,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Return strongly associated numeric parameter pairs."""

    numeric = frame[list(parameters)].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    names = list(correlations.columns)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            value = correlations.loc[first, second]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "Parameter 1": first,
                        "Parameter 2": second,
                        "Spearman Correlation": float(value),
                        "Absolute Correlation": float(abs(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values("Absolute Correlation", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["Parameter 1", "Parameter 2", "Spearman Correlation", "Absolute Correlation"])


def cross_validated_mixed_importance(
    frame: pd.DataFrame,
    numeric_parameters: Sequence[str],
    categorical_parameters: Sequence[str],
    objectives: Sequence[str],
    *,
    random_state: int = 42,
    n_estimators: int = 200,
    folds: int = 5,
    permutation_repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate mixed-type nonlinear importance on held-out folds.

    Importance is reported only alongside held-out model performance.  If the
    model does not beat a mean baseline, its importance values should not be
    treated as evidence about individual parameters.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = [parameter for parameter in numeric_parameters if parameter in frame.columns]
    categorical = [parameter for parameter in categorical_parameters if parameter in frame.columns]
    features = [*numeric, *categorical]
    if not features:
        raise ValueError("No available parameters were supplied for nonlinear screening")
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model_template = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=0.7,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    split = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    performance_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for objective in objectives:
        objective_values = pd.to_numeric(frame[objective], errors="coerce")
        usable = objective_values.notna()
        X = frame.loc[usable, features].copy()
        for parameter in numeric:
            X[parameter] = pd.to_numeric(X[parameter], errors="coerce")
        # Newer scikit-learn SimpleImputer versions reject native bool dtype.
        # Object dtype preserves True/False as categories and keeps missing
        # values available to the categorical imputer.
        for parameter in categorical:
            X[parameter] = X[parameter].astype(object)
        y = objective_values.loc[usable].astype(float).reset_index(drop=True)
        X = X.reset_index(drop=True)
        model_predictions = np.full(len(X), np.nan)
        baseline_predictions = np.full(len(X), np.nan)
        for split_number, (train, test) in enumerate(split.split(X), start=1):
            model = model_template.fit(X.iloc[train], y.iloc[train])
            baseline = DummyRegressor(strategy="mean").fit(
                np.zeros((len(train), 1)), y.iloc[train]
            )
            model_predictions[test] = model.predict(X.iloc[test])
            baseline_predictions[test] = baseline.predict(np.zeros((len(test), 1)))
            permutation = permutation_importance(
                model,
                X.iloc[test],
                y.iloc[test],
                scoring="neg_mean_absolute_error",
                n_repeats=permutation_repeats,
                random_state=random_state + split_number,
                n_jobs=1,
            )
            for parameter_index, parameter in enumerate(features):
                for importance in permutation.importances[parameter_index]:
                    importance_rows.append(
                        {
                            "Objective": objective,
                            "Fold": split_number,
                            "Parameter": parameter,
                            "Importance": float(importance),
                        }
                    )
        performance_rows.extend(
            [
                {
                    "Objective": objective,
                    "Model": "Mixed-type Random Forest",
                    "Runs": len(X),
                    "Parameters": len(features),
                    "Held-out R2": r2_score(y, model_predictions),
                    "Held-out MAE": mean_absolute_error(y, model_predictions),
                },
                {
                    "Objective": objective,
                    "Model": "Mean baseline",
                    "Runs": len(X),
                    "Parameters": 0,
                    "Held-out R2": r2_score(y, baseline_predictions),
                    "Held-out MAE": mean_absolute_error(y, baseline_predictions),
                },
            ]
        )
    raw_importance = pd.DataFrame(importance_rows)
    summary = (
        raw_importance.groupby(["Objective", "Parameter"], as_index=False)
        .agg(
            Mean_Importance=("Importance", "mean"),
            SD_Importance=("Importance", "std"),
            Positive_Fraction=("Importance", lambda values: float(values.gt(0).mean())),
        )
    )
    summary["Rank"] = summary.groupby("Objective")["Mean_Importance"].rank(
        method="min", ascending=False
    )
    return pd.DataFrame(performance_rows), summary


def readiness_summary(
    priors: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    confounding: pd.DataFrame,
) -> pd.DataFrame:
    """Combine maintained decisions with empirical support into plain-language readiness."""

    numeric_lookup = numeric_summary.set_index("Parameter") if not numeric_summary.empty else pd.DataFrame()
    categorical_counts = (
        categorical_summary.groupby("Parameter")["Runs"].min()
        if not categorical_summary.empty
        else pd.Series(dtype=float)
    )
    confounded = set(confounding.get("Parameter 1", [])) | set(confounding.get("Parameter 2", []))
    rows: list[dict[str, Any]] = []
    for prior in priors.itertuples(index=False):
        parameter = prior.parameter
        if prior.parameter_type in {"categorical", "boolean"}:
            support = categorical_counts.get(parameter, 0)
            empirical_status = "Supported categorical coverage" if support >= 10 else "Sparse categorical coverage"
        elif parameter not in numeric_lookup.index:
            empirical_status = "Insufficient archive evidence"
        else:
            evidence = numeric_lookup.loc[parameter]
            if evidence["Unique Values"] < 4 or evidence["Promising Count"] < 3:
                empirical_status = "Sparse numeric coverage"
            elif parameter in confounded:
                empirical_status = "Strongly confounded"
            else:
                empirical_status = "Supported empirical coverage"
        if prior.decision == "exclude":
            status = "Operationally or scientifically excluded"
        elif prior.decision == "review":
            status = "Needs scientific decision"
        elif prior.parameter_type in {"categorical", "boolean"}:
            if support < 10:
                status = "Needs exploration"
            elif prior.decision == "fixed":
                status = "Probably safe to fix"
            elif prior.decision == "reserve":
                status = "Held in reserve"
            else:
                status = "Ready to optimize"
        elif parameter not in numeric_lookup.index:
            status = "Insufficient evidence"
        else:
            if evidence["Unique Values"] < 4 or evidence["Promising Count"] < 3:
                status = "Needs exploration"
            elif parameter in confounded:
                if prior.decision == "fixed":
                    status = "Fix cautiously"
                elif prior.decision == "reserve":
                    status = "Held in reserve"
                else:
                    status = "Optimize cautiously"
            elif prior.decision == "fixed":
                status = "Probably safe to fix"
            elif prior.decision in {"active", "reserve"}:
                status = "Ready to optimize" if prior.decision == "active" else "Held in reserve"
            else:
                status = "Needs scientific decision"
        rows.append(
            {
                "Parameter": parameter,
                "Decision": prior.decision,
                "Bottom pH relevance": prior.bottom_ph_relevance,
                "Holistic relevance": prior.holistic_relevance,
                "Empirical Evidence": empirical_status,
                "Readiness": status,
                "Confounded": parameter in confounded,
            }
        )
    return pd.DataFrame(rows)


def budget_assessment(
    active_numeric: int,
    categorical_level_counts: Sequence[int],
    trial_budget: int,
) -> dict[str, str | int]:
    """Return a conservative, explicitly heuristic search-budget message."""

    categorical_burden = sum(max(0, count - 1) for count in categorical_level_counts)
    effective_dimensions = active_numeric + categorical_burden
    if effective_dimensions == 0:
        rating = "No active search space"
        message = "Select at least one active parameter before preparing a study."
    elif trial_budget < max(10, 2 * effective_dimensions):
        rating = "Limited"
        message = (
            "The study may find useful candidates, but stable parameter-range conclusions "
            "are unlikely at this budget."
        )
    elif trial_budget < max(20, 4 * effective_dimensions):
        rating = "Focused"
        message = "Reasonable for focused optimization; interaction conclusions remain tentative."
    else:
        rating = "Supported"
        message = "Budget is reasonably matched to the proposed search-space complexity."
    return {
        "Active numeric parameters": active_numeric,
        "Categorical burden": categorical_burden,
        "Effective dimensions (heuristic)": effective_dimensions,
        "Trial budget": trial_budget,
        "Assessment": rating,
        "Interpretation": message,
    }


# ---------------------------------------------------------------------------
# Optuna study safety and TPE interpretation
# ---------------------------------------------------------------------------


def search_space_fingerprint(
    objective_names: Sequence[str],
    numeric_bounds: Mapping[str, Sequence[float]],
    categorical_choices: Mapping[str, Sequence[Any]],
    *,
    sampler_settings: Mapping[str, Any] | None = None,
) -> str:
    """Hash objective and search-space definitions to prevent silent study reuse."""

    payload = {
        "objectives": list(objective_names),
        "numeric_bounds": {key: list(value) for key, value in sorted(numeric_bounds.items())},
        "categorical_choices": {
            key: list(value) for key, value in sorted(categorical_choices.items())
        },
        "sampler_settings": dict(sorted((sampler_settings or {}).items())),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def create_or_load_multiobjective_study(
    *,
    study_name: str,
    storage: str,
    objective_version: str,
    fingerprint: str,
    seed: int,
    n_startup_trials: int,
    multivariate: bool,
    historical_numeric_import_policy: str | None = None,
    load_if_exists: bool = True,
) -> Any:
    """Create/load a two-objective study and enforce version/fingerprint compatibility."""

    import optuna

    if historical_numeric_import_policy not in {None, "candidate_bounds", "observed_support"}:
        raise ValueError(
            "historical_numeric_import_policy must be 'candidate_bounds' or "
            "'observed_support'"
        )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        seed=seed,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        sampler=sampler,
        load_if_exists=load_if_exists,
    )
    expected = {
        "Objective Version": objective_version,
        "Search Space Fingerprint": fingerprint,
    }
    if historical_numeric_import_policy is not None:
        expected["Historical Numeric Import Policy"] = historical_numeric_import_policy
    for key, value in expected.items():
        stored = study.user_attrs.get(key)
        if stored is None:
            study.set_user_attr(key, value)
        elif stored != value:
            raise ValueError(
                f"Study {key} mismatch. Stored={stored!r}; configured={value!r}. "
                "Use a new study name and database rather than mixing definitions."
            )
    return study


def historical_numeric_support_table(
    frame: pd.DataFrame,
    numeric_bounds: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Compare observed archive support with hard bounds for future candidates."""

    records: list[dict[str, Any]] = []
    for parameter, bounds in numeric_bounds.items():
        candidate_low, candidate_high = float(bounds[0]), float(bounds[1])
        values = (
            pd.to_numeric(frame[parameter], errors="coerce")
            if parameter in frame.columns
            else pd.Series(dtype=float)
        )
        finite = values[np.isfinite(values)]
        observed_low = float(finite.min()) if not finite.empty else np.nan
        observed_high = float(finite.max()) if not finite.empty else np.nan
        support_low = min(candidate_low, observed_low) if np.isfinite(observed_low) else candidate_low
        support_high = max(candidate_high, observed_high) if np.isfinite(observed_high) else candidate_high
        outside = int(((finite < candidate_low) | (finite > candidate_high)).sum())
        records.append(
            {
                "Parameter": parameter,
                "Historical Observed Low": observed_low,
                "Historical Observed High": observed_high,
                "Historical Support Low": support_low,
                "Historical Support High": support_high,
                "Candidate Low": candidate_low,
                "Candidate High": candidate_high,
                "Historical Values Outside Candidate Bounds": outside,
            }
        )
    return pd.DataFrame(records)


def import_historical_multiobjective_trials(
    study: Any,
    frame: pd.DataFrame,
    *,
    numeric_bounds: Mapping[str, Sequence[float]],
    categorical_choices: Mapping[str, Sequence[Any]],
    objective_columns: Sequence[str] = ("Cost", "Bottom pH Objective"),
    objective_version: str,
    run_column: str = "Run Name",
    station_date_column: str = "Station Run Date",
    previous_optimizer_prefix: str = "OPTUNA_",
    numeric_import_policy: str = "candidate_bounds",
) -> pd.DataFrame:
    """Add eligible archive rows as two-objective completed trials.

    ``candidate_bounds`` requires archive values to lie within the future search
    bounds. ``observed_support`` represents imported numeric values on a wider,
    shared distribution while leaving future candidate suggestions constrained
    by ``numeric_bounds``.
    """

    import optuna

    if len(objective_columns) != 2:
        raise ValueError("This workflow requires exactly two objective columns")
    valid_policies = {"candidate_bounds", "observed_support"}
    if numeric_import_policy not in valid_policies:
        raise ValueError(
            f"numeric_import_policy must be one of {sorted(valid_policies)}"
        )
    support = historical_numeric_support_table(frame, numeric_bounds).set_index("Parameter")
    distributions: dict[str, Any] = {
        parameter: optuna.distributions.FloatDistribution(
            float(support.at[parameter, "Historical Support Low"]),
            float(support.at[parameter, "Historical Support High"]),
        )
        if numeric_import_policy == "observed_support"
        else optuna.distributions.FloatDistribution(float(bounds[0]), float(bounds[1]))
        for parameter, bounds in numeric_bounds.items()
    }
    distributions.update(
        {
            parameter: optuna.distributions.CategoricalDistribution(tuple(choices))
            for parameter, choices in categorical_choices.items()
        }
    )
    required_parameters = [*numeric_bounds, *categorical_choices]
    existing_names = {
        trial.user_attrs.get("Actual Run Name", trial.user_attrs.get("Run Name"))
        for trial in study.trials
    }
    sort_columns = [station_date_column] if station_date_column in frame.columns else []
    archive = frame.sort_values(sort_columns, kind="stable", na_position="last") if sort_columns else frame
    records: list[dict[str, Any]] = []
    imported_utc = utc_now_text()
    for _, row in archive.iterrows():
        run_name = str(row[run_column])
        candidate_bound_violations = [
            parameter
            for parameter, bounds in numeric_bounds.items()
            if parameter in row.index
            and pd.notna(row[parameter])
            and not (float(bounds[0]) <= float(row[parameter]) <= float(bounds[1]))
        ]
        status = "eligible"
        if run_name in existing_names:
            status = "already present"
        elif any(parameter not in row.index or pd.isna(row[parameter]) for parameter in required_parameters):
            status = "missing active parameter"
        elif any(objective not in row.index or not np.isfinite(pd.to_numeric(pd.Series([row[objective]]), errors="coerce").iloc[0]) for objective in objective_columns):
            status = "missing objective"
        elif numeric_import_policy == "candidate_bounds" and candidate_bound_violations:
            status = "outside configured bounds"
        elif any(row[parameter] not in choices for parameter, choices in categorical_choices.items()):
            status = "unknown category"

        if status == "eligible":
            params = {parameter: float(row[parameter]) for parameter in numeric_bounds}
            params.update({parameter: row[parameter] for parameter in categorical_choices})
            station_date = row.get(station_date_column, pd.NaT)
            station_date_text = (
                pd.Timestamp(station_date).date().isoformat() if pd.notna(station_date) else None
            )
            origin = (
                "Previous optimizer"
                if run_name.startswith(previous_optimizer_prefix)
                else "Historical archive"
            )
            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                values=[float(row[objective_columns[0]]), float(row[objective_columns[1]])],
                state=optuna.trial.TrialState.COMPLETE,
                user_attrs={
                    "Run Name": run_name,
                    "Actual Run Name": run_name,
                    "Origin": origin,
                    "Objective Version": objective_version,
                    "Station Run Date": station_date_text,
                    "Station Date Is Fallback": bool(row.get("Station Date Is Fallback", pd.isna(station_date))),
                    "Imported UTC": imported_utc,
                },
            )
            study.add_trial(trial)
            existing_names.add(run_name)
            status = "added"
        records.append(
            {
                "Run Name": run_name,
                "Import Status": status,
                "Within Candidate Bounds": not candidate_bound_violations,
                "Candidate Bound Violations": "|".join(candidate_bound_violations),
            }
        )
    return pd.DataFrame(records)


def ask_candidate_trials(
    study: Any,
    *,
    numeric_bounds: Mapping[str, Sequence[float]],
    categorical_choices: Mapping[str, Sequence[Any]],
    count: int,
    run_name_prefix: str,
    objective_version: str,
) -> pd.DataFrame:
    """Create explicit RUNNING trials and return their proposed parameters."""

    import optuna

    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    if running:
        raise ValueError(
            f"Cannot generate a new batch while {len(running)} trial(s) remain RUNNING."
        )
    rows: list[dict[str, Any]] = []
    for _ in range(int(count)):
        trial = study.ask()
        run_name = f"{run_name_prefix}{trial.number}"
        trial.set_user_attr("Run Name", run_name)
        trial.set_user_attr("Origin", "Multi-objective TPE")
        trial.set_user_attr("Objective Version", objective_version)
        params = {
            parameter: trial.suggest_float(parameter, float(bounds[0]), float(bounds[1]))
            for parameter, bounds in numeric_bounds.items()
        }
        params.update(
            {
                parameter: trial.suggest_categorical(parameter, list(choices))
                for parameter, choices in categorical_choices.items()
            }
        )
        rows.append({"Run Name": run_name, "Trial Number": trial.number, **params})
    return pd.DataFrame(rows)


def build_complete_candidate_summary(
    candidates: pd.DataFrame,
    *,
    study_name: str,
    objective_version: str,
    search_space_fingerprint: str,
    parameter_priors_file: Path | str,
    baseline_run_name: str,
    sampler_name: str,
    fixed_values: Mapping[str, Any],
    priors: pd.DataFrame,
) -> pd.DataFrame:
    """Add complete resolved-input and study provenance to a candidate batch.

    Candidate values always win over fixed/baseline values.  Conditional child
    applicability is recorded explicitly; no child value is silently invented.
    """

    required = {"Run Name", "Trial Number"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    if candidates["Run Name"].duplicated().any():
        raise ValueError("Candidate run names must be unique")
    if candidates["Trial Number"].duplicated().any():
        raise ValueError("Candidate trial numbers must be unique")

    summary = candidates.copy()
    active_parameters = [
        column for column in candidates.columns
        if column not in {"Run Name", "Trial Number"}
    ]
    resolved_fixed = {
        str(parameter): value
        for parameter, value in fixed_values.items()
        if parameter not in active_parameters
    }
    for parameter, value in sorted(resolved_fixed.items()):
        summary[parameter] = value

    generated_utc = utc_now_text()
    priors_path = Path(parameter_priors_file)
    metadata = {
        "Study Name": study_name,
        "Trial State": "RUNNING",
        "Origin": "Multi-objective TPE",
        "Generated UTC": generated_utc,
        "Objective Version": objective_version,
        "Search Space Fingerprint": search_space_fingerprint,
        "Parameter Priors SHA256": file_sha256(priors_path),
        "Baseline Run Name": baseline_run_name,
        "Sampler": sampler_name,
        "Active Parameters": "; ".join(active_parameters),
        "Fixed/Baseline Parameters": "; ".join(sorted(resolved_fixed)),
        "Candidate Summary Version": "mo_bio_complete_v1",
    }
    insertion_point = 2
    for column, value in metadata.items():
        summary.insert(insertion_point, column, value)
        insertion_point += 1

    conditional_rows = priors.loc[
        priors.get("parent_option", pd.Series(index=priors.index, dtype=str))
        .fillna("").astype(str).str.strip().ne("")
    ]
    applicable_text: list[str] = []
    inactive_text: list[str] = []
    for _, candidate in summary.iterrows():
        applicable: list[str] = []
        inactive: list[str] = []
        for prior in conditional_rows.itertuples(index=False):
            parent = str(prior.parent_option).strip()
            parent_value = candidate.get(parent, fixed_values.get(parent, False))
            enabled = pd.notna(parent_value) and bool(parent_value)
            (applicable if enabled else inactive).append(str(prior.parameter))
        applicable_text.append("; ".join(applicable))
        inactive_text.append("; ".join(inactive))
    summary["Applicable Conditional Parameters"] = applicable_text
    summary["Inactive Conditional Parameters"] = inactive_text
    return summary


def append_candidate_summary(
    new_candidates: pd.DataFrame, output_file: Path | str
) -> pd.DataFrame:
    """Append new candidate records without altering an existing trial record."""

    output = Path(output_file)
    keys = ["Study Name", "Trial Number"]
    if output.is_file():
        existing = pd.read_csv(output)
        missing = sorted(set(keys).difference(existing.columns))
        if missing:
            raise ValueError(f"Existing candidate summary is missing keys: {missing}")
    else:
        existing = pd.DataFrame(columns=new_candidates.columns)

    combined = existing.copy()
    existing_keys = {
        (str(row["Study Name"]), int(row["Trial Number"]))
        for _, row in existing.iterrows()
    }
    additions = []
    for _, row in new_candidates.iterrows():
        key = (str(row["Study Name"]), int(row["Trial Number"]))
        if key in existing_keys:
            old = existing.loc[
                existing["Study Name"].astype(str).eq(key[0])
                & pd.to_numeric(existing["Trial Number"], errors="coerce").eq(key[1])
            ].iloc[0]
            common = [column for column in new_candidates.columns if column in old.index]
            conflicts = [
                column for column in common
                if not (
                    (pd.isna(old[column]) and pd.isna(row[column]))
                    or str(old[column]) == str(row[column])
                )
            ]
            if conflicts:
                raise ValueError(
                    f"Candidate summary already contains conflicting record {key}: "
                    f"{conflicts}"
                )
            continue
        additions.append(row.to_dict())
        existing_keys.add(key)
    if additions:
        addition_frame = pd.DataFrame(additions)
        combined = (
            addition_frame.reset_index(drop=True)
            if combined.empty
            else pd.concat([combined, addition_frame], ignore_index=True)
        )
        atomic_write_csv(combined, output, index=False)
    elif not output.is_file():
        atomic_write_csv(combined, output, index=False)
    return combined


def synchronize_running_trials(
    study: Any,
    frame: pd.DataFrame,
    *,
    objective_columns: Sequence[str] = ("Cost", "Bottom pH Objective"),
    run_column: str = "Run Name",
    run_map: pd.DataFrame | None = None,
    study_name: str | None = None,
    commit: bool = False,
) -> pd.DataFrame:
    """Preview or complete RUNNING trials whose two objective values are available."""

    import optuna

    results = frame.drop_duplicates(subset=[run_column], keep="last").set_index(run_column)
    mapping_by_study_name: dict[str, tuple[str, int]] = {}
    if run_map is not None and not run_map.empty:
        required = {"Actual Run Name", "Study Name", "Study Run Name", "Trial Number"}
        missing = sorted(required.difference(run_map.columns))
        if missing:
            raise ValueError(f"Run map is missing columns: {missing}")
        selected = run_map.copy()
        if study_name is not None:
            selected = selected.loc[selected["Study Name"].astype(str).eq(str(study_name))]
        if selected["Study Run Name"].duplicated().any() or selected["Trial Number"].duplicated().any():
            raise ValueError("Run map contains duplicate study run names or trial numbers")
        mapping_by_study_name = {
            str(row["Study Run Name"]): (str(row["Actual Run Name"]), int(row["Trial Number"]))
            for _, row in selected.iterrows()
        }
    records: list[dict[str, Any]] = []
    for frozen in study.trials:
        if frozen.state != optuna.trial.TrialState.RUNNING:
            continue
        study_run_name = frozen.user_attrs.get("Run Name")
        if study_run_name in mapping_by_study_name:
            actual_run_name, mapped_trial = mapping_by_study_name[study_run_name]
            if mapped_trial != frozen.number:
                raise ValueError(
                    f"Run-map trial mismatch for {study_run_name!r}: "
                    f"mapped={mapped_trial}, stored={frozen.number}"
                )
        else:
            actual_run_name = frozen.user_attrs.get("Actual Run Name", study_run_name)
        if actual_run_name not in results.index:
            records.append(
                {
                    "Trial Number": frozen.number,
                    "Study Run Name": study_run_name,
                    "Actual Run Name": actual_run_name,
                    "Status": "results unavailable",
                }
            )
            continue
        row = results.loc[actual_run_name]
        values = [pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0] for column in objective_columns]
        if not np.isfinite(values).all():
            records.append(
                {
                    "Trial Number": frozen.number,
                    "Study Run Name": study_run_name,
                    "Actual Run Name": actual_run_name,
                    "Status": "objectives incomplete",
                }
            )
            continue
        status = "ready to complete"
        if commit:
            live = optuna.trial.Trial(study, frozen._trial_id)
            live.set_user_attr("Study Run Name", study_run_name)
            live.set_user_attr("Actual Run Name", actual_run_name)
            live.set_user_attr("Station Run Date", row.get("Station Run Date"))
            study.tell(frozen.number, [float(values[0]), float(values[1])])
            status = "completed"
        records.append(
            {
                "Trial Number": frozen.number,
                "Study Run Name": study_run_name,
                "Actual Run Name": actual_run_name,
                objective_columns[0]: values[0],
                objective_columns[1]: values[1],
                "Status": status,
            }
        )
    return pd.DataFrame(records)


def tpe_distribution_samples(
    study: Any,
    sampler: Any,
    search_space: Mapping[str, Any],
    *,
    sample_count: int = 20_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample Optuna's private below/above Parzen models for diagnostics.

    This function is intentionally isolated because it relies on private Optuna
    APIs.  The notebook must display the returned version and compatibility note.
    """

    import optuna
    from optuna.samplers._tpe.sampler import _split_trials
    from optuna.trial import TrialState

    if not str(optuna.__version__).startswith("4.9."):
        raise RuntimeError(
            "Exact TPE reconstruction is validated only for Optuna 4.9.x; "
            f"installed version is {optuna.__version__}. Use empirical diagnostics instead."
        )
    trials = study._get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE, TrialState.PRUNED),
        use_cache=False,
    )
    n = len(trials)
    if n < sampler._n_startup_trials:
        raise ValueError(
            f"TPE has {n} usable trials but requires {sampler._n_startup_trials} startup trials."
        )
    below, above = _split_trials(
        study,
        trials,
        sampler._gamma(n),
        sampler._constraints_func is not None,
    )
    below_model = sampler._build_parzen_estimator(study, dict(search_space), below, True)
    above_model = sampler._build_parzen_estimator(study, dict(search_space), above, False)
    below_raw = below_model.sample(np.random.RandomState(seed), sample_count)
    above_raw = above_model.sample(np.random.RandomState(seed + 1), sample_count)

    def externalize(samples: Mapping[str, np.ndarray]) -> pd.DataFrame:
        external: dict[str, list[Any]] = {}
        for name, values in samples.items():
            distribution = search_space[name]
            external[name] = [distribution.to_external_repr(value) for value in values]
        return pd.DataFrame(external)

    metadata = {
        "Optuna version": optuna.__version__,
        "Usable trials": n,
        "Promising trial count": len(below),
        "Less-promising trial count": len(above),
        "Promising trial numbers": [trial.number for trial in below],
        "Less-promising trial numbers": [trial.number for trial in above],
        "Private API": True,
    }
    return externalize(below_raw), externalize(above_raw), metadata


def trial_provenance_table(study: Any) -> pd.DataFrame:
    """Return explicit provenance and timing for every stored Optuna trial."""

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        rows.append(
            {
                "Trial Number": trial.number,
                "State": trial.state.name,
                "Run Name": trial.user_attrs.get("Run Name"),
                "Actual Run Name": trial.user_attrs.get("Actual Run Name"),
                "Origin": trial.user_attrs.get("Origin", trial.user_attrs.get("Source")),
                "Station Run Date": trial.user_attrs.get("Station Run Date"),
                "Station Date Is Fallback": trial.user_attrs.get("Station Date Is Fallback"),
                "Imported UTC": trial.user_attrs.get("Imported UTC"),
                "Optuna Started": trial.datetime_start,
                "Optuna Completed": trial.datetime_complete,
            }
        )
    return pd.DataFrame(rows)


def package_versions(package_names: Sequence[str]) -> pd.DataFrame:
    """Return installed package versions without failing on optional packages."""

    from importlib.metadata import PackageNotFoundError, version

    rows = []
    for package in package_names:
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        rows.append({"Package": package, "Version": installed})
    return pd.DataFrame(rows)
