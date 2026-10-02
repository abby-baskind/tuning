#!/usr/bin/env python3
"""Multi-year model skill analysis for ROMS-COSiNE cost-summary files.

The module calculates the normalized statistics described by Jolliff et al.
(2009), observation-standard-deviation-normalized cost used by Ward et al.
(2010), target and Taylor diagrams, and collocated time-series diagnostics.

All statistics use one joint finite-value mask for model/observation pairs.
Pooled statistics concatenate those pairs before calculating any moments, so
years and stations contribute in proportion to their valid observations.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_INPUT_ROOT = Path("/Users/akbaskind/Desktop/COST_FILES")
DEFAULT_OUTPUT_ROOT = Path("/Users/akbaskind/Desktop/SKILL")
DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("skill_priority_weights.csv")
DEPTHS = ("Surface", "Bottom", "Total")
DEPTH_MARKERS = {"Surface": "o", "Bottom": "s", "Total": "^", "Not applicable": "D"}
SOURCE_COLORS = {"NBFSMN": "#0072B2", "PLT": "#D55E00", "CHRP": "#009E73", "PLT Secchi": "#CC79A7"}
AUTHORITATIVE_COLORS = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7B3294", "#8C510A",
    "#CC79A7", "#3366CC", "#117733", "#44AA99", "#666666",
)


@dataclass(frozen=True)
class VariableSpec:
    key: str
    label: str
    source: str
    observation: str
    model: str
    time: str | None
    depth_resolved: bool
    station_dimension: str | None
    legacy_cost: str
    units: str
    optional_group: str | None = None


VARIABLE_SPECS = (
    VariableSpec("temperature", "Temperature", "NBFSMN", "temp_obs", "temp_mod", "Day", True, "Site", "temperature_cost", "°C"),
    VariableSpec("salinity", "Salinity", "NBFSMN", "salt_obs", "salt_mod", "Day", True, "Site", "salt_cost", "PSU"),
    VariableSpec("ph", "pH", "NBFSMN", "pH_obs", "pH_mod", "Day", True, "Site", "pH_cost", "NBS"),
    VariableSpec("oxygen", "Dissolved oxygen", "NBFSMN", "oxygen_obs", "oxygen_mod", "Day", True, "Site", "oxygen_cost", "mmol m⁻³"),
    VariableSpec("plt_no3", "NO₃", "PLT", "NO3_obs", "NO3_mod", "Date", True, None, "NO3_cost", "mmol m⁻³"),
    VariableSpec("plt_nh4", "NH₄", "PLT", "NH4_obs", "NH4_mod", "Date", True, None, "NH4_cost", "mmol m⁻³"),
    VariableSpec("plt_si", "Silicate", "PLT", "Si_obs", "Si_mod", "DateSi", True, None, "Si_cost", "mmol m⁻³"),
    VariableSpec("chrp_no3", "NO₃", "CHRP", "NO3_chrp_obs", "NO3_chrp_mod", "DateCHRP", True, "Station", "NO3_chrp_cost", "mmol m⁻³"),
    VariableSpec("chrp_nh4", "NH₄", "CHRP", "NH4_chrp_obs", "NH4_chrp_mod", "DateCHRP", True, "Station", "NH4_chrp_cost", "mmol m⁻³"),
    VariableSpec("chrp_si", "Silicate", "CHRP", "Si_chrp_obs", "Si_chrp_mod", "DateCHRP", True, "Station", "Si_chrp_cost", "mmol m⁻³"),
    VariableSpec("secchi", "Secchi depth", "PLT Secchi", "SD_obs", "SD_mod", "Week", False, None, "SD_cost", "m"),
    VariableSpec("primary_production", "Primary production", "CHRP", "PP_obs", "PP_mod", None, False, "Station", "PP_cost", "g C m⁻² yr⁻¹", "primary_production"),
    VariableSpec("sediment_oxygen", "Sediment oxygen flux", "Benthic", "sed_o2_obs", "sed_o2_mod", None, False, "Loc", "sed_o2_cost", "mmol m⁻² d⁻¹", "benthic"),
    VariableSpec("sediment_no3", "Sediment NO₃ flux", "Benthic", "sed_no3_obs", "sed_no3_mod", None, False, "Loc", "sed_no3_cost", "mmol m⁻² d⁻¹", "benthic"),
    VariableSpec("sediment_nh4", "Sediment NH₄ flux", "Benthic", "sed_nh4_obs", "sed_nh4_mod", None, False, "Loc", "sed_nh4_cost", "mmol m⁻² d⁻¹", "benthic"),
)


def selected_specs(include_optional: Iterable[str] = ()) -> tuple[VariableSpec, ...]:
    groups = set(include_optional)
    unknown = groups - {"primary_production", "benthic"}
    if unknown:
        raise ValueError(f"Unknown optional groups: {sorted(unknown)}")
    return tuple(s for s in VARIABLE_SPECS if s.optional_group is None or s.optional_group in groups)


def component_name(spec: VariableSpec, depth: str) -> str:
    suffix = depth.lower().replace(" ", "_")
    return f"{spec.key}__{suffix}"


def pairwise_finite(model: np.ndarray, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(model, dtype=float).ravel()
    observation = np.asarray(observation, dtype=float).ravel()
    if model.shape != observation.shape:
        raise ValueError(f"Model and observation shapes differ: {model.shape} != {observation.shape}")
    valid = np.isfinite(model) & np.isfinite(observation)
    return model[valid], observation[valid]


def calculate_skill(model: np.ndarray, observation: np.ndarray, ddof: int = 0) -> dict[str, float | int | str]:
    """Calculate dimensional and Jolliff-normalized metrics on valid pairs."""
    model, observation = pairwise_finite(model, observation)
    n = model.size
    base: dict[str, float | int | str] = {
        "n": int(n), "status": "calculated", "mean_model": np.nan,
        "mean_observation": np.nan, "std_model": np.nan,
        "std_observation": np.nan, "std_ratio": np.nan,
        "correlation": np.nan, "bias": np.nan, "bias_star": np.nan,
        "rmse": np.nan, "rmse_star": np.nan,
        "rmse_centered": np.nan, "rmse_centered_star": np.nan,
        "target_x": np.nan, "ward_cost": np.nan,
        "identity_error": np.nan,
    }
    if n == 0:
        base["status"] = "no_valid_pairs"
        return base

    mean_model = float(np.mean(model))
    mean_obs = float(np.mean(observation))
    residual = model - observation
    bias = mean_model - mean_obs
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered_residual = (model - mean_model) - (observation - mean_obs)
    rmse_centered = float(np.sqrt(np.mean(centered_residual**2)))
    std_model = float(np.std(model, ddof=ddof)) if n > ddof else np.nan
    std_obs = float(np.std(observation, ddof=ddof)) if n > ddof else np.nan
    base.update({
        "mean_model": mean_model, "mean_observation": mean_obs,
        "std_model": std_model, "std_observation": std_obs,
        "bias": bias, "rmse": rmse, "rmse_centered": rmse_centered,
    })

    if n > 1 and std_model > 0 and std_obs > 0:
        base["correlation"] = float(np.corrcoef(model, observation)[0, 1])
    if not np.isfinite(std_obs) or std_obs <= 0:
        base["status"] = "nonpositive_observation_std"
        return base

    bias_star = bias / std_obs
    rmse_star = rmse / std_obs
    rmse_centered_star = rmse_centered / std_obs
    std_ratio = std_model / std_obs
    target_sign = 1.0 if std_model >= std_obs else -1.0
    ward_cost = rmse_star**2
    identity_error = rmse_star**2 - (bias_star**2 + rmse_centered_star**2)
    base.update({
        "std_ratio": std_ratio, "bias_star": bias_star,
        "rmse_star": rmse_star, "rmse_centered_star": rmse_centered_star,
        "target_x": target_sign * rmse_centered_star,
        "ward_cost": ward_cost, "identity_error": identity_error,
    })
    return base


def _depth_values(array: xr.DataArray) -> list[str]:
    return [str(v) for v in array["Depth"].values.tolist()]


def extract_pairs(dataset: xr.Dataset, spec: VariableSpec, depth: str) -> tuple[np.ndarray, np.ndarray]:
    if spec.model not in dataset or spec.observation not in dataset:
        return np.array([], dtype=float), np.array([], dtype=float)
    model = dataset[spec.model]
    observation = dataset[spec.observation]
    if model.dims != observation.dims or model.shape != observation.shape:
        raise ValueError(f"{spec.key}: model and observation grids do not match")
    if spec.depth_resolved and depth != "Total":
        if "Depth" not in model.dims or depth not in _depth_values(model):
            return np.array([], dtype=float), np.array([], dtype=float)
        model = model.sel(Depth=depth)
        observation = observation.sel(Depth=depth)
    return pairwise_finite(model.values, observation.values)


def _period_label(years: Sequence[int]) -> str:
    years = sorted(set(int(y) for y in years))
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]} pooled"


def _metric_rows(
    datasets: Mapping[int, xr.Dataset], specs: Sequence[VariableSpec], ddof: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    coverage: list[dict] = []
    pair_cache: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    all_years = sorted(datasets)

    for year, dataset in datasets.items():
        for spec in specs:
            depths = DEPTHS if spec.depth_resolved else ("Not applicable",)
            available = spec.model in dataset and spec.observation in dataset
            for depth in depths:
                model, obs = extract_pairs(dataset, spec, depth)
                pair_cache[(year, spec.key, depth)] = (model, obs)
                metrics = calculate_skill(model, obs, ddof=ddof)
                rows.append({
                    "period": str(year), "period_type": "annual", "year": year,
                    "source": spec.source, "variable": spec.label,
                    "variable_key": spec.key, "depth": depth, "units": spec.units,
                    "component": component_name(spec, depth), **metrics,
                })
                coverage.append({
                    "year": year, "source": spec.source, "variable": spec.label,
                    "variable_key": spec.key, "depth": depth,
                    "model_variable": spec.model, "observation_variable": spec.observation,
                    "variables_available": available, "valid_pair_count": len(model),
                    "status": metrics["status"],
                })

    pooled_label = _period_label(all_years)
    for spec in specs:
        depths = DEPTHS if spec.depth_resolved else ("Not applicable",)
        for depth in depths:
            pieces = [pair_cache[(year, spec.key, depth)] for year in all_years]
            models = np.concatenate([p[0] for p in pieces]) if pieces else np.array([])
            observations = np.concatenate([p[1] for p in pieces]) if pieces else np.array([])
            metrics = calculate_skill(models, observations, ddof=ddof)
            rows.append({
                "period": pooled_label, "period_type": "pooled", "year": np.nan,
                "source": spec.source, "variable": spec.label,
                "variable_key": spec.key, "depth": depth, "units": spec.units,
                "component": component_name(spec, depth), **metrics,
            })
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def load_priority_weights(path: Path, specs: Sequence[VariableSpec]) -> pd.DataFrame:
    weights = pd.read_csv(path)
    required = {"component", "priority_weight"}
    if not required.issubset(weights.columns):
        raise ValueError(f"Weight file must contain columns {sorted(required)}")
    if weights["component"].duplicated().any():
        raise ValueError("Priority-weight component names must be unique")
    weights["priority_weight"] = pd.to_numeric(weights["priority_weight"], errors="raise")
    if (~np.isfinite(weights["priority_weight"]) | (weights["priority_weight"] < 0)).any():
        raise ValueError("Priority weights must be finite and nonnegative")
    expected = {
        component_name(spec, depth)
        for spec in specs
        for depth in (("Surface", "Bottom") if spec.depth_resolved else ("Not applicable",))
    }
    missing = expected - set(weights["component"])
    if missing:
        raise ValueError(f"Weight file is missing components: {sorted(missing)}")
    return weights


def _cost_tables(metrics: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    independent = metrics.loc[metrics["depth"] != "Total"].copy()
    independent = independent.merge(weights[["component", "priority_weight"]], on="component", how="left", validate="many_to_one")
    independent["included"] = (
        independent["ward_cost"].notna() & (independent["priority_weight"] > 0)
    )
    independent["exclusion_reason"] = np.where(
        independent["ward_cost"].isna(), independent["status"],
        np.where(independent["priority_weight"].eq(0), "zero_priority_weight", "included"),
    )
    independent["weighted_contribution"] = np.where(
        independent["included"], independent["ward_cost"] * independent["priority_weight"], np.nan
    )
    summaries = []
    for (period, period_type), group in independent.groupby(["period", "period_type"], sort=False):
        included = group.loc[group["included"]]
        weighted_sum = float(included["weighted_contribution"].sum())
        weight_sum = float(included["priority_weight"].sum())
        summaries.append({
            "period": period, "period_type": period_type,
            "weighted_cost_sum": weighted_sum,
            "weighted_cost_mean": weighted_sum / weight_sum if weight_sum > 0 else np.nan,
            "included_component_count": len(included),
            "excluded_component_count": len(group) - len(included),
            "included_priority_weight_sum": weight_sum,
            "expected_component_count": len(group),
        })
    return independent, pd.DataFrame(summaries)


def _legacy_rows(datasets: Mapping[int, xr.Dataset], specs: Sequence[VariableSpec], costs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, ds in datasets.items():
        annual = costs.loc[costs["period"].eq(str(year))]
        for spec in specs:
            depths = ("Surface", "Bottom") if spec.depth_resolved else ("Not applicable",)
            for depth in depths:
                legacy = np.nan
                status = "legacy_cost_unavailable"
                if spec.legacy_cost in ds:
                    value = ds[spec.legacy_cost]
                    if spec.depth_resolved and "Depth" in value.dims and depth in _depth_values(value):
                        value = value.sel(Depth=depth)
                    try:
                        legacy = float(value.item())
                        status = "available" if np.isfinite(legacy) else "legacy_cost_nan"
                    except ValueError:
                        pass
                component = component_name(spec, depth)
                match = annual.loc[annual["component"].eq(component)]
                rows.append({
                    "period": str(year), "source": spec.source, "variable": spec.label,
                    "variable_key": spec.key, "depth": depth, "component": component,
                    "ward_cost": match["ward_cost"].iloc[0] if len(match) else np.nan,
                    "legacy_cost": legacy, "legacy_status": status,
                    "legacy_minus_ward": legacy - match["ward_cost"].iloc[0]
                    if len(match) and np.isfinite(legacy) else np.nan,
                })
    return pd.DataFrame(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _point_label(row: pd.Series) -> str:
    prefix = {"NBFSMN": "", "PLT": "PLT ", "CHRP": "CHRP ", "PLT Secchi": ""}.get(row["source"], f"{row['source']} ")
    depth = "" if row["depth"] == "Not applicable" else f" {row['depth'][0]}"
    return f"{prefix}{row['variable']}{depth}"


def _target_panel(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    finite = data[np.isfinite(data["target_x"]) & np.isfinite(data["bias_star"])]
    magnitude = np.sqrt(finite["target_x"] ** 2 + finite["bias_star"] ** 2)
    limit = max(1.1, float(magnitude.max() * 1.18) if len(magnitude) else 1.1)
    for mef, color in ((0.75, "#117733"), (0.5, "#88CCEE"), (0.0, "#BBBBBB")):
        radius = math.sqrt(1 - mef)
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=color, lw=1.2, ls="--"))
        ax.text(radius / math.sqrt(2), radius / math.sqrt(2), f"MEF={mef:g}", color=color, fontsize=7)
    for _, row in finite.iterrows():
        ax.scatter(row["target_x"], row["bias_star"], s=45,
                   marker=DEPTH_MARKERS.get(row["depth"], "o"),
                   color=SOURCE_COLORS.get(row["source"], "#555555"), edgecolor="black", linewidth=0.5)
        ax.annotate(_point_label(row), (row["target_x"], row["bias_star"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
           xlabel="Signed centered RMSE*", ylabel="B*", title=title)
    ax.grid(alpha=0.2)


def plot_target_diagram(metrics: pd.DataFrame, period: str, output: Path, grouped: bool = True) -> None:
    data = metrics.loc[metrics["period"].eq(period)]
    groups = (("NBFSMN", ["NBFSMN"]), ("PLT", ["PLT", "PLT Secchi"]), ("CHRP", ["CHRP"])) if grouped else ((period, list(data["source"].unique())),)
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5.5), squeeze=False)
    for ax, (title, sources) in zip(axes.ravel(), groups):
        _target_panel(ax, data[data["source"].isin(sources)], title)
    fig.suptitle(f"{period}: Jolliff-style target diagram", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _authoritative_groups(metrics: pd.DataFrame, period: str) -> list[dict]:
    """Select one primary point per data type plus optional depth satellites."""
    pooled = metrics.loc[metrics["period"].eq(period)]
    groups = []
    for index, spec in enumerate(selected_specs(())):
        available = pooled.loc[pooled["variable_key"].eq(spec.key)]
        if available.empty:
            continue
        if not spec.depth_resolved:
            primary_depth = "Not applicable"
        elif spec.source == "CHRP":
            primary_depth = "Surface"
        else:
            primary_depth = "Total"
        primary = available.loc[available["depth"].eq(primary_depth)]
        if primary.empty or not np.isfinite(primary.iloc[0]["rmse"]):
            continue
        satellites = available.loc[
            available["depth"].isin(["Surface", "Bottom"])
            & available["rmse"].notna()
            & ~available["depth"].eq(primary_depth)
        ]
        groups.append({
            "id": chr(ord("A") + len(groups)),
            "color": AUTHORITATIVE_COLORS[index % len(AUTHORITATIVE_COLORS)],
            "spec": spec, "primary": primary.iloc[0], "satellites": satellites,
            "available": available,
        })
    return groups


def _rmse_text(row: pd.Series | None, units: str) -> str:
    if row is None or not np.isfinite(row.get("rmse", np.nan)):
        return "—"
    return f"{row['rmse']:.3g} {units}"


def _row_at_depth(data: pd.DataFrame, depth: str) -> pd.Series | None:
    match = data.loc[data["depth"].eq(depth)]
    return match.iloc[0] if len(match) else None


def _add_authoritative_table(ax: plt.Axes, groups: Sequence[dict], show_title: bool = True) -> None:
    ax.axis("off")
    rows = []
    for group in groups:
        spec = group["spec"]
        primary = group["primary"]
        basis = {"Total": "Total", "Surface": "Surface only", "Not applicable": "Scalar"}[primary["depth"]]
        data_type = "PLT Secchi depth" if spec.source == "PLT Secchi" else f"{spec.source} {spec.label}"
        rows.append([
            group["id"], data_type,
            f"{_rmse_text(primary, spec.units)}\n({basis})",
            _rmse_text(_row_at_depth(group["available"], "Surface"), spec.units),
            _rmse_text(_row_at_depth(group["available"], "Bottom"), spec.units),
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["ID", "Data type", "Primary RMSE\n(dimensional)", "Surface RMSE", "Bottom RMSE"],
        colWidths=[.06, .27, .27, .20, .20],
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for column in range(5):
        table[(0, column)].set_facecolor("#D9E2F3")
        table[(0, column)].set_text_props(weight="bold")
    for row_index, group in enumerate(groups, start=1):
        table[(row_index, 0)].set_facecolor(group["color"])
        table[(row_index, 0)].set_text_props(color="white", weight="bold", ha="center")
        for column in range(5):
            table[(row_index, column)].set_edgecolor("0.82")
    if show_title:
        ax.set_title("Point identity and dimensional RMSE", fontsize=11, weight="bold", pad=8)
    ax.text(
        0, .015,
        "Primary = pooled Total for depth-resolved NBFSMN/PLT data;\n"
        "CHRP uses Surface because bottom observations are unavailable.",
        transform=ax.transAxes, fontsize=8, color="0.3", va="bottom",
    )


def _depth_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25", markeredgecolor="black", markersize=10, label="Primary pooled result"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="0.45", alpha=.42, markersize=7, label="Surface satellite"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="0.45", alpha=.42, markersize=7, label="Bottom satellite"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, title="Point hierarchy", title_fontsize=8)


def plot_authoritative_target(metrics: pd.DataFrame, period: str, output: Path) -> None:
    groups = _authoritative_groups(metrics, period)
    fig = plt.figure(figsize=(18, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.15), wspace=.16)
    ax = fig.add_subplot(grid[0, 0]); table_ax = fig.add_subplot(grid[0, 1])
    all_rows = [g["primary"] for g in groups]
    all_rows.extend(row for g in groups for _, row in g["satellites"].iterrows())
    magnitude = [math.hypot(float(r["target_x"]), float(r["bias_star"])) for r in all_rows if np.isfinite(r["target_x"]) and np.isfinite(r["bias_star"])]
    limit = max(1.1, max(magnitude, default=1) * 1.16)
    for mef, color in ((0.75, "#117733"), (0.5, "#88CCEE"), (0.0, "#BBBBBB")):
        radius = math.sqrt(1 - mef)
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=color, lw=1.2, ls="--"))
        ax.text(radius / math.sqrt(2), radius / math.sqrt(2), f"MEF={mef:g}", color=color, fontsize=7)
    for group in groups:
        primary = group["primary"]; satellites = group["satellites"]; color = group["color"]
        connected = []
        for depth in ("Surface", "Total", "Bottom"):
            candidate = primary if primary["depth"] == depth else _row_at_depth(satellites, depth)
            if candidate is not None and np.isfinite(candidate["target_x"]) and np.isfinite(candidate["bias_star"]):
                connected.append(candidate)
        if len(connected) > 1:
            ax.plot([r["target_x"] for r in connected], [r["bias_star"] for r in connected], color=color, alpha=.25, lw=1)
        for _, row in satellites.iterrows():
            ax.scatter(row["target_x"], row["bias_star"], s=50,
                       marker="^" if row["depth"] == "Surface" else "v",
                       color=color, alpha=.38, edgecolor="black", linewidth=.5)
        ax.scatter(primary["target_x"], primary["bias_star"], s=150, marker="o", color=color, edgecolor="black", linewidth=.8, zorder=4)
        ax.text(primary["target_x"], primary["bias_star"], group["id"], color="white", weight="bold", fontsize=8, ha="center", va="center", zorder=5)
    ax.axhline(0, color="0.3", lw=.7); ax.axvline(0, color="0.3", lw=.7)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
           xlabel="Signed centered RMSE*", ylabel="B*",
           title="Large labeled points are authoritative pooled results")
    ax.grid(alpha=.2); _depth_legend(ax); _add_authoritative_table(table_ax, groups)
    fig.suptitle(f"{period}: authoritative Jolliff-style target diagram", fontsize=16, y=.98)
    fig.subplots_adjust(top=.90, bottom=.08, left=.05, right=.98, wspace=.16)
    fig.savefig(output, dpi=230, bbox_inches="tight"); plt.close(fig)


def _taylor_panel(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    finite = data[np.isfinite(data["std_ratio"]) & np.isfinite(data["correlation"])]
    max_std = max(1.6, float(finite["std_ratio"].max() * 1.15) if len(finite) else 1.6)
    theta = np.linspace(0, np.pi, 181)
    radius = np.linspace(0, max_std, 120)
    tt, rr = np.meshgrid(theta, radius)
    centered = np.sqrt(1 + rr**2 - 2 * rr * np.cos(tt))
    levels = [0.25, 0.5, 1.0, 1.5, 2.0]
    levels = [level for level in levels if level < max_std + 1]
    contours = ax.contour(tt, rr, centered, levels=levels, colors="0.65", linewidths=0.7)
    ax.clabel(contours, fontsize=6, fmt="%.2g")
    ax.scatter([0], [1], marker="*", s=120, color="black", label="Observations", zorder=5)
    for _, row in finite.iterrows():
        angle = math.acos(float(np.clip(row["correlation"], -1, 1)))
        ax.scatter(angle, row["std_ratio"], s=45,
                   marker=DEPTH_MARKERS.get(row["depth"], "o"),
                   color=SOURCE_COLORS.get(row["source"], "#555555"), edgecolor="black", linewidth=0.5)
        ax.annotate(_point_label(row), (angle, row["std_ratio"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    correlations = np.array([1, .9, .7, .5, 0, -.5, -.9, -1])
    ax.set_xticks(np.arccos(correlations))
    ax.set_xticklabels([f"{r:g}" for r in correlations])
    ax.set_thetamin(0); ax.set_thetamax(180); ax.set_ylim(0, max_std)
    ax.set_xlabel("Correlation: −1 at left, +1 at right", labelpad=12)
    ax.set_title(title, pad=18)
    ax.grid(alpha=0.25)


TAYLOR_GUIDE = (
    "HOW TO READ   ★ = perfect agreement (R=1, σM/σO=1)   •   "
    "angle = correlation   •   radius = σM/σO "
    "(<1 under-variable; >1 over-variable)   •   "
    "gray contours = centered RMSE* (smaller toward ★)"
)


def plot_taylor_diagram(metrics: pd.DataFrame, period: str, output: Path, grouped: bool = True) -> None:
    data = metrics.loc[metrics["period"].eq(period)]
    groups = (("NBFSMN", ["NBFSMN"]), ("PLT", ["PLT", "PLT Secchi"]), ("CHRP", ["CHRP"])) if grouped else ((period, list(data["source"].unique())),)
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5.7), subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, (title, sources) in zip(axes.ravel(), groups):
        _taylor_panel(ax, data[data["source"].isin(sources)], title)
    fig.suptitle(f"{period}: normalized Taylor diagram", fontsize=15, y=.98)
    fig.text(
        .5, .89, TAYLOR_GUIDE, ha="center", va="center", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F6F7", "edgecolor": "0.7"},
    )
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_authoritative_taylor(metrics: pd.DataFrame, period: str, output: Path) -> None:
    groups = _authoritative_groups(metrics, period)
    finite_rows = [g["primary"] for g in groups]
    finite_rows.extend(row for g in groups for _, row in g["satellites"].iterrows())
    ratios = [float(row["std_ratio"]) for row in finite_rows if np.isfinite(row["std_ratio"])]
    max_std = max(1.6, max(ratios, default=1) * 1.12)

    fig = plt.figure(figsize=(18, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.15), wspace=.16)
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    table_ax = fig.add_subplot(grid[0, 1])
    theta = np.linspace(0, np.pi, 181)
    radius = np.linspace(0, max_std, 140)
    tt, rr = np.meshgrid(theta, radius)
    centered = np.sqrt(1 + rr**2 - 2 * rr * np.cos(tt))
    levels = [level for level in (.25, .5, 1, 1.5, 2, 2.5, 3) if level < max_std + 1]
    contours = ax.contour(tt, rr, centered, levels=levels, colors="0.65", linewidths=.8)
    ax.clabel(contours, fontsize=7, fmt="%.2g")
    ax.scatter([0], [1], marker="*", s=170, color="black", zorder=6)

    for group in groups:
        primary = group["primary"]; satellites = group["satellites"]; color = group["color"]
        connected = []
        for depth in ("Surface", "Total", "Bottom"):
            candidate = primary if primary["depth"] == depth else _row_at_depth(satellites, depth)
            if candidate is not None and np.isfinite(candidate["std_ratio"]) and np.isfinite(candidate["correlation"]):
                connected.append(candidate)
        if len(connected) > 1:
            ax.plot(
                [math.acos(float(np.clip(r["correlation"], -1, 1))) for r in connected],
                [r["std_ratio"] for r in connected], color=color, alpha=.25, lw=1,
            )
        for _, row in satellites.iterrows():
            if not np.isfinite(row["std_ratio"]) or not np.isfinite(row["correlation"]):
                continue
            ax.scatter(math.acos(float(np.clip(row["correlation"], -1, 1))), row["std_ratio"], s=50,
                       marker="^" if row["depth"] == "Surface" else "v",
                       color=color, alpha=.38, edgecolor="black", linewidth=.5)
        angle = math.acos(float(np.clip(primary["correlation"], -1, 1)))
        ax.scatter(angle, primary["std_ratio"], s=150, marker="o", color=color,
                   edgecolor="black", linewidth=.8, zorder=4)
        ax.text(angle, primary["std_ratio"], group["id"], color="white", weight="bold",
                fontsize=8, ha="center", va="center", zorder=5)

    correlations = np.array([1, .9, .7, .5, 0, -.5, -.9, -1])
    ax.set_xticks(np.arccos(correlations)); ax.set_xticklabels([f"{r:g}" for r in correlations])
    ax.set_thetamin(0); ax.set_thetamax(180); ax.set_ylim(0, max_std)
    ax.set_xlabel("Correlation: −1 at left, +1 at right", labelpad=15)
    ax.grid(alpha=.25); _depth_legend(ax); _add_authoritative_table(table_ax, groups, show_title=False)
    fig.suptitle(f"{period}: authoritative normalized Taylor diagram", fontsize=16, y=.98)
    fig.text(
        .5, .91, TAYLOR_GUIDE, ha="center", va="center", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F6F7", "edgecolor": "0.7"},
    )
    fig.subplots_adjust(top=.84, bottom=.08, left=.04, right=.98, wspace=.16)
    fig.savefig(output, dpi=230, bbox_inches="tight"); plt.close(fig)


def _annual_diagram(metrics: pd.DataFrame, output: Path, kind: str) -> None:
    annual = metrics[(metrics["period_type"] == "annual") & (metrics["depth"] != "Total")]
    periods = annual["period"].drop_duplicates().tolist()
    subplot_kw = {"projection": "polar"} if kind == "taylor" else {}
    fig, axes = plt.subplots(1, len(periods), figsize=(6 * len(periods), 5.5), subplot_kw=subplot_kw, squeeze=False)
    for ax, period in zip(axes.ravel(), periods):
        subset = annual[annual["period"].eq(period)]
        (_taylor_panel if kind == "taylor" else _target_panel)(ax, subset, period)
    fig.suptitle(f"Annual {'Taylor' if kind == 'taylor' else 'target'} diagrams", fontsize=15, y=.98)
    if kind == "taylor":
        fig.text(
            .5, .89, TAYLOR_GUIDE, ha="center", va="center", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F6F7", "edgecolor": "0.7"},
        )
        fig.tight_layout(rect=(0, 0, 1, .86))
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _dates(dataset: xr.Dataset, coord: str) -> np.ndarray:
    values = dataset[coord].values
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values).to_numpy()
    if coord == "Day":
        year = int(dataset.attrs["year"])
        # Cost files store Day as one-based local day of year without a CF
        # ``units`` attribute, so generic datetime conversion would yield 1970.
        return (pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(values.astype(int) - 1, unit="D")).to_numpy()
    raise ValueError(f"Coordinate {coord!r} is numeric and has no supported date convention")


def _series(dataset: xr.Dataset, spec: VariableSpec, depth: str, station_index: int | None = None):
    obs = dataset[spec.observation]
    mod = dataset[spec.model]
    if spec.depth_resolved:
        obs = obs.sel(Depth=depth); mod = mod.sel(Depth=depth)
    if station_index is not None and spec.station_dimension:
        obs = obs.isel({spec.station_dimension: station_index})
        mod = mod.isel({spec.station_dimension: station_index})
    valid = np.isfinite(obs.values) & np.isfinite(mod.values)
    return obs.values[valid], mod.values[valid], _dates(dataset, spec.time)[valid]


def _station_labels(dataset: xr.Dataset, dimension: str) -> list[str]:
    if dimension in dataset.coords:
        values = dataset[dimension].values.tolist()
        return [str(v) for v in values]
    return [str(i + 1) for i in range(dataset.sizes[dimension])]


def _plot_model_line(ax: plt.Axes, dates: np.ndarray, values: np.ndarray, spec: VariableSpec, label: str | None) -> None:
    """Plot sampled model values without connecting long unsampled gaps."""
    if len(dates) == 0:
        return
    order = np.argsort(dates)
    dates = np.asarray(dates)[order]
    values = np.asarray(values)[order]
    gap_days = 14 if spec.source == "NBFSMN" else 60
    breaks = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > gap_days)[0] + 1
    for index, (date_piece, value_piece) in enumerate(zip(np.split(dates, breaks), np.split(values, breaks))):
        ax.plot(date_piece, value_piece, lw=.8, marker=".", ms=2.5,
                color="#0072B2", alpha=.85, label=label if index == 0 else None)


def _plot_station_grid(datasets: Mapping[int, xr.Dataset], spec: VariableSpec, depth: str, output: Path, residual: bool) -> None:
    first = next(ds for ds in datasets.values() if spec.observation in ds)
    labels = _station_labels(first, spec.station_dimension)
    count = len(labels); ncols = min(4, count); nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.8 * nrows), sharex=False, squeeze=False)
    legend_assigned = False
    for index, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        total_n = 0
        for year, ds in datasets.items():
            if spec.observation not in ds or ds.sizes.get(spec.time, 0) == 0:
                continue
            obs, mod, dates = _series(ds, spec, depth, index)
            total_n += len(obs)
            if residual:
                ax.scatter(dates, mod - obs, s=9, color="#D55E00", alpha=.75)
            else:
                legend_here = not legend_assigned and len(obs) > 0
                ax.scatter(dates, obs, s=8, color="black", alpha=.7,
                           label="Observation" if legend_here else None)
                _plot_model_line(ax, dates, mod, spec, "Model" if legend_here else None)
                legend_assigned = legend_assigned or legend_here
        if residual: ax.axhline(0, color="0.25", lw=.7)
        if total_n == 0:
            ax.set_visible(False)
            continue
        ax.set_title(f"{label} (n={total_n})", fontsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(alpha=.18)
    for ax in axes.ravel()[count:]: ax.set_visible(False)
    if not residual:
        for ax in axes.ravel():
            handles, legend_labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
                break
    ylabel = f"Model − observation ({spec.units})" if residual else spec.units
    fig.supylabel(ylabel)
    fig.suptitle(f"{spec.source} {spec.label} — {depth}" + (" residuals" if residual else ""), fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_nonstation_series(datasets: Mapping[int, xr.Dataset], spec: VariableSpec, output: Path, residual: bool) -> None:
    depths = ("Surface", "Bottom") if spec.depth_resolved else ("Not applicable",)
    fig, axes = plt.subplots(len(depths), 1, figsize=(11, 3.2 * len(depths)), squeeze=False)
    for ax, depth in zip(axes.ravel(), depths):
        total_n = 0
        legend_assigned = False
        for year, ds in datasets.items():
            if spec.observation not in ds or ds.sizes.get(spec.time, 0) == 0:
                continue
            obs, mod, dates = _series(ds, spec, depth)
            total_n += len(obs)
            if residual:
                ax.scatter(dates, mod - obs, s=16, color="#D55E00", alpha=.8)
            else:
                legend_here = not legend_assigned and len(obs) > 0
                ax.scatter(dates, obs, s=18, color="black", label="Observation" if legend_here else None)
                _plot_model_line(ax, dates, mod, spec, "Model" if legend_here else None)
                legend_assigned = legend_assigned or legend_here
        if residual: ax.axhline(0, color="0.25", lw=.7)
        ax.set_title(f"{depth} (n={total_n})")
        ax.set_ylabel(f"Model − observation ({spec.units})" if residual else spec.units)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(alpha=.2)
        if not residual and legend_assigned:
            ax.legend(fontsize=8)
    fig.suptitle(f"{spec.source} {spec.label}" + (" residuals" if residual else ""), fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(datasets: Mapping[int, xr.Dataset], specs: Sequence[VariableSpec], directory: Path) -> None:
    for spec in specs:
        if spec.time is None or not any(spec.observation in ds for ds in datasets.values()):
            continue
        source_dir = directory / _safe_name(spec.source.lower())
        residual_dir = directory / "residuals" / _safe_name(spec.source.lower())
        source_dir.mkdir(parents=True, exist_ok=True); residual_dir.mkdir(parents=True, exist_ok=True)
        if spec.station_dimension:
            depths = ("Surface", "Bottom") if spec.depth_resolved else ("Not applicable",)
            for depth in depths:
                # Skip all-empty CHRP bottom figures.
                if sum(len(extract_pairs(ds, spec, depth)[0]) for ds in datasets.values()) == 0:
                    continue
                stem = f"{spec.key}_{depth.lower().replace(' ', '_')}"
                _plot_station_grid(datasets, spec, depth, source_dir / f"{stem}.png", False)
                _plot_station_grid(datasets, spec, depth, residual_dir / f"{stem}_residuals.png", True)
        else:
            _plot_nonstation_series(datasets, spec, source_dir / f"{spec.key}.png", False)
            _plot_nonstation_series(datasets, spec, residual_dir / f"{spec.key}_residuals.png", True)


def plot_cost_components(costs: pd.DataFrame, period: str, output: Path) -> None:
    data = costs[(costs["period"].eq(period)) & costs["included"]].copy()
    data = data.sort_values("weighted_contribution")
    fig, ax = plt.subplots(figsize=(9, max(5, .35 * len(data))))
    labels = [f"{s}: {v} {d}" for s, v, d in zip(data.source, data.variable, data.depth)]
    ax.barh(labels, data["weighted_contribution"], color=[SOURCE_COLORS.get(s, "#777777") for s in data.source])
    ax.set_xlabel("Weighted Ward cost contribution")
    ax.set_title(f"{period}: component contributions")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def _write_outputs(
    output_dir: Path, metrics: pd.DataFrame, costs: pd.DataFrame,
    summaries: pd.DataFrame, coverage: pd.DataFrame, legacy: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    tables = output_dir / "tables"; tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "metrics.csv", index=False)
    costs.to_csv(tables / "ward_cost_components.csv", index=False)
    summaries.to_csv(tables / "ward_cost_summary.csv", index=False)
    coverage.to_csv(tables / "data_coverage.csv", index=False)
    legacy.to_csv(tables / "legacy_cost_comparison.csv", index=False)
    weights.to_csv(tables / "applied_priority_weights.csv", index=False)


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xarray": xr.__version__, "matplotlib": matplotlib.__version__}


def run_analysis(
    run_name: str,
    years: Sequence[int],
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    include_optional: Iterable[str] = (),
    ddof: int = 0,
    make_time_series: bool = True,
    make_annual_diagrams: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the complete analysis and return tables plus output paths."""
    years = sorted(set(int(y) for y in years))
    if not years or ddof < 0:
        raise ValueError("At least one year and a nonnegative ddof are required")
    input_root = Path(input_root); output_root = Path(output_root); weights_path = Path(weights_path)
    safe_run = _safe_name(run_name)
    year_range = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    output_dir = output_root / safe_run / year_range
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use overwrite=True to replace generated files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {year: input_root / f"{run_name}_{year}.nc" for year in years}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    datasets = {year: xr.open_dataset(path) for year, path in paths.items()}
    try:
        specs = selected_specs(include_optional)
        metrics, coverage = _metric_rows(datasets, specs, ddof)
        weights = load_priority_weights(weights_path, specs)
        costs, summaries = _cost_tables(metrics, weights)
        legacy = _legacy_rows(datasets, specs, costs)
        _write_outputs(output_dir, metrics, costs, summaries, coverage, legacy, weights)

        figures = output_dir / "figures"
        for subdir in ("target", "taylor", "cost", "time_series"):
            (figures / subdir).mkdir(parents=True, exist_ok=True)
        pooled = _period_label(years)
        target_png = figures / "target" / "target_diagram_pooled.png"
        taylor_png = figures / "taylor" / "taylor_diagram_pooled.png"
        # Unified authoritative figures use one dominant point per data type;
        # source-panel versions are retained as secondary diagnostics.
        plot_authoritative_target(metrics, pooled, target_png)
        plot_authoritative_taylor(metrics, pooled, taylor_png)
        plot_authoritative_target(metrics, pooled, figures / "target" / "target_diagram_pooled.pdf")
        plot_authoritative_taylor(metrics, pooled, figures / "taylor" / "taylor_diagram_pooled.pdf")
        plot_target_diagram(metrics, pooled, figures / "target" / "target_diagram_by_source.png", grouped=True)
        plot_taylor_diagram(metrics, pooled, figures / "taylor" / "taylor_diagram_by_source.png", grouped=True)
        if make_annual_diagrams:
            _annual_diagram(metrics, figures / "target" / "target_diagram_annual.png", "target")
            _annual_diagram(metrics, figures / "taylor" / "taylor_diagram_annual.png", "taylor")
        plot_cost_components(costs, pooled, figures / "cost" / "ward_cost_components.png")
        if make_time_series:
            plot_time_series(datasets, specs, figures / "time_series")

        manifest = {
            "run_name": run_name, "years": years, "period": pooled,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_files": {str(y): {"path": str(p), "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(), "size_bytes": p.stat().st_size} for y, p in paths.items()},
            "output_directory": str(output_dir), "weights_file": str(weights_path),
            "include_optional": sorted(include_optional), "ddof": ddof,
            "pooling": "concatenate all jointly finite model-observation pairs before calculating statistics",
            "ward_cost": "mean((model-observation)^2) / observation_standard_deviation^2",
            "priority_weighting": "component Ward cost multiplied by nonnegative priority weight",
            "total_depth_policy": "reported diagnostically; excluded from aggregate cost when surface and bottom components are present",
            "software_versions": _versions(),
        }
        (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        for dataset in datasets.values(): dataset.close()

    return {"metrics": metrics, "cost_components": costs, "cost_summary": summaries,
            "coverage": coverage, "legacy_comparison": legacy,
            "output_directory": output_dir, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, dest="run_name")
    parser.add_argument("--years", required=True, nargs="+", type=int)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--include", nargs="*", choices=("primary_production", "benthic"), default=[])
    parser.add_argument("--ddof", type=int, default=0)
    parser.add_argument("--no-time-series", action="store_true")
    parser.add_argument("--no-annual-diagrams", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        run_name=args.run_name, years=args.years, input_root=args.input_root,
        output_root=args.output_root, weights_path=args.weights,
        include_optional=args.include, ddof=args.ddof,
        make_time_series=not args.no_time_series,
        make_annual_diagrams=not args.no_annual_diagrams,
        overwrite=args.overwrite,
    )
    print(f"Analysis complete: {result['output_directory']}")
    print(result["cost_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
