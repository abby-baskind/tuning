"""Interpretive plots for the canonical MO-BIO notebook.

The functions in this module are deliberately side-effect light: they read
the supplied tables (and, for station diagnostics, cost-summary NetCDFs),
return any derived diagnostic tables, and save figures only when requested.
They never modify an Optuna study or a run inventory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


UPPER_BAY_STATIONS = {"PD", "BR", "CP"}


def _finish(fig, output: Path | None, *, dpi: int = 220) -> None:
    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _heatmap(
    ax,
    values: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    fmt: str | None = None,
) -> None:
    matrix = values.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(values.shape[1]), values.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(values.shape[0]), values.index)
    ax.set_title(title)
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(colorbar_label)
    if fmt and matrix.size <= 300:
        for row, column in np.argwhere(np.isfinite(matrix)):
            ax.text(column, row, format(matrix[row, column], fmt), ha="center", va="center", fontsize=7)


def plot_pareto_fronts(
    archive: pd.DataFrame,
    objective_columns: Sequence[str],
    *,
    maximum_rank: int,
    output: Path | None = None,
) -> None:
    """Restore explicit Pareto and near-Pareto rank curves."""

    cost, bottom_ph = objective_columns
    rank = pd.to_numeric(archive["Pareto Rank"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    outside = rank.gt(maximum_rank) | rank.isna()
    ax.scatter(
        archive.loc[outside, cost], archive.loc[outside, bottom_ph],
        color="lightgray", alpha=0.55, s=30,
        label=f"Outside selected ranks (>{maximum_rank})", zorder=1,
    )
    colors = ["crimson", "darkorange", "goldenrod", "seagreen", "royalblue", "mediumpurple"]
    for rank_number in range(maximum_rank + 1):
        front = archive.loc[rank.eq(rank_number)].sort_values(cost)
        if front.empty:
            continue
        exact = rank_number == 0
        ax.plot(
            front[cost], front[bottom_ph], marker="o",
            linestyle="-" if exact else "--",
            linewidth=2.2 if exact else 1.35,
            color=colors[rank_number % len(colors)],
            label="Pareto front (rank 0)" if exact else f"Near-Pareto front (rank {rank_number})",
            zorder=3,
        )
        for _, row in front.iterrows():
            ax.annotate(str(row["Run Name"]), (row[cost], row[bottom_ph]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7, alpha=0.85)
    ax.set(
        xlabel=f"{cost} (lower is better)",
        ylabel=f"{bottom_ph} (lower is better)",
        title=f"Canonical Pareto fronts through rank {maximum_rank}",
    )
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    _finish(fig, output)


def plot_model_year_context(
    archive: pd.DataFrame,
    year_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    objective_columns: Sequence[str],
    *,
    maximum_rank: int,
    output_directory: Path | None = None,
) -> None:
    """Plot global year context plus paired changes and match-group panels."""

    cost, bottom_ph = objective_columns
    year_values = pd.to_numeric(archive["dstart_year"], errors="coerce")
    years = sorted(year_values.dropna().astype(int).unique())
    colors = dict(zip(years, plt.cm.tab10(np.linspace(0, 1, max(1, len(years))))))
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), gridspec_kw={"width_ratios": [2.4, 1]})
    objective_axis, coverage_axis = axes
    rank = pd.to_numeric(archive["Pareto Rank"], errors="coerce")
    for year in years:
        rows = archive.loc[year_values.eq(year)]
        objective_axis.scatter(rows[cost], rows[bottom_ph], color=colors[year], alpha=0.55,
                               s=42, label=str(year), zorder=2)
        exact = rows["Pareto Rank"].eq(0)
        near = rows["Pareto Rank"].between(1, maximum_rank)
        objective_axis.scatter(rows.loc[near, cost], rows.loc[near, bottom_ph], facecolors="none",
                               edgecolors=colors[year], s=105, linewidths=1.6, zorder=3)
        objective_axis.scatter(rows.loc[exact, cost], rows.loc[exact, bottom_ph], marker="*",
                               color=colors[year], edgecolors="black", s=220, linewidths=0.8, zorder=4)
    for _, pair in comparisons.iterrows():
        objective_axis.annotate(
            "", xy=(pair[f"Later {cost}"], pair[f"Later {bottom_ph}"]),
            xytext=(pair[f"Earlier {cost}"], pair[f"Earlier {bottom_ph}"]),
            arrowprops={"arrowstyle": "->", "color": "0.35", "alpha": 0.45}, zorder=1,
        )
    matched = archive.loc[archive.get("Cross-Year Parameter Match", False).fillna(False)]
    if not matched.empty:
        objective_axis.scatter(matched[cost], matched[bottom_ph], facecolors="none", edgecolors="black",
                               s=145, linewidths=1.5, label="Exact cross-year match", zorder=5)
    objective_axis.set(xlabel=cost, ylabel=bottom_ph, title="Global canonical objective space")
    objective_axis.grid(alpha=0.25)
    objective_axis.legend(title="Model year / marker", fontsize=9)
    coverage_axis.bar(
        year_summary["Model Year"].astype(str), year_summary["Runs"],
        color=[colors.get(int(year), "steelblue") for year in year_summary["Model Year"]],
    )
    coverage_axis.set(xlabel="Model year", ylabel="Usable runs", title="Canonical archive coverage")
    coverage_axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Canonical year context and exact cross-year matches", fontsize=15)
    _finish(fig, None if output_directory is None else output_directory / "canonical_model_year_context.png")

    if comparisons.empty:
        return
    labels = comparisons["Earlier Run"].astype(str) + " → " + comparisons["Later Run"].astype(str)
    positions = np.arange(len(comparisons))
    fig, axes = plt.subplots(1, 2, figsize=(20, max(6, 0.55 * len(comparisons) + 2)), sharey=True)
    for axis, column in zip(axes, (f"Change in {cost}", f"Change in {bottom_ph}")):
        values = pd.to_numeric(comparisons[column], errors="coerce")
        point_colors = np.where(values <= 0, "seagreen", "firebrick")
        axis.axvline(0, color="black", linewidth=1.1)
        axis.hlines(positions, 0, values, color=point_colors, alpha=0.55)
        axis.scatter(values, positions, color=point_colors, s=60, zorder=3)
        axis.set(xlabel="Later minus earlier (negative is better)", title=column)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Exact parameter matches: canonical performance change", fontsize=15)
    _finish(fig, None if output_directory is None else output_directory / "canonical_matched_year_changes.png")

    if "Parameter Match Group" not in matched or matched.empty:
        return
    groups = list(matched.groupby("Parameter Match Group", sort=True))
    columns = 2
    rows = int(np.ceil(len(groups) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 5 * rows), squeeze=False)
    offsets = [(7, 8), (7, -15), (-7, 8), (-7, -15)]
    for panel, (axis, (group_id, members)) in enumerate(zip(axes.ravel(), groups), start=1):
        group_pairs = comparisons.loc[comparisons["Parameter Match Group"].eq(group_id)]
        for _, pair in group_pairs.iterrows():
            axis.annotate("", xy=(pair[f"Later {cost}"], pair[f"Later {bottom_ph}"]),
                          xytext=(pair[f"Earlier {cost}"], pair[f"Earlier {bottom_ph}"]),
                          arrowprops={"arrowstyle": "->", "color": "0.55", "alpha": 0.55})
        for position, (_, member) in enumerate(members.sort_values("dstart_year").iterrows()):
            year = int(member["dstart_year"])
            offset = offsets[position % len(offsets)]
            axis.scatter(member[cost], member[bottom_ph], color=colors.get(year, "steelblue"),
                         edgecolors="black", s=85, zorder=3)
            axis.annotate(str(member["Run Name"]), (member[cost], member[bottom_ph]),
                          xytext=offset, textcoords="offset points", fontsize=9,
                          ha="left" if offset[0] > 0 else "right",
                          bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "none"})
        axis.set(xlabel=cost, ylabel=bottom_ph, title=f"Match group {panel}")
        axis.margins(0.22)
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(groups):]:
        axis.set_visible(False)
    fig.suptitle("Matched runs grouped by identical biological parameters", fontsize=15)
    _finish(fig, None if output_directory is None else output_directory / "canonical_matched_parameter_groups.png")


def plot_parameter_evidence(
    archive: pd.DataFrame,
    numeric_parameters: Sequence[str],
    categorical_screening: pd.DataFrame,
    *,
    output_directory: Path | None = None,
) -> None:
    """Restore disjoint distributions, categorical intervals, parallel lines, and pairwise views."""

    parameters = [p for p in numeric_parameters if p in archive][:12]
    if parameters:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(10, 2.8 * len(parameters)), squeeze=False)
        for ax, parameter in zip(axes.ravel(), parameters):
            values = pd.to_numeric(archive[parameter], errors="coerce")
            good = values.loc[archive["Near Pareto"]].dropna()
            other = values.loc[~archive["Near Pareto"]].dropna()
            usable = values.dropna()
            bins = np.histogram_bin_edges(usable, bins=min(15, max(5, int(np.sqrt(len(usable))))))
            ax.hist(other, bins=bins, density=True, color="lightgray", alpha=0.75,
                    label=f"Other runs (n={len(other)})")
            ax.hist(good, bins=bins, density=True, color="seagreen", alpha=0.55,
                    label=f"Pareto + near-Pareto (n={len(good)})")
            ax.set(xlabel=parameter, ylabel="Empirical density")
            ax.legend(fontsize=8)
        fig.suptitle("Parameter values represented among promising canonical runs", fontsize=14)
        _finish(fig, None if output_directory is None else output_directory / "canonical_promising_numeric_distributions.png")

    if not categorical_screening.empty:
        for parameter, group in categorical_screening.groupby("Parameter", sort=False):
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = group["Category"].astype(str)
            values = group["P(Promising | Category)"].astype(float)
            ax.bar(categories, values, color="seagreen")
            ax.errorbar(categories, values,
                        yerr=[values - group["95% Low"], group["95% High"] - values],
                        fmt="none", ecolor="black", capsize=3)
            ax.set(ylim=(0, 1), ylabel="Probability of Pareto/near-Pareto status", title=str(parameter))
            ax.tick_params(axis="x", rotation=25)
            slug = str(parameter).replace("/", "_").replace(" ", "_")
            _finish(fig, None if output_directory is None else output_directory / f"canonical_promising_categorical_{slug}.png")

    parallel_parameters = parameters[:6]
    if len(parallel_parameters) >= 2:
        numeric = archive[parallel_parameters].apply(pd.to_numeric, errors="coerce")
        normalized = numeric.copy()
        for parameter in parallel_parameters:
            low, high = numeric[parameter].min(), numeric[parameter].max()
            normalized[parameter] = (numeric[parameter] - low) / (high - low) if high > low else 0.5
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(parallel_parameters))
        for index, row in normalized.loc[~archive["Near Pareto"]].iterrows():
            ax.plot(x, row, color="lightgray", alpha=0.18, linewidth=0.8)
        for index, row in normalized.loc[archive["Near Pareto"]].iterrows():
            ax.plot(x, row, color="seagreen", alpha=0.8, linewidth=1.8)
        ax.set_xticks(x, parallel_parameters, rotation=30, ha="right")
        ax.set(ylabel="Position within tested range", ylim=(-0.02, 1.02),
               title="Promising canonical parameter-space pattern (green)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        _finish(fig, None if output_directory is None else output_directory / "canonical_promising_parallel_coordinates.png")

    pair_parameters = parameters[:4]
    if len(pair_parameters) >= 2:
        size = len(pair_parameters)
        fig, axes = plt.subplots(size, size, figsize=(3 * size, 3 * size))
        for row_index, y_parameter in enumerate(pair_parameters):
            for column_index, x_parameter in enumerate(pair_parameters):
                ax = axes[row_index, column_index]
                if row_index == column_index:
                    ax.hist(pd.to_numeric(archive[x_parameter], errors="coerce").dropna(), bins=12, color="lightgray")
                else:
                    ax.scatter(archive[x_parameter], archive[y_parameter],
                               c=np.where(archive["Near Pareto"], "seagreen", "lightgray"),
                               alpha=0.65, s=22)
                if row_index == size - 1:
                    ax.set_xlabel(x_parameter, rotation=20, ha="right")
                else:
                    ax.set_xticklabels([])
                if column_index == 0:
                    ax.set_ylabel(y_parameter)
                else:
                    ax.set_yticklabels([])
        fig.suptitle("Pairwise parameter regions; green = Pareto/near-Pareto", fontsize=14)
        _finish(fig, None if output_directory is None else output_directory / "canonical_promising_pairwise_regions.png")


def plot_focused_parameter(
    archive: pd.DataFrame,
    parameter: str | None,
    *,
    maximum_rank: int,
    output_directory: Path | None = None,
) -> None:
    if not parameter:
        return
    if parameter not in archive:
        print(f"Focused parameter {parameter!r} is unavailable; plot omitted.")
        return
    values = pd.to_numeric(archive[parameter], errors="coerce")
    usable = values.dropna()
    if usable.empty:
        print(f"Focused parameter {parameter!r} has no numeric values; plot omitted.")
        return
    selected = pd.to_numeric(archive["Pareto Rank"], errors="coerce").le(maximum_rank)
    bins = np.histogram_bin_edges(usable, bins=min(15, max(5, int(np.sqrt(len(usable))))))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values.loc[~selected].dropna(), bins=bins, density=True, color="lightgray", alpha=0.75,
            label=f"Other runs (n={values.loc[~selected].notna().sum()})")
    ax.hist(values.loc[selected].dropna(), bins=bins, density=True, color="seagreen", alpha=0.55,
            label=f"Pareto ranks 0–{maximum_rank} (n={values.loc[selected].notna().sum()})")
    ax.set(xlabel=parameter, ylabel="Empirical density", title=f"Focused promising-value distribution: {parameter}")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    slug = parameter.replace("/", "_").replace(" ", "_")
    _finish(fig, None if output_directory is None else output_directory / f"canonical_focused_distribution_{slug}.png")


def plot_nonlinear_importance(
    performance: pd.DataFrame,
    importance: pd.DataFrame,
    objectives: Sequence[str],
    *,
    output_directory: Path | None = None,
) -> None:
    if performance.empty or importance.empty:
        return
    for objective in objectives:
        model = performance.loc[
            performance["Objective"].eq(objective)
            & performance["Model"].eq("Mixed-type Random Forest")
        ]
        if model.empty or float(model.iloc[0]["Held-out R2"]) <= 0:
            print(f"{objective}: validated importance is inconclusive; plot omitted.")
            continue
        data = importance.loc[importance["Objective"].eq(objective)].nsmallest(15, "Rank").sort_values("Mean_Importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(data["Parameter"], data["Mean_Importance"], xerr=data["SD_Importance"],
                color="tab:blue", alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Held-out MAE increase when permuted", title=f"Validated importance: {objective}")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        slug = objective.lower().replace(" ", "_")
        _finish(fig, None if output_directory is None else output_directory / f"canonical_validated_importance_{slug}.png")


def component_heatmaps(
    components: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    maximum_runs: int = 40,
    output_directory: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot selected and all-run canonical component contribution heatmaps."""

    valid = components.loc[
        components["include_in_total"].astype(bool)
        & components["status"].eq("calculated")
    ].copy()
    matrix = valid.pivot_table(index="Run Name", columns="label", values="weighted_contribution", aggfunc="first")
    ordering = archive.sort_values(["Pareto Rank", "Canonical Cost"])["Run Name"]
    matrix = matrix.reindex([name for name in ordering if name in matrix.index])
    selected = matrix.head(maximum_runs)
    if not selected.empty:
        fig, ax = plt.subplots(figsize=(15, max(7, 0.28 * len(selected))))
        _heatmap(ax, selected, title=f"Canonical component contributions: first {len(selected)} ranked runs",
                 colorbar_label="Weighted canonical contribution", cmap="magma")
        _finish(fig, None if output_directory is None else output_directory / "canonical_component_heatmap_ranked.png")
    if len(matrix) > len(selected):
        fig, ax = plt.subplots(figsize=(15, max(10, 0.18 * len(matrix))))
        _heatmap(ax, matrix, title="Canonical component contributions: all valid runs",
                 colorbar_label="Weighted canonical contribution", cmap="magma")
        ax.tick_params(axis="y", labelsize=5)
        _finish(fig, None if output_directory is None else output_directory / "canonical_component_heatmap_all_runs.png", dpi=250)
    return matrix, selected


def observation_year_context(
    components: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    output: Path | None = None,
) -> pd.DataFrame:
    context = components.loc[components["include_in_total"].astype(bool)].merge(
        archive[["Run Name", "dstart_year"]].drop_duplicates("Run Name"),
        on="Run Name", how="left", validate="many_to_one",
    )
    context["Model Year"] = pd.to_numeric(context["dstart_year"], errors="coerce").astype("Int64")
    context = context.loc[context["Model Year"].notna()].copy()
    summary = context.groupby(["Model Year", "label"], dropna=False).agg(
        Observation_Count=("observation_count", "median"),
        Observation_SD=("observation_std_ddof0", "median"),
        Observational_Error=("observational_error", "median"),
        Normalization_Variance=("normalization_variance", "median"),
        Runs=("Run Name", "nunique"),
    ).reset_index()
    metrics = [
        ("Observation_Count", "Median observation count", "viridis", ".0f"),
        ("Observation_SD", "Observed standard deviation", "cividis", ".2g"),
        ("Normalization_Variance", "Normalization variance", "magma", ".2g"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    for ax, (column, title, cmap, fmt) in zip(axes, metrics):
        table = summary.pivot(index="Model Year", columns="label", values=column)
        _heatmap(ax, table, title=title, colorbar_label=column.replace("_", " "), cmap=cmap, fmt=fmt)
    fig.suptitle("Observation and normalization context by model year", fontsize=15)
    _finish(fig, output)
    return summary


def station_diagnostics(
    archive: pd.DataFrame,
    run_names: Sequence[str],
    *,
    cost_root: Path,
    output_directory: Path | None = None,
) -> pd.DataFrame:
    """Calculate and plot station/depth pH and oxygen RMSE, bias, and counts."""

    if not run_names:
        return pd.DataFrame()
    from netCDF4 import Dataset

    records: list[dict[str, Any]] = []
    lookup = archive.drop_duplicates("Run Name").set_index("Run Name")
    for run_name in run_names:
        if run_name not in lookup.index:
            print(f"Station diagnostic run {run_name!r} is absent; skipped.")
            continue
        cost_name = str(lookup.loc[run_name, "Cost File"])
        path = Path(cost_name)
        if not path.is_absolute():
            path = cost_root / path
        if not path.is_file():
            print(f"Station diagnostic cost file is missing: {path}")
            continue
        with Dataset(path) as dataset:
            sites = [str(value) for value in dataset.variables["Site"][:]]
            depths = [str(value) for value in dataset.variables["Depth"][:]]
            for variable, model_name, observation_name in (
                ("pH", "pH_mod", "pH_obs"),
                ("Oxygen", "oxygen_mod", "oxygen_obs"),
            ):
                model = np.ma.asarray(dataset.variables[model_name][:], dtype=float).filled(np.nan)
                observation = np.ma.asarray(dataset.variables[observation_name][:], dtype=float).filled(np.nan)
                for site_index, site in enumerate(sites):
                    for depth_index, depth in enumerate(depths):
                        m = model[site_index, depth_index]
                        o = observation[site_index, depth_index]
                        paired = np.isfinite(m) & np.isfinite(o)
                        residual = m[paired] - o[paired]
                        records.append({
                            "Run Name": run_name, "Variable": variable, "Station": site, "Depth": depth,
                            "Count": int(paired.sum()),
                            "RMSE": float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan,
                            "Bias": float(np.mean(residual)) if residual.size else np.nan,
                        })
    diagnostics = pd.DataFrame(records)
    for run_name, run_data in diagnostics.groupby("Run Name", sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        for row, variable in enumerate(("pH", "Oxygen")):
            subset = run_data.loc[run_data["Variable"].eq(variable)]
            for column, metric in enumerate(("RMSE", "Bias", "Count")):
                table = subset.pivot(index="Station", columns="Depth", values=metric)
                station_order = [station for station in lookup_station_order(subset) if station in table.index]
                table = table.reindex(station_order)
                cmap = "magma" if metric == "RMSE" else ("coolwarm" if metric == "Bias" else "viridis")
                fmt = ".0f" if metric == "Count" else ".2g"
                _heatmap(axes[row, column], table, title=f"{variable} {metric}", colorbar_label=metric, cmap=cmap, fmt=fmt)
                for label in axes[row, column].get_yticklabels():
                    if label.get_text() in UPPER_BAY_STATIONS:
                        label.set_color("crimson")
                        label.set_fontweight("bold")
        fig.suptitle(f"Station-resolved diagnostics: {run_name} (upper bay in red)", fontsize=15)
        slug = str(run_name).replace("/", "_")
        _finish(fig, None if output_directory is None else output_directory / f"canonical_station_diagnostics_{slug}.png")
    return diagnostics


def lookup_station_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["BR", "MV", "QP", "TW", "GB", "GD", "PD", "CP", "NP", "PP", "MH", "SR"]
    present = frame["Station"].dropna().astype(str).unique().tolist()
    return [station for station in preferred if station in present] + sorted(set(present) - set(preferred))


def linked_run_trajectories(
    archive: pd.DataFrame,
    components: pd.DataFrame,
    objective_columns: Sequence[str],
    *,
    maximum_groups: int = 8,
    output_directory: Path | None = None,
) -> pd.DataFrame:
    """Plot multi-year objective and component trajectories for exact parameter groups."""

    if "Parameter Match Group" not in archive:
        return pd.DataFrame()
    matched = archive.loc[archive.get("Cross-Year Parameter Match", False).fillna(False)].copy()
    groups = [item for item in matched.groupby("Parameter Match Group", sort=True)
              if pd.to_numeric(item[1]["dstart_year"], errors="coerce").nunique() >= 2]
    groups = groups[:maximum_groups]
    if not groups:
        return pd.DataFrame()
    cost, bottom_ph = objective_columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    selected_names: list[str] = []
    for position, (group_id, members) in enumerate(groups, start=1):
        members = members.sort_values("dstart_year")
        selected_names.extend(members["Run Name"].tolist())
        label = f"Group {position}: " + " → ".join(members["Run Name"].astype(str))
        for axis, objective in zip(axes, (cost, bottom_ph)):
            axis.plot(members["dstart_year"], members[objective], marker="o", label=label)
            axis.set(xlabel="Model year", ylabel=objective, title=f"Linked-run trajectory: {objective}")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    _finish(fig, None if output_directory is None else output_directory / "canonical_linked_objective_trajectories.png")

    selected = components.loc[
        components["Run Name"].isin(selected_names)
        & components["include_in_total"].astype(bool)
        & components["status"].eq("calculated")
    ].merge(archive[["Run Name", "dstart_year"]], on="Run Name", how="left")
    matrix = selected.pivot_table(index=["dstart_year", "Run Name"], columns="label",
                                  values="weighted_contribution", aggfunc="first").sort_index()
    if not matrix.empty:
        matrix.index = [f"{int(year)} | {run}" for year, run in matrix.index]
        fig, ax = plt.subplots(figsize=(15, max(6, 0.35 * len(matrix))))
        _heatmap(ax, matrix, title="Linked-run component contributions by year",
                 colorbar_label="Weighted canonical contribution", cmap="magma")
        _finish(fig, None if output_directory is None else output_directory / "canonical_linked_component_trajectories.png")
    return matrix


def _pareto_points(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if dominated.any():
            keep[i] = False
    return points[keep]


def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    front = _pareto_points(points)
    front = front[np.argsort(front[:, 0])]
    area = 0.0
    previous_y = reference[1]
    for x, y in front:
        if x < reference[0] and y < previous_y:
            area += (reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def optimization_progress(
    study: Any,
    objective_columns: Sequence[str],
    *,
    output_directory: Path | None = None,
) -> pd.DataFrame:
    """Plot trial states, cumulative best objectives, and 2-D hypervolume."""

    try:
        import optuna
    except ImportError:
        print("Optuna is unavailable; optimization-progress plots were skipped.")
        return pd.DataFrame()
    trials = list(study.trials)
    if not trials:
        return pd.DataFrame()
    states = pd.Series([trial.state.name for trial in trials]).value_counts()
    origins = pd.Series([trial.user_attrs.get("Origin", "Unknown") for trial in trials]).value_counts()
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.values is not None and len(trial.values) >= 2
                and np.isfinite(trial.values[:2]).all()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(states.index, states.values, color="steelblue")
    axes[0].set(ylabel="Trials", title="Canonical study trial states")
    axes[1].bar(origins.index.astype(str), origins.values, color="seagreen")
    axes[1].set(ylabel="Trials", title="Historical versus generated trials")
    axes[1].tick_params(axis="x", rotation=25)
    _finish(fig, None if output_directory is None else output_directory / "canonical_trial_states.png")
    if not complete:
        return pd.DataFrame()
    complete.sort(key=lambda trial: trial.number)
    values = np.array([trial.values[:2] for trial in complete], dtype=float)
    reference = np.nanmax(values, axis=0) * 1.05
    reference = np.where(reference > np.nanmax(values, axis=0), reference, np.nanmax(values, axis=0) + 1.0)
    records = []
    for index, trial in enumerate(complete):
        observed = values[: index + 1]
        records.append({
            "Trial Number": trial.number,
            f"Best {objective_columns[0]}": observed[:, 0].min(),
            f"Best {objective_columns[1]}": observed[:, 1].min(),
            "Pareto Hypervolume": _hypervolume_2d(observed, reference),
            "Origin": trial.user_attrs.get("Origin", "Unknown"),
        })
    progress = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(progress["Trial Number"], progress[f"Best {objective_columns[0]}"], label=objective_columns[0])
    axes[0].plot(progress["Trial Number"], progress[f"Best {objective_columns[1]}"], label=objective_columns[1])
    axes[0].set(xlabel="Completed trial number", ylabel="Best attained value", title="Objective progress")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(progress["Trial Number"], progress["Pareto Hypervolume"], color="seagreen")
    axes[1].set(xlabel="Completed trial number", ylabel="Dominated hypervolume", title="Pareto-front progress")
    axes[1].grid(alpha=0.25)
    _finish(fig, None if output_directory is None else output_directory / "canonical_optimization_progress.png")
    return progress


def tpe_preferences(
    study: Any,
    sampler: Any,
    numeric_bounds: Mapping[str, tuple[float, float]],
    categorical_choices: Mapping[str, Sequence[Any]],
    *,
    seed: int,
    output_directory: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Restore the original TPE l(x) versus g(x) diagnostics."""

    if not numeric_bounds and not categorical_choices:
        return {}
    import optuna
    import candidate_parameter_utils as cpu

    search_space = {
        parameter: optuna.distributions.FloatDistribution(*bounds)
        for parameter, bounds in numeric_bounds.items()
    }
    search_space.update({
        parameter: optuna.distributions.CategoricalDistribution(choices)
        for parameter, choices in categorical_choices.items()
    })
    try:
        good, other, metadata = cpu.tpe_distribution_samples(study, sampler, search_space, seed=seed)
    except (RuntimeError, ValueError) as error:
        print(f"Exact TPE diagnostic unavailable: {error}")
        return {}
    display_tables: dict[str, pd.DataFrame] = {"metadata": pd.DataFrame([metadata])}
    for parameter, bounds in numeric_bounds.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(other[parameter], bins=30, density=True, color="lightgray", alpha=0.75,
                label="TPE less-promising model g(x)")
        ax.hist(good[parameter], bins=30, density=True, color="seagreen", alpha=0.55,
                label="TPE promising model l(x)")
        ax.axvline(bounds[0], color="darkorange", linestyle="--", label="Candidate bounds")
        ax.axvline(bounds[1], color="darkorange", linestyle="--")
        ax.set(xlabel=parameter, ylabel="Sampled model density", title=f"TPE preference: {parameter}")
        ax.legend(fontsize=8)
        slug = parameter.replace("/", "_").replace(" ", "_")
        _finish(fig, None if output_directory is None else output_directory / f"canonical_tpe_preference_{slug}.png")
    for parameter in categorical_choices:
        comparison = pd.concat([
            good[parameter].value_counts(normalize=True).rename("Promising l(x)"),
            other[parameter].value_counts(normalize=True).rename("Less-promising g(x)"),
        ], axis=1).fillna(0)
        comparison["Preference ratio l/g"] = comparison["Promising l(x)"] / comparison["Less-promising g(x)"].replace(0, np.nan)
        display_tables[parameter] = comparison
    return display_tables


def inactive_parameter_diagnostics(
    archive: pd.DataFrame,
    priors: pd.DataFrame,
    parameters: Sequence[str],
    *,
    maximum_rank: int,
    output_directory: Path | None = None,
) -> pd.DataFrame:
    """Plot fixed/reserve history with current values and proposed bounds."""

    if not parameters:
        return pd.DataFrame()
    lookup = priors.set_index("parameter", drop=False)
    rows = []
    promising = pd.to_numeric(archive["Pareto Rank"], errors="coerce").le(maximum_rank)
    for parameter in parameters:
        if parameter not in lookup.index or parameter not in archive:
            continue
        prior = lookup.loc[parameter]
        if prior["decision"] not in {"fixed", "reserve"}:
            continue
        values = pd.to_numeric(archive[parameter], errors="coerce")
        usable = values.dropna()
        if usable.empty:
            continue
        current = pd.to_numeric(pd.Series([prior.get("fixed_value")]), errors="coerce").iloc[0]
        low = pd.to_numeric(pd.Series([prior.get("proposed_low")]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([prior.get("proposed_high")]), errors="coerce").iloc[0]
        rows.append({"Parameter": parameter, "Decision": prior["decision"], "Current Value": current,
                     "Proposed Low": low, "Proposed High": high, "Usable Runs": len(usable),
                     "Promising Runs": int(values.loc[promising].notna().sum()),
                     "Tested Low": usable.min(), "Tested High": usable.max()})
        bins = np.histogram_bin_edges(usable, bins=min(15, max(5, int(np.sqrt(len(usable))))))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values.loc[~promising].dropna(), bins=bins, density=True, color="lightgray", alpha=0.75,
                label="Other historical runs")
        ax.hist(values.loc[promising].dropna(), bins=bins, density=True, color="seagreen", alpha=0.55,
                label=f"Historical Pareto ranks 0–{maximum_rank}")
        if np.isfinite(current):
            ax.axvline(current, color="royalblue", linestyle="--", linewidth=1.8, label=f"Fixed value ({current:.4g})")
        if np.isfinite(low) and np.isfinite(high):
            ax.axvline(low, color="darkorange", linestyle=":", linewidth=1.8, label="Proposed bounds")
            ax.axvline(high, color="darkorange", linestyle=":", linewidth=1.8)
        ax.set(xlabel=parameter, ylabel="Empirical density", title=f"Inactive diagnostic: {parameter} ({prior['decision']})")
        ax.legend(fontsize=8)
        _finish(fig, None if output_directory is None else output_directory / f"canonical_inactive_parameter_{parameter}.png")
    return pd.DataFrame(rows)
