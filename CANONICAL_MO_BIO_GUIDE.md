# Canonical MO-BIO audit and study

This workflow recalculates MO-BIO objectives without changing the running
legacy MO-BIO study, its database, its cache, or PHYS. Its default output root
is `/Users/akbaskind/Desktop/Optimization/MO-BIO-CANONICAL`.

`CandidateParameterSetsBio_Canonical.ipynb` is the interactive operational
entry point. It follows the legacy MO-BIO sequence with separate canonical run
modes for review, study preparation, candidate generation, result
registration, and ROMS input writing. The command-line audit remains useful
for quick recalculation and automation.

Canonical search-space decisions are maintained in
`canonical_parameter_priors.csv`. It was seeded from the legacy active,
reserve, and fixed parameters, plus the CACO3, RIVER_SEDIMENT, and USE_MOCSY
CPP controls. Every initial canonical decision is `review`; `legacy_decision`
preserves the earlier classification. The full legacy prior inventory remains
a read-only input when constructing exact cross-year configuration signatures.

## Scientific definition

Each model run remains one annual observation. For component `k`, finite
observations and their collocated model values are used to calculate

`J_k = mean((model - observation)^2) / (std(observation, ddof=0)^2 + error_k^2)`.

The rules are:

- Every finite observation must have a finite model value. Otherwise that
  component, and therefore a required objective, is invalid.
- Surface and bottom components are normalized independently.
- The total objective is `sum(J_k * weight_k^2)`.
- The weighted-mean value is saved as a diagnostic; it is not optimized.
- Objective two is the bottom-pH normalized RMSE, `sqrt(J_bottom_pH)`, including
  bias.
- The existing observational-error values are explicit in
  `canonical_mo_bio_components.csv`.
- CHRP nutrient metrics are calculated but remain diagnostic-only.
- Different model years remain separate annual trials. Cross-year pooling is
  diagnostic only and is not part of either objective.

The component CSV is the editable scientific profile. Changing it requires a
new objective version, study name, and database; its SHA-256 digest is included
in study compatibility checks.

## Interpretation workflow

Review mode retains every run with complete canonical objectives; there is no
legacy-style arbitrary total-cost threshold. Robust plot scaling is display
only. The notebook writes reusable tables and figures for:

- Exact and near-Pareto fronts and representative endpoint/compromise runs.
- Weighted component attribution for representative runs.
- Annual model-year coverage and exact cross-year parameter matches.
- Future U3/C4 policy context without excluding older configurations.
- Numeric associations, promising intervals, and historical distributions.
- Disjoint promising-versus-other distributions, parallel coordinates,
  pairwise parameter regions, and a user-selected focused parameter.
- Categorical and CPP-option success-rate intervals.
- Strong parameter confounding.
- Cross-validated mixed-type nonlinear importance, always accompanied by
  held-out model-versus-baseline performance.
- Canonical prior readiness and search-budget assessment.
- Ranked and all-run component-contribution heatmaps.
- Observation count, observed variability, and normalization context by year.
- Station/depth pH and oxygen RMSE and bias for selected runs; PD, BR, and CP
  are highlighted as Upper Bay stations.
- Multi-year objective and component trajectories for exact parameter matches.
- Trial states, cumulative objective progress, Pareto hypervolume, and TPE
  preference diagnostics when a canonical study is available.

The operating cell controls the potentially large diagnostic sets through
`COMPONENT_HEATMAP_MAX_RUNS`, `STATION_DIAGNOSTIC_RUNS`,
`MAX_LINKED_TRAJECTORY_GROUPS`, `FOCUSED_PARAMETER_TO_INSPECT`, and
`INACTIVE_PARAMETERS_TO_DIAGNOSE`. An empty `STATION_DIAGNOSTIC_RUNS` list uses
the representative runs selected by the notebook.

All durable diagnostic tables are under `audit/`; all displayed diagnostic
figures are under `figures/`.

## Continuations

The canonical notebook ranks source-year runs by canonical Pareto rank,
balanced objective performance, and parameter-space diversity while avoiding
configurations already represented in the target year. Selected rows are added
manually to the existing `MO-BIO Year Transitions` worksheet and tagged
`MO-BIO-CANONICAL` in `Notes`; the canonical writer ignores untagged legacy
rows. It writes only rows with `Status=planned`.

Continuation inputs preserve source biological parameters but override all
biological tracer advection to U3/C4. The manifest explicitly records that the
separate physical ROMS configuration must enforce U3/C4 for temperature and
salinity. Backcast writing is intentionally not included yet.

Ordinary canonical candidates follow the same policy: candidate summaries and
trial metadata record U3/C4, the biological writer overrides the A4/A4 baseline,
and every generated optics input must pass a 15-horizontal-U3 and
15-vertical-C4 validation before any file is written. Temperature and salinity
remain a separate physical ROMS configuration check.

## Safe sequence

Run an audit first:

```bash
python canonical_mo_bio.py audit
```

This performs `refresh_and_save`: it creates the separate folder and builds an
independent canonical cache directly from the cost and station NetCDF files.
It writes component metrics, annual objectives, failures, parameters, CPP
states, every tracer's advection scheme, policy-compliance flags, and an
objective manifest. It does not read the legacy MO-BIO cache or create an
Optuna study.

The notebook exposes three data modes:

- `cache`: read `cache/Model_Run_Summary_Bio_Canonical.csv` without opening
  source NetCDF files.
- `refresh`: rebuild in memory without replacing the cache.
- `refresh_and_save`: rebuild and atomically replace the canonical cache and
  refresh diagnostics.

After reviewing those files, initialize the separate canonical study:

```bash
python canonical_mo_bio.py prepare-study
```

Run study preparation in the project's Python 3.11+ environment from
`requirements.txt`; unlike the audit command, it requires Optuna.

Only completed runs with valid canonical objectives and complete active
parameters are imported. No placeholder is created for a legacy RUNNING trial.
When those trials finish, rerun `prepare-study`; import is idempotent by run
name, so existing canonical trials remain unchanged and newly completed runs
are added. The existing user-reviewed correction ledger is read and applied to
the import table, while its canonical issue and application audits are written
under the new output root.

Both commands accept `--output-root`, `--inventory-file`, `--cost-root`,
`--legacy-cache`, and `--components-file`. This keeps the implementation usable
for alternate studies without changing the active legacy notebook.

## Output layout

```text
MO-BIO-CANONICAL/
├── audit/
├── cache/Model_Run_Summary_Bio_Canonical.csv
├── candidates/
├── figures/
├── optuna/
├── provenance/
├── quality_control/
└── run_maps/
```

The canonical SQLite database is
`optuna/model_calibration_bio_canonical_v1.db`. The old MO-BIO database and all
legacy cost files are read-only inputs to this workflow.
