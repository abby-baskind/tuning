# ROMS–COSiNE multi-objective TPE tuning guide

This is the authoritative guide for `CandidateParameterSetsBio_MultiObjectiveTPE.ipynb`.
It is intentionally more detailed and somewhat redundant with the notebook. The
notebook should be sufficient for routine use; this guide is for setup, difficult
decisions, troubleshooting, recovery, maintenance, and future changes by people
or AI assistants.

## 1. Purpose

The workflow tunes ROMS–COSiNE biological parameters against two minimization
objectives:

1. **Total cost**, calculated exactly as in `CandidateParameterSetsBio.ipynb`.
2. **Bottom pH Objective**, the absolute value of the existing signed normalized
   RMSD for bottom pH.

Total cost already includes a bottom-pH cost component. This overlap is
intentional:

- Total cost asks, “Does the system look good holistically?”
- Bottom pH asks, “Does the most important tuning target look good?”

`Non-bottom-pH Cost` is retained as a diagnostic, not a third optimization
objective. It asks whether improved Bottom pH came at the expense of everything
else represented by total cost.

The workflow also addresses a question that precedes optimization: **which
parameters should be active?** The active list is not copied from the reference
notebook. Existing evidence and user-supplied scientific priors are reviewed
before a new study is created.

## 2. Files delivered with the workflow

| File | Role | Edit manually? | Regenerable? |
|---|---|---:|---:|
| `CandidateParameterSetsBio_MultiObjectiveTPE.ipynb` | Guided user interface | Only marked `USER INPUT` cells | Yes, from version control |
| `candidate_parameter_utils.py` | Stable mechanics | Usually no | Yes, from version control |
| `test_candidate_parameter_utils.py` | Fast regression tests using `unittest` | Usually no | Yes, from version control |
| `parameter_priors.csv` | Scientific parameter decisions | **Yes** | No; it contains user reasoning |
| `MO-BIO/quality_control/data_quality_issues.csv` | Generated suspicious-value log | No | Yes |
| `MO-BIO/quality_control/data_corrections.csv` | Verified correction ledger | **Yes** | No; it contains verification work |
| `MULTIOBJECTIVE_TPE_GUIDE.md` | Authoritative documentation | When behavior changes | Yes, from version control |

The original `CandidateParameterSetsBio.ipynb` and its single-objective database
are intentionally left unchanged.

Run the fast helper regression tests with:

```bash
conda run -n my_env python test_candidate_parameter_utils.py
```

## 3. Quick start

1. Select the `my_env` kernel.
2. Open `CandidateParameterSetsBio_MultiObjectiveTPE.ipynb`.
3. Search for `USER INPUT` and review the control panel and scientific settings.
4. Leave `RUN_MODE = "review"`.
5. Use `DATA_LOAD_MODE = "cache"` for normal work or `"refresh"` to reread all
   source NetCDF files without updating the cache.
6. Run through the quality-control and parameter-screening sections.
7. Resolve genuine data issues in
   `MO-BIO/quality_control/data_corrections.csv`.
8. Edit `parameter_priors.csv` to identify active, fixed, reserve, and excluded
   parameters.
9. Rerun from the prior-loading section and review readiness and budget warnings.
10. Only after the search space is satisfactory, use
    `RUN_MODE = "prepare_study"`.
11. Return to `RUN_MODE = "review"` after the intended write action.

## 4. User-input reference

The notebook marks decision cells with `USER INPUT`. The following variables are
the ones a routine user should expect to review.

### 4.1 Operating controls

| Variable | Meaning | Typical value | Review when |
|---|---|---|---|
| `RUN_MODE` | Selects the one permitted workflow action | `"review"` | Every session |
| `DATA_LOAD_MODE` | Cache or fresh source-file behavior | `"cache"` | New files arrive or cache seems stale |
| `OUTPUT_DETAIL` | Controls diagnostic verbosity | `"concise"` | More detail is needed |
| `TRIAL_BUDGET` | Number of simulations affordable in this phase | User decision | Search space changes |
| `N_CANDIDATES` | Number of candidates in one generated batch | `4` | Before candidate generation |
| `RANDOM_SEED` | Reproducibility seed | `42` | Normally never |
| `RUN_NONLINEAR_SCREENING` | Enables slower cross-validated screening | `True` | Runtime is inconvenient |
| `HISTORICAL_NUMERIC_IMPORT_POLICY` | Separates historical numeric support from future candidate bounds | `"observed_support"` | Only for a deliberate compatibility experiment |
| `N_SCREENING_PARAMETERS_TO_SHOW` | Numeric-screening rows shown in concise mode | `25` | You want a shorter or longer review table |
| `N_CONFOUNDING_PAIRS_TO_SHOW` | Confounded pairs shown outside debug mode | `15` | You want more diagnostic pairs |
| `N_PARAMETER_DISTRIBUTIONS_TO_PLOT` | Numeric distributions drawn in Section 12 | `20` | The figure set is too short or long |
| `NEAR_PARETO_RANK` | Number of non-dominated layers treated as nearby evidence | `3` | Sensitivity analysis |
| `COST_THRESHOLD` | Failure/unreasonable-cost filter | `200` | Cost policy changes |

### 4.2 Paths

| Variable | Expected input/output |
|---|---|
| `PROJECT_DIR` | Directory containing this notebook and accessory files |
| `OPT_DIR` | Shared input spreadsheets used by multiple workflows |
| `COST_DIR` | Cost-summary and station NetCDF files |
| `FILE_LIST` | Workbook with `Run Name`, `Cost File`, and `Station File` |
| `PARAMETER_LIST` | CSV with a `Parameter Name` column |
| `MO_BIO_OUTPUT_DIRECTORY` | Root for every generated or workflow-maintained MO-BIO file |
| `OUTPUT_PATHS` | Structured MO-BIO path definitions derived from the output root |
| `RUN_CACHE_FILE` | Validated run-summary cache |
| `PARAMETER_PRIORS_FILE` | User-maintained parameter decisions |
| `QUALITY_ISSUES_FILE` | Generated suspicious-value log |
| `CORRECTIONS_FILE` | User-maintained verified correction ledger |
| `STORAGE_FILE` | New multi-objective Optuna SQLite database |
| `OPTICS_TEMPLATE` | ROMS water-column/optics template |
| `SEDIMENT_TEMPLATE` | ROMS sediment template |
| `CANDIDATE_OUTPUT_DIR` | Generated candidate input files |

`ROMS_OPT_DIR` and `ROMS_COST_DIR` environment variables can override the
default Desktop locations without editing notebook code.

### 4.3 Output layout

The authoritative output root is:

`/Users/akbaskind/Desktop/Optimization/MO-BIO`

```text
MO-BIO/
├── audit/
├── candidates/
│   └── input_files/
├── provenance/
├── quality_control/
├── figures/
├── cache/
└── optuna/
```

Important files include:

| Location | Purpose |
|---|---|
| `audit/OUTPUT_MIGRATION_LOG.txt` | One-time migration provenance and recovery instructions |
| `audit/mo_bio_*.csv` | Durable analysis, Pareto, screening, confounding, and readiness tables |
| `candidates/mo_bio_candidate_summary.csv` | Cumulative complete candidate records |
| `candidates/input_files/` | Generated optics and sediment input files |
| `provenance/mo_bio_trial_provenance.csv` | Optuna origin, timing, state, and run identity |
| `provenance/mo_bio_historical_import.csv` | Per-run import result and candidate-bound violations |
| `provenance/mo_bio_historical_numeric_support.csv` | Historical support compared with hard future candidate bounds |
| `quality_control/data_quality_issues.csv` | Generated suspicious-value log |
| `quality_control/data_corrections.csv` | User-maintained verified corrections |
| `cache/Model_Run_Summary_Bio.csv` | MO-BIO's independent validated cache |
| `optuna/model_calibration_bio_cost_bottom_ph.db` | Multi-objective Optuna study |

The older shared `Model_Run_Summary_Bio.csv` remains in the parent Optimization
directory because legacy notebooks still use it. It was copied—not moved—during
migration. Future MO-BIO cache refreshes use only the organized copy.

Old MO-BIO output paths are not fallback locations. If an old correction file,
candidate table, candidate-input directory, or multi-objective database appears,
the notebook reports it but does not silently choose between copies.

### 4.4 Study identity

| Variable | Purpose |
|---|---|
| `STUDY_NAME` | Human-readable Optuna study identity |
| `STORAGE_FILE` | Physical SQLite database |
| `OBJECTIVE_VERSION` | Scientific definition identity |

Change all three when objective definitions are changed. A search-space
fingerprint additionally prevents silent reuse when active parameters, bounds,
categories, or sampler settings change.

### 4.5 Candidate-file controls

| Variable | Purpose |
|---|---|
| `BASELINE_RUN_NAME` | Supplies values for inactive template parameters |
| `REMOTE_SEDIMENT_DIRECTORY` | Path written into `BSEDPARNAM` |
| `OVERWRITE_CANDIDATE_FILES` | Allows replacing existing generated files |

Leave overwrite disabled unless replacement is deliberate.

## 5. Run modes and side effects

### `review`

- Loads and analyzes data.
- Does not create or modify an Optuna study.
- Does not write candidate files.
- Does not replace the run cache.
- Writes regenerable audit CSV tables when `SAVE_AUDIT_OUTPUTS=True`.
- This is the default and should be restored after every action.

### `prepare_study`

- Creates or loads the separate two-objective study.
- Validates objective version and search-space fingerprint.
- Imports eligible historical rows as completed two-objective trials.
- Writes to the SQLite database.

Use this only after the active search space has been reviewed. A changed search
space should usually begin a new study rather than mutate an existing one.

The default `HISTORICAL_NUMERIC_IMPORT_POLICY="observed_support"` deliberately
uses two numeric ranges:

- **Historical support:** the minimum and maximum valid values observed in the
  archive, expanded when necessary to include the candidate range. Imported
  completed trials use this wider Optuna distribution.
- **Candidate bounds:** `proposed_low` and `proposed_high` from
  `parameter_priors.csv`. Every future `suggest_float` call uses these hard
  limits.

This lets TPE learn from valid prior experiments without permitting a new
recommendation outside the literature-supported range. Historical values are
never clipped or silently changed. The notebook saves the exact comparison to
`provenance/mo_bio_historical_numeric_support.csv`, and the import ledger records
which runs violate candidate bounds.

Categorical choices remain exact: a historical category absent from
`allowed_choices` is not imported. Unlike numeric distributions, categorical
distributions cannot be safely narrowed between imported and future trials in
this workflow.

The alternative policy `candidate_bounds` recreates the conservative old
behavior and excludes historical runs outside the candidate bounds. It is
retained mainly for explicit comparison, not recommended for this study.

### `generate_candidates`

- Loads the configured study.
- Refuses to run while previous trials remain `RUNNING`.
- Creates new `RUNNING` trials using multi-objective TPE.
- Appends complete candidate records to
  `MO-BIO/candidates/mo_bio_candidate_summary.csv`.

The complete summary includes the official run name, trial and study identity,
generation time, objective/search-space provenance, prior-file fingerprint,
sampler, all active proposals, and every fixed or baseline template value. It
also identifies applicable and inactive conditional parameters. Existing trial
records are never silently replaced by conflicting values.

### `register_results`

- Finds `RUNNING` trials whose run names now appear in the validated archive.
- Requires finite total cost and Bottom pH Objective.
- Permanently changes eligible trials to `COMPLETE`.
- Uses the generated MO-BIO run name as the official model run name.

Use a fresh data read before this action if new NetCDF results have arrived.
MO-BIO does not maintain a separate run map. Pair a MO-BIO run with a PHYS
candidate in the PHYS run map.

### `write_inputs`

- Loads existing `RUNNING` trial parameters.
- Builds optics and sediment files in memory.
- Validates every replacement before writing.
- Writes to `CANDIDATE_OUTPUT_DIR`.
- Refuses to replace existing files unless overwrite is explicitly enabled.

CPP-option changes are not supported by candidate generation. CACO3 remains
excluded until executable selection, compile-time configuration, and compatible
templates can be managed together.

## 6. Cache behavior

`DATA_LOAD_MODE` has three values:

- `cache`: use `RUN_CACHE_FILE`.
- `refresh`: reread every listed source file and rebuild data in memory; leave the
  cache untouched.
- `refresh_and_save`: reread, validate, and atomically replace the cache.

An atomic replacement writes a temporary file first, then replaces the old file
only after successful serialization. A failed refresh cannot partially overwrite
a valid cache.

The enhanced refresh records:

- Total cost.
- Bottom pH cost contribution.
- Non-bottom-pH cost.
- Individual labeled cost contributions.
- All objective columns.
- Station-history run date.
- Whether the date was unavailable.
- Raw station-history text.

The older cache may not contain all enhanced diagnostic columns. The notebook
will still perform Pareto analysis and will explain which diagnostics require a
fresh rebuild.

## 7. Output policy

The workflow is quiet unless something is actionable.

### `concise`

Reports:

- Files that could not be read.
- Parameters or objectives that are completely missing.
- Unreasonable costs.
- Invalid prior or correction entries.
- Database/search-space conflicts.
- Trustworthiness warnings.

It does not print one message per successful file or repeat ordinary paired-data
mismatch warnings.

### `detailed`

Adds aggregated diagnostic counts and supporting tables.

### `debug`

Adds low-level and per-run diagnostic records. Use this only when investigating a
specific failure.

Helper functions return structured diagnostics and should not print during normal
operation.

## 8. Parameter-prior file

`parameter_priors.csv` is the source of truth for scientific parameter decisions.
It can be edited in a spreadsheet or text editor.

### 8.1 Columns

| Column | Valid content |
|---|---|
| `parameter` | Exact archive/template parameter name |
| `parameter_type` | `float`, `integer`, `categorical`, or `boolean` |
| `decision` | `active`, `fixed`, `reserve`, `exclude`, or `review` |
| `bottom_ph_relevance` | `high`, `medium`, `low`, or `unknown` |
| `holistic_relevance` | `high`, `medium`, `low`, or `unknown` |
| `confidence_in_current_value` | `high`, `medium`, `low`, or `unknown` |
| `scale` | `linear`, `log`, or `categorical` |
| `allowed_choices` | Pipe-separated categories, e.g. `A|B|C` |
| `proposed_low` | Numeric lower bound enforced for future Optuna proposals |
| `proposed_high` | Numeric upper bound enforced for future Optuna proposals |
| `fixed_value` | Value required when `decision=fixed` |
| `parent_option` | Parent of a conditional parameter, if any |
| `scientific_rationale` | Why this decision makes sense |
| `future_question` | Evidence that could change the decision |
| `notes` | Free-form reminder |

### 8.2 Decisions

- `active`: TPE samples this parameter now.
- `fixed`: candidate files use the supplied fixed value.
- `reserve`: plausible but intentionally held out of this focused phase.
- `exclude`: scientifically or operationally unavailable.
- `review`: a decision is still required.

“Fixed for this round” does not mean scientifically unimportant. Reserve is often
the right choice when the run budget cannot support a large search space.

The proposed numeric bounds govern new candidates, not whether a valid
historical run can teach TPE. With the default observed-support policy, an old
value outside these bounds is retained as evidence while all new values remain
inside them. A preference that piles up at a candidate boundary may partly
reflect better historical outcomes beyond that boundary; treat that as a reason
to inspect the evidence, not as automatic permission to override the literature.

### 8.3 Choosing an initial search space

Combine:

1. Mechanistic plausibility.
2. Genuine uncertainty in the current value.
3. Operational controllability.
4. Existing empirical signal.
5. Independent historical variation.
6. Available trial budget.

For a limited budget, approximately 5–8 numeric parameters and at most one
categorical parameter is a reasonable starting target. This is a practical
heuristic, not a theorem.

Do not narrow a range because of one strong run. Use multiple supported runs,
scientific plausibility, and boundary behavior. If TPE accumulates at a bound,
the bound may need expansion rather than further narrowing.

The proposed bounds are not retrospective data-quality limits. Historically
tested values outside them remain valid evidence in Sections 8–13. Bounds affect
future candidate generation and historical-study import, not whether an observed
value is considered corrupted.

### 8.4 Readiness categories

Section 13 lists every non-excluded parameter by name and sorts it into an action
category. The category combines the maintained `decision` with archive coverage
and confounding:

- `Probably safe to fix`: already marked `fixed`, with adequate varied and
  promising observations and no strong confounding detected. This does not prove
  that the exact `fixed_value` is optimal.
- `Fix cautiously`: already marked `fixed`, but strongly confounded; the archive
  cannot independently validate that fixed decision.
- `Ready to optimize`: active and supported by basic archive coverage.
- `Optimize cautiously`: active but strongly confounded.
- `Held in reserve`: intentionally omitted from the current search; the evidence
  column still reports whether it is confounded.
- `Needs exploration`: too few distinct numeric values, too few promising
  observations, or sparse categorical coverage.
- `Insufficient evidence`: no usable archive screening evidence.
- `Needs scientific decision`: the prior remains `review`.

These are conservative workflow labels, not causal or scientific verdicts. The
notebook also displays the underlying `Empirical Evidence` and a short guidance
statement for each parameter.

## 9. Conditional parameters

Conditional children are not ordinary independent numeric parameters.

When CACO3 is disabled, values such as `cacopf`, `cacodr`, and `omega_thresh` are
**not applicable**. They are represented as missing in fresh archives, not as
evidence that zero is favorable.

Rules:

1. Analyze the parent option first.
2. Analyze children only among runs where the parent is enabled.
3. Report the smaller effective sample size.
4. Do not allow zero-filled disabled children to acquire false importance.
5. Suggest child values only if the parent is enabled in a future conditional
   candidate workflow.

CACO3 is currently operationally excluded because candidate generation cannot
select a compatible executable or modify compile-time CPP configuration.
Operational exclusion does not prevent historical inspection: Section 10 includes
`CACO3` as a categorical parent and reports separate `True` and `False` rows. This
comparison describes whether CACO3-enabled runs are overrepresented among the
selected promising ranks. It does not evaluate the CACO3-specific child values,
which remain excluded from general screening.

## 10. Data-quality issue and correction workflow

### 10.1 Generated issue log

`MO-BIO/quality_control/data_quality_issues.csv` contains:

| Column | Meaning |
|---|---|
| `issue_id` | Stable identifier derived from run, field, issue, and value |
| `run_name` | Affected run |
| `source_file` | Source station/cost file |
| `field` | Parameter or objective |
| `observed_value` | Raw value |
| `issue_type` | Completely missing or suspiciously tiny value |
| `reason` | Why it was flagged |
| `expected_low`, `expected_high` | Reserved schema fields; blank under current rules |
| `detected_utc` | Detection time |
| `resolution_status` | Current resolution |

Do not edit the generated issue log.

The quality detector intentionally does not compare historical values with
`proposed_low` or `proposed_high`. It flags suspiciously tiny nonzero numeric
values (currently absolute value below `1e-250`) because those can indicate a
station-file writing or extraction failure. Legitimate zero values are not caught
by that rule.

### 10.2 User-maintained correction ledger

`MO-BIO/quality_control/data_corrections.csv` contains:

| Column | Meaning |
|---|---|
| `issue_id` | Exact ID from the issue log |
| `run_name` | Must match the issue |
| `field` | Must match the issue |
| `original_value` | Value independently checked |
| `action` | Resolution action |
| `replacement_value` | Required for `replace` |
| `evidence` | Lab notebook, archived input, email, etc. |
| `notes` | Additional explanation |
| `reviewed_by` | Reviewer |
| `reviewed_date` | Date of verification |

Valid actions:

- `replace`: use a verified replacement.
- `accept`: retain the unusual raw value.
- `set_missing`: treat the value as unknown.
- `exclude_run`: remove the complete run from analysis.
- `unresolved`: keep it excluded while investigation continues.

Example:

```csv
issue_id,run_name,field,original_value,action,replacement_value,evidence,notes,reviewed_by,reviewed_date
d1ab96a69424549e,2005_CTRL,bUmaxSi_nspc0,6.3187126644846e-310,replace,0.1,Lab notebook 4 page 37,Confirmed input value,AB,2026-09-20
```

Corrections never modify source NetCDF files. Raw data are retained separately;
corrections apply only to the analysis dataframe. If a source value changes, an
old correction fails its original-value check rather than silently applying.

Unresolved suspicious values are excluded by default.

## 11. Historical provenance

Every run should retain:

- Original run name.
- Study run name and trial number when known.
- Origin: historical archive, previous optimizer, or new multi-objective TPE.
- Station-file run date parsed from the `history` attribute.
- Whether that station date was unavailable.
- Date imported into the new study.
- Native Optuna start/completion times for new trials.

The station-history date is treated as the historical run/completion date, but it
is not the same as an Optuna completion timestamp. Missing dates remain unknown;
they are not assigned an artificial year for chronological interpretation.

The initial historical archive is evidence used to initialize TPE. It must not be
described as trials sequentially learned by the new study. Genuine learning over
time begins with trials whose origin is `Multi-objective TPE`.

## 12. Pareto interpretation

A run is Pareto-optimal when no other run is at least as good on both objectives
and strictly better on one. Dominated runs can be rejected for two-objective
selection because another observed run improves both targets.

The notebook reports:

- Best total-cost endpoint.
- Best Bottom pH endpoint.
- Normalized balanced compromise.
- Exact Pareto front.
- Nearby non-dominated ranks used as additional parameter evidence.

Near-Pareto runs are retained because the exact front may contain very few runs
and can be sensitive to sampling. Near-front status is evidence, not equivalence
to the exact front.

Because Bottom pH already contributes to total cost, the objectives are not
independent. That is intentional prioritization. Candidate comparisons should
always inspect non-bottom-pH cost when available.

## 13. What TPE learns

TPE models two parameter distributions:

- `l(x)`: parameter settings associated with relatively promising trials.
- `g(x)`: settings associated with less-promising trials.

It favors proposals where `l(x) / g(x)` is large.

For multiple objectives, Optuna 4.9 ranks trials through non-domination and uses
hypervolume-related weighting. With few Pareto trials and many active dimensions,
the promising model may be based on sparse evidence. This is why a focused search
space matters.

Numeric plots compare samples from the two Parzen models. Categorical tables show
probabilities and preference ratios. These describe TPE's search preference over
the configured bounds and observations; they do not prove a scientifically true
parameter value.

The notebook follows the TPE plots with a separate inactive-parameter diagnostic.
It can compare selected numeric `fixed` and `reserve` parameters using the same
green/gray visual language, but those distributions are empirical historical
evidence rather than TPE `l(x)` and `g(x)` models. The diagnostic also reports the
current fixed or baseline value, proposed bounds, tested range, sample counts,
relevance, readiness, and confounding. Use it to decide what deserves a future
tuning round; it does not modify the current search space or study.

Exact TPE reconstruction uses private Optuna APIs because no stable public API
exposes the fitted Parzen models. It is isolated in one helper function and
validated for Optuna 4.9.x. If the version changes, empirical promising-run plots
remain the stable fallback.

## 14. Parameter screening and its limits

The notebook provides:

- Numeric coverage and tested range.
- Promising-run 10th, median, and 90th percentiles.
- Spearman associations with both objectives.
- Categorical promising probabilities and uncertainty intervals.
- Strong parameter co-variation.
- Cross-validated mixed-type nonlinear importance.

Guiding rules:

1. Correlation is not causation.
2. Historically correlated parameters cannot be interpreted independently.
3. A nonlinear importance ranking is not trustworthy when held-out prediction
   does not beat a mean-only baseline.
4. Sparse categories need wide uncertainty and cautious language.
5. TPE preference is not parameter identifiability.
6. Evidence applies only within tested bounds and categories.
7. An apparently narrow good range based on one value is not convergence.

The current archive has strong historical confounding. Previous analysis found
some parameter correlations above 0.99. Bottom pH has been more predictable than
total cost; total-cost importance may remain inconclusive until additional runs
provide more independent variation.

## 15. Trial-budget interpretation

The notebook reports a heuristic effective dimension:

```text
numeric active parameters + extra categorical levels
```

Budget labels mean:

- **Limited:** candidates may improve, but stable ranges and interactions are
  unlikely.
- **Focused:** useful for optimization; interaction conclusions remain tentative.
- **Supported:** more reasonably matched to search-space complexity.

Warnings do not prevent execution. They state what can and cannot reasonably be
concluded from the requested budget.

When time is limited, prefer a scientifically informed 5–8 parameter search and
retain other plausible parameters as reserves. If the Pareto front or TPE
preferences plateau, start a clearly versioned later phase with selected reserve
parameters.

## 16. Candidate-file behavior

Candidate generation and input writing are separate actions.

1. TPE creates RUNNING trials and appends complete records to the cumulative
   candidate summary.
2. `write_inputs` reloads those RUNNING trials.
3. Active parameters come from trial suggestions.
4. Parameters marked `fixed` use `fixed_value` from the prior file.
5. Other inactive template parameters inherit the selected baseline run.
6. Water-column and sediment text is built and validated in memory.
7. Files are written only after all replacements succeed.

Special representative biological settings are applied to all 15 biological
tracers when active:

- `nl_tnu2_tracer9` controls the shared biological `TNU2` value.
- `tracer9_advection_scheme` controls all biological horizontal and vertical
  advection lines.

The remote sediment path written to `BSEDPARNAM` is controlled by
`REMOTE_SEDIMENT_DIRECTORY`.

The generated `Run Name` is the official model run name. It must remain attached
to the trial, both generated input files, archive row, and resulting outputs.
MO-BIO therefore does not need a separate run map.

### Paired initial MO-BIO and PHYS runs

The initial four MO-BIO candidates may be paired one-to-one with a four-cell PHYS
temperature/salinity test. The biological-tracer advection and diffusion settings
remain untouched in this phase. PHYS evaluates temperature and salinity, while
MO-BIO evaluates the biological objectives.

This design assumes the tested temperature/salinity changes are small enough that
the biological comparison remains informative. Verify that assumption using the
physical cost and its four components. If a paired run has anomalous physical
performance, interpret its biological objectives cautiously. The exact pairing
belongs in the PHYS run map using the official MO-BIO run name and MO-BIO trial
number.

## 17. Expected issues and responses

### Cache is missing

Use `DATA_LOAD_MODE = "refresh"` to test a fresh read or
`"refresh_and_save"` to build a new cache.

### Cache lacks non-bottom-pH diagnostics

The cache was produced by the older notebook. Use a fresh rebuild when those
diagnostics are needed.

### A cost or station file cannot be read

Confirm the run inventory filename and `COST_DIR`. The run remains visible in the
failure table and is excluded when its objectives are unusable.

### A parameter/objective is completely missing

Check spelling, parameter extraction, source-file version, and whether the field
is conditional. Do not substitute zero for unknown data.

### A suspicious value is real

Add an `accept` row to `MO-BIO/quality_control/data_corrections.csv` with
independent evidence.

### A suspicious value has a verified replacement

Add a `replace` row with the exact original value and replacement. Keep the
evidence citation specific enough to revisit later.

### A correction becomes stale

The source issue changed or disappeared. Reinspect the source before creating a
new correction. Do not reuse an old issue ID blindly.

### Total-cost model does not validate

Treat parameter importance for cost as inconclusive. Use scientific priors,
confounding information, and reserve decisions rather than forcing a ranking.

### Too many active parameters for the budget

Move lower-priority parameters to `reserve`, narrow only scientifically justified
bounds, or accept that range conclusions will remain tentative.

### Search-space fingerprint mismatch

The existing study was created with different parameters, bounds, choices, or
sampler settings. Use a new study name and database. Do not bypass the check.

### Objective-version mismatch

Objective definitions differ. Use a new study identity. Never mix objective
versions in one study.

### RUNNING trials block a new batch

Run or resolve the outstanding candidates, refresh their results, and use
`register_results`. Do not create overlapping batches accidentally.

### Exact TPE diagnostic is unavailable

Check the installed Optuna version. Use empirical promising-run distributions
until the private-API adapter is reviewed for the new version.

### Candidate replacement fails

The template layout or parameter inventory differs from expectations. Inspect the
template; do not weaken exact replacement-count checks merely to force output.

## 18. Known limitations and future improvements

| Limitation | Current impact | Workaround | Future improvement |
|---|---|---|---|
| Historical parameters changed together | Individual effects are confounded | Report groups and use scientific priors | Targeted independent variation |
| Total cost is difficult to predict | Cost importance may be unreliable | Require held-out validation | More diverse focused runs |
| Few CACO3-enabled runs | Weak CACO3 inference | Mark low confidence | Balanced enabled runs |
| CPP choices cannot be applied to candidates | CACO3 cannot be active | Exclude operationally | Executable/template/CPP workflow |
| Exact TPE view uses private APIs | Version-sensitive | Pin/check Optuna 4.9.x | Adopt public API if exposed |
| Legacy cache lacks enhanced diagnostics | Non-bottom-pH diagnostic absent | Fresh rebuild | Use enhanced cache thereafter |
| Pareto front may be very small | Good-value distributions are sparse | Include near-Pareto rank | Add focused observations |
| Candidate provenance from old studies is incomplete | Some origins inferred from names | Label inference clearly | Import original study metadata |

This table should be updated whenever a limitation is resolved or discovered.

## 19. Recovery and preservation

### Irreplaceable user work

Back up or commit:

- `parameter_priors.csv`
- `MO-BIO/quality_control/data_corrections.csv`
- `MO-BIO/optuna/model_calibration_bio_cost_bottom_ph.db`
- `MO-BIO/candidates/mo_bio_candidate_summary.csv`
- PHYS run-map records connecting MO-BIO runs to PHYS candidates

### Regenerable outputs

These can be rebuilt from source data and configuration:

- `MO-BIO/quality_control/data_quality_issues.csv`
- Run-summary caches
- Figures and tables
- Candidate-summary CSV files
- Candidate ROMS input files, provided templates and study trials are preserved

### If the notebook fails halfway through

1. Return `RUN_MODE` to `review`.
2. Restart the kernel.
3. Run top to bottom through the review sections.
4. Inspect the final successful action and database trial states.
5. Do not repeat a write action until its existing result is understood.

SQLite study writes are stateful. A notebook restart does not undo them.

## 20. Maintenance brief for developers and AI assistants

This section is a compact architecture contract for future modifications.

### 20.1 Sources of truth

- Scientific objective mappings and weights: notebook scientific-definition cell.
- Parameter decisions and user rationale: `parameter_priors.csv`.
- Verified data corrections: `MO-BIO/quality_control/data_corrections.csv`.
- Stable mechanics: `candidate_parameter_utils.py`.
- Study state: the configured multi-objective SQLite database.
- Comprehensive behavior and rationale: this guide.

### 20.2 Invariants

Future changes must preserve these unless the user explicitly changes policy:

1. The original notebook and original study remain untouched.
2. Review mode must not create or modify an Optuna study or candidate files.
3. Total cost must remain parity-compatible with the reference calculation unless
   objective version and study identity change.
4. Both Optuna objectives are minimized.
5. Non-bottom-pH cost remains diagnostic unless explicitly redesigned.
6. Raw source files are never modified by corrections.
7. Unresolved suspicious values are excluded from analysis.
8. Conditional children are not treated as ordinary zero-valued observations.
9. Objective and search-space mismatches fail loudly.
10. Helper functions remain quiet during normal operation.
11. Candidate replacement counts remain strict.
12. Scientific decisions remain visible and editable without rewriting helpers.
13. Generated MO-BIO run names are official run names; MO-BIO does not use a
    separate run map.
14. Old output paths are reported but never used as silent fallbacks.

### 20.3 Architecture

```text
Source NetCDF files and inventories
              ↓
Raw cache / fresh extraction
              ↓
Generated issue detection + user correction ledger
              ↓
Validated analysis dataframe
              ↓
Pareto, screening, confounding, and readiness
              ↓
User-approved parameter priors
              ↓
Versioned two-objective Optuna study
              ↓
TPE candidates → ROMS files → completed results
              ↺
```

### 20.4 Extension points

- Add an objective by extending the objective registry and deliberately redesigning
  study dimensionality, Pareto displays, versioning, and trial import.
- Add a parameter by adding it to the extraction/template inventory and prior file.
- Add a conditional family by defining its parent, child extraction, applicability,
  and candidate-generation mechanism.
- Add a quality rule in `detect_data_quality_issues`; preserve stable issue IDs.
- Add a plot in the notebook or a reusable calculation in the helper module.
- Add a new write action as an explicit run mode; never hide it in review mode.

### 20.5 Documentation synchronization

Any change to the following must update both the notebook guidance and this guide:

- User-input variables.
- Accessory-file schemas.
- Run modes or side effects.
- Objective definitions.
- Study versioning.
- Candidate-file behavior.
- Correction semantics.
- Known limitations.

Intentional redundancy is a feature. The notebook should hold the user's hand;
this guide should explain the entire system when routine guidance is insufficient.

## 21. Glossary

- **Active parameter:** sampled by TPE in the current phase.
- **Reserve parameter:** plausible but intentionally excluded from this phase.
- **Pareto front:** observed solutions not dominated on both objectives.
- **Near-Pareto:** nearby non-domination ranks retained as supporting evidence.
- **TPE:** Tree-structured Parzen Estimator sampler.
- **`l(x)`:** TPE parameter distribution for promising trials.
- **`g(x)`:** TPE parameter distribution for less-promising trials.
- **Hypervolume:** measure of objective-space region dominated by a Pareto set.
- **Confounding:** inability to distinguish parameter effects because parameters
  changed together.
- **Search-space fingerprint:** hash of objectives, parameters, bounds, categories,
  and sampler settings.
- **Objective version:** human-readable identity for scientific calculations.
- **Raw value:** unmodified source-file value.
- **Analysis value:** raw value after explicit correction policy is applied.
