# PHYS constrained physical-tuning guide

This is the authoritative guide for `CandidateParameterSetsPhysical.ipynb`
(“PHYS”). The notebook repeats the essential instructions beside the relevant
cells; this document intentionally includes more detail for troubleshooting,
future revisions, and AI-assisted maintenance.

## Purpose and scientific scope

PHYS is a single-objective workflow. Lower cost is better. The existing physical
cost definition is preserved exactly:

1. surface temperature cost;
2. bottom temperature cost;
3. surface salinity cost; and
4. bottom salinity cost.

The four components are summed with equal weight. The individual components are
retained so a low total can still be checked for an unacceptable tradeoff.

The workflow answers five questions:

1. Is Upstream3–Centered4 or Akima4–Akima4 preferable?
2. Is the shared physical-tracer `nl_tnu2` value 0.5 or 1.0 preferable?
3. With `GLS_MIXING + RI_SPLINES + N2S2_HORAVG` required, should any additional
   tracked CPP options be enabled?
4. Does the archive already contain a strong run satisfying the candidate rules?
5. Which other numeric parameters differ from the latest completed run?

Historical association is not automatically a causal effect. Many settings were
changed together. PHYS therefore displays all-run evidence, required-bundle-only
evidence, exact joint configurations, replication counts, and cost components.
Sparse or confounded comparisons should be treated as reasons for additional
testing rather than proof that a setting is good or bad.

## Files that define the workflow

These remain in the Git repository:

- `CandidateParameterSetsPhysical.ipynb`: guided user interface and analysis;
- `physical_candidate_utils.py`: parsing, validation, provenance, and table
  mechanics;
- `physical_tuning_spec.toml`: authoritative editable scientific rules;
- `PHYSICAL_TUNING_GUIDE.md`: this guide; and
- `test_physical_candidate_utils.py`: regression tests.

Do not put scientific choices into the helper merely because editing Python is
possible. Allowed values and combinations belong in the TOML file.

## Inputs and path controls

The notebook control panel contains the path variables. The initial values are:

```python
COST_DIRECTORY = Path("/Users/akbaskind/Desktop/COST_FILES")
OPTIMIZATION_DIRECTORY = Path("/Users/akbaskind/Desktop/Optimization")
FILE_LIST = OPTIMIZATION_DIRECTORY / "FileNames.xlsx"
PARAMETER_LIST = OPTIMIZATION_DIRECTORY / "PhysicalParameterList.csv"
PHYS_OUTPUT_DIRECTORY = Path("/Users/akbaskind/Desktop/Optimization/PHYS")
SPEC_FILE = Path("physical_tuning_spec.toml")
```

`FileNames.xlsx` must contain `Run Name`, `Cost File`, and `Station File`.
`PhysicalParameterList.csv` must contain `Variable`. Cost and station files are
resolved under `COST_DIRECTORY`, matching the established MO-BIO path convention.

## Output layout

All generated or workflow-maintained output is written beneath:

`/Users/akbaskind/Desktop/Optimization/PHYS`

The notebook creates these subdirectories when needed:

```text
PHYS/
├── audit/
├── candidates/
├── provenance/
├── quality_control/
├── figures/
├── cache/
└── optuna/
```

Important outputs are:

| Output | Meaning |
|---|---|
| `audit/physical_archive_inventory.csv` | Parsed historical settings, costs, and provenance |
| `audit/physical_observed_orientation_pairs.csv` | Exact observed UV/TS pairs and performance |
| `audit/physical_observed_optional_cpp_combinations.csv` | Exact optional-option states and performance |
| `audit/physical_observed_advection_schemes.csv` | Exact temperature/salinity advection bundles |
| `audit/physical_varied_numeric_parameters.csv` | Numeric fields that actually varied |
| `audit/physical_numeric_parameter_differences.csv` | Per-run differences from the latest run |
| `audit/physical_factorial_audit.csv` | Every currently allowed candidate cell and whether it was observed exactly |
| `quality_control/physical_data_quality_issues.csv` | Missing, malformed, inconsistent, or suspicious source values |
| `quality_control/physical_data_corrections.csv` | User-verified overrides; original files remain unchanged |
| `candidates/physical_candidate_summary.csv` | Complete settings for each requested candidate |
| `candidates/physical_candidate_change_manifest.csv` | Manual changes required relative to the latest run |
| `provenance/physical_candidate_run_map.csv` | PHYS trial → official MO-BIO run and MO-BIO trial |
| `optuna/physical_tuning.db` | Single-objective Optuna study |

The cache files are implementation details and may be regenerated. The
corrections file and run map are user-maintained provenance and must not be
treated as disposable cache.

## Normal workflow

### 1. Audit safely

Keep the notebook controls at:

```python
ACTION_MODE = "audit"
FORCE_REFRESH = False
```

Run the notebook. It may write refreshed audit tables but will not ask Optuna for
new candidates or modify trial states.

Set `FORCE_REFRESH = True` when station files, cost files, the file manifest, or
the scientific specification have changed and a completely fresh read is
desired. The cache also invalidates automatically when tracked source metadata or
the TOML fingerprint changes.

### 2. Review the TOML decision points

Open `physical_tuning_spec.toml`. The two initial review gates are:

- `mixing_orientation.review_status`; and
- `optional_cpp_options.review_status`.

The initial lists reflect combinations found by the first archive audit, not an
automatic recommendation that every listed state deserves a new run. Compare
them with the notebook tables, executable availability, runtime concerns, and
scientific knowledge. Remove unwanted entries. Then set the relevant status to
`"approved"`.

Candidate generation is blocked while either status is pending. This is a safety
feature, not an error.

### 3. Inspect candidate-space size

Rerun in audit mode. `component_factorial` crosses:

- candidate-allowed advection bundles;
- approved shared `nl_tnu2` values;
- approved observed orientation pairs; and
- approved observed optional-CPP states.

The required mass-conservation bundle is fixed and does not multiply the grid.
The notebook reports total cells, cells found exactly in the archive, and new
cells. A component factorial may create a complete cross-component configuration
not previously run, although each internal component state is approved.

`observed_configurations_only` is stricter: it retains only complete tuples found
in the archive. The initial audit found only one eligible exact tuple, so this
mode may be too restrictive for learning.

### 4. Choose factorial or TPE

Set `workflow.candidate_strategy` in the TOML:

- `factorial`: request every allowed cell; best when the table is small enough;
- `tpe`: request only `workflow.trial_budget` cells using a categorical,
  single-objective TPE sampler.

Changing the candidate space or strategy changes the specification fingerprint.
Use a fresh study name/database when a previously populated study is no longer
compatible. PHYS checks the stored fingerprint and refuses to mix incompatible
studies.

### 5. Generate candidate records

After reviewing the full table, set:

```python
ACTION_MODE = "generate_candidates"
```

Rerun the candidate section. PHYS records the selected configurations in Optuna
and writes the candidate summary, manual change manifest, and run-map rows. It
does not generate or edit ROMS input files.

`CANDIDATE_BATCH_LABEL` in the control panel defaults to `batch_001`. Rerunning
generation with the same label recovers the existing batch instead of duplicating
it, including completed or failed candidate records. Change the label only when
you intentionally want a distinct later batch under the same scientific study.

Return `ACTION_MODE` to `"audit"` immediately afterward to reduce the chance of
accidental repeat actions.

### 6. Build and name runs manually

Use the change manifest as a checklist and the full summary as the authoritative
configuration. After assigning a real model-run name, edit:

`provenance/physical_candidate_run_map.csv`

For example:

```text
candidate_id,optuna_trial_number,actual_run_name,mo_bio_study_name,mo_bio_trial_number,status
PHYS_000,0,OPTUNA_BIO_MO_107,model_calibration_bio_cost_bottom_ph,107,prepared
```

Keep `candidate_id` and `optuna_trial_number` unchanged. Valid status labels are
documentary rather than enforced workflow states; recommended values are
`planned`, `prepared`, `running`, `complete`, `failed`, and `excluded`.
For paired MO-BIO/PHYS runs, `actual_run_name` is the official MO-BIO run name.
The two MO-BIO columns are redundant by design: they make the cross-study pairing
auditable without decoding the run name.

### 7. Register results

Add completed run names and files to the established archive manifest. Rerun with
`FORCE_REFRESH = True`, inspect the parsed cost and mapping, and only then enable
the notebook’s explicit completion action. A run is never matched by guessing a
filename; the run map is the identity bridge.

## Scientific rules encoded initially

### `nl_tnu2`

`nl_tnu2_tracer0` and `nl_tnu2_tracer1` must be equal. They are represented as
one conceptual parameter, `nl_tnu2_shared`. New candidates permit only 0.5 and
1.0. Historical 2.0 and 4.0 values remain valid evidence and are not quality
errors, but they are excluded from new candidates.

### Advection

Temperature and salinity must have identical horizontal schemes and identical
vertical schemes. Allowed new pairs are:

- Upstream3 / Centered4 (`U3_C4`); and
- Akima4 / Akima4 (`A4_A4`).

MPDATA/MPDATA and HSIMT/HSIMT remain visible historically but are excluded from
new testing because of runtime and stability concerns. The exact archive spelling
is `HSIMT`.

### Mixing orientation

Exactly one option from each group must be present:

- UV: `MIX_S_UV`, `MIX_ISO_UV`, or `MIX_GEO_UV`;
- TS: `MIX_S_TS`, `MIX_ISO_TS`, or `MIX_GEO_TS`.

Only exact UV/TS pairs present in the TOML may become candidates. PHYS never
creates the independent 3×3 cross-product implicitly.

### Required mass-conservation bundle

Every new candidate enables all three:

- `GLS_MIXING`;
- `RI_SPLINES`; and
- `N2S2_HORAVG`.

This is an externally established invariant. Runs lacking it remain in broad
historical screening because their other settings may still be informative.

### Optional CPP options

`KANTHA_CLAYSON`, `CRAIG_BANNER`, and `CANUTO_A` are tracked. They are represented
as named joint states so observed dependencies are visible. Be aware that fixing
`RI_SPLINES=True` while selecting an optional state historically seen with
`RI_SPLINES=False` creates a new complete CPP combination. The notebook marks
whether the resulting full candidate tuple was observed exactly.

### Other numeric parameters

All numeric values are read. Only historically varied fields are highlighted.
Unless a future TOML revision explicitly promotes another parameter, new
candidates use its value from the latest completed station file. “Latest” means
the timestamp in the station file’s `history` attribute, not filename order or
filesystem modification time.

## Interpreting the main evidence

- Prefer medians and full distributions over a single best run.
- Always inspect run counts. A level represented once is an anecdote.
- Compare the all-run and required-bundle-only tables. A reversal suggests
  confounding or interaction.
- Inspect the four cost components before accepting a low total.
- Repeated identical costs may represent true duplicate configurations or reused
  outputs; use the exact configuration and provenance tables to distinguish them.
- An unobserved factorial cell is a proposed experiment, not an inferred winner.
- Historical screening can identify promising choices, but sparse archives may
  be unable to separate factors that always changed together.

## Quality issues and verified corrections

Routine successful reads are not printed. The notebook reports only meaningful
problems: unreadable files, completely missing required fields, coupling or
exclusive-option violations, suspicious tiny nonzero values, and costs above the
configured threshold. Full details go to
`quality_control/physical_data_quality_issues.csv`.

To correct a verified archive value without editing the source file, add one row
to `physical_data_corrections.csv`:

| Column | Required meaning |
|---|---|
| `run_name` | Exact `Run Name` from the manifest |
| `field` | Exact parsed dataframe column |
| `corrected_value` | Verified replacement |
| `reason` | Why the source value is wrong |
| `verified_by` | Person or source used to verify it |
| `verified_date` | Date of verification |

Corrections are validated strictly. Unknown runs, unknown fields, malformed
Boolean values, and ambiguous duplicate run names stop execution. The source
NetCDF file is never changed. Coupled and categorical conceptual fields are
recomputed after corrections. The quality log intentionally continues to record
the original source-file problem, while the applied-corrections table shows what
the analysis used.

## CPP executable availability

CPP options are compile-time settings. A scientifically valid candidate is not
necessarily runnable with an executable already on hand. PHYS specifies desired
CPP states but does not compile executables or verify binaries. Before running a
candidate, record executable availability in your operational notes and verify
that the binary contains the intended options.

## Git snapshot and recovery

The pre-redesign PHYS notebook is preserved in commit:

`fad2127` — `snapshot PHYS before constrained tuning redesign`

Inspect the old notebook without changing the working copy:

```bash
git show fad2127:CandidateParameterSetsPhysical.ipynb
```

Recover it to a separate file:

```bash
git show fad2127:CandidateParameterSetsPhysical.ipynb > CandidateParameterSetsPhysical_pre_redesign.ipynb
```

Restoring over the active notebook discards uncommitted notebook edits, so first
save or commit anything you need. Then, only if replacement is truly intended:

```bash
git restore --source fad2127 -- CandidateParameterSetsPhysical.ipynb
```

## Maintenance principles

1. Preserve the original cost definition unless a new objective version and new
   study are deliberately created.
2. Keep scientific rules in the TOML, stable mechanics in the helper, and guided
   decisions in the notebook.
3. Never silently normalize malformed station values or guess an intended CPP or
   advection setting.
4. Keep historical evidence even when a value is prohibited for future trials.
5. Treat changes to allowed levels, coupling, fixed options, candidate mode, or
   strategy as study-defining changes.
6. Add tests whenever parsing or constraint behavior changes.
7. Keep output concise; detailed anomalies belong in CSV logs.
8. Candidate tables specify what to build but do not prove that manually edited
   ROMS files match the specification.
