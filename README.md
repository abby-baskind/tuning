# ROMS–COSiNE Narragansett Bay tuning

Notebook-based workflows for evaluating and tuning physical and biological
parameters in the ROMS–COSiNE Narragansett Bay model. The repository includes
cost analysis, candidate generation, constrained physical tuning, and
multi-objective biological tuning against total model cost and bottom-pH
performance.

The model archives and generated tuning outputs are intentionally kept outside
Git. This repository contains the analysis code, scientific configuration,
small provenance tables, and documentation needed to reproduce the workflows
against those local archives.

## Main workflows

| Workflow | Entry point | Supporting files |
|---|---|---|
| Multi-objective biological tuning | `CandidateParameterSetsBio_MultiObjectiveTPE.ipynb` | `candidate_parameter_utils.py`, `parameter_priors.csv`, `MULTIOBJECTIVE_TPE_GUIDE.md` |
| Constrained physical tuning | `CandidateParameterSetsPhysical.ipynb` | `physical_candidate_utils.py`, `physical_tuning_spec.toml`, `PHYSICAL_TUNING_GUIDE.md` |
| Cost calculation | `cost_function_v2.ipynb` | `run_cost_batch.sh`, `cost_runs.csv`, `cost_analysis_history.csv` |
| Parameter analysis | `ParameterAnalysis.ipynb` | Historical run and cost inputs |

`CandidateParameterSets.ipynb` and `CandidateParameterSetsBio.ipynb` are the
earlier candidate-generation workflows retained for continuity with existing
studies.

## Requirements

- Python 3.11 or newer
- JupyterLab or Jupyter Notebook
- Access to the local ROMS optimization, cost, and station-file archives

Install the Python dependencies into an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Several notebooks start with machine-specific default paths. Review their
control cells before running them. The multi-objective workflow also supports
the `ROMS_OPT_DIR` and `ROMS_COST_DIR` environment variables; see its guide for
details.

## Getting started

1. Clone the repository and create the environment above.
2. Run the fast regression tests:

   ```bash
   python -m unittest discover -v
   ```

3. Choose the biological or physical workflow from the table above.
4. Read its guide and review every user-input or action-mode control before
   executing the notebook.
5. Begin in the workflow's non-mutating review or audit mode.

The tuning notebooks can create Optuna studies, candidate files, caches, audit
tables, and quality-control ledgers. Their guides distinguish regenerable files
from user-maintained scientific decisions and provenance.

## Repository conventions

- Scientific choices belong in `parameter_priors.csv`,
  `physical_tuning_spec.toml`, or clearly marked notebook controls.
- Reusable parsing, validation, provenance, and candidate mechanics belong in
  the Python helper modules.
- Generated notebooks, batch logs, caches, databases, and large model outputs
  do not belong in Git.
- Changes to a helper module should include or update its corresponding
  `test_*.py` regression tests.

## Documentation

- [Multi-objective biological tuning guide](MULTIOBJECTIVE_TPE_GUIDE.md)
- [Physical tuning guide](PHYSICAL_TUNING_GUIDE.md)

## Status

This is an active research repository. Paths, archive schemas, and scientific
decisions are tailored to the Narragansett Bay ROMS–COSiNE tuning workflow and
should be reviewed before reuse in another model configuration.
