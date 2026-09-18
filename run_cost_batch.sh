#!/usr/bin/env bash

# Execute cost_function_v2.ipynb once for each row in a two-column CSV
# manifest. Papermill needs an output notebook while a run is active, but all
# such notebooks are written under a guarded temporary directory and deleted.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_PATH="${SCRIPT_DIR}/cost_function_v2.ipynb"
DEFAULT_MANIFEST="${SCRIPT_DIR}/cost_runs.csv"
HISTORY_PATH="${SCRIPT_DIR}/cost_analysis_history.csv"
PYTHON_ENV="/Users/akbaskind/opt/anaconda3/envs/my_env"
PAPERMILL="${PYTHON_ENV}/bin/papermill"

DRY_RUN=false
VALIDATE_ONLY=false
MANIFEST_PATH=""


usage() {
    cat <<'EOF'
Usage:
  ./run_cost_batch.sh [--dry-run] [--validate-only] [manifest.csv]

Options:
  --dry-run       Execute every run but inject SAVE_COST_FILE=false.
  --validate-only Validate configuration and manifest rows without execution.
  -h, --help      Show this help message.

If manifest.csv is omitted, cost_runs.csv beside this script is used.

Normal execution writes cost NetCDF files through the notebook, retains one
text log per run, writes a batch summary CSV, and appends every attempted run
to cost_analysis_history.csv. Executed notebooks are never retained.
EOF
}


die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}


while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "Unknown option: $1"
            ;;
        *)
            if [[ -n "${MANIFEST_PATH}" ]]; then
                die "Only one manifest path may be provided."
            fi
            MANIFEST_PATH="$1"
            ;;
    esac
    shift
done

if [[ -z "${MANIFEST_PATH}" ]]; then
    MANIFEST_PATH="${DEFAULT_MANIFEST}"
elif [[ "${MANIFEST_PATH}" != /* ]]; then
    MANIFEST_PATH="$(cd -- "$(dirname -- "${MANIFEST_PATH}")" && pwd)/$(basename -- "${MANIFEST_PATH}")"
fi

[[ -f "${NOTEBOOK_PATH}" ]] || die "Notebook not found: ${NOTEBOOK_PATH}"
[[ -f "${MANIFEST_PATH}" ]] || die "Manifest not found: ${MANIFEST_PATH}"
[[ -x "${PAPERMILL}" ]] || die "Papermill is not executable: ${PAPERMILL}"

# Read the header separately. Removing a trailing carriage return supports CSV
# files saved with Windows line endings without relaxing the required schema.
IFS= read -r manifest_header < "${MANIFEST_PATH}" || die "Manifest is empty."
manifest_header="${manifest_header%$'\r'}"
[[ "${manifest_header}" == "runname,runyear" ]] || die \
    "Manifest header must be exactly: runname,runyear"

declare -a RUN_NAMES=()
declare -a RUN_YEARS=()
declare -a SEEN_RUNS=()
seen_count=0

line_number=1
while IFS=, read -r runname runyear extra; do
    ((line_number += 1))
    runyear="${runyear%$'\r'}"

    # Empty lines are ignored; partially empty or extra-column rows are not.
    if [[ -z "${runname}" && -z "${runyear}" && -z "${extra:-}" ]]; then
        continue
    fi
    [[ -z "${extra:-}" ]] || die \
        "Line ${line_number} has more than two columns."
    [[ "${runname}" =~ ^[A-Za-z0-9_.-]+$ ]] || die \
        "Line ${line_number} has an invalid runname: '${runname}'"
    [[ "${runyear}" =~ ^[0-9]{4}$ ]] || die \
        "Line ${line_number} has an invalid runyear: '${runyear}'"
    ((10#${runyear} >= 2005 && 10#${runyear} <= 2024)) || die \
        "Line ${line_number} has an unsupported year: ${runyear}"

    run_key="${runname},${runyear}"
    # macOS ships Bash 3.2, which predates associative arrays. The manifests
    # are intentionally small, so this portable indexed-array check is ample.
    if ((seen_count > 0)); then
        for seen_key in "${SEEN_RUNS[@]}"; do
            [[ "${seen_key}" != "${run_key}" ]] || die \
                "Duplicate manifest row: ${run_key}"
        done
    fi
    SEEN_RUNS+=("${run_key}")
    ((seen_count += 1))
    RUN_NAMES+=("${runname}")
    RUN_YEARS+=("${runyear}")
done < <(tail -n +2 -- "${MANIFEST_PATH}")

((${#RUN_NAMES[@]} > 0)) || die "Manifest contains no runs."

printf 'Manifest validation passed\n'
printf '  Manifest: %s\n' "${MANIFEST_PATH}"
printf '  Notebook: %s\n' "${NOTEBOOK_PATH}"
printf '  Runs:     %d\n' "${#RUN_NAMES[@]}"
for index in "${!RUN_NAMES[@]}"; do
    printf '    %2d. %s (%s)\n' \
        "$((index + 1))" "${RUN_NAMES[index]}" "${RUN_YEARS[index]}"
done

if [[ "${VALIDATE_ONLY}" == true ]]; then
    printf 'Validation-only mode complete; no runs were executed.\n'
    exit 0
fi

if [[ "${DRY_RUN}" == true ]]; then
    save_cost_file=false
    batch_mode="dry_run"
else
    save_cost_file=true
    batch_mode="production"
fi

# This ledger persists across timestamped batch directories. Quoting every CSV
# field makes paths safe even if a future directory or filename contains a
# comma. Validation-only invocations are omitted because they analyze no runs.
HISTORY_HEADER="runname,station_file,cost_file,analysis_date_utc,runyear,mode,status,batch_id"
if [[ -e "${HISTORY_PATH}" && ! -f "${HISTORY_PATH}" ]]; then
    die "Analysis history path is not a regular file: ${HISTORY_PATH}"
fi
if [[ -s "${HISTORY_PATH}" ]]; then
    IFS= read -r history_header < "${HISTORY_PATH}" || die \
        "Could not read analysis history: ${HISTORY_PATH}"
    history_header="${history_header%$'\r'}"
    [[ "${history_header}" == "${HISTORY_HEADER}" ]] || die \
        "Unexpected analysis history header in ${HISTORY_PATH}"
else
    printf '%s\n' "${HISTORY_HEADER}" > "${HISTORY_PATH}"
fi

csv_field() {
    # RFC 4180-style escaping: double embedded quotes, then quote the field.
    local value="$1"
    value="${value//\"/\"\"}"
    printf '"%s"' "${value}"
}

batch_started_epoch="$(date +%s)"
batch_id="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${SCRIPT_DIR}/batch_output/${batch_id}"
LOG_DIR="${OUTPUT_DIR}/logs"
SUMMARY_PATH="${OUTPUT_DIR}/batch_summary.csv"
mkdir -p -- "${LOG_DIR}"

# Keep temporary notebooks on the same local filesystem as the logs. The
# guarded cleanup refuses to delete anything outside the expected prefix.
BATCH_TEMP_DIR="$(mktemp -d "${OUTPUT_DIR}/.tmp.cost-batch.XXXXXX")"
cleanup() {
    case "${BATCH_TEMP_DIR:-}" in
        "${OUTPUT_DIR}"/.tmp.cost-batch.*)
            rm -rf -- "${BATCH_TEMP_DIR}"
            ;;
        *)
            printf 'WARNING: Refused unexpected temporary path: %s\n' \
                "${BATCH_TEMP_DIR:-unset}" >&2
            ;;
    esac
}
trap cleanup EXIT INT TERM

printf 'runname,runyear,status,exit_code,start_utc,end_utc,duration_seconds,log_file\n' \
    > "${SUMMARY_PATH}"

if [[ "${DRY_RUN}" == true ]]; then
    printf 'Mode: dry run; cost NetCDF files will not be written.\n'
else
    printf 'Mode: production; existing cost NetCDF files may be overwritten.\n'
fi
printf 'Batch output: %s\n' "${OUTPUT_DIR}"
printf 'Analysis history: %s\n' "${HISTORY_PATH}"

success_count=0
failure_count=0

for index in "${!RUN_NAMES[@]}"; do
    runname="${RUN_NAMES[index]}"
    runyear="${RUN_YEARS[index]}"
    run_id="${runname}_${runyear}"
    log_path="${LOG_DIR}/${run_id}.log"
    relative_log="logs/${run_id}.log"
    parameter_path="${BATCH_TEMP_DIR}/${run_id}.yaml"
    executed_path="${BATCH_TEMP_DIR}/${run_id}.ipynb"
    run_started_epoch="$(date +%s)"
    run_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # These YAML values are deliberately unquoted where type matters. The
    # notebook independently rejects string-valued Booleans before processing.
    {
        printf 'runname: %s\n' "${runname}"
        printf 'runyear: %s\n' "${runyear}"
        printf 'SAVE_COST_FILE: %s\n' "${save_cost_file}"
        printf 'OVERWRITE_EXISTING_COST_FILE: true\n'
    } > "${parameter_path}"

    printf '\n[%d/%d] Starting %s\n' \
        "$((index + 1))" "${#RUN_NAMES[@]}" "${run_id}"

    set +e
    IPYTHONDIR="${BATCH_TEMP_DIR}/ipython" \
    MPLCONFIGDIR="${BATCH_TEMP_DIR}/matplotlib" \
    "${PAPERMILL}" \
        "${NOTEBOOK_PATH}" \
        "${executed_path}" \
        --kernel python3 \
        --cwd "${SCRIPT_DIR}" \
        --parameters_file "${parameter_path}" \
        --no-progress-bar \
        --log-output \
        --log-level INFO \
        2>&1 | tee "${log_path}"
    run_exit_code="${PIPESTATUS[0]}"
    set -e

    # Remove the temporary notebook and parameter file immediately. The EXIT
    # trap is a second cleanup layer for interruptions or unexpected failures.
    rm -f -- "${executed_path}" "${parameter_path}"

    run_finished_epoch="$(date +%s)"
    run_finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_duration="$((run_finished_epoch - run_started_epoch))"

    if ((run_exit_code == 0)); then
        status="success"
        ((success_count += 1))
        printf '[%d/%d] SUCCESS %s (%d seconds)\n' \
            "$((index + 1))" "${#RUN_NAMES[@]}" "${run_id}" "${run_duration}"
    else
        status="failed"
        ((failure_count += 1))
        printf '[%d/%d] FAILED %s (exit %d; %d seconds)\n' \
            "$((index + 1))" "${#RUN_NAMES[@]}" "${run_id}" \
            "${run_exit_code}" "${run_duration}" >&2
    fi

    # These values come from the notebook after it resolves FileNames.xlsx and
    # its year-specific fallback. A failure before path resolution leaves the
    # corresponding fields blank while status preserves the attempted run.
    station_file="$(sed -n \
        's/^.*Station input:[[:space:]]*//p' "${log_path}" | tail -n 1)"
    cost_file="$(sed -n \
        's/^.*Cost output:[[:space:]]*//p' "${log_path}" | tail -n 1)"

    {
        csv_field "${runname}"
        printf ','
        csv_field "${station_file}"
        printf ','
        csv_field "${cost_file}"
        printf ','
        csv_field "${run_finished_utc}"
        printf ','
        csv_field "${runyear}"
        printf ','
        csv_field "${batch_mode}"
        printf ','
        csv_field "${status}"
        printf ','
        csv_field "${batch_id}"
        printf '\n'
    } >> "${HISTORY_PATH}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${runname}" "${runyear}" "${status}" "${run_exit_code}" \
        "${run_started_utc}" "${run_finished_utc}" "${run_duration}" \
        "${relative_log}" >> "${SUMMARY_PATH}"
done

batch_finished_epoch="$(date +%s)"
batch_duration="$((batch_finished_epoch - batch_started_epoch))"

printf '\nBatch complete\n'
printf '  Successful runs: %d\n' "${success_count}"
printf '  Failed runs:     %d\n' "${failure_count}"
printf '  Duration:        %d seconds\n' "${batch_duration}"
printf '  Summary:         %s\n' "${SUMMARY_PATH}"
printf '  Analysis history:%s\n' " ${HISTORY_PATH}"

if ((failure_count > 0)); then
    exit 1
fi
