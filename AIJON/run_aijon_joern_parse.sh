#!/usr/bin/env bash
set -euo pipefail

# Usage: ./parse_one.sh SUBDIR_NAME
# Example: ./parse_one.sh NOT_HELPFUL_UNREACHED_SATURATED_500

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBDIR="${1:-}"

RAW_PARENT="${REPO_ROOT}/AIJON/data/${SUBDIR}/raw_code"
JOERN_DIR="${REPO_ROOT}/code-slicer/joern"
JOERN_BIN="${JOERN_DIR}/joern-parse"
PARSED_PARENT="${REPO_ROOT}/AIJON/data/${SUBDIR}/parsed"
SUBDIR="${1:-}"
LOG_FILE="${REPO_ROOT}/joern-errors.log"

if [[ -z "${SUBDIR}" ]]; then
  echo "Usage: $0 SUBDIR_NAME" >&2
  exit 1
fi

RAW_DIR="${RAW_PARENT}"
OUT_DIR="${PARSED_PARENT}"

# Checks
[[ -d "${RAW_DIR}" ]] || { echo "[error] Missing input dir: ${RAW_DIR}" >&2; exit 1; }
[[ -x "${JOERN_BIN}" ]] || { echo "[error] Missing or not executable ${JOERN_BIN}" >&2; exit 1; }
[[ -d "${JOERN_DIR}/projects/extensions/joern-fuzzyc/build" ]] || { echo "[error] fuzzyc not built." >&2; exit 1; }

# Clean only this subdir output
rm -rf "${OUT_DIR}"

pushd "${REPO_ROOT}/AIJON/data/${SUBDIR}" >/dev/null
echo "Parsing ${SUBDIR} ..."
"${JOERN_BIN}" "raw_code" 2>> "${LOG_FILE}"
mv "parsed/raw_code/"* "${OUT_DIR}/" 2>/dev/null || true
popd >/dev/null


# Remove leftover raw_code directory
rm -rf "${PARSED_PARENT}/raw_code"

echo "Done. Output: ${OUT_DIR}"
echo "Warnings/errors: ${LOG_FILE}"
