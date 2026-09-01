#!/usr/bin/env bash
# Download BPI Challenge 2014 dataset CSVs into data/raw/.
#
# Direct file endpoints resolved from the 4TU.ResearchData RO-CRATE metadata
# (https://data.4tu.nl/v3/datasets/<id>/versions/1/ro-crate-metadata.json).
# Files are semicolon-delimited CSVs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
mkdir -p "${RAW_DIR}"

# name => direct download URL
declare -A FILES=(
  ["Detail_Incident.csv"]="https://data.4tu.nl/file/a752c1b5-9732-4bbc-8949-bf24581f9034/740deeae-5014-48bf-8c86-4a4c3e0355a9"
  ["Detail_Incident_Activity.csv"]="https://data.4tu.nl/file/657fb1d6-b4c2-4adc-ba48-ed25bf313025/bd6cfa31-44f8-4542-9bad-f1f70c894728"
  ["Detail_Change.csv"]="https://data.4tu.nl/file/e9c00fe9-c87a-450e-8bd6-d5e06a6b309a/9d7cd406-9826-4adf-8418-c751b475f556"
  ["Detail_Interaction.csv"]="https://data.4tu.nl/file/f1f0a188-f31a-45af-a7c8-29727b318adf/e5872002-415b-4a37-9377-c82cccae902d"
)

echo "Downloading BPI Challenge 2014 into ${RAW_DIR}"
for name in "${!FILES[@]}"; do
  url="${FILES[$name]}"
  out="${RAW_DIR}/${name}"
  if [[ -f "${out}" && -s "${out}" ]]; then
    echo "  [skip] ${name} already present"
    continue
  fi
  echo "  [get]  ${name}"
  if ! curl -fL --retry 3 -o "${out}" "${url}"; then
    echo "  [FAIL] download failed for ${name}"
    echo "         manual URL: ${url}"
    rm -f "${out}"
  fi
done

echo
echo "Files in ${RAW_DIR}:"
ls -la "${RAW_DIR}"
