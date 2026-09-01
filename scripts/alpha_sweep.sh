#!/usr/bin/env bash
set -e

echo "alpha | late-fusion results"

for a in 0.5 0.6 0.7 0.8 0.85 0.9 0.95; do
    echo "Running alpha=$a"
    result=$(uv run python scripts/11_run_sn_text_retrieval.py --alpha "$a" 2>&1 | grep "Late-fusion results:" | sed 's/.*Late-fusion results: //')
    echo "alpha=$a | $result"
done

