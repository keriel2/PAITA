#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
    --strategy static \
    --model meta-llama/Meta-Llama-3-8B \
    --data_path /path/to/train.csv \
    --epochs 4 \
    --output_dir outputs/static
