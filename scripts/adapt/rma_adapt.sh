#!/bin/bash
set -euo pipefail

EXP_DIR="${EXP_DIR:?}"

cd "$(dirname "$0")/.."
python adapt/rma_adapt.py --exp_dir "$EXP_DIR"
