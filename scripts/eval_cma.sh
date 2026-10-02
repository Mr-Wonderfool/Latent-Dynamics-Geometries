#!/bin/bash
set -euo pipefail

AGENT="${AGENT:-rma}"
BASE_EXP_DIR="${BASE_EXP_DIR:?}"
MODE="${MODE:-normal}"

cd "$(dirname "$0")"
python exp/eval_cma.py --agent "$AGENT" --base_exp_dir "$BASE_EXP_DIR" --mode "$MODE"
