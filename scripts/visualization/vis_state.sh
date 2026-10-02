#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?}"

cd "$(dirname "$0")/.."
python visualization/vis_state.py --model "$MODEL_PATH"
