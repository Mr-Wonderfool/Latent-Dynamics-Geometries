#!/bin/bash
set -euo pipefail

MONITOR_DIR="${MONITOR_DIR:?}"
OUTPUT_PATH="${OUTPUT_PATH:-reward_curve.png}"

cd "$(dirname "$0")/.."
python visualization/plot_rewards.py --monitor_dir "$MONITOR_DIR" --output "$OUTPUT_PATH"
