#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n ldr python=3.10 -y
conda activate ldr
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .