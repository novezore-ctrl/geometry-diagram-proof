#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
mkdir -p "$PROJECT_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# Verified GPU combination on this machine (RTX 5060 Laptop, sm120):
# Python 3.10 + torch 2.11.0/cu130 + torchvision 0.26.0/cu130.
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
python -m pip install Pillow numpy
printf '%s\n' "$VENV_DIR" > "$PROJECT_DIR/.venv_path"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
