#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$(cat "$PROJECT_DIR/.venv_path")"
source "$VENV_DIR/bin/activate"
python "$PROJECT_DIR/ml/synthetic_dataset.py" --out "$PROJECT_DIR/ml/data/synthetic" --count 32
python "$PROJECT_DIR/ml/train.py" --data "$PROJECT_DIR/ml/data/synthetic" --epochs 1 --out "$PROJECT_DIR/ml/checkpoints/geometry_unet_smoke.pt"
