param(
  [string]$Distribution = "Ubuntu-22.04",
  [string]$Project = "/mnt/e/codex/geometry_diagram_parser",
  [string]$Python = "$Project/.venv/bin/python",
  [string]$Checkpoint = "$Project/ml/checkpoints/geometry_unet_pgdp5k_gpu.pt"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting local GPU geometry service on http://127.0.0.1:8765"
Write-Host "Safety limits: CUDA required, batch=1, process memory fraction=0.25"
wsl -d $Distribution --cd $Project -- $Python ml/serve.py `
  --checkpoint $Checkpoint `
  --host 127.0.0.1 `
  --port 8765 `
  --cuda-memory-fraction 0.25
