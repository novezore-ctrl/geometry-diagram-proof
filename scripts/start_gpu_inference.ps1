$ErrorActionPreference = "Stop"
$Project = "/mnt/e/codex/geometry_diagram_parser"
$Python = "/mnt/d/AI_Projects/geometry_diagram_proof/.venv/bin/python"
$Checkpoint = "/mnt/d/AI_Projects/geometry_diagram_proof/ml/checkpoints/geometry_unet_pgdp5k_gpu.pt"

Write-Host "Starting local GPU geometry service on http://127.0.0.1:8765"
Write-Host "Safety limits: CUDA required, batch=1, process memory fraction=0.25"
wsl -d Ubuntu-22.04 --cd $Project -- $Python ml/serve.py `
  --checkpoint $Checkpoint `
  --host 127.0.0.1 `
  --port 8765 `
  --cuda-memory-fraction 0.25
