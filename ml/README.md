# Geometry segmentation model

This folder contains the next recognition pipeline: a small U-Net with a
MobileNet-style depthwise-separable encoder. It predicts pixel masks for
`background`, `solid_line`, `dashed_line`, `circle_or_ellipse`, `point`, and
`text`.

The model does not directly invent labels or relations. After segmentation,
the deterministic geometry layer fits primitives, calculates intersections,
and exports a JSON graph. Arrowheads are split from ordinary point blobs by a
conservative line-direction postprocessor, so a tip becomes ray direction
instead of a fake point. The web app can later load an exported Core ML model.

## Local training

Create an isolated Python environment in the project directory, then run:

```bash
bash scripts/setup_wsl_home_venv.sh
bash scripts/run_ml_smoke_wsl.sh
```

Synthetic data only verifies that the pipeline works. Real accuracy requires
annotated geometry-question images.

## End-to-end inference

Run the trained checkpoint on one cropped diagram and export both a color mask
and deterministic `geometry-diagram/v2` JSON:

```bash
.venv/bin/python ml/infer.py input.png \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --out-mask ml/outputs/input.mask.png \
  --out-json ml/outputs/input.graph.json
```

The command defaults to CUDA and fails instead of silently falling back to CPU.
Version 2 adds `arrows` and `attachments`. Coordinates remain visual estimates;
an attachment is exact only after the user confirms an equal-division choice
in the web interface.

## PGDP5K test evaluation

The project machine is safety-capped at batch size 16:

```bash
.venv/bin/python ml/evaluate.py \
  --data data/pgdp5k/processed \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --split test --batch-size 16 --num-workers 2 \
  --out ml/evaluation/pgdp5k_test_epoch5.json
```

The report contains six-class pixel metrics plus point and connected-edge
precision/recall computed from predicted and ground-truth masks.

## Mobile-export intermediate

On Windows/WSL, verify the 320x320 traced model before moving the checkpoint to
a Core ML conversion environment:

```bash
.venv/bin/python ml/export_coreml.py \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --trace-out ml/exports/geometry_unet_pgdp5k_gpu_320x320.torchscript.pt \
  --trace-only
```

Final `.mlpackage` conversion still requires `coremltools` on macOS or another
compatible environment.

## Local GPU web service

The browser UI now calls the checkpoint through a local CUDA bridge. Start it
from Windows PowerShell with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_gpu_inference.ps1
```

The service requires CUDA, loads the epoch-5 checkpoint once, uses mixed
precision with batch size 1, and limits its CUDA allocator fraction to 0.25.
It listens on `127.0.0.1:8765`; `GET /health` reports the active device,
checkpoint epoch, batch size, and current CUDA memory.

For the photographed Figure 2 regression, run:

```bash
/mnt/d/AI_Projects/geometry_diagram_proof/.venv/bin/python \
  tests/figure2_gpu_regression.py \
  --checkpoint /mnt/d/AI_Projects/geometry_diagram_proof/ml/checkpoints/geometry_unet_pgdp5k_gpu.pt
```

Expected acceptance is 7 points, 10 required segments, 0 arrows, and 4
point-on-line relations. The names A-G are assigned by matching the explicit
problem-statement topology to U-Net candidates; this is intentionally labelled
as question-constrained naming rather than glyph OCR.

## Production cross-device browser inference

The CUDA web service above is now an optional desktop diagnostic route. The
production correction page runs the same epoch-5 segmentation model on the
phone itself from:

```text
public/models/geometry_unet_pgdp5k_epoch5_320.onnx
```

`app/mobileInference.ts` selects WebGPU on compatible Windows and Android
browsers and falls back to single-thread WASM. iPhone/iPad Safari therefore
runs on the Apple A/M-series CPU; Windows Edge/Chrome normally uses WebGPU. The
six-class mask is postprocessed locally by `app/mobileQuestionConstraints.ts`;
no selected image is sent to port 8765 or another inference API. Both WebGPU
and WASM regressions remain 7 points, 10 segments, 0 arrows, and 4 point-on-line
relations with the PC GPU service stopped. Apple Neural Engine use would
require a separate native Core ML application and is not claimed here.

The ONNX Runtime 1.27.0 `.mjs` and `.wasm` files are loaded from a
version-pinned CDN so each platform downloads only its selected backend. The
U-Net model remains a same-origin static asset, and image pixels are never sent
to the CDN or an inference service.
