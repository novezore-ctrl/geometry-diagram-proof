# Geometry segmentation model

This folder contains the next recognition pipeline: a small U-Net with a
MobileNet-style depthwise-separable encoder. It predicts pixel masks for
`background`, `solid_line`, `dashed_line`, `circle_or_ellipse`, `point`, and
`text`.

The model does not directly invent labels or relations. After segmentation,
the deterministic geometry layer fits primitives, calculates intersections,
and exports a JSON graph. The web app can later load an exported Core ML model.

## Local training

Create an isolated Python environment in the project directory, then run:

```bash
bash scripts/setup_wsl_home_venv.sh
bash scripts/run_ml_smoke_wsl.sh
```

Synthetic data only verifies that the pipeline works. Real accuracy requires
annotated geometry-question images.
