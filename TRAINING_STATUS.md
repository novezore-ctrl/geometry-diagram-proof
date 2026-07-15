# Geometry diagram training handoff

Updated: 2026-07-15

## Current data

- PGDP5K source archive: `D:/AI_Projects/geometry_diagram_proof/data/pgdp5k/PGDP5K.zip`
- PGDP5K converted pairs: `D:/AI_Projects/geometry_diagram_proof/data/pgdp5k/processed`
- Converted count: 5,000 image/mask pairs (train 3,500, val 500, test 1,000)
- Geometry3K converted pairs: 47
- Geo170K images: 12,785

## Code already in place

- `ml/prepare_pgdp5k.py`: converts PGDP5K primitive annotations to masks
- `ml/train.py`: mixes synthetic and real data; uses named PGDP5K train/val splits when available
- `ml/extract_graph.py`: detects points and creates point-connected line relations

## Training status

The command below was started with PGDP5K, but was stopped by the user after it exceeded the expected CPU runtime. No completed PGDP5K checkpoint was produced.

```text
python ml/train.py --data ml/data/synthetic_v2 --real-data data/pgdp5k/processed --real-repeat 1 --epochs 2 --batch-size 32 --out ml/checkpoints/geometry_unet_pgdp5k.pt
```

The process was terminated cleanly. Existing checkpoints remain:

- `ml/checkpoints/geometry_unet_balanced.pt`
- `ml/checkpoints/geometry_unet_mixed.pt`

## Resume recommendation

Start with one epoch and a larger batch if memory permits, then inspect `line_p`, `line_r`, `point_p`, and `point_r` before deciding on more epochs:

```text
python ml/train.py --data ml/data/synthetic_v2 --real-data data/pgdp5k/processed --real-repeat 1 --epochs 1 --batch-size 64 --out ml/checkpoints/geometry_unet_pgdp5k_epoch1.pt
```

## Verified GPU environment

This machine's other working AI project was checked directly and reported:

- WSL Ubuntu-22.04
- Python 3.11 in that project environment
- `torch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `torch.version.cuda == 13.0`
- `torch.cuda.is_available() == True`
- `NVIDIA GeForce RTX 5060 Laptop GPU`

For this geometry project, the project-local Python 3.10 environment is being switched to the same CUDA 13.0 wheel family (`torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`). Do not use the old CPU-index command or cu126 for RTX 5060 sm120.
