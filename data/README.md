# Geometry diagram datasets

## Geometry3K (downloaded)

- Source: https://huggingface.co/datasets/LibraTree/geometry3k
- Local role: structured-annotation validation set and rasterization source.
- Current archive: `D:/AI_Projects/geometry_diagram_proof/data/geometry3k/geometry3k_data.zip`
- The extracted subset contains 47 images and matching `ex.json` files. The annotations include image dimensions, point coordinates, line instances, circle instances, and diagram logic forms.
- The Hugging Face card marks the license as `other`; do not redistribute this copy until the original Geometry3K terms are confirmed.

## Geo170K (downloaded)

- Source: https://huggingface.co/datasets/Luckyjhg/Geo170K
- Local role: larger image/text pretraining and description-generation corpus; it is not a pixel-level segmentation set.
- Downloaded files: `images.zip` (146,943,927 bytes), `qa_tuning.parquet` (19,804,893 bytes).
- Extracted image count: 12,785 PNG files.
- The dataset card states an MIT license. Verify the upstream terms again before publishing derivatives.

## Training boundary

Geometry3K is the high-value source for point/line/circle structure, while Geo170K broadens visual and language coverage. Neither source directly provides the exact six-class pixel masks used by the current U-Net, so a converter must rasterize Geometry3K primitives and synthetic masks remain useful for initial training.

## PGDP5K (downloaded)

- Source mirror: https://huggingface.co/datasets/PeijieWang/PGDP5K (original CASIA page: https://nlpr.ia.ac.cn/databases/CASIA-PGDP5K/)
- Local archive: `D:/AI_Projects/geometry_diagram_proof/data/pgdp5k/PGDP5K.zip`
- 5,000 diagrams: 3,500 train, 500 validation, 1,000 test.
- Primitive annotations include points, line endpoints, circles/arcs, and point-to-line/point-to-circle relations. The source describes solid, dashed and mixed line types, but the compact `geos` records used by this converter do not expose a reliable per-line style field; converted line masks are therefore marked solid.
- License/use boundary: the Hugging Face mirror says MIT, while the original CASIA page says academic research is free under an agreement and commercial use requires contact. Treat commercial redistribution as restricted until the original terms are confirmed.
- Converted output: `D:/AI_Projects/geometry_diagram_proof/data/pgdp5k/processed` with 5,000 image/mask pairs.
