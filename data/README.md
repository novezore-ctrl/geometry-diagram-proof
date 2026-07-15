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
