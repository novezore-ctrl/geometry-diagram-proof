# Geometry diagram recognition handoff

Updated: 2026-07-16

## Goal and product boundary

Build an iPad-oriented geometry-diagram parser whose main job is to recover the diagram correctly before GPT solves the problem. The user will crop/select the diagram region manually, so full-page object-detection IoU is not the priority.

The important outputs are:

1. accurate points and point positions;
2. accurate solid/dashed lines and endpoint/intersection connectivity;
3. circles/ellipses and their relations to points;
4. editable labels (Latin upper/lowercase, superscript/subscript, and common Greek letters);
5. deterministic structured JSON that can be shared to GPT through the iOS share sheet.

Human correction is part of the intended workflow: the user can tap a detected point and choose/correct its label. Do not spend the next iteration building a general OCR or general-purpose scene detector.

## Project locations

- Training/runtime project: `D:/AI_Projects/geometry_diagram_proof`
- Source repository: `E:/codex/geometry_diagram_parser`
- WSL distribution: `Ubuntu-22.04`
- Project-local virtual environment: `D:/AI_Projects/geometry_diagram_proof/.venv`

Keep all project dependencies inside this project-local environment. Do not modify WSL system Python or shared environments.

## Current data

- PGDP5K archive: `data/pgdp5k/PGDP5K.zip`
- PGDP5K processed pairs: `data/pgdp5k/processed`
- PGDP5K count: 5,000 image/mask pairs (train 3,500, val 500, test 1,000)
- Geometry3K processed pairs: 47 (one malformed source JSON was skipped)
- Geo170K extracted images: 12,785
- Synthetic data: `ml/data/synthetic_v2`

PGDP5K is the primary real-data source. The current converter maps points, straight lines, and curves into segmentation masks. Its compact geometry annotations did not expose a sufficiently reliable dashed-line style, so converted PGDP5K lines currently use the solid-line class. Dashed-line accuracy therefore still needs synthetic data or a separately annotated dataset.

## Code already implemented

- `ml/model.py`: compact U-Net with six classes: background, solid line, dashed line, circle/ellipse, point, text.
- `ml/prepare_geometry3k.py`: Geometry3K preprocessing.
- `ml/prepare_pgdp5k.py`: PGDP5K primitive annotations to segmentation masks.
- `ml/train.py`: mixes synthetic and real data, uses named PGDP5K train/validation splits, resizes to 320x320, and reports class metrics.
- `ml/extract_graph.py`: turns predicted masks into point candidates and point-to-point line-support relations for structured graph output.
- `ml/infer.py`: runs a checkpoint on one cropped diagram and exports a color mask plus scaled `geometry-diagram/v1` JSON.
- `ml/evaluate.py`: evaluates all six mask classes and a mask-derived point/edge topology proxy; this machine is code-capped at batch 16.
- `ml/export_coreml.py`: reads the checkpoint input size instead of using the old incompatible 256x256 constant; the epoch-5 model therefore exports at 320x320 when `coremltools` is available.
- `scripts/setup_wsl_home_venv.sh`: records the verified CUDA wheel family. Do not restore the old CPU index or cu126 combination.

The metrics most relevant to the product are `line_p`, `line_r`, `point_p`, `point_r`, `curve_p`, and `curve_r`. mIoU can be retained as a diagnostic, but it is not the main acceptance criterion.

## Completed training and checkpoints

Existing checkpoints:

- `ml/checkpoints/geometry_unet_balanced.pt`
- `ml/checkpoints/geometry_unet_mixed.pt`
- `ml/checkpoints/geometry_unet_pgdp5k_gpu.pt` (completed PGDP5K GPU training, epoch 5)
- `ml/checkpoints/geometry_unet_pgdp5k_gpu.epoch1.pt` through `.epoch5.pt`

Previous PGDP5K CPU attempts were stopped because they were too slow. One batch-64 CPU attempt exited abnormally, and a batch-16 attempt was also stopped when the work switched to GPU setup. No checkpoint from those CPU attempts is complete; in particular, the nonexistent filename `geometry_unet_pgdp5k.pt` must not be reported as trained.

The PGDP5K GPU run was completed on 2026-07-16 using the project `.venv`, CUDA mixed precision, batch size 16, two data-loader workers, and a 75% CUDA allocator cap. The final checkpoint is `geometry_unet_pgdp5k_gpu.pt`.

Validation metrics by epoch:

| Epoch | Loss | mIoU | Line P/R | Point P/R | Curve P/R | Epoch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4053 | 0.3339 | 0.568 / 0.717 | 0.694 / 0.486 | 0.244 / 0.294 | 330.7 s |
| 2 | 0.1016 | 0.4207 | 0.594 / 0.839 | 0.611 / 0.835 | 0.543 / 0.754 | 85.6 s |
| 3 | 0.0695 | 0.4358 | 0.554 / 0.908 | 0.678 / 0.836 | 0.548 / 0.876 | 54.4 s |
| 4 | 0.0614 | 0.4166 | 0.502 / 0.929 | 0.631 / 0.878 | 0.472 / 0.924 | 54.0 s |
| 5 | 0.0575 | 0.4783 | 0.545 / 0.923 | 0.631 / 0.894 | 0.558 / 0.894 | 53.5 s |

The epoch-5 checkpoint was reloaded on the RTX 5060 and successfully ran a real validation image (`val_1000.png`). Its output was finite with shape `[1, 6, 320, 320]`.

## Completed end-to-end inference and test evaluation

The image-to-model-to-mask-to-graph path is now implemented and was verified on `test_100.png` using CUDA and mixed precision. It produced three points and three correct mask-supported connections in `ml/outputs/test_100.graph.json`, plus a color mask in `ml/outputs/test_100.mask.png`.

The full 1,000-image PGDP5K test split was evaluated with batch size 16 and two data-loader workers. The formal report is `ml/evaluation/pgdp5k_test_epoch5.json`.

Test results:

- pixel accuracy: `0.983167`;
- mean IoU across the four classes present in the converted test masks: `0.652959`;
- solid-line precision/recall: `0.539047 / 0.928956`;
- curve precision/recall: `0.560995 / 0.904029`;
- point-mask precision/recall: `0.626496 / 0.889827`;
- mask-derived point topology precision/recall: `0.895616 / 0.975170`;
- mask-derived connected-edge precision/recall: `0.749347 / 0.933475`;
- exact mask-derived graph rate: `0.301`.

Topology claim boundary: these topology metrics compare graphs extracted from predicted and ground-truth segmentation masks at 320x320 with a 10-pixel point tolerance and 0.58 line-support threshold. They are a reproducible proxy, not direct scoring against the original PGDP5K relation annotations. The converted PGDP5K masks contain no dashed-line or text class pixels, so this test does not validate dashed-line classification or label OCR.

A manual audit of the highest-error representative `test_1012.png` confirmed the main failure mechanism: arrowheads can be classified as point blobs, some small marked points are missed, and residual text/annotation strokes can become false primitive pixels. This agrees with the high edge recall but lower edge precision and makes false-connection suppression the next model/parser priority.

## Completed arrow and attachment semantics (2026-07-16)

The current parser now exports `geometry-diagram/v2` semantics in both the web prototype and Python graph layer:

- arrowheads are represented as ray-direction tips, not ordinary geometry points;
- the six-class GPU checkpoint is unchanged, but Python mask postprocessing separates a conservative one-sided arrow candidate from dots/vertices by checking the surrounding shaft directions;
- the audited `test_1012` mask now yields three arrow tips (the three visible arrowheads) and six ordinary point candidates instead of treating all nine blobs as points;
- a T-junction in the web graph becomes a line-inside-line attachment candidate;
- the default attachment state is `approximate_location` with uncertain coordinates and ratio;
- the user can confirm 2 through 10 equal parts and choose the indexed division point from the displayed first endpoint;
- GPT sharing includes a Chinese explanation plus structured JSON, and explicitly says only `user_confirmed_division` is exact;
- the built-in case contains one arrow ray and one T-junction for direct browser verification.

This is a parser/postprocessing improvement and did not restart GPU training. The checkpoint still has six segmentation classes and no separately trained arrow class. Representative manual audits remain necessary for photographed or noisy diagrams.

The synchronized D-drive runtime was then verified with an actual CUDA + mixed-precision inference on `test_1012.png`. It wrote `ml/outputs/test_1012_v2.graph.json` and `ml/outputs/test_1012_v2.mask.png`, reporting 6 ordinary points, 10 point-to-point connections, and 3 arrow tips. Python currently leaves `attachments` empty for this crowded multi-line diagram; the web correction layer is the authoritative place to confirm a T-junction's equal-division or uncertain-position semantics.

## Completed multi-position edit regression fix (2026-07-16)

The web correction layer previously required a junction to have exactly three incident graph edges. Adding `P5` and the long segment `P5-P3` therefore removed the old `P3 on P1-P2` candidate and did not discover that `P4` lies inside `P3-P5`.

Attachment derivation now uses geometric collinearity plus any supporting whole or partial segment instead of a degree-3-only rule. It can expose several independent point-on-segment relations at once, keeps a user-confirmed division choice across surrounding edits, and treats a long segment passing through an existing point as a valid host. The exact live regression state now yields both `P3@P1-P2` and `P4@P3-P5`; the production page was rebuilt and verified with both selectors visible simultaneously.

The corrected 320x320 mobile-export intermediate was also executed, saved, reloaded, and checked for finite `[1, 6, 320, 320]` output. The verified artifact is `ml/exports/geometry_unet_pgdp5k_gpu_320x320.torchscript.pt`; its maximum traced-vs-eager difference on the verification input was `0.0`. Final Core ML `.mlpackage` conversion remains incomplete because it requires `coremltools` on macOS or another compatible conversion environment.

## Verified GPU environment

Verified live on 2026-07-16 from the geometry project's own `.venv`:

- Python environment: `D:/AI_Projects/geometry_diagram_proof/.venv`
- `torch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `torch.version.cuda == 13.0`
- `torch.cuda.is_available() == True`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- NVIDIA/CUDA dedicated framebuffer memory: `8151 MiB` (about 8 GiB).
- Windows DirectX reports `7899 MB` dedicated memory, `18347 MB` shared system memory, and `26246 MB` total display memory for the NVIDIA adapter. Shared system memory is not equivalent to dedicated CUDA VRAM.

The RTX 5060 uses the newer sm120 architecture. The previously tried cu126 package warned that this GPU architecture was unsupported. Use the verified cu130 combination above.

Old duplicate CPU/cu126 package metadata was removed, temporary pip/CUDA download files were cleaned, and WSL was shut down after setup to release memory. There is no residual training or pip process at this handoff.

## Completed GPU action and next validation

The one-epoch baseline and the continuation through epoch 5 are complete. The safe continuation command, if later experiments require more epochs, is:

```bash
cd /mnt/d/AI_Projects/geometry_diagram_proof
PYTHONUNBUFFERED=1 .venv/bin/python ml/train.py \
  --data ml/data/synthetic_v2 \
  --real-data data/pgdp5k/processed \
  --real-repeat 1 \
  --epochs 6 \
  --batch-size 16 \
  --num-workers 2 \
  --amp \
  --require-cuda \
  --cuda-memory-fraction 0.75 \
  --resume ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --out ml/checkpoints/geometry_unet_pgdp5k_gpu.pt
```

Do not increase the batch size for CUDA training on this machine. The RTX 5060 exposes `8151 MiB` of dedicated framebuffer memory to NVIDIA/CUDA; Windows' larger total-display-memory figure includes shared system memory and must not be treated as equally fast training VRAM. A live mixed-precision probe measured about 4.00 GiB reserved at batch 16, 5.96 GiB at batch 24, and 7.87 GiB at batch 32. Batch 32 exhausted dedicated VRAM and slowed the median GPU step from 0.223 s at batch 16 to 4.016 s as WDDM fell under memory pressure. This near-exhaustion is the strongest reproduced explanation for the previous whole-computer freeze risk; batch 64 is unsafe here.

The training and evaluation scripts now refuse batch sizes above 16 on this <=9 GiB dedicated-VRAM device. The detailed hardware and per-batch risk record is `D:/AI_Projects/电脑遇到过的问题/2026-07-16_几何模型GPU训练batch与显存边界记录.md`.

The automated mask-derived proxy is complete. Product acceptance still requires representative original-annotation or manual graph audits, not just pixel metrics:

- point missed / false point;
- correct intersection location;
- correct endpoint pairing;
- solid versus dashed line;
- circle/ellipse boundary and center relation;
- graph JSON matches the visible topology.

## Host performance note

On 2026-07-16, mouse lag was investigated. At the time of inspection there was no training process or GPU compute process. CPU was about 22-33%, committed memory about 36%, disk activity about 1-7%, and GPU utilization about 3%. The lag was strongest when switching Codex conversations, not during WSL training.

Local Codex data showed:

- all session JSONL files: about 325 MB;
- this long task's session file: about 7.6 MB;
- several individual old session files: about 10-29 MB;
- `logs_2.sqlite`: about 99 MB;
- Codex package cache: about 537 MB, much of it application binaries and web-engine data.

The leading diagnosis is long-conversation parsing/rendering plus accumulated Chromium/WebView cache, not a damaged CUDA/WSL environment. Any cleanup must preserve `.codex/sessions`, login data, project files, datasets, and checkpoints. Only disposable UI caches should be rebuilt after Codex is fully closed.

## Safety rules for the next AI

- Do not delete or move datasets, checkpoints, `.venv`, or session records.
- Do not reinstall PyTorch unless the live verification above fails.
- Do not touch WSL system Python.
- Keep training artifacts under `D:/AI_Projects/geometry_diagram_proof`.
- Report real metrics and failed/partial runs candidly.
- The user's priority is geometry topology accuracy, especially lines and points, rather than full-page box IoU.

## Completed question-aware GPT sharing and missing-line guard (2026-07-16)

The photographed Figure 2 exposed a share/render gap rather than a JSON syntax failure. The previous JSON already listed all seven points and ten visible connections, including `E-G`, but the GPT-generated picture wrote `EG` in its legend and still omitted the line on the canvas. It also drew two false rays from machine-detected arrow candidates even though the textbook diagram contains no visible arrowheads. Without the problem statement, GPT also had no reliable way to know that `EG` is the bisector of `angle BED` meeting line `CD` at `G`, or that `CD` is the fold line.

The web exporter is now `geometry-diagram/v3` and treats the input as a geometry-problem figure instead of an isolated drawing:

- a dedicated question-stem field is shared together with the graph JSON;
- GPT is instructed to extract explicit fold, intersection, angle-bisector, angle, and collinearity conditions from the stem before reasoning or redrawing;
- every currently retained segment becomes a `must_be_visible` item and is repeated in `drawingContract.mustRenderSegmentLabels`;
- the post-render contract requires GPT to find every listed segment in the final image, explicitly preventing a listed-but-unrendered `E-G`;
- point-on-segment relations are also exported as ordered collinear chains such as `C-D-G`, `B-D-A`, `B-F-C`, and `D-F-E`;
- detected arrows and circles are excluded by default and enter the share document only after an explicit user confirmation;
- normalized visual coordinates and a preserve-layout rule reduce arbitrary point reordering, while remaining explicitly approximate.

The built-in `Figure 2 question-linked case` contains the photographed problem stem, seven points, ten required visible segments (`E-G`, `B-C`, `B-D`, `D-A`, `C-D`, `D-G`, `C-A`, `E-D`, `B-E`, `E-C`), four point-on-segment relations, and zero arrows. Browser verification confirmed all of these states on `http://127.0.0.1:3000/`. The semantic regression test also confirmed that two synthetic false-arrow candidates are excluded, `E-G` remains in the required list, `C-D-G` is exported in order, and the earlier `P3@P1-P2` plus `P4@P3-P5` edit regression still passes. The production build and both rendered-page tests pass.

Remaining boundary: the site does not yet OCR a full Chinese problem statement automatically. The user should paste or correct the relevant sub-question in the question-stem field. The stronger JSON and rendering contract substantially reduce omission risk, but a generative GPT redraw still requires its own final visual self-check and cannot be made mathematically deterministic by prompt text alone.

## Completed zoomable correction canvas and clarified point-label boundary (2026-07-16)

Full-page problem photos can now remain intact while the diagram is enlarged for manual correction. The web canvas supports 100%-600% zoom, `-` / `+` controls, `Fit canvas`, mouse-wheel zoom around the pointer, and a two-touch pinch-and-pan path for phones and tablets. Touch handling delays any edit until a single-finger drag or tap is established, so a second finger switches to zoom instead of accidentally adding, deleting, or moving a point.

Superseded on 2026-07-16 after a live regression: direct original-photograph cropping introduced a second coordinate system and made the overlay depend on rotation and crop mapping. Recognition now uses a clean, oriented copy of the same intrinsic 760-pixel canvas used by the correction overlay. A selected small diagram is then enlarged to a 760-pixel detector edge, so the diagram is not kept tiny even when the full problem statement is present in the photograph.

`P1`, `P2`, and similar names are explicitly documented in the interface as temporary topology IDs. The current browser detector finds likely letter-box locations near graph points but does not classify the glyph inside each box, so it has not actually recognized `A`, `B`, `C`, or `D`. Users can enlarge the canvas and rename these temporary IDs. Automatic point-letter OCR/classification remains an unfinished recognition item and must not be represented as complete.

Verification completed on the production build:

- `npm run build`: passed;
- semantic regression: passed, including all ten Figure 2 required lines and both multi-position attachment cases;
- rendered HTML tests: 2/2 passed;
- live desktop browser: wheel zoom changed the displayed level from 100% to 114%;
- the real two-finger path is implemented and compiles, but still requires a phone or tablet touch-screen acceptance check because the desktop browser controller cannot generate two simultaneous physical touches.

The updated local site is running at `http://127.0.0.1:3000/` with the built-in Figure 2 case loaded.

## Completed image rotation and one-click recognition reset (2026-07-16)

The correction canvas now has left-90-degree and right-90-degree controls. Rotation is not a display-only CSS transform: the canvas dimensions, source photograph, detected points, arrow tips and direction vectors, circles, label rectangles, and current selection rectangle all rotate into the same coordinate system. Zoom returns to `Fit canvas` after a turn. Recognition renders a clean copy of this already-oriented intrinsic canvas and crops it in the same coordinate system; it no longer inverse-maps the selection through original-photograph coordinates.

The `Clear all recognition` action removes detected and manually added points, segments, arrows, circles, labels, attachments, and confirmation state in one click. It deliberately keeps the source image, question stem, orientation, and existing selection rectangle, allowing an immediate clean re-run. If no selection existed, the interface tells the user to draw one again.

Production and browser verification completed:

- production build passed;
- semantic regression passed, including all Figure 2 required lines and both multi-position attachment cases;
- rendered HTML tests passed 2/2;
- Figure 2 right rotation changed the canvas from `760x520` to `520x760`, reported `90 degrees`, and retained all 7 points plus all 10 connections;
- one-click clearing reduced both the point and connection lists to zero while preserving the question stem and 90-degree orientation;
- the arrow/attachment sample was then analyzed while rotated 90 degrees and still produced the expected 4 points, 5 connections, 1 arrow tip, and 1 point-on-segment relation.

The updated local page remains at `http://127.0.0.1:3000/`.

## Completed crowded-photo duplicate suppression and safe arrow overlay (2026-07-16)

The photographed Figure 2 regression produced a visibly unusable overlay after adding source-resolution crop analysis: 14 points, 26 connections, 13 arrow candidates, and 22 point-on-segment candidates. Candidate suppression reduced the clutter but did not solve the regression. A later live inspection proved that the detector could find seven points while mapping them into a compressed area, so the original-resolution crop/rotation coordinate path, not display zoom itself, was the primary regression source.

The client-side topology detector now applies four safeguards before presenting results:

- near-parallel Hough variants of one thick or blurred stroke do not create intersections;
- collinear Hough estimates with different endpoints are merged when their centerlines describe the same stroke;
- nearby point estimates use connected-neighbour clustering, preventing one blurry junction from becoming a chain such as `P5/P8/P13`;
- automatic arrow and point-on-segment candidates are confidence-sorted and bounded, while manually confirmed relations are retained.

Unconfirmed machine arrow candidates are no longer drawn as full orange rays across the photograph. Only a small translucent tip marker is shown until the user confirms the arrow; confirmed or manually added arrows still render the full directed ray.

Verification completed:

- semantic regression passed: 4 points, 5 segments, 1 arrow, and 1 point-on-segment relation;
- the previous `P4@P3-P5` and `P3@P1-P2` edit regressions still pass;
- production build passed;
- rendered HTML tests passed 2/2 before the final clustering-only patch, and the final production build passed again;
- live browser check produced the expected 4/5/1/1 sample result and showed no unconfirmed full-length orange overlay.

Acceptance boundary: the temporary source photograph was no longer present at its original WeChat cache path after the page reload, so the exact photographed crop still requires one user re-import and re-run. Automatic `A/B/C/D` glyph reading remains unfinished; `P1/P2/...` are still temporary topology IDs rather than recognized point letters.

## Corrected zoom/rotation recognition regression (2026-07-16)

The user correctly reported that recognition worked before zoom support and failed after it. Live inspection preserved the failing page before reload and recorded: `760x428` intrinsic canvas, `114%` display zoom, `270 degrees` rotation, and a 7-point/9-edge result whose overlay was compressed into a small part of Figure 2. This showed that further threshold tuning was the wrong response.

Root cause: the pre-zoom detector and overlay shared one 760-pixel canvas coordinate system. The zoom implementation later changed the detector input to an original-resolution photograph crop, added inverse rotation/source-crop transforms, and then mapped results back into the display canvas. Display size, oriented canvas size, and original-photo crop size could therefore diverge.

Correction:

- CSS zoom is display-only and never enters recognition coordinates;
- pointer positions are converted to intrinsic canvas coordinates by one tested helper;
- recognition renders a clean copy of the same oriented intrinsic canvas used by the overlay;
- the selected crop is normalized both upward and downward to a 760-pixel detector edge, so a small diagram inside a full photographed page is enlarged for recognition;
- detector results map directly from the normalized crop back to the selected intrinsic-canvas rectangle, with no original-photo inverse transform.

Verification passed:

- 100% and 200% display rectangles map the same logical point to identical intrinsic coordinates;
- the 190x100 selected crop normalizes to 760x400 for recognition;
- semantic regression remains 4 points, 5 segments, 1 arrow, and 1 point-on-segment relation;
- production build passed;
- live browser acceptance at `270 degrees` rotation and `120%` display zoom still produced the expected 4/5/1/1 result.

The exact user photograph must still be re-imported once after the service restart for final real-photo acceptance. Do not restore the superseded original-resolution inverse-mapping path unless it is implemented as a separately tested transform with real-photo fixtures.

## Completed CUDA U-Net web integration and Figure 2 acceptance (2026-07-16)

The previous browser detector was no longer adequate for the photographed Figure 2. On the fixed real-photo crop it produced 10 points, 13 segments, 4 false arrows, and 8 attachment candidates. Running the epoch-5 U-Net alone was also insufficient: its raw mask extractor produced 20 points, 41 pairwise connections, 1 false arrow, and no attachments. The model was active, but printed letters and angle marks were entering the point class and the old pairwise-support extractor then amplified them into a false graph.

The production page now uses the trained GPU U-Net through a local-only inference service instead of the client-side Hough detector:

- `ml/gpu_runtime.py` loads `geometry_unet_pgdp5k_gpu.pt` once on CUDA, requires GPU, uses mixed precision, and serializes batch-1 inference;
- `ml/serve.py` exposes only local `/health` and `/infer` endpoints, limits request size, and keeps the CUDA allocator fraction at `0.25`;
- `app/gpuInference.ts` converts the Python graph into the editable web graph;
- `ml/question_constraints.py` recognizes the explicit Figure 2 construction from several independent phrases in the problem stem and assigns only seven U-Net point candidates to the required topology;
- point names `A` through `G` are marked as `question_topology` assignments. This is not claimed to be glyph OCR, and the web interface says that the names still require human review;
- absent or incomplete question text does not receive the Figure 2 template silently; it returns raw GPU candidates for manual correction.

The original-resolution crop is now restored without reviving the old inverse-coordinate bug. The browser first renders the original photograph into one already-oriented source-resolution canvas, crops the selection in that same oriented coordinate frame, sends the crop to the GPU, and maps the response directly back into the intrinsic 760-pixel correction canvas. CSS zoom remains display-only. Therefore enlarging the image for manual work does not require restoring 100% size before recognition and does not reduce the pixels sent to the model.

Permanent regression assets:

- photographed fixture: `tests/fixtures/figure2_photo.jpg`;
- expected problem text and topology: `tests/fixtures/figure2_expected.json`;
- CUDA acceptance test: `tests/figure2_gpu_regression.py`;
- one-click real-photo browser case: `public/figure2-photo.jpg`.

Verified acceptance with the final checkpoint and with the actual browser-to-service path:

- device: `NVIDIA GeForce RTX 5060 Laptop GPU`;
- inference batch size: `1`;
- points: `7` with names `A,B,C,D,E,F,G`;
- required visible segments: `10`, including `E-G`;
- arrows: `0`;
- point-on-line locations: `4` (`D@A-B`, `D@C-G`, `F@B-C`, `F@E-D`);
- mean required-line mask support in the direct CUDA regression: `0.9832`;
- at 120% display zoom, a second web inference still returned `7/10/0/4` and retained `E-G`;
- production build passed and the final browser console had no warnings or errors.

Post-inference monitoring showed `769 MiB` GPU memory in use, `2%` GPU utilization, and `43 C` after the one-shot regression process exited. The resident service health endpoint reports batch `1`, checkpoint epoch `5`, CUDA device identity, and the `0.25` allocator fraction. This is intentionally separate from the completed training configuration, which remains batch `16` and must not be increased on this device.

The local page is running at `http://127.0.0.1:3000/` and the CUDA bridge at `http://127.0.0.1:8765/`. Use `scripts/start_gpu_inference.ps1` to restart the GPU bridge after a reboot.

## Cross-device browser inference supersedes the CUDA bridge (2026-07-16)

The production web interface no longer calls the PC CUDA bridge. The trained
epoch-5 U-Net was exported to the static ONNX model
`public/models/geometry_unet_pgdp5k_epoch5_320.onnx` (opset 18, fixed
`1x3x320x320` input, six output classes, 2,202,748 bytes). The page loads the
model with `onnxruntime-web` and performs preprocessing, U-Net execution,
mask argmax, question-constrained topology assignment, and coordinate mapping
inside the browser. The selected photograph is not encoded or posted to an
inference endpoint.

One URL now selects the runtime automatically:

- iPhone/iPad Safari uses the Apple A/M-series CPU through single-thread WASM;
- Windows Edge/Chrome prefers the Windows GPU through WebGPU and falls back to
  Windows CPU/WASM;
- compatible Android Chromium prefers phone WebGPU and falls back to phone
  CPU/WASM;
- the first run downloads the 2.2 MB model from this site and only the selected
  ONNX Runtime 1.27.0 binary from a version-pinned CDN; later loads can use the
  browser cache;
- `ml/serve.py` and `scripts/start_gpu_inference.ps1` remain only as optional
  desktop diagnostics and are not imported or contacted by the production UI.

Acceptance was performed with the PC GPU service stopped. Both browser paths
returned the Figure 2 target result:

- 7 named points (`A` through `G`);
- 10 required segments, including `E-G`;
- 0 arrows;
- 4 point-on-line relations;
- production build passed.

Final Windows browser timings were 1,190 ms through WebGPU and 985 ms through
the forced single-thread WASM path used as the Apple-browser equivalent. Both
had no console warnings or errors. Physical iPhone/iPad performance still
depends on the exact Apple device and Safari version and requires device-side
acceptance.

This proves the browser-side execution path is independent of the PC CUDA
service. Apple Neural Engine access is not claimed by the web build; a native
Core ML app would be required for that accelerator.

The runtime binaries are intentionally not packaged into the Sites deployment.
This keeps the app archive small and prevents iPhone/iPad clients from
downloading the Windows WebGPU binary (and vice versa). The CDN receives only a
request for the public runtime library; the problem image and model tensors
never leave the browser.
