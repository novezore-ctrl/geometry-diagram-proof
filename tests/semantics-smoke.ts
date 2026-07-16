import { buildGeometryDocument, toGptShareText } from "../app/geometryExport";
import { deriveAttachments, detectorRasterSize, detectDiagram, viewportToCanvasPoint, type PointNode, type SegmentEdge } from "../app/vision";

const normalizedLargeCrop = detectorRasterSize(2200, 1100);
if (normalizedLargeCrop.width !== 760 || normalizedLargeCrop.height !== 380) throw new Error(`大图选区没有归一化到稳定识别尺寸: ${JSON.stringify(normalizedLargeCrop)}`);
const unchangedReferenceCrop = detectorRasterSize(520, 760);
if (unchangedReferenceCrop.width !== 520 || unchangedReferenceCrop.height !== 760) throw new Error("参考尺寸选区不应再次缩放");
const enlargedSmallCrop = detectorRasterSize(190, 100);
if (enlargedSmallCrop.width !== 760 || enlargedSmallCrop.height !== 400 || enlargedSmallCrop.scale !== 4) throw new Error(`小图选区没有被放大到稳定识别尺寸: ${JSON.stringify(enlargedSmallCrop)}`);

const logicalAt100 = viewportToCanvasPoint(310, 140, { left: 10, top: 20, width: 760, height: 380 }, 760, 380);
const logicalAt200 = viewportToCanvasPoint(610, 260, { left: 10, top: 20, width: 1520, height: 760 }, 760, 380);
if (Math.abs(logicalAt100.x - logicalAt200.x) > .001 || Math.abs(logicalAt100.y - logicalAt200.y) > .001) throw new Error(`显示缩放改变了画布坐标: ${JSON.stringify({ logicalAt100, logicalAt200 })}`);

const width = 760, height = 520;
const data = new Uint8ClampedArray(width * height * 4);
for (let i = 0; i < data.length; i += 4) data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 255;

function ink(x: number, y: number, radius = 2) {
  for (let oy = -radius; oy <= radius; oy += 1) for (let ox = -radius; ox <= radius; ox += 1) {
    const nx = Math.round(x + ox), ny = Math.round(y + oy);
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
    const index = (ny * width + nx) * 4;
    data[index] = data[index + 1] = data[index + 2] = 10;
  }
}

function line(x1: number, y1: number, x2: number, y2: number) {
  const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
  for (let i = 0; i <= steps; i += 1) ink(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps);
}

// Triangle A-C-B with C-H meeting the interior of host A-B, and a ray
// continuing from H through C to the arrow tip.
line(100, 420, 380, 180);
line(380, 180, 680, 420);
line(100, 420, 680, 420);
line(380, 420, 380, 28);
for (let y = 28; y <= 55; y += 1) {
  const half = Math.round((y - 28) * .48);
  for (let x = 380 - half; x <= 380 + half; x += 1) ink(x, y, 0);
}

const result = detectDiagram({ width, height, data, colorSpace: "srgb" } as ImageData);
if (result.arrows.length < 1) throw new Error(`箭头尖端候选不足: ${result.arrows.length}`);
if (result.arrows.length > 4) throw new Error(`箭头候选没有限制在可人工复核的数量: ${result.arrows.length}`);
if (result.attachments.length < 1) throw new Error(`线内连接候选不足: ${result.attachments.length}`);
if (result.attachments[0].position.kind !== "approximate" || !result.attachments[0].position.coordinateUncertain) throw new Error("未默认标注坐标不确定");

result.points.forEach((point, index) => { point.label = ["A", "B", "C", "H"][index] || point.label; });
result.attachments[0].position = { kind: "division", parts: 3, index: 1, coordinateUncertain: false };
const exportContext = { questionText: "在三角形中作一条带箭头的射线。", confirmedArrowIds: result.arrows.map((arrow) => arrow.id) };
const document = buildGeometryDocument(result, exportContext);
const shareText = toGptShareText(result, exportContext);
if (document.primitives.rays.length < 1) throw new Error("GPT JSON 缺少射线");
if (document.attachments[0].position.kind !== "user_confirmed_division") throw new Error("GPT JSON 未保留用户确认的等分条件");
if (!shareText.includes("用户已确认的射线方向")) throw new Error("分享说明没有区分已确认箭头");

// Regression for the user's live edit: P5-P3 is one long line through P4,
// while the left half P2-P3 is no longer an explicit graph edge. Both P3 on
// P1-P2 and P4 on P3-P5 must still be offered independently.
const editedPoints: PointNode[] = [
  { id: "P1", label: "P1", x: 680, y: 420, confidence: 1, source: "manual" },
  { id: "P2", label: "P2", x: 100, y: 420, confidence: 1, source: "manual" },
  { id: "P3", label: "P3", x: 380, y: 420, confidence: 1, source: "manual" },
  { id: "P4", label: "P4", x: 380, y: 180, confidence: 1, source: "manual" },
  { id: "P5", label: "P5", x: 380, y: 28, confidence: 1, source: "manual" },
];
const originalSegments: SegmentEdge[] = [
  ["P1", "P3"], ["P3", "P2"], ["P4", "P3"], ["P4", "P2"], ["P1", "P4"],
].map(([a, b], index) => ({ id: `S${index + 1}`, a, b, confidence: 1, source: "manual" }));
const editedSegments: SegmentEdge[] = [
  ["P1", "P3"], ["P4", "P3"], ["P4", "P2"], ["P1", "P4"], ["P5", "P3"],
].map(([a, b], index) => ({ id: `E${index + 1}`, a, b, confidence: 1, source: "manual" }));
const beforeEdit = deriveAttachments(editedPoints.slice(0, 4), originalSegments);
const confirmedBase = beforeEdit.find((item) => item.junction === "P3" && new Set([item.hostA, item.hostB]).has("P1") && new Set([item.hostA, item.hostB]).has("P2"));
if (!confirmedBase) throw new Error("编辑前没有识别 P3 在 P1-P2 上");
confirmedBase.position = { kind: "division", parts: 2, index: 1, coordinateUncertain: false };
confirmedBase.source = "manual";
const afterEdit = deriveAttachments(editedPoints, editedSegments, beforeEdit);
const p3OnBase = afterEdit.find((item) => item.junction === "P3" && [item.hostA, item.hostB].includes("P1") && [item.hostA, item.hostB].includes("P2"));
const p4OnVertical = afterEdit.find((item) => item.junction === "P4" && [item.hostA, item.hostB].includes("P3") && [item.hostA, item.hostB].includes("P5"));
if (!p3OnBase || p3OnBase.position.kind !== "division") throw new Error("新增点和线后丢失了原 P3/P1-P2 选项或用户选择");
if (!p4OnVertical) throw new Error("没有识别 P4 在新增线 P3-P5 上");

// One accidental half-edge is insufficient evidence for a point-on-segment
// relation. Otherwise a few false points create a combinatorial relation list.
const incompleteHost = deriveAttachments(editedPoints.slice(0, 3), [originalSegments[0]]);
if (incompleteHost.length) throw new Error(`单侧残缺线错误生成线内关系: ${incompleteHost.map((item) => item.id).join(",")}`);

// Regression for the photographed problem in this turn. The visible segment
// inventory contains E-G, while two machine arrow candidates are unconfirmed.
// Export must force E-G into the post-render checklist and exclude both arrows.
const problemPoints: PointNode[] = [
  ["E",404,488],["G",640,90],["B",640,370],["C",160,370],["A",160,90],["D",400,230],["F",402,370],
].map(([id, x, y]) => ({ id: String(id), label: String(id), x: Number(x), y: Number(y), confidence: 1, source: "manual" }));
const problemSegments: SegmentEdge[] = [
  ["E","G"],["B","C"],["B","D"],["D","A"],["C","D"],["D","G"],["C","A"],["E","D"],["B","E"],["E","C"],
].map(([a, b], index) => ({ id: `Q${index + 1}`, a, b, confidence: 1, source: "manual" }));
const problemAttachments = deriveAttachments(problemPoints, problemSegments);
const problemResult = {
  points: problemPoints,
  segments: problemSegments,
  arrows: [
    { id: "false-A", x: 700, y: 370, tailX: 640, tailY: 370, directionX: 1, directionY: 0, fromPoint: "B", confidence: .7, source: "detected" as const },
    { id: "false-E", x: 700, y: 40, tailX: 640, tailY: 90, directionX: .7, directionY: -.7, fromPoint: "E", confidence: .6, source: "detected" as const },
  ],
  attachments: problemAttachments,
  circles: [], labels: [], threshold: 0,
};
const problemQuestion = "连接BE，∠BED的平分线交直线CD于点G。";
const problemDocument = buildGeometryDocument(problemResult, { questionTitle: "图2", questionText: problemQuestion });
const problemShare = toGptShareText(problemResult, { questionTitle: "图2", questionText: problemQuestion });
if (!problemDocument.drawingContract.mustRenderSegmentLabels.includes("E—G")) throw new Error("必画线清单漏掉 E-G");
if (problemDocument.primitives.rays.length !== 0 || problemDocument.excludedUnconfirmedCandidates.arrows.length !== 2) throw new Error("未确认箭头没有被隔离");
if (!problemShare.includes("逐条必画清单：E—G") || !problemShare.includes(problemQuestion)) throw new Error("分享内容没有同时携带题干和 E-G 必画约束");
if (!problemDocument.collinearChains.some((chain) => chain.pointsInOrder.join("-") === "C-D-G")) throw new Error("没有输出 C-D-G 共线顺序");

console.log(JSON.stringify({
  points: result.points.length,
  segments: result.segments.length,
  arrows: result.arrows.length,
  attachments: result.attachments.length,
  exportedSchema: document.schema,
  problemMustRender: problemDocument.drawingContract.mustRenderSegmentLabels,
  excludedProblemArrows: problemDocument.excludedUnconfirmedCandidates.arrows,
  editedAttachmentLocations: afterEdit.map((item) => `${item.junction}@${item.hostA}-${item.hostB}`),
}, null, 2));
