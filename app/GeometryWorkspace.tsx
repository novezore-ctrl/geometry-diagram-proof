"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { buildGeometryDocument, toGptShareText } from "./geometryExport";
import { initialMobileStatus, inspectClientStatus, requestMobileInference, type MobileInferenceStatus } from "./mobileInference";
import { deriveAttachments, viewportToCanvasPoint, type Detection, type PointNode, type SegmentEdge } from "./vision";

type Tool = "move" | "point" | "connect" | "erase";
const EMPTY: Detection = { points: [], segments: [], arrows: [], attachments: [], circles: [], labels: [], threshold: 0 };
const MIN_ZOOM = 1;
const MAX_ZOOM = 6;

function drawOrientedImage(ctx: CanvasRenderingContext2D, image: HTMLImageElement, width: number, height: number, rotation: number) {
  ctx.save();
  if (rotation === 90) {
    ctx.translate(width, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(image, 0, 0, height, width);
  } else if (rotation === 180) {
    ctx.translate(width, height); ctx.rotate(Math.PI); ctx.drawImage(image, 0, 0, width, height);
  } else if (rotation === 270) {
    ctx.translate(0, height); ctx.rotate(-Math.PI / 2); ctx.drawImage(image, 0, 0, height, width);
  } else ctx.drawImage(image, 0, 0, width, height);
  ctx.restore();
}

export function GeometryWorkspace() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragging = useRef<string | null>(null);
  const selecting = useRef(false);
  const selectionStart = useRef<{ x: number; y: number } | null>(null);
  const zoomRef = useRef(1);
  const touchPoints = useRef(new Map<number, { clientX: number; clientY: number }>());
  const pendingTouch = useRef<{ pointerId: number; clientX: number; clientY: number; x: number; y: number; target: HTMLCanvasElement } | null>(null);
  const activeTouch = useRef<number | null>(null);
  const touchStartSelection = useRef<{ x: number; y: number; w: number; h: number } | null>(null);
  const touchStartFirstPoint = useRef<string | null>(null);
  const pinchState = useRef<{ startDistance: number; startZoom: number; anchorX: number; anchorY: number } | null>(null);
  const [result, setResult] = useState<Detection>(EMPTY);
  const [tool, setTool] = useState<Tool>("move");
  const [firstPoint, setFirstPoint] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [fileName, setFileName] = useState("");
  const [hasImage, setHasImage] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [selection, setSelection] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [message, setMessage] = useState("导入清晰的印刷二维几何图，或先运行示例。");
  const [showBoxes, setShowBoxes] = useState(true);
  const [questionText, setQuestionText] = useState("");
  const [confirmedArrowIds, setConfirmedArrowIds] = useState<string[]>([]);
  const [confirmedCircleIds, setConfirmedCircleIds] = useState<string[]>([]);
  const [zoom, setZoom] = useState(1);
  const [fitSize, setFitSize] = useState({ width: 0, height: 0 });
  const [rotation, setRotation] = useState(0);
  const [mobileStatus, setMobileStatus] = useState<MobileInferenceStatus>(() => initialMobileStatus());

  const pointMap = new Map(result.points.map((point) => [point.id, point]));
  const pointName = (id: string) => pointMap.get(id)?.label || id;

  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current, image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawOrientedImage(ctx, image, canvas.width, canvas.height, rotation);
    const byId = new Map(result.points.map((point) => [point.id, point]));
    ctx.lineCap = "round";
    for (const segment of result.segments) {
      const a = byId.get(segment.a), b = byId.get(segment.b); if (!a || !b) continue;
      ctx.strokeStyle = segment.source === "manual" ? "#0b9870" : "#1769d2";
      ctx.lineWidth = segment.source === "manual" ? 4 : 3;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (const arrow of result.arrows) {
      const from = arrow.fromPoint ? byId.get(arrow.fromPoint) : null;
      const startX = from?.x ?? arrow.tailX, startY = from?.y ?? arrow.tailY;
      const confirmed = confirmedArrowIds.includes(arrow.id) || arrow.source === "manual";
      ctx.strokeStyle = confirmed ? "#0b9870" : "#d45a2a88"; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = confirmed ? 4 : 3;
      if (confirmed) { ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(arrow.x, arrow.y); ctx.stroke(); }
      const headLength = confirmed ? 17 : 11, headWidth = confirmed ? 8 : 5;
      const baseX = arrow.x - arrow.directionX * headLength, baseY = arrow.y - arrow.directionY * headLength;
      const sideX = -arrow.directionY * headWidth, sideY = arrow.directionX * headWidth;
      ctx.beginPath(); ctx.moveTo(arrow.x, arrow.y); ctx.lineTo(baseX + sideX, baseY + sideY); ctx.lineTo(baseX - sideX, baseY - sideY); ctx.closePath(); ctx.fill();
    }
    for (const attachment of result.attachments) {
      const junction = byId.get(attachment.junction); if (!junction) continue;
      ctx.strokeStyle = attachment.position.kind === "division" ? "#0b9870" : "#e99137"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(junction.x, junction.y, 12, 0, Math.PI * 2); ctx.stroke();
    }
    for (const circle of result.circles) {
      const confirmed = confirmedCircleIds.includes(circle.id) || circle.source === "manual";
      ctx.strokeStyle = confirmed ? "#0b9870" : "#9a55d888"; ctx.lineWidth = confirmed ? 4 : 3; ctx.setLineDash(confirmed ? [] : [8, 6]);
      ctx.beginPath(); ctx.arc(circle.cx, circle.cy, circle.r, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
    if (showBoxes) for (const label of result.labels) {
      ctx.strokeStyle = "#ed8a2d"; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.strokeRect(label.x - 2, label.y - 2, label.w + 4, label.h + 4); ctx.setLineDash([]);
    }
    for (const point of result.points) {
      ctx.fillStyle = point.id === firstPoint ? "#ed315a" : point.source === "manual" ? "#0b9870" : "#1769d2";
      ctx.strokeStyle = "white"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(point.x, point.y, point.id === firstPoint ? 9 : 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.font = "600 16px Arial"; ctx.lineWidth = 4; ctx.strokeStyle = "white"; ctx.strokeText(point.label, point.x + 10, point.y - 10);
      ctx.fillStyle = "#14283e"; ctx.fillText(point.label, point.x + 10, point.y - 10);
    }
    if (selection && !hasAnalyzed) {
      ctx.save(); ctx.fillStyle = "#1769d21c"; ctx.strokeStyle = "#1769d2"; ctx.lineWidth = 2; ctx.setLineDash([8, 5]);
      ctx.fillRect(selection.x, selection.y, selection.w, selection.h); ctx.strokeRect(selection.x, selection.y, selection.w, selection.h); ctx.restore();
    }
  }, [confirmedArrowIds, confirmedCircleIds, firstPoint, result, rotation, showBoxes, selection, hasAnalyzed]);

  useEffect(() => renderCanvas(), [renderCanvas]);
  useEffect(() => {
    setMobileStatus(inspectClientStatus());
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => undefined);
  }, []);

  const fitCanvas = useCallback(() => {
    const canvas = canvasRef.current, stage = stageRef.current;
    if (!canvas || !stage || !canvas.width || !canvas.height) return;
    const availableWidth = Math.max(260, stage.clientWidth - 32);
    const availableHeight = Math.max(320, window.innerHeight * .68);
    const scale = Math.min(1, availableWidth / canvas.width, availableHeight / canvas.height);
    setFitSize({ width: Math.max(1, Math.round(canvas.width * scale)), height: Math.max(1, Math.round(canvas.height * scale)) });
  }, []);

  useEffect(() => {
    window.addEventListener("resize", fitCanvas);
    return () => window.removeEventListener("resize", fitCanvas);
  }, [fitCanvas]);

  const setZoomAt = useCallback((requested: number, clientX?: number, clientY?: number) => {
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, requested));
    const previous = zoomRef.current;
    if (Math.abs(next - previous) < .001) return;
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const viewportX = (clientX ?? rect.left + rect.width / 2) - rect.left;
    const viewportY = (clientY ?? rect.top + rect.height / 2) - rect.top;
    const contentX = (stage.scrollLeft + viewportX) / previous;
    const contentY = (stage.scrollTop + viewportY) / previous;
    zoomRef.current = next;
    setZoom(next);
    requestAnimationFrame(() => {
      stage.scrollLeft = contentX * next - viewportX;
      stage.scrollTop = contentY * next - viewportY;
    });
  }, []);

  const wheelZoom = (event: React.WheelEvent<HTMLDivElement>) => {
    if (!hasImage) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.14 : 1 / 1.14;
    setZoomAt(zoomRef.current * factor, event.clientX, event.clientY);
  };

  const resetZoom = () => {
    zoomRef.current = 1; setZoom(1);
    requestAnimationFrame(() => { if (stageRef.current) { stageRef.current.scrollLeft = 0; stageRef.current.scrollTop = 0; } });
  };

  const rotateImage = (direction: -1 | 1) => {
    const canvas = canvasRef.current; if (!canvas || !imageRef.current || busy) return;
    const oldWidth = canvas.width, oldHeight = canvas.height;
    const rotatePoint = (x: number, y: number) => direction === 1 ? { x: oldHeight - y, y: x } : { x: y, y: oldWidth - x };
    const rotateRect = (rect: { x: number; y: number; w: number; h: number }) => direction === 1
      ? { x: oldHeight - rect.y - rect.h, y: rect.x, w: rect.h, h: rect.w }
      : { x: rect.y, y: oldWidth - rect.x - rect.w, w: rect.h, h: rect.w };
    canvas.width = oldHeight; canvas.height = oldWidth;
    setResult((current) => ({
      ...current,
      points: current.points.map((point) => ({ ...point, ...rotatePoint(point.x, point.y) })),
      arrows: current.arrows.map((arrow) => {
        const tip = rotatePoint(arrow.x, arrow.y), tail = rotatePoint(arrow.tailX, arrow.tailY);
        return { ...arrow, ...tip, tailX: tail.x, tailY: tail.y, directionX: direction === 1 ? -arrow.directionY : arrow.directionY, directionY: direction === 1 ? arrow.directionX : -arrow.directionX };
      }),
      circles: current.circles.map((circle) => { const center = rotatePoint(circle.cx, circle.cy); return { ...circle, cx: center.x, cy: center.y }; }),
      labels: current.labels.map((label) => ({ ...label, ...rotateRect(label) })),
    }));
    setSelection((current) => current ? rotateRect(current) : null);
    setRotation((current) => (current + (direction === 1 ? 90 : 270)) % 360);
    setFirstPoint(null); dragging.current = null; selecting.current = false; selectionStart.current = null;
    resetZoom();
    requestAnimationFrame(fitCanvas);
    setMessage(`图片已${direction === 1 ? "向右" : "向左"}旋转 90°，识别结果和选区已同步旋转。`);
  };

  const clearAllDetections = () => {
    setResult(EMPTY); setHasAnalyzed(false); setConfirmedArrowIds([]); setConfirmedCircleIds([]);
    setFirstPoint(null); setTool("move"); dragging.current = null; selecting.current = false; selectionStart.current = null;
    setMessage(selection ? "已清除全部识别和人工校正结果，原图与选区已保留，可立即重新识别。" : "已清除全部识别和人工校正结果，原图已保留；请重新框选后识别。");
  };

  const loadImage = useCallback((src: string, name: string, autoSelect = false, prepared?: Detection) => {
    const image = new Image();
    image.onload = () => {
      const canvas = canvasRef.current; if (!canvas) return;
      const scale = Math.min(1, 760 / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(320, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(240, Math.round(image.naturalHeight * scale));
      imageRef.current = image; setHasImage(true); setResult(prepared || EMPTY); setFileName(name); setHasAnalyzed(Boolean(prepared));
      setConfirmedArrowIds([]); setConfirmedCircleIds([]);
      setRotation(0);
      zoomRef.current = 1; setZoom(1);
      setSelection(autoSelect ? { x: 0, y: 0, w: canvas.width, h: canvas.height } : null);
      const autoSelectMessage = name.includes("图2")
        ? "真实图2照片已载入，题干与全图选区已就绪；点击“开始识别”将在这台手机上运行 U-Net。"
        : "箭头与线内连接案例已载入，选区已就绪，点击“开始识别”。";
      setMessage(prepared ? "题干联动案例已载入：图中没有箭头，E—G 被列为必须绘制的角平分线连接。" : autoSelect ? autoSelectMessage : "图片已载入，请框选区域后点击“开始识别”。");
      requestAnimationFrame(() => { fitCanvas(); renderCanvas(); if (stageRef.current) { stageRef.current.scrollLeft = 0; stageRef.current.scrollTop = 0; } });
    };
    image.src = src;
  }, [fitCanvas, renderCanvas]);

  const importFile = (file?: File) => {
    if (!file) return;
    setHasAnalyzed(false); setSelection(null); setResult(EMPTY);
    const reader = new FileReader(); reader.onload = () => loadImage(String(reader.result), file.name); reader.readAsDataURL(file);
  };

  const sample = () => {
    setHasAnalyzed(false); setSelection(null); setResult(EMPTY);
    const c = document.createElement("canvas"); c.width = 900; c.height = 620;
    const ctx = c.getContext("2d"); if (!ctx) return;
    ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height); ctx.strokeStyle = "#111"; ctx.fillStyle = "#111"; ctx.lineWidth = 4; ctx.font = "30px Georgia";
    const a = { x: 140, y: 510 }, b = { x: 760, y: 510 }, top = { x: 450, y: 225 }, h = { x: 450, y: 510 }, tip = { x: 450, y: 38 };
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(top.x, top.y); ctx.lineTo(b.x, b.y); ctx.lineTo(a.x, a.y); ctx.moveTo(h.x, h.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
    const baseY = tip.y + 24;
    ctx.beginPath(); ctx.moveTo(tip.x, tip.y); ctx.lineTo(tip.x - 13, baseY); ctx.lineTo(tip.x + 13, baseY); ctx.closePath(); ctx.fill();
    for (const [label, x, y] of [["A",102,548],["B",775,548],["C",470,218],["H",463,548]] as const) ctx.fillText(label, x, y);
    ctx.font = "20px 'Microsoft YaHei'"; ctx.fillStyle = "#657a73"; ctx.fillText("案例：射线方向 + C—H 在线段 A—B 上的连接位置", 190, 595);
    loadImage(c.toDataURL("image/png"), "箭头与线内连接案例", true);
  };

  const questionSample = () => {
    const question = "在△ABC中，∠ACB=90°，点D是边AB上一点，将△ACD沿直线CD翻折得到△ECD，DE与射线CB交于点F。如图2，连接BE，∠BED的平分线交直线CD于点G，且∠ADC=2∠G。若∠A=∠AGD，请说明∠CBE=∠G。";
    setQuestionText(question);
    loadImage("/figure2-photo.jpg", "真实照片 · 图2 手机端回归案例", true);
  };

  const analyze = async () => {
    const image = imageRef.current, visible = canvasRef.current; if (!image || !visible) return;
    if (!selection || selection.w < 12 || selection.h < 12) { setMessage("请先在图片上拖出需要解析的区域。"); return; }
    setBusy(true); setMessage("正在下载或读取缓存中的手机模型，并在本机芯片上分离几何线与噪声……");
    try {
      // CSS zoom is display-only.  Recognition crops the corresponding region
      // from the oriented source-resolution photo, so zooming for manual work
      // never reduces or changes the pixels sent to the phone-side model.
      const quarterTurn = rotation === 90 || rotation === 270;
      const naturalWidth = quarterTurn ? image.naturalHeight : image.naturalWidth;
      const naturalHeight = quarterTurn ? image.naturalWidth : image.naturalHeight;
      const maxSourceEdge = 3000;
      const sourceScale = Math.min(1, maxSourceEdge / Math.max(naturalWidth, naturalHeight));
      const clean = document.createElement("canvas");
      clean.width = Math.max(1, Math.round(naturalWidth * sourceScale));
      clean.height = Math.max(1, Math.round(naturalHeight * sourceScale));
      const cleanCtx = clean.getContext("2d");
      if (!cleanCtx) throw new Error("无法读取旋转后的原始图片");
      cleanCtx.fillStyle = "white"; cleanCtx.fillRect(0, 0, clean.width, clean.height);
      drawOrientedImage(cleanCtx, image, clean.width, clean.height, rotation);
      const scaleX = clean.width / visible.width, scaleY = clean.height / visible.height;
      const crop = document.createElement("canvas");
      crop.width = Math.max(12, Math.round(selection.w * scaleX));
      crop.height = Math.max(12, Math.round(selection.h * scaleY));
      const cropCtx = crop.getContext("2d");
      if (!cropCtx) throw new Error("无法读取所选区域");
      cropCtx.fillStyle = "white"; cropCtx.fillRect(0, 0, crop.width, crop.height);
      cropCtx.imageSmoothingEnabled = true; cropCtx.imageSmoothingQuality = "high";
      cropCtx.drawImage(
        clean,
        selection.x * scaleX, selection.y * scaleY, selection.w * scaleX, selection.h * scaleY,
        0, 0, crop.width, crop.height,
      );
      const response = await requestMobileInference(crop, questionText);
      const next = response.detection;
      setMobileStatus(response.status);
      const mapX = selection.w / crop.width, mapY = selection.h / crop.height, mapR = (mapX + mapY) / 2;
      const mapped = {
        ...next,
        points: next.points.map((p) => ({ ...p, x: selection.x + p.x * mapX, y: selection.y + p.y * mapY })),
        arrows: next.arrows.map((arrow) => ({ ...arrow, x: selection.x + arrow.x * mapX, y: selection.y + arrow.y * mapY, tailX: selection.x + arrow.tailX * mapX, tailY: selection.y + arrow.tailY * mapY })),
        circles: next.circles.map((circle) => ({ ...circle, cx: selection.x + circle.cx * mapX, cy: selection.y + circle.cy * mapY, r: circle.r * mapR })),
        labels: next.labels.map((label) => ({ ...label, x: selection.x + label.x * mapX, y: selection.y + label.y * mapY, w: label.w * mapX, h: label.h * mapY })),
      };
      setResult(mapped); setHasAnalyzed(true); setBusy(false);
      const constrained = response.metadata.question_constraints_applied === true;
      const backendName = `${response.status.deviceLabel} ${response.status.backend === "webgpu" ? "WebGPU" : "CPU/WASM"}`;
      setMessage(`${backendName} 本地识别完成（${response.metadata.inference_ms} ms）：${next.points.length}个点、${next.segments.length}条连接、${next.arrows.length}个箭头、${next.attachments.length}个线内位置。${constrained ? "已用题干约束筛除字母和角标误检；点名来自题干拓扑匹配，请人工复核。" : "题干不足以建立拓扑模板，当前是手机端模型候选，请人工复核。"}`);
    } catch (error) {
      setBusy(false);
      const detail = error instanceof Error ? error.message : String(error);
      setMessage(`手机本地识别失败：${detail}。首次使用请保持网络畅通，让浏览器下载模型和运行库。`);
    }
  };

  const coordinatesAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current!, rect = canvas.getBoundingClientRect();
    return viewportToCanvasPoint(clientX, clientY, rect, canvas.width, canvas.height);
  };
  const nearest = (x: number, y: number) => result.points.map((p) => ({ p, d: Math.hypot(p.x - x, p.y - y) })).filter((v) => v.d < 25).sort((a, b) => a.d - b.d)[0]?.p;
  const updateTopology = (change: (current: Detection) => Detection) => setResult((current) => {
    const next = change(current);
    return { ...next, attachments: deriveAttachments(next.points, next.segments, current.attachments) };
  });

  const beginInteraction = (x: number, y: number, pointerId: number, target: HTMLCanvasElement) => {
    if (!imageRef.current) return;
    if (!hasAnalyzed) {
      selecting.current = true; selectionStart.current = { x, y }; setSelection({ x, y, w: 0, h: 0 });
      if (!target.hasPointerCapture(pointerId)) target.setPointerCapture(pointerId);
      return;
    }
    const hit = nearest(x, y);
    if (tool === "point") {
      const point: PointNode = { id: `P${Date.now()}`, label: `P${result.points.length + 1}`, x, y, confidence: 1, source: "manual" };
      updateTopology((r) => ({ ...r, points: [...r.points, point] })); setMessage("已补点，可在右侧修改名称。");
    } else if (tool === "move" && hit) {
      dragging.current = hit.id;
      if (!target.hasPointerCapture(pointerId)) target.setPointerCapture(pointerId);
    } else if (tool === "erase" && hit) {
      updateTopology((r) => ({ ...r, points: r.points.filter((p) => p.id !== hit.id), segments: r.segments.filter((s) => s.a !== hit.id && s.b !== hit.id), arrows: r.arrows.filter((arrow) => arrow.fromPoint !== hit.id), labels: r.labels.filter((l) => l.nearPoint !== hit.id) }));
      setMessage("已删除点及其连接。");
    } else if (tool === "connect" && hit) {
      if (!firstPoint) { setFirstPoint(hit.id); setMessage(`已选择${hit.label}，请再点另一个点。`); }
      else if (firstPoint !== hit.id) {
        const existing = result.segments.find((s) => (s.a === firstPoint && s.b === hit.id) || (s.b === firstPoint && s.a === hit.id));
        if (existing) { updateTopology((r) => ({ ...r, segments: r.segments.filter((s) => s.id !== existing.id) })); setMessage("已断开这条连接。"); }
        else { const edge: SegmentEdge = { id: `S${Date.now()}`, a: firstPoint, b: hit.id, confidence: 1, source: "manual" }; updateTopology((r) => ({ ...r, segments: [...r.segments, edge] })); setMessage("已补充人工连接。"); }
        setFirstPoint(null);
      }
    }
  };

  const moveInteraction = (clientX: number, clientY: number) => {
    if (selecting.current && selectionStart.current) {
      const { x, y } = coordinatesAt(clientX, clientY), start = selectionStart.current;
      setSelection({ x: Math.min(start.x, x), y: Math.min(start.y, y), w: Math.abs(x - start.x), h: Math.abs(y - start.y) }); return;
    }
    if (!dragging.current) return;
    const { x, y } = coordinatesAt(clientX, clientY), id = dragging.current, canvas = canvasRef.current!;
    updateTopology((r) => ({ ...r, points: r.points.map((p) => p.id === id ? { ...p, x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)), source: "manual" } : p) }));
  };
  const endInteraction = (pointerId: number, target: HTMLCanvasElement) => {
    if (selecting.current) {
      selecting.current = false; selectionStart.current = null;
      if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
      setMessage("选区已确定，可以开始识别。"); return;
    }
    if (!dragging.current) return; dragging.current = null;
    if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
    setMessage("点的位置已校正。");
  };

  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!imageRef.current) return;
    if (event.pointerType !== "touch") {
      const { x, y } = coordinatesAt(event.clientX, event.clientY);
      beginInteraction(x, y, event.pointerId, event.currentTarget);
      return;
    }
    event.preventDefault();
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.setPointerCapture(event.pointerId);
    touchPoints.current.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
    if (touchPoints.current.size === 1) {
      const { x, y } = coordinatesAt(event.clientX, event.clientY);
      pendingTouch.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, x, y, target: event.currentTarget };
      touchStartSelection.current = selection;
      touchStartFirstPoint.current = firstPoint;
      return;
    }
    if (touchPoints.current.size === 2) {
      selecting.current = false; selectionStart.current = null; dragging.current = null; activeTouch.current = null; pendingTouch.current = null;
      setSelection(touchStartSelection.current); setFirstPoint(touchStartFirstPoint.current);
      const [a, b] = Array.from(touchPoints.current.values());
      const midX = (a.clientX + b.clientX) / 2, midY = (a.clientY + b.clientY) / 2;
      const stage = stageRef.current; if (!stage) return;
      const rect = stage.getBoundingClientRect(), viewportX = midX - rect.left, viewportY = midY - rect.top;
      pinchState.current = {
        startDistance: Math.max(1, Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)),
        startZoom: zoomRef.current,
        anchorX: (stage.scrollLeft + viewportX) / zoomRef.current,
        anchorY: (stage.scrollTop + viewportY) / zoomRef.current,
      };
    }
  };

  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerType !== "touch") { moveInteraction(event.clientX, event.clientY); return; }
    event.preventDefault();
    if (!touchPoints.current.has(event.pointerId)) return;
    touchPoints.current.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
    if (pinchState.current && touchPoints.current.size >= 2) {
      const [a, b] = Array.from(touchPoints.current.values());
      const distance = Math.max(1, Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY));
      const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, pinchState.current.startZoom * distance / pinchState.current.startDistance));
      const midX = (a.clientX + b.clientX) / 2, midY = (a.clientY + b.clientY) / 2;
      const stage = stageRef.current; if (!stage) return;
      const rect = stage.getBoundingClientRect(), viewportX = midX - rect.left, viewportY = midY - rect.top;
      zoomRef.current = next; setZoom(next);
      requestAnimationFrame(() => {
        if (!pinchState.current) return;
        stage.scrollLeft = pinchState.current.anchorX * next - viewportX;
        stage.scrollTop = pinchState.current.anchorY * next - viewportY;
      });
      return;
    }
    const pending = pendingTouch.current;
    if (pending?.pointerId === event.pointerId && Math.hypot(event.clientX - pending.clientX, event.clientY - pending.clientY) >= 7) {
      activeTouch.current = event.pointerId; pendingTouch.current = null;
      beginInteraction(pending.x, pending.y, event.pointerId, pending.target);
    }
    if (activeTouch.current === event.pointerId) moveInteraction(event.clientX, event.clientY);
  };

  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerType !== "touch") { endInteraction(event.pointerId, event.currentTarget); return; }
    event.preventDefault();
    const wasPinching = Boolean(pinchState.current);
    touchPoints.current.delete(event.pointerId);
    if (wasPinching) {
      pinchState.current = null; pendingTouch.current = null; activeTouch.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      setMessage(`已缩放至 ${Math.round(zoomRef.current * 100)}%，双指可继续缩放和移动。`);
      return;
    }
    const pending = pendingTouch.current;
    if (pending?.pointerId === event.pointerId) {
      pendingTouch.current = null;
      if (hasAnalyzed) {
        const { x, y } = coordinatesAt(event.clientX, event.clientY);
        beginInteraction(x, y, event.pointerId, event.currentTarget);
        endInteraction(event.pointerId, event.currentTarget);
      } else setMessage("请在图上拖动框选需要识别的区域；双指可先放大。 ");
    } else if (activeTouch.current === event.pointerId) {
      activeTouch.current = null; endInteraction(event.pointerId, event.currentTarget);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const pointerCancel = (event: React.PointerEvent<HTMLCanvasElement>) => {
    touchPoints.current.delete(event.pointerId); pendingTouch.current = null; activeTouch.current = null; pinchState.current = null;
    selecting.current = false; selectionStart.current = null; dragging.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const rename = (id: string, label: string) => setResult((r) => ({ ...r, points: r.points.map((p) => p.id === id ? { ...p, label: label.slice(0, 4), source: "manual" } : p) }));
  const updateAttachmentKind = (id: string, value: string) => setResult((current) => ({ ...current, attachments: current.attachments.map((item) => {
    if (item.id !== id) return item;
    if (value === "approximate") {
      const center = pointMap.get(item.junction), a = pointMap.get(item.hostA), b = pointMap.get(item.hostB);
      const da = center && a ? Math.hypot(center.x - a.x, center.y - a.y) : 1;
      const db = center && b ? Math.hypot(center.x - b.x, center.y - b.y) : 1;
      return { ...item, position: { kind: "approximate", approximateT: da / Math.max(1, da + db), coordinateUncertain: true }, source: "manual" };
    }
    const parts = Number(value);
    return { ...item, position: { kind: "division", parts, index: Math.min(parts - 1, item.position.kind === "division" ? item.position.index : 1), coordinateUncertain: false }, source: "manual" };
  }) }));
  const updateAttachmentIndex = (id: string, index: number) => setResult((current) => ({ ...current, attachments: current.attachments.map((item) => item.id === id && item.position.kind === "division" ? { ...item, position: { ...item.position, index }, source: "manual" } : item) }));
  const toggleConfirmed = (kind: "arrow" | "circle", id: string) => {
    const update = (items: string[]) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id];
    if (kind === "arrow") setConfirmedArrowIds(update);
    else setConfirmedCircleIds(update);
  };

  const shareToGpt = async () => {
    const shareContext = { questionTitle: fileName, questionText, confirmedArrowIds, confirmedCircleIds };
    const text = toGptShareText(result, shareContext), json = JSON.stringify(buildGeometryDocument(result, shareContext), null, 2);
    try {
      if (navigator.share) {
        const file = new File([json], "geometry-problem-figure-v3.json", { type: "application/json" });
        if (!navigator.canShare || navigator.canShare({ files: [file] })) await navigator.share({ title: "题干与校正后的几何图语义", text, files: [file] });
        else await navigator.share({ title: "题干与校正后的几何图语义", text });
        setMessage("已打开系统分享面板，可发送给 GPT。"); return;
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
    }
    await navigator.clipboard.writeText(text);
    setMessage("题干、必画线清单和结构化 JSON 已一起复制，可直接粘贴到 GPT。");
  };

  const unconfirmedArrows = result.arrows.filter((arrow) => arrow.source !== "manual" && !confirmedArrowIds.includes(arrow.id)).length;
  const unconfirmedCircles = result.circles.filter((circle) => circle.source !== "manual" && !confirmedCircleIds.includes(circle.id)).length;
  const segmentChecklist = result.segments.map((segment) => `${pointName(segment.a)}—${pointName(segment.b)}`);
  const temporaryPointNames = result.points.filter((point) => /^P\d+$/i.test(point.label.trim())).length;
  const hasRecognition = hasAnalyzed || result.points.length > 0 || result.segments.length > 0 || result.arrows.length > 0 || result.circles.length > 0 || result.labels.length > 0;

  const tools: Array<[Tool, string, string]> = [["move","↖","移动点"],["point","●","补点"],["connect","╱","连接/断开"],["erase","×","删除点"]];

  return <main className="app-shell">
    <header className="topbar">
      <div><span className="eyebrow">本地处理 · 不上传题图</span><h1>几何图校对器</h1></div>
      <span className={`version gpu-state ${mobileStatus.ready ? "ready" : "offline"}`}><i />{mobileStatus.ready ? `${mobileStatus.deviceLabel} · ${mobileStatus.backend === "webgpu" ? "WebGPU" : "CPU/WASM"} · 题图不上传` : `${mobileStatus.deviceLabel} · 首次识别加载模型`}</span>
    </header>

    <section className="intro-card">
      <div><p className="step-label">题干 + 图形 + 人工确认</p><h2>先明确这是哪道题的图，再让 GPT 按清单一条不漏地理解</h2><p>题干给语义，图形给连接；未确认箭头和圆不会进入分享内容。</p></div>
      <div className="scope-pills"><span>题干联合理解</span><span>全部线段必画</span><span>误检候选隔离</span></div>
    </section>

    <section className="workspace-grid">
      <div className="canvas-panel">
        <div className="panel-head"><div><span className="panel-index">01</span><strong>题图与结构叠加</strong><small>{fileName || "尚未载入图片"}</small></div>
          <div className="import-actions"><input ref={inputRef} hidden type="file" accept="image/*" capture="environment" onChange={(e) => importFile(e.target.files?.[0])}/><button className="button secondary" onClick={() => inputRef.current?.click()}>导入/拍照</button><button className="button ghost" onClick={questionSample}>打开图2案例</button><button className="button ghost" onClick={sample}>箭头案例</button></div>
        </div>
        <div className="zoom-toolbar" data-testid="zoom-toolbar">
          <div className="toolbar-actions"><button aria-label="缩小" disabled={!hasImage || zoom <= MIN_ZOOM} onClick={() => setZoomAt(zoomRef.current / 1.2)}>−</button><output data-testid="zoom-level">{Math.round(zoom * 100)}%</output><button aria-label="放大" disabled={!hasImage || zoom >= MAX_ZOOM} onClick={() => setZoomAt(zoomRef.current * 1.2)}>＋</button><button className="fit-button" disabled={!hasImage} onClick={resetZoom}>适应画布</button><span className="toolbar-divider"/><button className="rotate-button" aria-label="向左旋转90度" disabled={!hasImage || busy} onClick={() => rotateImage(-1)}>↶ 左转</button><output className="rotation-level" data-testid="rotation-level">方向 {rotation}°</output><button className="rotate-button" aria-label="向右旋转90度" disabled={!hasImage || busy} onClick={() => rotateImage(1)}>右转 ↷</button><button className="clear-button" disabled={!hasImage || busy || !hasRecognition} onClick={clearAllDetections}>清除全部识别</button></div>
          <small>电脑滚轮缩放；手机/平板双指缩放并移动</small>
        </div>
        <div ref={stageRef} className={`canvas-stage ${hasImage ? "has-image" : "empty"}`} onWheel={wheelZoom}>
          {!hasImage && <button className="empty-prompt" onClick={() => inputRef.current?.click()}><b>＋</b><strong>选择一道几何题图片</strong><small>正面拍摄、背景干净、线条清楚</small></button>}
          <div className="canvas-zoom-surface" style={hasImage ? { width: `${fitSize.width * zoom}px`, height: `${fitSize.height * zoom}px` } : undefined}>
            <canvas ref={canvasRef} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerCancel} aria-label="几何图校正画布" />
          </div>
          {busy && <div className="busy-overlay"><i />正在用手机芯片本地识别</div>}
        </div>
        <div className="status-row"><span /><p>{message}</p><button className="button primary" disabled={!hasImage || busy} onClick={analyze}>{busy ? "识别中…" : "开始识别"}</button></div>
      </div>

      <aside className="control-panel">
        <div className="panel-head compact"><div><span className="panel-index">02</span><strong>人工校正</strong></div></div>
        <section className="question-context">
          <div><strong>题干（与图一起分享）</strong><span className={questionText.trim() ? "context-state ok" : "context-state warn"}>{questionText.trim() ? "已填写" : "待填写"}</span></div>
          <textarea data-testid="question-context" value={questionText} onChange={(event) => setQuestionText(event.target.value)} placeholder="粘贴这幅图所属小题的题干、已知条件和问题。题干越完整，GPT 越能判断某条线的作用。" />
          <small>分享时会先要求 GPT 从题干提取翻折、交点、角平分线等语义，再用图形结构核对。例如本题的 E—G 不是普通装饰线，而是 ∠BED 的平分线与直线 CD 相交所形成的关键连接。</small>
        </section>
        <div className="tool-grid">{tools.map(([value, icon, label]) => <button key={value} className={tool === value ? "tool active" : "tool"} onClick={() => { setTool(value); setFirstPoint(null); }}><b>{icon}</b>{label}</button>)}</div>
        <label className="box-toggle"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)}/>显示橙色标签候选框</label>
        <div className="candidate-scroll">
          <Candidate title={`点 ${result.points.length}`} tone="blue">{result.points.length ? result.points.map((p) => <div className="candidate-row" key={p.id}><i className={p.source}/><input aria-label={`${p.label} 点名`} value={p.label} onChange={(e) => rename(p.id, e.target.value)}/><small>{/^P\d+$/i.test(p.label.trim()) ? "临时编号 · 未识字" : p.source === "manual" ? "已校正" : p.source === "question_constrained" ? "题干拓扑命名 · 待复核" : `${Math.round(p.confidence * 100)}%`}</small></div>) : <Empty />}<p className={`ocr-note ${temporaryPointNames ? "active" : ""}`}><strong>为什么可能显示 P1/P2？</strong>题干足够完整时会按拓扑匹配 A、B、C……，但这不是把字形 OCR 当作绝对事实；题干不足时 P1/P2 仍只是临时编号，请人工确认。</p></Candidate>
          <Candidate title={`连接 ${result.segments.length}`} tone="green">{result.segments.length ? result.segments.map((s) => <div className="candidate-row relation" key={s.id}><i className={s.source}/><strong>{pointName(s.a)} — {pointName(s.b)}</strong><button onClick={() => updateTopology((r) => ({ ...r, segments: r.segments.filter((v) => v.id !== s.id) }))}>删除</button></div>) : <Empty />}</Candidate>
          <Candidate title={`箭头尖端 ${result.arrows.length}`} tone="arrow">{result.arrows.length ? result.arrows.map((arrow) => { const confirmed = arrow.source === "manual" || confirmedArrowIds.includes(arrow.id); return <div className={`candidate-row relation review ${confirmed ? "confirmed" : "pending"}`} key={arrow.id}><i className="arrow"/><strong>射线 {arrow.fromPoint ? pointName(arrow.fromPoint) : "起点未匹配"} → 箭头方向<small>{confirmed ? "会分享" : "机器候选，默认不分享"}</small></strong><div className="candidate-actions"><button className="confirm-action" onClick={() => toggleConfirmed("arrow", arrow.id)}>{confirmed ? "取消确认" : "确认"}</button><button onClick={() => { setConfirmedArrowIds((ids) => ids.filter((id) => id !== arrow.id)); setResult((r) => ({ ...r, arrows: r.arrows.filter((item) => item.id !== arrow.id) })); }}>删除</button></div></div>; }) : <Empty />}</Candidate>
          <Candidate title={`线内位置 ${result.attachments.length}`} tone="junction">{result.attachments.length ? result.attachments.map((item) => <div className="attachment-card" key={item.id}>
            <strong>{item.kind === "point_on_segment" ? `${pointName(item.junction)} 在线段 ${pointName(item.hostA)}—${pointName(item.hostB)} 上` : `${pointName(item.branch || "")}—${pointName(item.junction)} 接到 ${pointName(item.hostA)}—${pointName(item.hostB)}`}</strong>
            <label>连接位置<select value={item.position.kind === "approximate" ? "approximate" : String(item.position.parts)} onChange={(event) => updateAttachmentKind(item.id, event.target.value)}><option value="approximate">大致位置（坐标不确定）</option>{Array.from({ length: 9 }, (_, index) => index + 2).map((parts) => <option key={parts} value={parts}>{parts === 2 ? "中点（2等分）" : `${parts}等分点`}</option>)}</select></label>
            {item.position.kind === "division" ? <label>从 {pointName(item.hostA)} 起<select value={item.position.index} onChange={(event) => updateAttachmentIndex(item.id, Number(event.target.value))}>{Array.from({ length: item.position.parts - 1 }, (_, index) => index + 1).map((index) => <option key={index} value={index}>第 {index} 个分点</option>)}</select></label> : <small>{item.support === "remembered" ? "周围线条已变化，关系为你保留，请重新确认。" : "只告诉 GPT“这个点在线上”，不把图上比例当作精确条件。"}</small>}
          </div>) : <Empty />}</Candidate>
          <Candidate title={`圆候选 ${result.circles.length}`} tone="purple">{result.circles.length ? result.circles.map((c, i) => { const confirmed = c.source === "manual" || confirmedCircleIds.includes(c.id); return <div className={`candidate-row relation review ${confirmed ? "confirmed" : "pending"}`} key={c.id}><i className="circle"/><strong>圆{i + 1} · r≈{c.r.toFixed(0)}<small>{confirmed ? "会分享" : "机器候选，默认不分享"}</small></strong><div className="candidate-actions"><button className="confirm-action" onClick={() => toggleConfirmed("circle", c.id)}>{confirmed ? "取消确认" : "确认"}</button><button onClick={() => { setConfirmedCircleIds((ids) => ids.filter((id) => id !== c.id)); setResult((r) => ({ ...r, circles: r.circles.filter((v) => v.id !== c.id) })); }}>删除</button></div></div>; }) : <Empty />}</Candidate>
          <Candidate title={`标签框 ${result.labels.length}`} tone="orange">{result.labels.length ? result.labels.slice(0, 12).map((l) => <div className="candidate-row relation" key={l.id}><i className="label"/><strong>{l.id} → {l.nearPoint ? pointName(l.nearPoint) : "未匹配"}</strong><small>{l.w}×{l.h}</small></div>) : <Empty />}</Candidate>
        </div>
        <div className="share-audit" data-testid="share-audit">
          <strong>分享前检查</strong>
          <span className={questionText.trim() ? "ok" : "warn"}>{questionText.trim() ? "题干会与图形一起发送" : "尚未填写题干，GPT 无法结合题意校验"}</span>
          <span className="ok">{segmentChecklist.length} 条线全部列入“必须画出”：{segmentChecklist.join("、") || "无"}</span>
          {(unconfirmedArrows > 0 || unconfirmedCircles > 0) && <span className="safe">已隔离 {unconfirmedArrows} 个未确认箭头、{unconfirmedCircles} 个未确认圆</span>}
        </div>
        <button className="button export" disabled={!result.points.length} onClick={shareToGpt}>复制题干 + 图形结构给 GPT</button>
      </aside>
    </section>
    <footer><strong>当前痛点：</strong>只传图会让 GPT 猜线的作用，只传 JSON 又可能“清单写了但图没画”。现在分享内容同时包含题干、必画线清单、共线顺序和误检候选隔离规则。</footer>
  </main>;
}

function Candidate({ title, tone, children }: { title: string; tone: string; children: React.ReactNode }) { return <section className="candidate-section"><h3><i className={tone}/>{title}</h3>{children}</section>; }
function Empty() { return <p className="empty-line">暂无候选</p>; }
