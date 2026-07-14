"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { detectDiagram, type Detection, type PointNode, type SegmentEdge } from "./vision";

type Tool = "move" | "point" | "connect" | "erase";
const EMPTY: Detection = { points: [], segments: [], circles: [], labels: [], threshold: 0 };

export function GeometryWorkspace() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragging = useRef<string | null>(null);
  const [result, setResult] = useState<Detection>(EMPTY);
  const [tool, setTool] = useState<Tool>("move");
  const [firstPoint, setFirstPoint] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [fileName, setFileName] = useState("");
  const [hasImage, setHasImage] = useState(false);
  const [message, setMessage] = useState("导入清晰的印刷二维几何图，或先运行示例。");
  const [showBoxes, setShowBoxes] = useState(true);

  const pointMap = new Map(result.points.map((point) => [point.id, point]));
  const pointName = (id: string) => pointMap.get(id)?.label || id;

  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current, image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const byId = new Map(result.points.map((point) => [point.id, point]));
    ctx.lineCap = "round";
    for (const segment of result.segments) {
      const a = byId.get(segment.a), b = byId.get(segment.b); if (!a || !b) continue;
      ctx.strokeStyle = segment.source === "manual" ? "#0b9870" : "#1769d2";
      ctx.lineWidth = segment.source === "manual" ? 4 : 3;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (const circle of result.circles) {
      ctx.strokeStyle = "#9a55d8"; ctx.lineWidth = 3; ctx.setLineDash([8, 6]);
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
  }, [firstPoint, result, showBoxes]);

  useEffect(() => renderCanvas(), [renderCanvas]);
  useEffect(() => { if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => undefined); }, []);

  const loadImage = useCallback((src: string, name: string) => {
    const image = new Image();
    image.onload = () => {
      const canvas = canvasRef.current; if (!canvas) return;
      const scale = Math.min(1, 760 / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(320, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(240, Math.round(image.naturalHeight * scale));
      imageRef.current = image; setHasImage(true); setResult(EMPTY); setFileName(name); setMessage("图片已载入，点击“开始识别”。");
      requestAnimationFrame(renderCanvas);
    };
    image.src = src;
  }, [renderCanvas]);

  const importFile = (file?: File) => {
    if (!file) return;
    const reader = new FileReader(); reader.onload = () => loadImage(String(reader.result), file.name); reader.readAsDataURL(file);
  };

  const sample = () => {
    const c = document.createElement("canvas"); c.width = 900; c.height = 620;
    const ctx = c.getContext("2d"); if (!ctx) return;
    ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height); ctx.strokeStyle = "#111"; ctx.fillStyle = "#111"; ctx.lineWidth = 4; ctx.font = "32px Georgia";
    ctx.beginPath(); ctx.moveTo(130, 510); ctx.lineTo(440, 90); ctx.lineTo(770, 510); ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(440, 90); ctx.lineTo(440, 510); ctx.stroke();
    ctx.beginPath(); ctx.arc(440, 350, 128, 0, Math.PI * 2); ctx.stroke();
    for (const [label, x, y] of [["A",98,552],["B",428,66],["C",786,552],["D",450,552],["O",452,344]] as const) ctx.fillText(label, x, y);
    loadImage(c.toDataURL("image/png"), "内置三角形与圆示例");
  };

  const analyze = () => {
    const image = imageRef.current, visible = canvasRef.current; if (!image || !visible) return;
    setBusy(true); setMessage("正在本机寻找线段、交点、圆和标签位置……");
    setTimeout(() => {
      const c = document.createElement("canvas"); c.width = visible.width; c.height = visible.height;
      const ctx = c.getContext("2d", { willReadFrequently: true }); if (!ctx) return;
      ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height); ctx.drawImage(image, 0, 0, c.width, c.height);
      const next = detectDiagram(ctx.getImageData(0, 0, c.width, c.height)); setResult(next); setBusy(false);
      setMessage(`检测完成：${next.points.length}个点、${next.segments.length}条连接、${next.circles.length}个圆候选、${next.labels.length}个标签框。蓝色结果需要你确认。`);
    }, 50);
  };

  const coordinates = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!, rect = canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width * canvas.width, y: (event.clientY - rect.top) / rect.height * canvas.height };
  };
  const nearest = (x: number, y: number) => result.points.map((p) => ({ p, d: Math.hypot(p.x - x, p.y - y) })).filter((v) => v.d < 25).sort((a, b) => a.d - b.d)[0]?.p;

  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!imageRef.current) return;
    const { x, y } = coordinates(event), hit = nearest(x, y);
    if (tool === "point") {
      const point: PointNode = { id: `P${Date.now()}`, label: `P${result.points.length + 1}`, x, y, confidence: 1, source: "manual" };
      setResult((r) => ({ ...r, points: [...r.points, point] })); setMessage("已补点，可在右侧修改名称。");
    } else if (tool === "move" && hit) {
      dragging.current = hit.id; event.currentTarget.setPointerCapture(event.pointerId);
    } else if (tool === "erase" && hit) {
      setResult((r) => ({ ...r, points: r.points.filter((p) => p.id !== hit.id), segments: r.segments.filter((s) => s.a !== hit.id && s.b !== hit.id), labels: r.labels.filter((l) => l.nearPoint !== hit.id) }));
      setMessage("已删除点及其连接。");
    } else if (tool === "connect" && hit) {
      if (!firstPoint) { setFirstPoint(hit.id); setMessage(`已选择${hit.label}，请再点另一个点。`); }
      else if (firstPoint !== hit.id) {
        const existing = result.segments.find((s) => (s.a === firstPoint && s.b === hit.id) || (s.b === firstPoint && s.a === hit.id));
        if (existing) { setResult((r) => ({ ...r, segments: r.segments.filter((s) => s.id !== existing.id) })); setMessage("已断开这条连接。"); }
        else { const edge: SegmentEdge = { id: `S${Date.now()}`, a: firstPoint, b: hit.id, confidence: 1, source: "manual" }; setResult((r) => ({ ...r, segments: [...r.segments, edge] })); setMessage("已补充人工连接。"); }
        setFirstPoint(null);
      }
    }
  };

  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragging.current) return;
    const { x, y } = coordinates(event), id = dragging.current, canvas = canvasRef.current!;
    setResult((r) => ({ ...r, points: r.points.map((p) => p.id === id ? { ...p, x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)), source: "manual" } : p) }));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragging.current) return; dragging.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setMessage("点的位置已校正。");
  };

  const rename = (id: string, label: string) => setResult((r) => ({ ...r, points: r.points.map((p) => p.id === id ? { ...p, label: label.slice(0, 4), source: "manual" } : p) }));
  const structureText = () => [
    "经用户校正的二维几何图结构：",
    `点：${result.points.map((p) => p.label).join("、") || "无"}`,
    `连接：${result.segments.map((s) => `${pointName(s.a)}-${pointName(s.b)}`).join("、") || "无"}`,
    `圆候选：${result.circles.length}个。`,
    "坐标只表示相对位置；不要从视觉比例推断垂直、平行或等长。",
  ].join("\n");

  const tools: Array<[Tool, string, string]> = [["move","↖","移动点"],["point","●","补点"],["connect","╱","连接/断开"],["erase","×","删除点"]];

  return <main className="app-shell">
    <header className="topbar">
      <div><span className="eyebrow">本地处理 · 不上传题图</span><h1>几何图校对器</h1></div>
      <span className="version"><i />清晰印刷二维图原型</span>
    </header>

    <section className="intro-card">
      <div><p className="step-label">第一版只解决三件事</p><h2>找候选、看连接、让你快速纠错</h2><p>蓝色是自动检测结果，绿色是你人工确认或修改的结果。</p></div>
      <div className="scope-pills"><span>印刷二维几何图</span><span>点 / 线 / 圆 / 标签框</span><span>人工校正连接</span></div>
    </section>

    <section className="workspace-grid">
      <div className="canvas-panel">
        <div className="panel-head"><div><span className="panel-index">01</span><strong>题图与结构叠加</strong><small>{fileName || "尚未载入图片"}</small></div>
          <div className="import-actions"><input ref={inputRef} hidden type="file" accept="image/*" capture="environment" onChange={(e) => importFile(e.target.files?.[0])}/><button className="button secondary" onClick={() => inputRef.current?.click()}>导入/拍照</button><button className="button ghost" onClick={sample}>运行示例</button></div>
        </div>
        <div className={`canvas-stage ${hasImage ? "has-image" : "empty"}`}>
          {!hasImage && <button className="empty-prompt" onClick={() => inputRef.current?.click()}><b>＋</b><strong>选择一道几何题图片</strong><small>正面拍摄、背景干净、线条清楚</small></button>}
          <canvas ref={canvasRef} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp} aria-label="几何图校正画布" />
          {busy && <div className="busy-overlay"><i />正在本地识别</div>}
        </div>
        <div className="status-row"><span /><p>{message}</p><button className="button primary" disabled={!hasImage || busy} onClick={analyze}>{busy ? "识别中…" : "开始识别"}</button></div>
      </div>

      <aside className="control-panel">
        <div className="panel-head compact"><div><span className="panel-index">02</span><strong>人工校正</strong></div></div>
        <div className="tool-grid">{tools.map(([value, icon, label]) => <button key={value} className={tool === value ? "tool active" : "tool"} onClick={() => { setTool(value); setFirstPoint(null); }}><b>{icon}</b>{label}</button>)}</div>
        <label className="box-toggle"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)}/>显示橙色标签候选框</label>
        <div className="candidate-scroll">
          <Candidate title={`点 ${result.points.length}`} tone="blue">{result.points.length ? result.points.map((p) => <div className="candidate-row" key={p.id}><i className={p.source}/><input value={p.label} onChange={(e) => rename(p.id, e.target.value)}/><small>{p.source === "manual" ? "已校正" : `${Math.round(p.confidence * 100)}%`}</small></div>) : <Empty />}</Candidate>
          <Candidate title={`连接 ${result.segments.length}`} tone="green">{result.segments.length ? result.segments.map((s) => <div className="candidate-row relation" key={s.id}><i className={s.source}/><strong>{pointName(s.a)} — {pointName(s.b)}</strong><button onClick={() => setResult((r) => ({ ...r, segments: r.segments.filter((v) => v.id !== s.id) }))}>删除</button></div>) : <Empty />}</Candidate>
          <Candidate title={`圆候选 ${result.circles.length}`} tone="purple">{result.circles.length ? result.circles.map((c, i) => <div className="candidate-row relation" key={c.id}><i className="circle"/><strong>圆{i + 1} · r≈{c.r.toFixed(0)}</strong><button onClick={() => setResult((r) => ({ ...r, circles: r.circles.filter((v) => v.id !== c.id) }))}>删除</button></div>) : <Empty />}</Candidate>
          <Candidate title={`标签框 ${result.labels.length}`} tone="orange">{result.labels.length ? result.labels.slice(0, 12).map((l) => <div className="candidate-row relation" key={l.id}><i className="label"/><strong>{l.id} → {l.nearPoint ? pointName(l.nearPoint) : "未匹配"}</strong><small>{l.w}×{l.h}</small></div>) : <Empty />}</Candidate>
        </div>
        <button className="button export" disabled={!result.points.length} onClick={async () => { await navigator.clipboard.writeText(structureText()); setMessage("校正后的结构说明已复制。"); }}>复制校正后的结构说明</button>
      </aside>
    </section>
    <footer><strong>当前边界：</strong>只识别视觉结构，不把“看起来垂直、平行、等长”当成题目条件。</footer>
  </main>;
}

function Candidate({ title, tone, children }: { title: string; tone: string; children: React.ReactNode }) { return <section className="candidate-section"><h3><i className={tone}/>{title}</h3>{children}</section>; }
function Empty() { return <p className="empty-line">暂无候选</p>; }
