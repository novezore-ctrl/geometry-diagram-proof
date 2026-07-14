import { detectDiagram } from "../app/vision";

const width = 600;
const height = 420;
const data = new Uint8ClampedArray(width * height * 4);
for (let i = 0; i < data.length; i += 4) data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 255;

function ink(x: number, y: number, radius = 2) {
  for (let oy = -radius; oy <= radius; oy += 1) for (let ox = -radius; ox <= radius; ox += 1) {
    const nx = Math.round(x + ox), ny = Math.round(y + oy);
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
    const i = (ny * width + nx) * 4;
    data[i] = data[i + 1] = data[i + 2] = 12;
  }
}

function line(x1: number, y1: number, x2: number, y2: number) {
  const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
  for (let i = 0; i <= steps; i += 1) ink(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps);
}

function circle(cx: number, cy: number, r: number) {
  for (let i = 0; i < 720; i += 1) { const a = i * Math.PI * 2 / 720; ink(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 1); }
}

function letterA(x: number, y: number) {
  line(x, y + 18, x + 7, y); line(x + 7, y, x + 14, y + 18); line(x + 3, y + 11, x + 11, y + 11);
}

line(90, 350, 300, 55);
line(300, 55, 520, 350);
line(90, 350, 520, 350);
line(300, 55, 300, 350);
circle(300, 245, 78);
letterA(62, 360); letterA(291, 18); letterA(535, 360);

const result = detectDiagram({ width, height, data, colorSpace: "srgb" } as ImageData);
console.log(JSON.stringify({
  points: result.points.length,
  segments: result.segments.length,
  circles: result.circles.length,
  circleDetails: result.circles.map((circle) => ({ cx: circle.cx, cy: circle.cy, r: circle.r, confidence: Number(circle.confidence.toFixed(2)) })),
  labels: result.labels.length,
  threshold: result.threshold,
}, null, 2));

if (result.points.length < 3) throw new Error(`点候选不足: ${result.points.length}`);
if (result.segments.length < 3) throw new Error(`连接候选不足: ${result.segments.length}`);
if (result.circles.length < 1) throw new Error("未检测到合成圆");
if (result.labels.length < 1) throw new Error("未检测到标签框");
