export type PointNode = { id: string; label: string; x: number; y: number; confidence: number; source: "detected" | "manual" };
export type SegmentEdge = { id: string; a: string; b: string; confidence: number; source: "detected" | "manual" };
export type CircleNode = { id: string; cx: number; cy: number; r: number; confidence: number; source: "detected" | "manual" };
export type LabelBox = { id: string; x: number; y: number; w: number; h: number; nearPoint?: string };
export type Detection = { points: PointNode[]; segments: SegmentEdge[]; circles: CircleNode[]; labels: LabelBox[]; threshold: number };
type Line = { theta: number; x1: number; y1: number; x2: number; y2: number; confidence: number };

const dist = (ax: number, ay: number, bx: number, by: number) => Math.hypot(ax - bx, ay - by);

function binarize(image: ImageData) {
  const gray = new Uint8Array(image.width * image.height);
  const histogram = new Uint32Array(256);
  for (let i = 0, p = 0; i < image.data.length; i += 4, p += 1) {
    gray[p] = Math.round(image.data[i] * .299 + image.data[i + 1] * .587 + image.data[i + 2] * .114);
    histogram[gray[p]] += 1;
  }
  let totalSum = 0;
  for (let i = 0; i < 256; i += 1) totalSum += i * histogram[i];
  let count = 0, partial = 0, best = 145, bestVariance = -1;
  for (let i = 0; i < 256; i += 1) {
    count += histogram[i]; partial += i * histogram[i];
    const rest = gray.length - count;
    if (!count || !rest) continue;
    const variance = count * rest * (partial / count - (totalSum - partial) / rest) ** 2;
    if (variance > bestVariance) { bestVariance = variance; best = i; }
  }
  const threshold = Math.max(70, Math.min(205, best));
  return { threshold, binary: gray.map((value) => value < threshold ? 1 : 0) };
}

function detectLines(binary: Uint8Array, width: number, height: number) {
  const ink: Array<[number, number]> = [];
  let inkCount = 0;
  for (const value of binary) inkCount += value;
  const stride = Math.max(1, Math.ceil(Math.sqrt(inkCount / 8500)));
  let seen = 0;
  for (let y = 2; y < height - 2; y += 1) for (let x = 2; x < width - 2; x += 1) {
    if (binary[y * width + x] && seen++ % stride === 0) ink.push([x, y]);
  }
  const angleStep = 3, angles = 60, rhoStep = 2, diagonal = Math.ceil(Math.hypot(width, height));
  const rhoBins = Math.ceil(diagonal * 2 / rhoStep) + 1;
  const acc = new Uint16Array(angles * rhoBins), cos = new Float32Array(angles), sin = new Float32Array(angles);
  for (let t = 0; t < angles; t += 1) { cos[t] = Math.cos(t * angleStep * Math.PI / 180); sin[t] = Math.sin(t * angleStep * Math.PI / 180); }
  for (const [x, y] of ink) for (let t = 0; t < angles; t += 1) {
    const r = Math.round((x * cos[t] + y * sin[t] + diagonal) / rhoStep);
    acc[t * rhoBins + r] += 1;
  }
  let maxVote = 0;
  for (const vote of acc) maxVote = Math.max(maxVote, vote);
  const peaks: Array<{ t: number; r: number; vote: number }> = [];
  const cutoff = Math.max(15, maxVote * .27);
  for (let t = 0; t < angles; t += 1) for (let r = 0; r < rhoBins; r += 1) {
    const vote = acc[t * rhoBins + r]; if (vote >= cutoff) peaks.push({ t, r, vote });
  }
  peaks.sort((a, b) => b.vote - a.vote);
  const chosen: typeof peaks = [];
  for (const peak of peaks) {
    if (!chosen.some((p) => Math.abs(p.t - peak.t) <= 2 && Math.abs(p.r - peak.r) <= 5)) chosen.push(peak);
    if (chosen.length === 24) break;
  }
  const lines: Line[] = [];
  for (const peak of chosen) {
    const c = cos[peak.t], s = sin[peak.t], rho = peak.r * rhoStep - diagonal;
    const projection = ink.filter(([x, y]) => Math.abs(x * c + y * s - rho) < 3.3).map(([x, y]) => -x * s + y * c).sort((a, b) => a - b);
    let runStart = 0, start = 0, end = 0;
    for (let i = 1; i <= projection.length; i += 1) if (i === projection.length || projection[i] - projection[i - 1] > 14) {
      if (projection[i - 1] - projection[runStart] > end - start) { start = projection[runStart]; end = projection[i - 1]; }
      runStart = i;
    }
    if (end - start < Math.min(width, height) * .15) continue;
    const baseX = rho * c, baseY = rho * s;
    const line: Line = { theta: peak.t * angleStep, x1: baseX - start * s, y1: baseY + start * c, x2: baseX - end * s, y2: baseY + end * c, confidence: Math.min(.99, peak.vote / maxVote) };
    const duplicate = lines.some((other) => {
      const angle = Math.min(Math.abs(other.theta - line.theta), 180 - Math.abs(other.theta - line.theta));
      const direct = dist(other.x1, other.y1, line.x1, line.y1) + dist(other.x2, other.y2, line.x2, line.y2);
      const reverse = dist(other.x1, other.y1, line.x2, line.y2) + dist(other.x2, other.y2, line.x1, line.y1);
      return angle < 5 && Math.min(direct, reverse) < 36;
    });
    if (!duplicate) lines.push(line);
    if (lines.length === 16) break;
  }
  return lines;
}

function intersection(a: Line, b: Line) {
  const d = (a.x1 - a.x2) * (b.y1 - b.y2) - (a.y1 - a.y2) * (b.x1 - b.x2);
  if (Math.abs(d) < .001) return null;
  const ca = a.x1 * a.y2 - a.y1 * a.x2, cb = b.x1 * b.y2 - b.y1 * b.x2;
  const x = (ca * (b.x1 - b.x2) - (a.x1 - a.x2) * cb) / d;
  const y = (ca * (b.y1 - b.y2) - (a.y1 - a.y2) * cb) / d;
  const inside = (l: Line) => x >= Math.min(l.x1, l.x2) - 10 && x <= Math.max(l.x1, l.x2) + 10 && y >= Math.min(l.y1, l.y2) - 10 && y <= Math.max(l.y1, l.y2) + 10;
  return inside(a) && inside(b) ? { x, y } : null;
}

function topology(lines: Line[], width: number, height: number) {
  const raw: Array<{ x: number; y: number; confidence: number }> = [];
  for (const line of lines) raw.push({ x: line.x1, y: line.y1, confidence: line.confidence }, { x: line.x2, y: line.y2, confidence: line.confidence });
  for (let i = 0; i < lines.length; i += 1) for (let j = i + 1; j < lines.length; j += 1) {
    const hit = intersection(lines[i], lines[j]);
    if (hit && hit.x >= 0 && hit.x <= width && hit.y >= 0 && hit.y <= height) raw.push({ ...hit, confidence: Math.min(lines[i].confidence, lines[j].confidence) });
  }
  const groups: Array<{ x: number; y: number; confidence: number; n: number }> = [];
  for (const p of raw) {
    const group = groups.find((g) => dist(g.x, g.y, p.x, p.y) < 12);
    if (group) { group.x = (group.x * group.n + p.x) / (group.n + 1); group.y = (group.y * group.n + p.y) / (group.n + 1); group.confidence = Math.max(group.confidence, p.confidence); group.n += 1; }
    else groups.push({ ...p, n: 1 });
  }
  const points: PointNode[] = groups.filter((g) => g.n > 1 || g.confidence > .45).map((g, i) => ({ id: `P${i + 1}`, label: `P${i + 1}`, x: g.x, y: g.y, confidence: Math.min(.99, g.confidence + Math.min(.15, g.n * .03)), source: "detected" }));
  const segments: SegmentEdge[] = [];
  for (const line of lines) {
    const dx = line.x2 - line.x1, dy = line.y2 - line.y1, len2 = dx * dx + dy * dy || 1;
    const ordered = points.map((p) => ({ p, t: ((p.x - line.x1) * dx + (p.y - line.y1) * dy) / len2, d: Math.abs(dy * p.x - dx * p.y + line.x2 * line.y1 - line.y2 * line.x1) / Math.sqrt(len2) })).filter((v) => v.t > -.05 && v.t < 1.05 && v.d < 10).sort((a, b) => a.t - b.t);
    for (let i = 0; i < ordered.length - 1; i += 1) {
      const a = ordered[i].p.id, b = ordered[i + 1].p.id;
      if (!segments.some((s) => (s.a === a && s.b === b) || (s.a === b && s.b === a))) segments.push({ id: `S${segments.length + 1}`, a, b, confidence: line.confidence, source: "detected" });
    }
  }
  return { points, segments };
}

function labelBoxes(binary: Uint8Array, width: number, height: number, points: PointNode[]) {
  const visited = new Uint8Array(binary.length), boxes: LabelBox[] = [], steps = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
  for (let y = 1; y < height - 1; y += 1) for (let x = 1; x < width - 1; x += 1) {
    const origin = y * width + x; if (!binary[origin] || visited[origin]) continue;
    const queue = [origin]; visited[origin] = 1;
    let at = 0, area = 0, minX = x, maxX = x, minY = y, maxY = y;
    while (at < queue.length && area < 2200) {
      const index = queue[at++], cy = Math.floor(index / width), cx = index - cy * width; area += 1;
      minX = Math.min(minX, cx); maxX = Math.max(maxX, cx); minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
      for (const [ox, oy] of steps) { const nx = cx + ox, ny = cy + oy, next = ny * width + nx; if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 && binary[next] && !visited[next]) { visited[next] = 1; queue.push(next); } }
    }
    const w = maxX - minX + 1, h = maxY - minY + 1;
    if (area < 5 || area > 650 || w < 2 || h < 5 || w > 52 || h > 52 || w / h > 2.4 || h / w > 4.5) continue;
    const cx = minX + w / 2, cy = minY + h / 2;
    const near = points.map((p) => ({ p, d: dist(cx, cy, p.x, p.y) })).sort((a, b) => a.d - b.d)[0];
    if (near && near.d > 7 && near.d < Math.max(58, Math.min(width, height) * .12)) boxes.push({ id: `L${boxes.length + 1}`, x: minX, y: minY, w, h, nearPoint: near.p.id });
  }
  return boxes.filter((box, i) => !boxes.slice(0, i).some((b) => b.nearPoint === box.nearPoint && dist(b.x + b.w / 2, b.y + b.h / 2, box.x + box.w / 2, box.y + box.h / 2) < 12)).slice(0, Math.max(16, points.length * 2));
}

function circles(binary: Uint8Array, width: number, height: number) {
  const found: Array<{ cx: number; cy: number; r: number; score: number }> = [], min = Math.max(18, Math.round(Math.min(width, height) * .055));
  const max = Math.round(Math.min(width, height) * .42), cStep = Math.max(12, Math.round(Math.min(width, height) / 36)), rStep = Math.max(8, Math.round(Math.min(width, height) / 52));
  const offsets = [[0,0],[-3,0],[3,0],[0,-3],[0,3],[-2,-2],[2,-2],[-2,2],[2,2]];
  const hit = (x: number, y: number) => { const ix = Math.round(x), iy = Math.round(y); for (const [ox, oy] of offsets) { const nx = ix + ox, ny = iy + oy; if (nx >= 0 && nx < width && ny >= 0 && ny < height && binary[ny * width + nx]) return 1; } return 0; };
  for (let cy = min; cy < height - min; cy += cStep) for (let cx = min; cx < width - min; cx += cStep) for (let r = min; r <= max; r += rStep) {
    if (cx - r < 1 || cy - r < 1 || cx + r >= width - 1 || cy + r >= height - 1) continue;
    const samples = 32, required = 19;
    let hits = 0, rejected = false;
    for (let i = 0; i < samples; i += 1) {
      const a = i * Math.PI * 2 / samples; hits += hit(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      if (hits + samples - i - 1 < required) { rejected = true; break; }
    }
    if (!rejected && hits >= required) found.push({ cx, cy, r, score: hits / samples });
  }
  found.sort((a, b) => b.score - a.score);
  const output: CircleNode[] = [];
  for (const c of found) { if (!output.some((o) => dist(o.cx, o.cy, c.cx, c.cy) < cStep * 1.5 && Math.abs(o.r - c.r) < rStep * 1.8)) output.push({ id: `C${output.length + 1}`, ...c, confidence: c.score, source: "detected" }); if (output.length === 4) break; }
  return output;
}

export function detectDiagram(image: ImageData): Detection {
  const { binary, threshold } = binarize(image), lines = detectLines(binary, image.width, image.height), graph = topology(lines, image.width, image.height);
  return { ...graph, labels: labelBoxes(binary, image.width, image.height, graph.points), circles: circles(binary, image.width, image.height), threshold };
}
