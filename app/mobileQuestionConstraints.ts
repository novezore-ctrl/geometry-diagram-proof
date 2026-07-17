import type { AttachmentRelation, Detection, PointNode, SegmentEdge } from "./vision";

type Candidate = { x: number; y: number; confidence: number; members: number };
type Template = {
  name: string;
  labels: readonly string[];
  segments: ReadonlyArray<readonly [string, string]>;
  orderedChains: ReadonlyArray<readonly [string, string, string]>;
};

const FIGURE2: Template = {
  name: "fold_angle_bisector_figure2",
  labels: ["A", "B", "C", "D", "E", "F", "G"],
  segments: [
    ["E", "G"], ["B", "C"], ["B", "D"], ["D", "A"], ["C", "D"],
    ["D", "G"], ["C", "A"], ["E", "D"], ["B", "E"], ["E", "C"],
  ],
  orderedChains: [["A", "D", "B"], ["C", "D", "G"], ["B", "F", "C"], ["E", "F", "D"]],
};

function templateFromQuestion(questionText: string): Template | null {
  const compact = (questionText || "").toUpperCase().replace(/\s+/g, "");
  const evidence = [
    compact.includes("D") && compact.includes("AB"),
    (compact.includes("翻折") || compact.includes("折叠")) && compact.includes("CD"),
    compact.includes("DE") && compact.includes("CB") && compact.includes("F"),
    compact.includes("连接BE"),
    compact.includes("BED") && compact.includes("平分线") && compact.includes("G"),
  ];
  return evidence.every(Boolean) ? FIGURE2 : null;
}

function components(mask: Uint8Array, width: number, height: number, classId: number, minimumArea = 5) {
  const seen = new Uint8Array(mask.length);
  const result: Array<Array<[number, number]>> = [];
  const queue = new Int32Array(mask.length);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const origin = y * width + x;
    if (seen[origin] || mask[origin] !== classId) continue;
    let head = 0, tail = 1;
    queue[0] = origin; seen[origin] = 1;
    const pixels: Array<[number, number]> = [];
    while (head < tail) {
      const index = queue[head++], cy = Math.floor(index / width), cx = index - cy * width;
      pixels.push([cx, cy]);
      for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx;
        if (!seen[next] && mask[next] === classId) { seen[next] = 1; queue[tail++] = next; }
      }
    }
    if (pixels.length >= minimumArea) result.push(pixels);
  }
  return result;
}

function pointCandidates(mask: Uint8Array, width: number, height: number): Candidate[] {
  const linePixels: Array<[number, number]> = [];
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const value = mask[y * width + x];
    if (value === 1 || value === 2) linePixels.push([x, y]);
  }
  const points: Candidate[] = [];
  for (const pixels of components(mask, width, height, 4, 5)) {
    let sumX = 0, sumY = 0, minX = width, maxX = 0, minY = height, maxY = 0;
    for (const [x, y] of pixels) {
      sumX += x; sumY += y; minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const cx = sumX / pixels.length, cy = sumY / pixels.length;
    const inner = Math.max(4.5, Math.max(maxX - minX + 1, maxY - minY + 1) * .62);
    const histogram = new Int32Array(32);
    for (const [x, y] of linePixels) {
      const dx = x - cx, dy = y - cy, radius = Math.hypot(dx, dy);
      if (radius < inner || radius > inner + 22) continue;
      const angle = (Math.atan2(dy, dx) + Math.PI * 2) % (Math.PI * 2);
      histogram[Math.min(31, Math.floor(angle / (Math.PI * 2) * 32))] += 1;
    }
    const smooth = Array.from(histogram, (value, index) => value + histogram[(index + 31) % 32] + histogram[(index + 1) % 32]);
    const maximum = Math.max(0, ...smooth);
    const peakOrder = smooth.map((_, index) => index).sort((a, b) => smooth[b] - smooth[a]);
    const peaks: number[] = [];
    for (const index of peakOrder) {
      if (smooth[index] < Math.max(3, maximum * .34)) break;
      if (peaks.every((other) => Math.min(Math.abs(index - other), 32 - Math.abs(index - other)) >= 3)) peaks.push(index);
      if (peaks.length === 3) break;
    }
    const oneSidedArrow = peaks.length === 1 && maximum >= 5 && pixels.length >= 12;
    if (!oneSidedArrow) points.push({ x: cx, y: cy, confidence: Math.min(1, pixels.length / 40), members: 1 });
  }
  return points;
}

function mergeCandidates(points: Candidate[], radius = 7.5): Candidate[] {
  const parent = points.map((_, index) => index);
  const find = (index: number): number => parent[index] === index ? index : (parent[index] = find(parent[index]));
  const union = (a: number, b: number) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[rb] = ra; };
  for (let i = 0; i < points.length; i += 1) for (let j = i + 1; j < points.length; j += 1) {
    if (Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y) <= radius) union(i, j);
  }
  const groups = new Map<number, Candidate[]>();
  points.forEach((point, index) => { const key = find(index), group = groups.get(key) || []; group.push(point); groups.set(key, group); });
  return [...groups.values()].map((group) => {
    const weights = group.map((item) => Math.max(.15, item.confidence));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return {
      x: group.reduce((sum, item, index) => sum + item.x * weights[index], 0) / total,
      y: group.reduce((sum, item, index) => sum + item.y * weights[index], 0) / total,
      confidence: Math.max(...group.map((item) => item.confidence)), members: group.length,
    };
  });
}

function dilatedLineMask(mask: Uint8Array, width: number, height: number, radius = 2) {
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const value = mask[y * width + x];
    if (value !== 1 && value !== 2) continue;
    for (let dy = -radius; dy <= radius; dy += 1) for (let dx = -radius; dx <= radius; dx += 1) {
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) output[ny * width + nx] = 1;
    }
  }
  return output;
}

function lineSupport(lineMask: Uint8Array, width: number, height: number, first: Candidate, second: Candidate) {
  let valid = 0, hits = 0;
  for (let index = 0; index < 220; index += 1) {
    const t = index / 219, x = Math.round(first.x + (second.x - first.x) * t), y = Math.round(first.y + (second.y - first.y) * t);
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    valid += 1; hits += lineMask[y * width + x];
  }
  return valid ? hits / valid : 0;
}

function orderedCollinearity(first: Candidate, middle: Candidate, last: Candidate) {
  const dx = last.x - first.x, dy = last.y - first.y, length2 = dx * dx + dy * dy;
  if (length2 < 36) return 0;
  const t = ((middle.x - first.x) * dx + (middle.y - first.y) * dy) / length2;
  const offset = Math.hypot(middle.x - (first.x + t * dx), middle.y - (first.y + t * dy));
  const tolerance = Math.max(3.5, Math.min(10, Math.sqrt(length2) * .045));
  const alignment = Math.exp(-((offset / tolerance) ** 2));
  const between = t > .04 && t < .96 ? 1 : Math.max(0, 1 - Math.abs(t - .5) * 2);
  return alignment * between;
}

function pairKey(a: number, b: number) { return a < b ? `${a}:${b}` : `${b}:${a}`; }

function assignmentScore(assignment: Record<string, number>, candidates: Candidate[], template: Template, support: Map<string, number>) {
  let score = 0;
  for (const index of Object.values(assignment)) {
    const candidate = candidates[index];
    score += .07 * candidate.confidence + .025 * Math.min(2, candidate.members);
  }
  for (const [first, second] of template.segments) if (first in assignment && second in assignment) {
    const value = support.get(pairKey(assignment[first], assignment[second])) || 0;
    score += 2.8 * value - 1.1 * Math.max(0, .52 - value);
  }
  for (const [first, middle, last] of template.orderedChains) if ([first, middle, last].every((label) => label in assignment)) {
    const value = orderedCollinearity(candidates[assignment[first]], candidates[assignment[middle]], candidates[assignment[last]]);
    score += 5.2 * value - 2 * (1 - value);
  }
  return score;
}

export function extractQuestionConstrained(mask: Uint8Array, width: number, height: number, questionText: string): Detection | null {
  const template = templateFromQuestion(questionText);
  if (!template) return null;
  const margin = Math.max(3, Math.min(width, height) * .018);
  const candidates = mergeCandidates(pointCandidates(mask, width, height)).filter((item) =>
    item.x >= margin && item.x <= width - margin && item.y >= margin && item.y <= height - margin,
  );
  if (candidates.length < template.labels.length) throw new Error(`模型只找到 ${candidates.length} 个有效点候选，无法匹配题干中的 ${template.labels.length} 个点`);

  const lineMask = dilatedLineMask(mask, width, height, 2);
  const support = new Map<string, number>();
  for (let i = 0; i < candidates.length; i += 1) for (let j = i + 1; j < candidates.length; j += 1) {
    support.set(pairKey(i, j), lineSupport(lineMask, width, height, candidates[i], candidates[j]));
  }
  const roleOrder = ["D", "F", "A", "B", "C", "G", "E"];
  let beam: Array<{ score: number; assignment: Record<string, number> }> = [{ score: 0, assignment: {} }];
  for (const role of roleOrder) {
    const expanded: typeof beam = [];
    for (const item of beam) {
      const used = new Set(Object.values(item.assignment));
      for (let index = 0; index < candidates.length; index += 1) if (!used.has(index)) {
        const assignment = { ...item.assignment, [role]: index };
        expanded.push({ score: assignmentScore(assignment, candidates, template, support), assignment });
      }
    }
    expanded.sort((a, b) => b.score - a.score);
    beam = expanded.slice(0, 2400);
  }
  if (!beam.length) throw new Error("题干约束的点位匹配失败");
  const assignment = beam[0].assignment;
  const byLabel = new Map(template.labels.map((label) => [label, candidates[assignment[label]]]));
  const points: PointNode[] = template.labels.map((label) => {
    const candidate = byLabel.get(label)!;
    return { id: label, label, x: candidate.x, y: candidate.y, confidence: Math.max(.55, Math.min(.98, candidate.confidence)), source: "question_constrained" };
  });
  const segments: SegmentEdge[] = template.segments.map(([a, b], index) => ({
    id: `S${index + 1}`, a, b, confidence: support.get(pairKey(assignment[a], assignment[b])) || 0, source: "question_constrained",
  }));
  const attachments: AttachmentRelation[] = template.orderedChains.map(([hostA, junction, hostB], index) => {
    const first = byLabel.get(hostA)!, middle = byLabel.get(junction)!, last = byLabel.get(hostB)!;
    const dx = last.x - first.x, dy = last.y - first.y, length2 = Math.max(1e-9, dx * dx + dy * dy);
    const approximateT = Math.max(0, Math.min(1, ((middle.x - first.x) * dx + (middle.y - first.y) * dy) / length2));
    return {
      id: `O${index + 1}`, kind: "point_on_segment", junction, hostA, hostB,
      position: { kind: "approximate", approximateT, coordinateUncertain: true },
      confidence: Math.max(.55, orderedCollinearity(first, middle, last)), source: "question_constrained", support: "question_and_collinearity",
    };
  });
  return { points, segments, arrows: [], attachments, circles: [], labels: [], threshold: 0 };
}
