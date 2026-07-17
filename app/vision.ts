export type CandidateSource = "detected" | "question_constrained" | "manual";
export type PointNode = { id: string; label: string; x: number; y: number; confidence: number; source: CandidateSource };
export type SegmentEdge = { id: string; a: string; b: string; confidence: number; source: CandidateSource };
export type CircleNode = { id: string; cx: number; cy: number; r: number; confidence: number; source: CandidateSource };
export type LabelBox = { id: string; x: number; y: number; w: number; h: number; nearPoint?: string };
export type ArrowTip = {
  id: string;
  x: number;
  y: number;
  tailX: number;
  tailY: number;
  directionX: number;
  directionY: number;
  fromPoint?: string;
  confidence: number;
  source: CandidateSource;
};
export type AttachmentPosition =
  | { kind: "approximate"; approximateT: number; coordinateUncertain: true }
  | { kind: "division"; parts: number; index: number; coordinateUncertain: false };
export type AttachmentRelation = {
  id: string;
  kind: "point_on_segment" | "branch_attachment";
  junction: string;
  hostA: string;
  hostB: string;
  branch?: string;
  position: AttachmentPosition;
  confidence: number;
  source: CandidateSource;
  support: "collinear_points" | "topology" | "remembered" | "question_and_collinearity";
};
export type Detection = {
  points: PointNode[];
  segments: SegmentEdge[];
  arrows: ArrowTip[];
  attachments: AttachmentRelation[];
  circles: CircleNode[];
  labels: LabelBox[];
  threshold: number;
};
type Line = { theta: number; x1: number; y1: number; x2: number; y2: number; confidence: number };
type ArrowCandidate = ArrowTip & { lineIndex: number };

export const DETECTOR_REFERENCE_EDGE = 760;
export function detectorRasterSize(width: number, height: number) {
  // The detector thresholds are calibrated in a fixed pixel space. Normalize
  // both large and small selections so a small diagram inside a full page is
  // enlarged for recognition.
  const scale = DETECTOR_REFERENCE_EDGE / Math.max(1, width, height);
  return { width: Math.max(12, Math.round(width * scale)), height: Math.max(12, Math.round(height * scale)), scale };
}

export function viewportToCanvasPoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  canvasWidth: number,
  canvasHeight: number,
) {
  return {
    x: (clientX - rect.left) / Math.max(1, rect.width) * canvasWidth,
    y: (clientY - rect.top) / Math.max(1, rect.height) * canvasHeight,
  };
}

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
  // Short text strokes often vote weakly in the Hough space. Keep only the
  // strongest structural lines so labels do not become fake geometry edges.
  const cutoff = Math.max(22, maxVote * .42);
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
    if (end - start < Math.min(width, height) * .24) continue;
    const baseX = rho * c, baseY = rho * s;
    const line: Line = { theta: peak.t * angleStep, x1: baseX - start * s, y1: baseY + start * c, x2: baseX - end * s, y2: baseY + end * c, confidence: Math.min(.99, peak.vote / maxVote) };
    const duplicate = lines.some((other) => {
      const angle = Math.min(Math.abs(other.theta - line.theta), 180 - Math.abs(other.theta - line.theta));
      const direct = dist(other.x1, other.y1, line.x1, line.y1) + dist(other.x2, other.y2, line.x2, line.y2);
      const reverse = dist(other.x1, other.y1, line.x2, line.y2) + dist(other.x2, other.y2, line.x1, line.y1);
      const lineMidX = (line.x1 + line.x2) / 2, lineMidY = (line.y1 + line.y2) / 2;
      const otherMidX = (other.x1 + other.x2) / 2, otherMidY = (other.y1 + other.y2) / 2;
      const sameStroke = pointToLine(lineMidX, lineMidY, other).distance < 7
        && pointToLine(otherMidX, otherMidY, line).distance < 7;
      return angle < 5 && (Math.min(direct, reverse) < 36 || sameStroke);
    });
    if (!duplicate) lines.push(line);
    if (lines.length === 12) break;
  }
  return lines;
}

function pointToLine(x: number, y: number, line: Line) {
  const dx = line.x2 - line.x1, dy = line.y2 - line.y1, len2 = dx * dx + dy * dy || 1;
  const t = ((x - line.x1) * dx + (y - line.y1) * dy) / len2;
  const px = line.x1 + t * dx, py = line.y1 + t * dy;
  return { t, distance: dist(x, y, px, py) };
}

function arrowScore(binary: Uint8Array, width: number, height: number, tipX: number, tipY: number, innerX: number, innerY: number) {
  const length = dist(tipX, tipY, innerX, innerY) || 1;
  const ux = (tipX - innerX) / length, uy = (tipY - innerY) / length;
  const vx = -uy, vy = ux;
  let left = 0, right = 0, maxSpread = 0, backwardDepth = 0;
  const radius = 22;
  for (let y = Math.max(0, Math.floor(tipY - radius)); y <= Math.min(height - 1, Math.ceil(tipY + radius)); y += 1) {
    for (let x = Math.max(0, Math.floor(tipX - radius)); x <= Math.min(width - 1, Math.ceil(tipX + radius)); x += 1) {
      if (!binary[y * width + x]) continue;
      const rx = x - tipX, ry = y - tipY;
      const along = rx * ux + ry * uy, across = rx * vx + ry * vy;
      if (along < -21 || along > 5 || Math.abs(across) > 15) continue;
      if (along < -2 && Math.abs(across) >= 3.2) {
        if (across < 0) left += 1; else right += 1;
        maxSpread = Math.max(maxSpread, Math.abs(across));
        backwardDepth = Math.max(backwardDepth, -along);
      }
    }
  }
  const wings = left + right;
  if (left < 5 || right < 5 || wings < 16 || maxSpread < 5.5 || backwardDepth < 8) return 0;
  return Math.min(.98, .52 + Math.min(.2, wings / 150) + Math.min(.14, (maxSpread - 5) / 24) + Math.min(.1, (backwardDepth - 8) / 50));
}

function detectArrowTips(binary: Uint8Array, width: number, height: number, lines: Line[]) {
  const candidates: ArrowCandidate[] = [];
  lines.forEach((line, lineIndex) => {
    const ends = [
      { x: line.x1, y: line.y1, innerX: line.x2, innerY: line.y2 },
      { x: line.x2, y: line.y2, innerX: line.x1, innerY: line.y1 },
    ];
    for (const end of ends) {
      const joined = lines.some((other, otherIndex) => {
        if (otherIndex === lineIndex) return false;
        const hit = pointToLine(end.x, end.y, other);
        return hit.t > -.06 && hit.t < 1.06 && hit.distance < 11;
      });
      if (joined) continue;
      const score = arrowScore(binary, width, height, end.x, end.y, end.innerX, end.innerY);
      if (!score) continue;
      const length = dist(end.x, end.y, end.innerX, end.innerY) || 1;
      const directionX = (end.x - end.innerX) / length, directionY = (end.y - end.innerY) / length;
      candidates.push({
        id: "", x: end.x, y: end.y, tailX: end.innerX, tailY: end.innerY,
        directionX, directionY, confidence: Math.min(score, line.confidence + .2),
        source: "detected", lineIndex,
      });
    }
  });
  candidates.sort((a, b) => b.confidence - a.confidence);
  const output: ArrowCandidate[] = [];
  for (const candidate of candidates) {
    if (!output.some((arrow) => dist(arrow.x, arrow.y, candidate.x, candidate.y) < 18)) output.push(candidate);
  }
  return output.map((arrow, index) => ({ ...arrow, id: `A${index + 1}` }));
}

function intersection(a: Line, b: Line) {
  const angle = Math.min(Math.abs(a.theta - b.theta), 180 - Math.abs(a.theta - b.theta));
  // Slightly different Hough estimates of one thick/blurred printed line can
  // cross near a real junction. Treat near-parallel estimates as one stroke,
  // otherwise they manufacture a fan of duplicate intersection points.
  if (angle < 7) return null;
  const d = (a.x1 - a.x2) * (b.y1 - b.y2) - (a.y1 - a.y2) * (b.x1 - b.x2);
  if (Math.abs(d) < .001) return null;
  const ca = a.x1 * a.y2 - a.y1 * a.x2, cb = b.x1 * b.y2 - b.y1 * b.x2;
  const x = (ca * (b.x1 - b.x2) - (a.x1 - a.x2) * cb) / d;
  const y = (ca * (b.y1 - b.y2) - (a.y1 - a.y2) * cb) / d;
  const inside = (l: Line) => x >= Math.min(l.x1, l.x2) - 10 && x <= Math.max(l.x1, l.x2) + 10 && y >= Math.min(l.y1, l.y2) - 10 && y <= Math.max(l.y1, l.y2) + 10;
  return inside(a) && inside(b) ? { x, y } : null;
}

function topology(lines: Line[], arrows: ArrowCandidate[], width: number, height: number) {
  const raw: Array<{ x: number; y: number; confidence: number }> = [];
  for (const line of lines) {
    for (const endpoint of [{ x: line.x1, y: line.y1 }, { x: line.x2, y: line.y2 }]) {
      if (!arrows.some((arrow) => dist(arrow.x, arrow.y, endpoint.x, endpoint.y) < 18)) raw.push({ ...endpoint, confidence: line.confidence });
    }
  }
  for (let i = 0; i < lines.length; i += 1) for (let j = i + 1; j < lines.length; j += 1) {
    const hit = intersection(lines[i], lines[j]);
    if (hit && hit.x >= 0 && hit.x <= width && hit.y >= 0 && hit.y <= height) raw.push({ ...hit, confidence: Math.min(lines[i].confidence, lines[j].confidence) });
  }
  // Merge by connected neighbourhood rather than a moving centroid. In a
  // blurry photo, estimates of one junction form a short chain; centroid-first
  // grouping can split the two ends into P5/P8/P13 even though every adjacent
  // estimate belongs to the same physical point.
  const parent = raw.map((_, index) => index);
  const find = (index: number): number => parent[index] === index ? index : (parent[index] = find(parent[index]));
  const union = (a: number, b: number) => { const rootA = find(a), rootB = find(b); if (rootA !== rootB) parent[rootB] = rootA; };
  for (let i = 0; i < raw.length; i += 1) for (let j = i + 1; j < raw.length; j += 1) {
    if (dist(raw[i].x, raw[i].y, raw[j].x, raw[j].y) < 18) union(i, j);
  }
  const grouped = new Map<number, typeof raw>();
  raw.forEach((point, index) => { const root = find(index), points = grouped.get(root) || []; points.push(point); grouped.set(root, points); });
  const groups = Array.from(grouped.values()).map((members) => ({
    x: members.reduce((sum, point) => sum + point.x, 0) / members.length,
    y: members.reduce((sum, point) => sum + point.y, 0) / members.length,
    confidence: Math.max(...members.map((point) => point.confidence)),
    n: members.length,
  }));
  // A true vertex is normally supported by at least two observations:
  // two line endpoints, or an endpoint plus an intersection. Single Hough
  // endpoints are usually text strokes and are intentionally discarded.
  const points: PointNode[] = groups.filter((g) => g.n > 1).map((g, i) => ({ id: `P${i + 1}`, label: `P${i + 1}`, x: g.x, y: g.y, confidence: Math.min(.99, g.confidence + Math.min(.15, g.n * .03)), source: "detected" }));
  const segments: SegmentEdge[] = [];
  for (const line of lines) {
    const dx = line.x2 - line.x1, dy = line.y2 - line.y1, len2 = dx * dx + dy * dy || 1;
    const ordered = points.map((p) => ({ p, t: ((p.x - line.x1) * dx + (p.y - line.y1) * dy) / len2, d: Math.abs(dy * p.x - dx * p.y + line.x2 * line.y1 - line.y2 * line.x1) / Math.sqrt(len2) })).filter((v) => v.t > -.05 && v.t < 1.05 && v.d < 10).sort((a, b) => a.t - b.t);
    for (let i = 0; i < ordered.length - 1; i += 1) {
      const a = ordered[i].p.id, b = ordered[i + 1].p.id;
      if (!segments.some((s) => (s.a === a && s.b === b) || (s.a === b && s.b === a))) segments.push({ id: `S${segments.length + 1}`, a, b, confidence: line.confidence, source: "detected" });
    }
  }
  const resolvedArrows: ArrowTip[] = arrows.map(({ lineIndex, ...arrow }) => {
    const line = lines[lineIndex];
    const onShaft = points.map((point) => ({ point, metric: pointToLine(point.x, point.y, line) }))
      .filter(({ point, metric }) => metric.t > -.08 && metric.t < 1.08 && metric.distance < 12 && dist(point.x, point.y, arrow.x, arrow.y) > 22)
      .sort((a, b) => dist(a.point.x, a.point.y, arrow.tailX, arrow.tailY) - dist(b.point.x, b.point.y, arrow.tailX, arrow.tailY))[0]?.point;
    return onShaft ? { ...arrow, fromPoint: onShaft.id } : arrow;
  });
  // Photographed triangle vertices and handwritten angle marks can resemble
  // arrow wings. Keep only the strongest few machine candidates for review;
  // unconfirmed candidates are excluded from export by the workspace.
  resolvedArrows.sort((a, b) => b.confidence - a.confidence);
  return { points, segments, arrows: resolvedArrows.slice(0, 4), attachments: deriveAttachments(points, segments) };
}

export function deriveAttachments(points: PointNode[], segments: SegmentEdge[], previous: AttachmentRelation[] = []) {
  const byId = new Map(points.map((point) => [point.id, point]));
  const edgeKey = (a: string, b: string) => [a, b].sort().join(":");
  const edgeMap = new Map(segments.map((segment) => [edgeKey(segment.a, segment.b), segment]));
  const locationKey = (junction: string, a: string, b: string) => `${junction}:${[a, b].sort().join(":")}`;
  const previousByLocation = new Map(previous.map((item) => [locationKey(item.junction, item.hostA, item.hostB), item]));
  const raw: Array<AttachmentRelation & { angle: number; hostLength: number }> = [];

  // A connection location is a geometric relation between three collinear
  // points, not a property of a degree-3 graph node. This keeps the relation
  // valid when a user adds another point or draws one long segment through an
  // existing point (for example P4 lying inside P3-P5).
  for (let i = 0; i < points.length; i += 1) for (let j = i + 1; j < points.length; j += 1) {
    const first = points[i], second = points[j];
    const dx = second.x - first.x, dy = second.y - first.y, hostLength = Math.hypot(dx, dy);
    if (hostLength < 24) continue;
    for (const center of points) {
      if (center.id === first.id || center.id === second.id) continue;
      const t = ((center.x - first.x) * dx + (center.y - first.y) * dy) / (hostLength * hostLength);
      if (t <= .035 || t >= .965) continue;
      const projectedX = first.x + t * dx, projectedY = first.y + t * dy;
      const offset = dist(center.x, center.y, projectedX, projectedY);
      const tolerance = Math.max(7, Math.min(15, hostLength * .028));
      if (offset > tolerance) continue;

      const direct = edgeMap.get(edgeKey(first.id, second.id));
      const firstHalf = edgeMap.get(edgeKey(first.id, center.id));
      const secondHalf = edgeMap.get(edgeKey(center.id, second.id));
      // A detected host line needs complete support: either one explicit long
      // segment through the center, or both adjacent halves. Accepting only one
      // half turns every accidental collinear point into many downstream
      // "point on segment" relations in crowded photographs.
      if (!direct && !(firstHalf && secondHalf)) continue;
      const evidence = [direct, firstHalf, secondHalf].filter(Boolean) as SegmentEdge[];

      const [hostA, hostB] = [first.id, second.id].sort();
      const a = byId.get(hostA)!, b = byId.get(hostB)!;
      const hostDx = b.x - a.x, hostDy = b.y - a.y, orderedLength = Math.hypot(hostDx, hostDy) || 1;
      const approximateT = ((center.x - a.x) * hostDx + (center.y - a.y) * hostDy) / (orderedLength * orderedLength);
      const key = locationKey(center.id, hostA, hostB), prior = previousByLocation.get(key);
      const geometricConfidence = Math.max(.45, 1 - offset / tolerance);
      raw.push({
        id: `O:${key}`, kind: "point_on_segment", junction: center.id, hostA, hostB,
        position: prior?.position.kind === "division" ? prior.position : { kind: "approximate", approximateT: Math.max(0, Math.min(1, approximateT)), coordinateUncertain: true },
        confidence: Math.min(.99, geometricConfidence * .7 + Math.max(...evidence.map((edge) => edge.confidence)) * .3),
        source: evidence.some((edge) => edge.source === "manual") || center.source === "manual" ? "manual" : "detected",
        support: "collinear_points", angle: Math.atan2(hostDy, hostDx), hostLength,
      });
    }
  }

  // If several wider/narrower endpoint pairs describe the same line through
  // one point, expose only the widest pair. Perpendicular lines remain separate.
  raw.sort((a, b) => b.hostLength - a.hostLength);
  const relations: AttachmentRelation[] = [];
  for (const candidate of raw) {
    const duplicateDirection = rawDirectionMatch(relations, candidate, byId);
    if (!duplicateDirection) {
      const { angle: _angle, hostLength: _hostLength, ...relation } = candidate;
      void _angle; void _hostLength;
      relations.push(relation);
    }
  }

  // Preserve a user-confirmed relation if its points still exist, even while
  // surrounding lines are being edited. It stays visibly marked for review.
  for (const prior of previous) {
    const exists = byId.has(prior.junction) && byId.has(prior.hostA) && byId.has(prior.hostB) && (!prior.branch || byId.has(prior.branch));
    const alreadyPresent = relations.some((item) => locationKey(item.junction, item.hostA, item.hostB) === locationKey(prior.junction, prior.hostA, prior.hostB));
    if (exists && !alreadyPresent && (prior.source === "manual" || prior.position.kind === "division")) relations.push({ ...prior, support: "remembered", confidence: Math.min(prior.confidence, .65) });
  }
  const automatic = relations.filter((relation) => relation.source !== "manual").sort((a, b) => b.confidence - a.confidence).slice(0, 8);
  const manual = relations.filter((relation) => relation.source === "manual");
  return [...manual, ...automatic];
}

function rawDirectionMatch(relations: AttachmentRelation[], candidate: AttachmentRelation, byId: Map<string, PointNode>) {
  const candidateA = byId.get(candidate.hostA), candidateB = byId.get(candidate.hostB); if (!candidateA || !candidateB) return false;
  const cdx = candidateB.x - candidateA.x, cdy = candidateB.y - candidateA.y, clen = Math.hypot(cdx, cdy) || 1;
  return relations.some((relation) => {
    if (relation.junction !== candidate.junction) return false;
    const a = byId.get(relation.hostA), b = byId.get(relation.hostB); if (!a || !b) return false;
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
    return Math.abs((dx * cdx + dy * cdy) / (len * clen)) > .985;
  });
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
  const { binary, threshold } = binarize(image), lines = detectLines(binary, image.width, image.height), arrows = detectArrowTips(binary, image.width, image.height, lines), graph = topology(lines, arrows, image.width, image.height);
  return { ...graph, labels: labelBoxes(binary, image.width, image.height, graph.points), circles: circles(binary, image.width, image.height), threshold };
}
