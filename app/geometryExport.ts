import type { ArrowTip, AttachmentRelation, Detection, PointNode } from "./vision";

export type GeometryShareContext = {
  questionTitle?: string;
  questionText?: string;
  confirmedArrowIds?: string[];
  confirmedCircleIds?: string[];
  sourceImageAttached?: boolean;
  includeQuestionText?: boolean;
};

const rounded = (value: number) => Math.round(value * 1000) / 1000;

function roughRegion(t: number) {
  if (t < .34) return "靠近第一端点";
  if (t > .66) return "靠近第二端点";
  return "线段中部附近";
}

function coordinateNormalizer(points: PointNode[]) {
  const xs = points.map((point) => point.x), ys = points.map((point) => point.y);
  const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
  const width = Math.max(1, maxX - minX), height = Math.max(1, maxY - minY);
  return (x: number, y: number) => ({ x: rounded((x - minX) / width), y: rounded((y - minY) / height) });
}

function throughPoints(arrow: ArrowTip, points: PointNode[]) {
  const from = arrow.fromPoint ? points.find((point) => point.id === arrow.fromPoint) : null;
  if (!from) return [];
  return points
    .filter((point) => point.id !== from.id)
    .map((point) => {
      const dx = point.x - from.x, dy = point.y - from.y;
      const along = dx * arrow.directionX + dy * arrow.directionY;
      const across = Math.abs(dx * arrow.directionY - dy * arrow.directionX);
      return { point, along, across };
    })
    .filter((item) => item.along > 8 && item.across < Math.max(9, item.along * .035))
    .sort((a, b) => a.along - b.along)
    .map((item) => item.point.label);
}

export function buildGeometryDocument(result: Detection, context: GeometryShareContext = {}) {
  const byId = new Map(result.points.map((point) => [point.id, point]));
  const name = (id?: string) => id ? byId.get(id)?.label || id : null;
  const normalize = coordinateNormalizer(result.points);
  const confirmedArrowIds = new Set([
    ...result.arrows.filter((arrow) => arrow.source === "manual").map((arrow) => arrow.id),
    ...(context.confirmedArrowIds || []),
  ]);
  const confirmedCircleIds = new Set([
    ...result.circles.filter((circle) => circle.source === "manual").map((circle) => circle.id),
    ...(context.confirmedCircleIds || []),
  ]);
  const confirmedArrows = result.arrows.filter((arrow) => confirmedArrowIds.has(arrow.id));
  const confirmedCircles = result.circles.filter((circle) => confirmedCircleIds.has(circle.id));
  const questionText = context.questionText?.trim() || "";
  const includeQuestionText = context.includeQuestionText === true;

  const attachment = (item: AttachmentRelation) => {
    const hostA = name(item.hostA), hostB = name(item.hostB), junction = name(item.junction), branch = name(item.branch);
    const relationStatement = item.kind === "point_on_segment"
      ? `点 ${junction} 在线段 ${hostA}—${hostB} 上`
      : `线 ${branch}—${junction} 在 ${junction} 处连接到线段 ${hostA}—${hostB}`;
    const position = item.position.kind === "division"
      ? {
          kind: "user_confirmed_division" as const,
          parts: item.position.parts,
          indexFromFirstEndpoint: item.position.index,
          fromEndpoint: hostA,
          exact: true,
          statement: `${junction} 是 ${hostA}—${hostB} 从 ${hostA} 起第 ${item.position.index} 个 ${item.position.parts} 等分点`,
        }
      : {
          kind: "approximate_location" as const,
          roughRegion: roughRegion(item.position.approximateT),
          coordinateUncertain: true,
          ratioUncertain: true,
          exact: false,
          statement: `${junction} 只确定在线段 ${hostA}—${hostB} 上；具体坐标和分点比例未知`,
        };
    return {
      id: item.id,
      relationKind: item.kind,
      statement: relationStatement,
      branch: branch ? { from: branch, toJunction: junction } : null,
      host: { firstEndpoint: hostA, secondEndpoint: hostB },
      junction,
      position,
      support: item.support,
      source: item.source,
      confidence: rounded(item.confidence),
    };
  };

  const attachments = result.attachments.map(attachment);
  const mustRenderSegments = result.segments.map((segment) => ({
    id: segment.id,
    endpoints: [name(segment.a), name(segment.b)] as [string | null, string | null],
    visibleLabel: `${name(segment.a)}—${name(segment.b)}`,
    requirement: "must_be_visible" as const,
    source: segment.source,
    confidence: rounded(segment.confidence),
  }));
  const collinearChains = result.attachments
    .filter((item) => item.kind === "point_on_segment")
    .map((item) => ({
      pointsInOrder: [name(item.hostA), name(item.junction), name(item.hostB)],
      statement: `${name(item.hostA)}—${name(item.junction)}—${name(item.hostB)} 三点共线，${name(item.junction)} 位于另外两点之间`,
      positionExact: item.position.kind === "division",
    }));

  return {
    schema: "geometry-diagram/v5",
    purpose: "随原始题图提供详细的辅助说明，供 GPT 使用多模态看图能力联合题干理解几何图形；这不是题目证明结论",
    sourceImage: {
      attachedWithThisShare: context.sourceImageAttached === true,
      role: "题干原文是最高依据；原始题图用于多模态核对，识别出的结构说明和 JSON 只作低优先级辅助，不能替代题干或看图。",
      multimodalInstruction: "必须先使用自身的多模态视觉能力观察原始题图，再把视觉观察与题干、人工校正结构联合理解。",
      missingImageRule: "如果实际没有收到或无法看见原始题图，必须明确说明并要求用户补图，不得假装已经看图。",
    },
    questionContext: {
      figureRole: "这些结构描述属于随附原题图片中的这道题，不是脱离题干的独立示意图",
      title: context.questionTitle?.trim() || null,
      stemProvided: Boolean(questionText && includeQuestionText),
      stemRepeatedOutsideImage: Boolean(questionText && includeQuestionText),
      stemMustBeReadFromSourceImage: !includeQuestionText,
      stem: questionText && includeQuestionText ? questionText : null,
      authorityRule: "原题图片中的印刷题干是最高依据。AI 对几何图形的直接视觉观察其次；用户校正或机器识别出的结构文字与 JSON 只作辅助。结构说明可能因用户疏忽、漏线或误连而出错；发现冲突时必须在回答中逐项指出，并明确说明采用原图题干作为依据，不得为了迁就结构说明而改写题干。",
      requiredReasoningOrder: [
        "先从附带的原题图片直接读取印刷题干，提取点、线、角、交点、翻折、平分线等显式条件",
        "再用自身多模态能力观察原图，并核对结构文字、线段清单、共线关系和 attachments 是否与题干一致",
        "若发现漏线、误连、错字或关系冲突，先在回答中列出冲突，并声明后续采用题干原文",
        "最后才进行证明、计算或重绘",
      ],
    },
    coordinatePolicy: "坐标只用于保留图形的相对布局和方向，默认不精确，不能据此推出长度、角度或比例关系",
    coordinateFrame: {
      origin: "top-left",
      xDirection: "right",
      yDirection: "down",
      normalized: true,
      renderingRule: "重绘时保留相对方位、共线顺序和交点关系；只允许整体缩放和平移，不要重新优化布局。",
    },
    points: result.points.map((point) => ({
      id: point.id,
      name: point.label,
      visualCoordinate: { x: rounded(point.x), y: rounded(point.y) },
      normalizedVisualCoordinate: normalize(point.x, point.y),
      coordinateStatus: "approximate",
      source: point.source,
    })),
    primitives: {
      segments: mustRenderSegments,
      rays: confirmedArrows.map((arrow) => ({
        id: arrow.id,
        from: name(arrow.fromPoint),
        throughPoints: throughPoints(arrow, result.points),
        arrowTipVisualCoordinate: { x: rounded(arrow.x), y: rounded(arrow.y) },
        normalizedArrowTipCoordinate: normalize(arrow.x, arrow.y),
        directionVector: { x: rounded(arrow.directionX), y: rounded(arrow.directionY) },
        meaning: name(arrow.fromPoint)
          ? `从 ${name(arrow.fromPoint)} 出发，沿箭头尖端方向无限延伸的射线`
          : "沿箭头尖端方向无限延伸的射线；起点尚未匹配",
        confirmation: "user_confirmed",
        source: arrow.source,
        confidence: rounded(arrow.confidence),
      })),
      circles: confirmedCircles.map((circle) => ({
        id: circle.id,
        visualCenter: { x: rounded(circle.cx), y: rounded(circle.cy) },
        normalizedVisualCenter: normalize(circle.cx, circle.cy),
        visualRadius: rounded(circle.r),
        measurementStatus: "approximate",
        confirmation: "user_confirmed",
      })),
    },
    collinearChains,
    attachments,
    excludedUnconfirmedCandidates: {
      arrows: result.arrows.filter((arrow) => !confirmedArrowIds.has(arrow.id)).map((arrow) => arrow.id),
      circles: result.circles.filter((circle) => !confirmedCircleIds.has(circle.id)).map((circle) => circle.id),
      rule: "这些只是机器候选，不是题图中的已确认几何元素，禁止在重绘或推理中使用。",
    },
    drawingContract: {
      expectedPointCount: result.points.length,
      expectedVisibleSegmentCount: mustRenderSegments.length,
      mustRenderEveryListedSegment: true,
      appliesAfterQuestionConflictAudit: true,
      authorityCondition: "线段清单只在不与题干原文冲突时作为重绘约束；若冲突，必须报告冲突并按题干修正，不得盲从清单。",
      mustRenderSegmentPairs: mustRenderSegments.map((segment) => segment.endpoints),
      mustRenderSegmentLabels: mustRenderSegments.map((segment) => segment.visibleLabel),
      collinearChains,
      forbiddenActions: [
        "不得跳过题干与 mustRenderSegmentPairs 的冲突检查；发现错误关系时必须先报告并采用题干",
        "对经题干核对无冲突的 mustRenderSegmentPairs，不得漏画任何一条线，例如 E—G 也必须清晰可见",
        "不得新增未列出的线、射线、圆、垂直、平行、等长、等角或中点条件",
        "不得把线段清单文字写对但在最终图中漏画",
      ],
      postRenderVerification: "先排除或修正与题干冲突的结构项，再逐条回读其余 mustRenderSegmentLabels；只有题干条件、原图观察和最终成图一致，才可交付。",
    },
    uncertainties: [
      ...attachments.filter((item) => !item.position.exact).map((item) => item.position.statement),
      ...(questionText || context.sourceImageAttached ? [] : ["没有可用的题干或原题图片，因此不能利用题意校验图形。"]),
      "用户可能疏忽未发现识别出的漏线、误连或错误共线关系；结构说明和 JSON 必须先与题干核对。",
      "没有题干文字或人工确认时，不从图形外观推断垂直、平行、等长、等角或精确比例。",
    ],
    instructionsForGpt: [
      "以下详细说明的目的，是帮助你更准确地理解原始题图中的几何图形，不是用文字或 JSON 替代原图。",
      "必须使用自身的多模态看图能力先观察原始题图，再把视觉观察、题干与该 JSON 当作同一道题的互补输入联合理解。",
      "如果实际没有收到或看不到原图，必须先明确说明并要求补图，不得假装已经看图。",
      "题干不在结构说明中重复转写，必须直接从原题图片读取；原图中的印刷题干是最高依据。",
      "对图形的直接视觉观察其次；points、segments、collinearChains、attachments 和 JSON 只作辅助，可能包含用户漏看的识别错误。",
      "先复述题干显式条件，再核对 points、segments、collinearChains 和 attachments；发现错误关系时必须在回答中逐项指出，并明确告知用户后续采用题干原文。",
      "重绘时只对经题干核对无冲突的 drawingContract.mustRenderSegmentPairs 逐条画出并自检；不得为了迁就错误线段关系而改写题干。",
      "只有 user_confirmed_division 才是精确等分条件；approximate_location 不能当作中点或精确比例。",
      "只有 primitives.rays 中的项目才是已确认射线；excludedUnconfirmedCandidates 中的箭头或圆必须忽略。",
    ],
  };
}

export function toGptShareText(result: Detection, context: GeometryShareContext = {}) {
  const document = buildGeometryDocument(result, context);
  const imageNotice = context.sourceImageAttached
    ? "随本内容附带了这道题的原始题图。"
    : "本说明必须与这道题的原始题图一起使用；如果当前消息没有附图，请先要求用户补图。";
  const lines = [
    "【给 GPT 的多模态理解说明】",
    imageNotice,
    "以下点线连接、共线顺序和结构化 JSON，是对随附原题图片中几何图形的辅助描述，目的是帮助你更准确地理解该题中的图形，不是用文字说明或 JSON 替代原图。",
    "请务必利用你自身的多模态看图能力先观察原始题图，再把视觉观察与下面的详细描述结合起来理解题目；不得只读说明或只看 JSON。",
    "如果你没有实际收到或看不到原始题图，请先明确说明并要求用户补图，不得假装已经看图。",
    "【最高优先级规则】题干不在这里重复转写，请直接从原题图片读取。原图中的印刷题干是最高依据；对图形的直接视觉观察其次；识别出的线段关系、共线说明和 JSON 只作辅助，可能因为用户疏忽而含有漏线、误连或错字。",
    "请先逐项检查辅助说明是否与原图题干和图形一致。若发现冲突，必须在回答中明确列出错误关系，告知用户将采用原图题干作为依据，然后再继续解题或重绘；不得静默采纳错误结构，也不得为了迁就它而改写题干。",
    "这是某道几何题中的一个配图。请把原始题图和下面的校正结构作为同一道题联合理解。",
    "结构化图形负责辅助说明点、线和交点怎样连接，但其权威低于题干。",
    "题干不在此处重复发送；必须直接读取随附原题图片中的印刷文字。",
    "",
  ];
  lines.push(
    `辅助结构清单：当前记录了 ${document.drawingContract.expectedVisibleSegmentCount} 条线；必须先与题干核对，冲突项要报告并按题干修正。`,
    `核对后无冲突的逐条必画清单：${document.drawingContract.mustRenderSegmentLabels.join("、") || "无"}`,
  );
  if (document.collinearChains.length) {
    lines.push("共线与点在线上：", ...document.collinearChains.map((chain) => `- ${chain.statement}`));
  }
  if (document.primitives.rays.length) {
    lines.push("用户已确认的射线方向：", ...document.primitives.rays.map((ray) => `- ${ray.meaning}`));
  }
  const excluded = document.excludedUnconfirmedCandidates;
  if (excluded.arrows.length || excluded.circles.length) {
    lines.push(`已排除机器候选：${excluded.arrows.length} 个未确认箭头、${excluded.circles.length} 个未确认圆；禁止把它们画进题图。`);
  }
  lines.push(
    "作图后自检：先确认已报告并排除与题干冲突的结构项，再在最终图中逐条找到其余必画线，并核对题干条件、点数和线数。",
    "不要从图形外观补出题干未给出的垂直、平行、等长、等角、中点或精确比例。",
    "",
    "结构化 JSON：",
    JSON.stringify(document, null, 2),
  );
  return lines.join("\n");
}
