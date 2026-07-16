import { extractQuestionConstrained } from "./mobileQuestionConstraints";
import { detectDiagram, type Detection } from "./vision";

const MODEL_SIZE = 320;
const MODEL_URL = "/models/geometry_unet_pgdp5k_epoch5_320.onnx";
const ORT_RUNTIME_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist";

export type MobileBackend = "webgpu" | "wasm";
export type MobileInferenceStatus = {
  ready: boolean;
  backend?: MobileBackend;
  deviceLabel: string;
};

type OrtModule = typeof import("onnxruntime-web/wasm");
type Runtime = {
  ort: OrtModule;
  session: Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>;
  backend: MobileBackend;
};

let runtimePromise: Promise<Runtime> | null = null;

function clientDeviceLabel() {
  if (typeof navigator === "undefined") return "本机芯片";
  const userAgent = navigator.userAgent;
  const appleTouch = /iPhone|iPad|iPod/i.test(userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (appleTouch) return "iPhone/iPad A/M 系列芯片";
  if (/Windows/i.test(userAgent)) return "Windows 电脑";
  if (/Android/i.test(userAgent)) return "安卓手机/平板";
  if (/Macintosh|Mac OS X/i.test(userAgent)) return "Mac 电脑";
  return "本机芯片";
}

function forceWasmForDiagnostics() {
  return typeof window !== "undefined" && new URLSearchParams(window.location.search).get("backend") === "wasm";
}

async function configureWasm(ort: OrtModule, variant: "asyncify" | "plain") {
  // One thread works on iOS and on ordinary HTTP origins without requiring
  // cross-origin isolation. It also avoids stealing every phone CPU core.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const suffix = variant === "asyncify" ? ".asyncify" : "";
  const mjs = `${ORT_RUNTIME_BASE}/ort-wasm-simd-threaded${suffix}.mjs`;
  const wasm = `${ORT_RUNTIME_BASE}/ort-wasm-simd-threaded${suffix}.wasm`;
  const response = await fetch(wasm, { mode: "cors", cache: "force-cache" });
  if (!response.ok) throw new Error(`无法下载手机运行库（${response.status}）`);
  // Passing the bytes explicitly avoids WebAssembly.compileStreaming MIME
  // problems on lightweight local servers that label .wasm as octet-stream.
  ort.env.wasm.wasmPaths = { mjs, wasm };
  ort.env.wasm.wasmBinary = await response.arrayBuffer();
}

async function createRuntime(): Promise<Runtime> {
  if (!forceWasmForDiagnostics() && "gpu" in navigator && window.isSecureContext) {
    try {
      const ort = await import("onnxruntime-web/webgpu") as unknown as OrtModule;
      await configureWasm(ort, "asyncify");
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["webgpu"], graphOptimizationLevel: "all",
      });
      return { ort, session, backend: "webgpu" };
    } catch (error) {
      console.warn("WebGPU unavailable for this model; falling back to phone CPU/WASM.", error);
    }
  }
  const ort = await import("onnxruntime-web/wasm") as OrtModule;
  await configureWasm(ort, "plain");
  const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  return { ort, session, backend: "wasm" };
}

function runtime() {
  if (!runtimePromise) runtimePromise = createRuntime().catch((error) => {
    runtimePromise = null;
    throw error;
  });
  return runtimePromise;
}

function modelInput(source: HTMLCanvasElement) {
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE; canvas.height = MODEL_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("手机浏览器无法建立识别画布");
  context.fillStyle = "white"; context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const image = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const plane = MODEL_SIZE * MODEL_SIZE, data = new Float32Array(plane * 3);
  for (let index = 0; index < plane; index += 1) {
    data[index] = image.data[index * 4] / 255;
    data[plane + index] = image.data[index * 4 + 1] / 255;
    data[plane * 2 + index] = image.data[index * 4 + 2] / 255;
  }
  return { data, image };
}

function argmaxMask(logits: Float32Array) {
  const plane = MODEL_SIZE * MODEL_SIZE;
  if (logits.length !== plane * 6) throw new Error(`手机模型输出尺寸异常：${logits.length}`);
  const mask = new Uint8Array(plane);
  for (let pixel = 0; pixel < plane; pixel += 1) {
    let bestClass = 0, bestValue = logits[pixel];
    for (let classId = 1; classId < 6; classId += 1) {
      const value = logits[classId * plane + pixel];
      if (value > bestValue) { bestValue = value; bestClass = classId; }
    }
    mask[pixel] = bestClass;
  }
  return mask;
}

function scaleDetection(detection: Detection, width: number, height: number): Detection {
  const scaleX = width / MODEL_SIZE, scaleY = height / MODEL_SIZE, scaleR = (scaleX + scaleY) / 2;
  return {
    ...detection,
    points: detection.points.map((point) => ({ ...point, x: point.x * scaleX, y: point.y * scaleY })),
    arrows: detection.arrows.map((arrow) => ({ ...arrow, x: arrow.x * scaleX, y: arrow.y * scaleY, tailX: arrow.tailX * scaleX, tailY: arrow.tailY * scaleY })),
    circles: detection.circles.map((circle) => ({ ...circle, cx: circle.cx * scaleX, cy: circle.cy * scaleY, r: circle.r * scaleR })),
    labels: detection.labels.map((label) => ({ ...label, x: label.x * scaleX, y: label.y * scaleY, w: label.w * scaleX, h: label.h * scaleY })),
  };
}

export function initialMobileStatus(): MobileInferenceStatus {
  // Keep the server-rendered and first client-rendered text identical. Device
  // detection happens after hydration through inspectClientStatus().
  return { ready: false, deviceLabel: "当前设备" };
}

export function inspectClientStatus(): MobileInferenceStatus {
  const canTryWebGpu = typeof navigator !== "undefined" && "gpu" in navigator && typeof window !== "undefined" && window.isSecureContext && !forceWasmForDiagnostics();
  return {
    ready: false, backend: canTryWebGpu ? "webgpu" : "wasm",
    deviceLabel: clientDeviceLabel(),
  };
}

export async function requestMobileInference(source: HTMLCanvasElement, questionText: string) {
  const startedAt = performance.now();
  const active = await runtime();
  const input = modelInput(source);
  const tensor = new active.ort.Tensor("float32", input.data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  const outputMap = await active.session.run({ [active.session.inputNames[0]]: tensor });
  const output = outputMap[active.session.outputNames[0]];
  const logits = await output.getData() as Float32Array;
  const mask = argmaxMask(logits);
  const constrained = extractQuestionConstrained(mask, MODEL_SIZE, MODEL_SIZE, questionText);
  // When no known question template applies, the legacy detector still runs
  // on the phone. It is only a conservative fallback and never calls the PC.
  const detection = constrained || detectDiagram(input.image);
  const elapsedMs = Math.round(performance.now() - startedAt);
  return {
    detection: scaleDetection(detection, source.width, source.height),
    metadata: {
      device: "phone", backend: active.backend, model: "geometry_unet_pgdp5k_epoch5_320",
      question_constraints_applied: Boolean(constrained), inference_ms: elapsedMs,
    },
    status: {
      ready: true, backend: active.backend,
      deviceLabel: clientDeviceLabel(),
    } satisfies MobileInferenceStatus,
  };
}
