"""Local-only HTTP bridge between the browser UI and CUDA U-Net runtime."""
from __future__ import annotations

import argparse
import base64
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

from PIL import Image

try:
    from .gpu_runtime import GPUInferenceRuntime
except ImportError:
    from gpu_runtime import GPUInferenceRuntime


MAX_REQUEST_BYTES = 12 * 1024 * 1024
ALLOWED_ORIGINS = {"http://127.0.0.1:3000", "http://localhost:3000"}
PRIVATE_LAN_ORIGIN = re.compile(
    r"^http://(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}):3000$"
)


def handler_factory(runtime: GPUInferenceRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GeometryGPU/1.0"

        def _origin(self) -> str:
            origin = self.headers.get("Origin", "")
            allowed = origin in ALLOWED_ORIGINS or bool(PRIVATE_LAN_ORIGIN.fullmatch(origin))
            return origin if allowed else "http://127.0.0.1:3000"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", self._origin())
            self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self._origin())
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, runtime.health())
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/infer":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request size is invalid or exceeds 12 MiB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                encoded = str(payload.get("imageBase64", ""))
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                image_bytes = base64.b64decode(encoded, validate=True)
                with Image.open(BytesIO(image_bytes)) as source:
                    image = source.convert("RGB")
                if image.width * image.height > 18_000_000:
                    raise ValueError("selected image exceeds 18 megapixels")
                graph = runtime.infer(image, str(payload.get("questionText", "")))
                self._send(200, {"graph": graph, "health": runtime.health()})
            except (ValueError, json.JSONDecodeError, OSError) as error:
                self._send(400, {"error": "invalid_request", "detail": str(error)})
            except Exception as error:  # keep CUDA/service errors visible to UI
                self._send(500, {"error": "inference_failed", "detail": str(error)})

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"[geometry-gpu] {self.address_string()} {format_string % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    args = parser.parse_args()
    runtime = GPUInferenceRuntime(args.checkpoint, args.cuda_memory_fraction)
    print(json.dumps({"service": "geometry-gpu", **runtime.health()}, ensure_ascii=False), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(runtime))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
