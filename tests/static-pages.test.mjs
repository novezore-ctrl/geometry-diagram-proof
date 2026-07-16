import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectBase = new URL("https://example.test/geometry-diagram-proof/");

test("PWA shell stays inside a GitHub project Pages scope", async () => {
  const indexHtml = await readFile(new URL("../dist-static/index.html", import.meta.url), "utf8");
  assert.match(indexHtml, /(?:src|href)="\.\/assets\//);
  assert.doesNotMatch(indexHtml, /(?:src|href)="\/assets\//);

  const manifest = JSON.parse(await readFile(new URL("../dist-static/manifest.webmanifest", import.meta.url), "utf8"));
  assert.equal(manifest.id, "./");
  assert.equal(manifest.start_url, "./");
  assert.equal(manifest.scope, "./");
  assert.equal(new URL(manifest.start_url, projectBase).pathname, "/geometry-diagram-proof/");

  const serviceWorker = await readFile(new URL("../dist-static/sw.js", import.meta.url), "utf8");
  assert.match(serviceWorker, /self\.registration\.scope/);
  assert.doesNotMatch(serviceWorker, /\["\/",\s*"\/manifest\.webmanifest"\]/);

  // Dataset-derived weights and photographed textbook fixtures are deliberately
  // not required in the public build. See THIRD_PARTY_NOTICES.md.
  await assert.rejects(access(new URL("../dist-static/models/geometry_unet_pgdp5k_epoch5_320.onnx", import.meta.url)));
  await assert.rejects(access(new URL("../dist-static/figure2-photo.jpg", import.meta.url)));
});

test("runtime assets are resolved from the deployed document path", async () => {
  const inference = await readFile(new URL("../app/mobileInference.ts", import.meta.url), "utf8");
  const workspace = await readFile(new URL("../app/GeometryWorkspace.tsx", import.meta.url), "utf8");
  assert.match(inference, /new URL\(MODEL_PATH, document\.baseURI\)/);
  assert.doesNotMatch(inference, /MODEL_URL\s*=\s*["']\/models\//);
  assert.match(inference, /classical_browser_fallback/);
  assert.match(workspace, /公开演示：由已核对拓扑重新绘制/);
  assert.doesNotMatch(workspace, /figure2-photo\.jpg/);
});
