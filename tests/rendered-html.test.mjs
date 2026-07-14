import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("server-renders the geometry workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>几何图校对器<\/title>/);
  assert.match(html, /第一版只解决三件事/);
  assert.match(html, /开始识别/);
  assert.match(html, /点 \d+/);
  assert.match(html, /人工校正/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/);
});

test("ships the local PWA shell and no starter preview", async () => {
  const manifest = await readFile(new URL("../public/manifest.webmanifest", import.meta.url), "utf8");
  const serviceWorker = await readFile(new URL("../public/sw.js", import.meta.url), "utf8");
  assert.match(manifest, /几何图校对器/);
  assert.match(manifest, /standalone/);
  assert.match(serviceWorker, /caches\.open/);
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});
