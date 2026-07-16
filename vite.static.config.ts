import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const PUBLIC_SHELL = ["favicon.svg", "manifest.webmanifest", "sw.js"];

function copyPublicShell(): Plugin {
  return {
    name: "copy-public-shell-without-private-assets",
    apply: "build",
    async closeBundle() {
      const project = process.cwd();
      const output = resolve(project, "dist-static");
      await mkdir(output, { recursive: true });
      await Promise.all(PUBLIC_SHELL.map((file) => copyFile(
        resolve(project, "public", file),
        resolve(output, file),
      )));
    },
  };
}

export default defineConfig({
  root: "static",
  plugins: [react(), copyPublicShell()],
  base: "./",
  // Only the explicit shell files above enter the public Pages artifact.
  // Locally ignored textbook photos and dataset-derived weights must never be
  // copied just because they happen to exist on the developer machine.
  publicDir: false,
  build: {
    outDir: "../dist-static",
    emptyOutDir: true,
  },
});
