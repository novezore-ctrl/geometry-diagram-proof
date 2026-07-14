import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "static",
  plugins: [react()],
  base: "./",
  publicDir: "../public",
  build: {
    outDir: "../dist-static",
    emptyOutDir: true,
  },
});
