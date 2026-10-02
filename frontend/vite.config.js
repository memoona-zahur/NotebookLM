import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands in ../static, which is what FastAPI already serves, so the
// production image needs no second asset path and `docker compose up` runs the
// same files a local `npm run build` produces.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    // Keep the emitted filenames predictable so index.html stays readable and a
    // stale cached bundle is easy to reason about.
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: {
    port: 5173,
    // `npm run dev` proxies the API so the frontend talks to a real backend on
    // 8000 instead of needing a mock.
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  test: {
    // The components render into a DOM, so the environment is jsdom rather
    // than node. CSS imports are stubbed: the build handles those, and
    // asserting on class names is not the point of these tests.
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.js",
    css: false,
    restoreMocks: true,
  },
});