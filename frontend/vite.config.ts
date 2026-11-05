import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { cpSync, existsSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { resolve, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
const demoMode = process.env.VITE_DEMO_MODE === "true";

function demoBuildPlugin(): Plugin {
  return {
    name: "saer-demo-static-output",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace("</head>", '  <meta name="robots" content="noindex,nofollow" />\n  </head>');
    },
    closeBundle() {
      const outDir = resolve(frontendRoot, "dist-demo");
      const sourceDir = resolve(frontendRoot, "src");
      const references = new Set<string>(["/index.html", "/robots.txt"]);
      const pathPattern = /\/(?:ui|assets|fonts|brands|demo)\/[A-Za-z0-9_./-]+\.(?:png|jpe?g|webp|svg|ttf|json)/gi;
      const activeSources = [
        "App.tsx", "ReferenceApp.tsx", "LoginPage.tsx", "config.ts", "style.css",
        "hamoud-live-original.ts", "hamoud-live-original.css", "hamoud-live-scoped.css", "hamoud-live-controls.css",
      ];
      for (const name of activeSources) {
        const path = resolve(sourceDir, name);
        if (!existsSync(path)) continue;
        for (const match of readFileSync(path, "utf8").matchAll(pathPattern)) references.add(match[0]);
      }
      const indexSource = readFileSync(resolve(frontendRoot, "index.html"), "utf8");
      for (const match of indexSource.matchAll(pathPattern)) references.add(match[0]);
      const builtIndex = readFileSync(resolve(outDir, "index.html"), "utf8");
      for (const match of builtIndex.matchAll(/(?:src|href)=["'](\/assets\/[^"']+\.(?:js|css))["']/gi)) references.add(match[1]);
      const layouts = JSON.parse(readFileSync(resolve(sourceDir, "data", "reference-layout.json"), "utf8"));
      for (const page of ["sell", "details", "budget", "compare"]) {
        const layout = layouts[page];
        if (layout?.background) references.add(layout.background);
        for (const part of Object.values(layout?.parts || {}) as { src?: string }[]) if (part.src) references.add(part.src);
      }
      references.delete("/ui/details/form.png");
      for (const path of ["/ui/saer-header.png", "/ui/saer-logo.png", "/assets/road-cars/car1.png", "/assets/road-cars/car2.png", "/assets/road-cars/car3.png", "/assets/road-cars/car4.png", "/assets/road-cars/car5.png", "/fonts/Cairo.ttf"]) references.add(path);
      references.add("/ui/skyline.png");
      references.add("/fonts/OFL.txt");
      references.add("/demo/placeholder.svg");
      references.add("/ui/login/login-reference.jpg");
      const demoJsonFiles = new Set([
        "/demo/listings.json", "/demo/meta.json", "/demo/pricing-options.json",
        "/demo/recommendations.json", "/demo/market.json",
      ]);
      const removeUnused = (directory: string) => {
        for (const entry of readdirSync(directory)) {
          const path = resolve(directory, entry);
          if (statSync(path).isDirectory()) {
            removeUnused(path);
            if (readdirSync(path).length === 0) rmSync(path, { recursive: true, force: true });
            continue;
          }
          const key = "/" + relative(outDir, path).split(sep).join("/");
          const isRuntimeChunk = /^\/assets\/[^/]+\.(?:js|css)$/i.test(key);
          const demoPayload = key.startsWith("/demo/img/") || demoJsonFiles.has(key);
          const required = references.has(key) || demoPayload || isRuntimeChunk || ["/_headers", "/_redirects", "/robots.txt", "/index.html"].includes(key);
          if (!required || /(?:^|\/)mock(?:\/|$)/i.test(key) || /\.map$/i.test(key) || /\.env(?:\.|$)/i.test(key)) rmSync(path, { force: true });
        }
      };
      removeUnused(outDir);
      for (const name of ["_headers", "_redirects", "robots.txt"]) cpSync(resolve(frontendRoot, name), resolve(outDir, name));
    },
  };
}

function removeDemoPayloadFromRegularBuild(): Plugin {
  return {
    name: "exclude-static-demo-payload-from-regular-build",
    apply: "build",
    closeBundle() {
      if (demoMode) return;
      rmSync(resolve(frontendRoot, "dist", "demo"), { recursive: true, force: true });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(demoMode ? [demoBuildPlugin()] : [removeDemoPayloadFromRegularBuild()])],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: demoMode ? { outDir: "dist-demo", emptyOutDir: true, sourcemap: false } : undefined,
});
