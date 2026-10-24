// 오프라인 렌더 드라이버: 헤드리스 Chromium에서 씬을 띄우고 프레임을 한 장씩 렌더해 PNG로 저장한다.
//
//   node render.mjs <job.json>
//
// job.json: { model, width, height, background, camera, lights, frames: [{bones, blendshapes, dt}], outDir }
// 네트워크는 쓰지 않는다. http://render.local/ 요청을 로컬 파일로 응답한다.
import { chromium } from "playwright";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const job = JSON.parse(await readFile(process.argv[2], "utf8"));
await mkdir(job.outDir, { recursive: true });

// GPU에 따라 결과가 달라지지 않도록 소프트웨어 GL(SwiftShader)을 고정한다.
const browser = await chromium.launch({
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: job.width, height: job.height } });
page.on("console", (msg) => { if (msg.type() === "error") console.error("[page]", msg.text()); });
page.on("pageerror", (err) => console.error("[page]", err.message));

const files = {
  "/index.html": { file: path.join(here, "src", "index.html"), type: "text/html" },
  "/scene.js": { file: path.join(here, "dist", "scene.js"), type: "text/javascript" },
  "/model.vrm": { file: path.resolve(job.model), type: "application/octet-stream" },
};
await page.route("http://render.local/**", async (route) => {
  const entry = files[new URL(route.request().url()).pathname];
  if (!entry) return route.fulfill({ status: 404 });
  return route.fulfill({ status: 200, contentType: entry.type, body: await readFile(entry.file) });
});
await page.goto("http://render.local/index.html");

const info = await page.evaluate((cfg) => AvatarScene.init(cfg), {
  width: job.width, height: job.height, modelUrl: "http://render.local/model.vrm",
  background: job.background, camera: job.camera, lights: job.lights, style: job.style,
});
console.error(`model: ${info.humanBones.length} bones, ${info.expressions.length} expressions, style ${JSON.stringify(info.style)}`);

for (let i = 0; i < job.frames.length; i++) {
  const missing = await page.evaluate((f) => AvatarScene.applyFrame(f), job.frames[i]);
  if (missing.length && i === 0) console.error(`모델에 없는 표정: ${missing.join(", ")}`);
  const dataUrl = await page.evaluate(() => AvatarScene.renderFrame());
  await writeFile(path.join(job.outDir, `${String(i).padStart(6, "0")}.png`), Buffer.from(dataUrl.split(",")[1], "base64"));
}
await browser.close();
