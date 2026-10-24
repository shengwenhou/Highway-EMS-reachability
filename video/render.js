// Render video/scene.html frame by frame with headless Chromium, then encode with ffmpeg.
// usage: node video/render.js <mode> <seconds> <out.mp4>   (serve the video/ folder on :8766 first)
const { chromium } = require("playwright");
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

(async () => {
  const [mode = "real", seconds = "8", out = "out.mp4"] = process.argv.slice(2);
  const fps = 30, n = Math.round(+seconds * fps);
  const dir = fs.mkdtempSync(path.join(require("os").tmpdir(), "frames-"));
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(`http://127.0.0.1:8766/scene.html?mode=${mode}`);
  await page.waitForFunction(() => window.ready === true, null, { timeout: 120000 });
  for (let i = 0; i < n; i++) {
    await page.evaluate((t) => window.frame(t), i / (n - 1));
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(5, "0")}.jpg`), type: "jpeg", quality: 92 });
    if (i % 60 === 0) console.log(mode, i, "/", n);
  }
  await browser.close();
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(fps), "-i", path.join(dir, "f%05d.jpg"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out]);
  fs.rmSync(dir, { recursive: true });
  console.log("wrote", out);
})();
