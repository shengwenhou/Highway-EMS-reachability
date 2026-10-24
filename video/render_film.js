// Render video/film.html to numbered JPEG frames with headless Chromium, in parallel workers.
// usage: node video/render_film.js <frames dir> [workers]   (serve the video/ folder on :8766 first)
// Frames already on disk are skipped, so an interrupted render resumes where it stopped.
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const FPS = 30;
(async () => {
  const [dir, workers = "3"] = process.argv.slice(2);
  fs.mkdirSync(dir, { recursive: true });
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const open = async () => {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    page.on("pageerror", (e) => console.error("page error:", e.message));
    await page.goto("http://127.0.0.1:8766/film.html");
    await page.waitForFunction(() => window.ready === true, null, { timeout: 180000 });
    return page;
  };
  const first = await open();
  const n = Math.round((await first.evaluate(() => window.TOTAL)) * FPS);
  const pages = [first, ...(await Promise.all(Array.from({ length: +workers - 1 }, open)))];
  let next = 0, done = 0;
  const t0 = Date.now();
  await Promise.all(pages.map(async (page) => {
    while (next < n) {
      const i = next++;
      const f = path.join(dir, `f${String(i).padStart(5, "0")}.jpg`);
      if (fs.existsSync(f)) { done++; continue; }
      await page.evaluate((t) => window.frame(t), i / FPS);
      await page.screenshot({ path: f + ".tmp", type: "jpeg", quality: 92 });
      fs.renameSync(f + ".tmp", f);
      if (++done % 300 === 0) console.log(`${done}/${n} frames, ${((Date.now() - t0) / 60000).toFixed(1)} min`);
    }
  }));
  await browser.close();
  console.log(`done: ${n} frames`);
})();
