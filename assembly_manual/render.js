// Renders pages/*.html to preview/page_NN.png and merges them into placebo-booklet.pdf (A4).
// Usage: npm i playwright pdf-lib && npx playwright install chromium && node render.js
const { chromium } = require("playwright");
const { PDFDocument } = require("pdf-lib");
const fs = require("fs"), path = require("path");
const W = 794, H = 1123; // A4 at 96 dpi
(async () => {
  const dir = __dirname;
  const pages = fs.readdirSync(path.join(dir, "pages")).filter(f => f.endsWith(".html")).sort();
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  fs.mkdirSync(path.join(dir, "preview"), { recursive: true });
  const merged = await PDFDocument.create();
  for (let i = 0; i < pages.length; i++) {
    await page.goto("file://" + path.join(dir, "pages", pages[i]));
    await page.waitForLoadState("networkidle");
    await page.evaluate(() => document.fonts.ready);
    const n = String(i + 1).padStart(2, "0");
    await page.screenshot({ path: path.join(dir, "preview", `page_${n}.png`), clip: { x: 0, y: 0, width: W, height: H } });
    const buf = await page.pdf({ width: `${W}px`, height: `${H}px`, printBackground: true, pageRanges: "1" });
    const src = await PDFDocument.load(buf);
    const [p] = await merged.copyPages(src, [0]);
    merged.addPage(p);
    console.log("rendered", pages[i]);
  }
  merged.setTitle("PLACEBÖ assembly instructions – PlaceboGPT hardware edition");
  merged.setAuthor("PharmaTools.AI");
  fs.writeFileSync(path.join(dir, "placebo-booklet.pdf"), await merged.save());
  await browser.close();
  console.log("wrote placebo-booklet.pdf");
})();
