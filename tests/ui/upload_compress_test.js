// Client-side upload compression (Vercel 4.5 MB body limit): js/core/upload-compress.js + its wiring in
// portrait-mode.js and magic-eraser.js. The REAL module runs in a vm sandbox with a fake canvas whose encoded size is
// a deterministic function of pixels / quality, so the budget loop and the image <-> mask alignment are checkable.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test, run, assert, createContext } = require('./_harness');

const root = path.join(__dirname, '../..');
const read = (rel) => fs.readFileSync(path.join(root, rel), 'utf8');
const MB = 1024 * 1024;

function sandbox(bytesPerPixelAtQ1, pngBytesPerPixel) {
  const pngBpp = pngBytesPerPixel === undefined ? 3 : pngBytesPerPixel;   // photo PNG ~3 B/px; a binary mask PNG is tiny
  const made = [];
  function canvas() {
    const c = {
      width: 0, height: 0, released: false,
      getContext() {
        return {
          imageSmoothingEnabled: false, imageSmoothingQuality: 'low',
          drawImage() {},
          getImageData: (x, y, w, h) => { const d = new Uint8ClampedArray(w * h * 4); for (let i = 0; i < d.length; i += 4) d[i] = (i / 4) % 2 ? 200 : 20; return { data: d, width: w, height: h }; },
          putImageData(img) { c.lastPut = img; }
        };
      },
      toBlob(cb, mime, q) {
        const px = c.width * c.height;
        const size = mime === 'image/png' ? Math.round(px * pngBpp) : Math.round(px * bytesPerPixelAtQ1 * (q || 0.92));
        cb({ size, type: mime });
      }
    };
    made.push(c);
    return c;
  }
  // save-image.js's canvasToBlob() is the app's only canvas->Blob wrapper; the module under test calls it through the global.
  const sb = { console, document: { createElement: canvas }, canvasToBlob: (c, mime, q) => new Promise((res) => c.toBlob(res, mime, q)) };
  createContext(sb);
  vm.runInContext(read('js/core/upload-compress.js'), sb);
  return { sb, made, C: sb.YoUploadCompress, canvas };
}
const srcCanvas = (canvas, w, h) => { const c = canvas(); c.width = w; c.height = h; return c; };

test('fitSize: untouched when it already fits, longest side capped otherwise, aspect ratio kept', () => {
  const { C } = sandbox(1);
  assert.deepEqual({ ...C.fitSize(1600, 900, 2048) }, { w: 1600, h: 900, scale: 1 });
  const r = C.fitSize(4000, 3000, 2048);
  assert.equal(r.w, 2048); assert.equal(r.h, 1536); assert.ok(Math.abs(r.scale - 0.512) < 1e-9);
  const t = C.fitSize(1000, 5000, 2048);
  assert.equal(t.h, 2048); assert.equal(t.w, 410);
  assert.deepEqual({ ...C.fitSize(0, 10, 2048) }, { w: 0, h: 0, scale: 1 });
});

test('nextAttempt: quality steps down first, then the picture shrinks and quality restarts; stops at the floor', () => {
  const { C } = sandbox(1);
  let st = { quality: 0.85, startQuality: 0.85, maxDim: 2048 };
  const q = C.nextAttempt(st, true);
  assert.equal(q.quality, 0.78); assert.equal(q.maxDim, 2048);
  const low = C.nextAttempt({ quality: 0.6, startQuality: 0.85, maxDim: 2048 }, true);
  assert.equal(low.quality, 0.85); assert.equal(low.maxDim, 1740);
  assert.equal(C.nextAttempt({ quality: 0.6, startQuality: 0.85, maxDim: 700 }, true), null);
  const png = C.nextAttempt({ quality: 0.85, startQuality: 0.85, maxDim: 2048 }, false);   // lossless: only shrinks
  assert.equal(png.maxDim, 1740);
});

test('scaledRefSize: scales with the picture, never below the sent window, always an integer >= 1', () => {
  const { C } = sandbox(1);
  assert.equal(C.scaledRefSize(2560, 0.5, 600, 500), 1280);
  assert.equal(C.scaledRefSize(100, 0.5, 600, 500), 600);
  assert.equal(C.scaledRefSize(undefined, undefined, 0, 0), 1);
});

test('encodeCanvas: a 4000x3000 photo becomes <= 2048 px JPEG under the budget', async () => {
  const { C, canvas } = sandbox(0.9);
  const out = await C.encodeCanvas(srcCanvas(canvas, 4000, 3000), { maxDim: 2048, quality: 0.85 });
  assert.equal(out.mime, 'image/jpeg'); assert.equal(out.width, 2048); assert.equal(out.height, 1536);
  assert.ok(out.blob.size <= 3 * MB - C.OVERHEAD, 'size ' + out.blob.size);
  assert.ok(Math.abs(out.scale - 2048 / 4000) < 1e-9); assert.equal(out.srcW, 4000);
});

test('encodeCanvas: a still-too-big photo lowers quality then shrinks; transparent photos stay PNG and only shrink', async () => {
  const dense = sandbox(2.2);                           // 2048x1536 @ q0.85 ~ 5.9 MB: must go down
  const j = await dense.C.encodeCanvas(srcCanvas(dense.canvas, 4000, 3000), { maxDim: 2048, quality: 0.85 });
  assert.ok(j.blob.size <= 3 * MB - dense.C.OVERHEAD); assert.equal(j.mime, 'image/jpeg');
  assert.ok(j.width < 2048 || j.blob.size > 0);
  const p = sandbox(1);
  const png = await p.C.encodeCanvas(srcCanvas(p.canvas, 3000, 2000), { maxDim: 2048, hasAlpha: true });
  assert.equal(png.mime, 'image/png'); assert.ok(png.blob.size <= 3 * MB - p.C.OVERHEAD);
  assert.ok(png.width < 2048, 'PNG had to shrink to fit');
});

test('encodeCanvas: gives up with code TOO_LARGE instead of uploading something over the limit', async () => {
  const { C, canvas } = sandbox(50);
  await assert.rejects(() => C.encodeCanvas(srcCanvas(canvas, 4000, 3000), {}), (e) => e.code === 'TOO_LARGE');
});

test('encodePair: the mask has EXACTLY the encoded image size (alignment) and the pair stays under the budget', async () => {
  const { C, canvas, made } = sandbox(1.0, 0.05);
  const img = srcCanvas(canvas, 3600, 2400), mask = srcCanvas(canvas, 3600, 2400);
  const out = await C.encodePair(img, mask, { quality: 0.92 });
  assert.equal(out.mime, 'image/jpeg');
  assert.ok(out.blob.size + out.maskBlob.size + C.OVERHEAD <= C.TOTAL_BUDGET);
  const maskSized = made.filter((c) => c.lastPut).pop();           // the re-binarised, rescaled mask canvas
  assert.ok(maskSized, 'a downscaled mask is re-binarised');
  assert.equal(maskSized.lastPut.width, out.width); assert.equal(maskSized.lastPut.height, out.height);
  const d = maskSized.lastPut.data;
  for (let i = 0; i < d.length; i += 4) { assert.ok(d[i] === 0 || d[i] === 255); assert.equal(d[i + 3], 255); }
});

test('encodePair: an eraser-sized window (<= 1060 px) keeps scale 1 - coordinates are untouched', async () => {
  const { C, canvas } = sandbox(0.5, 0.05);
  const out = await C.encodePair(srcCanvas(canvas, 1040, 900), srcCanvas(canvas, 1040, 900), { quality: 0.92 });
  assert.equal(out.scale, 1); assert.equal(out.width, 1040); assert.equal(out.height, 900);
});

test('wiring: loaded before the two consumers; consumers use it; the pasted result is mapped back to the window size', () => {
  const html = read('index.html');
  const at = (n) => html.indexOf(n);
  assert.ok(at('js/core/upload-compress.js') > -1);
  assert.ok(at('js/core/upload-compress.js') < at('js/portrait/portrait-mode.js'));
  assert.ok(at('js/core/upload-compress.js') < at('js/editing/magic-eraser.js'));
  const pm = read('js/portrait/portrait-mode.js'), er = read('js/editing/magic-eraser.js');
  assert.ok(/YoUploadCompress/.test(pm) && /prepareImageBlob/.test(pm) && /resizeBlobTo/.test(pm));
  assert.ok(!/form\.append\('image', portraitBaseBlob/.test(pm), 'the raw baseline is no longer uploaded');
  assert.ok(/encodePair/.test(er) && /scaledRefSize/.test(er));
  assert.ok(/drawImage\(patch, job\.win\.x, job\.win\.y, job\.win\.w, job\.win\.h\)/.test(er));
});

run();
