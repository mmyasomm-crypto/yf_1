// upload-compress.js - keeps every request to the AI backend safely below Vercel's 4.5 MB body limit.
//
// The server (FastAPI on Vercel) rejects any request body over 4.5 MB with a plain-text 413 that never reaches our
// code, so the photo is resized / re-encoded HERE, before upload:
//   * longest side  <= MAX_DIM (2048 px)
//   * photos        -> JPEG, quality 0.85 (alpha-free); transparent photos stay PNG (JPEG would flatten them)
//   * whole upload  <  TOTAL_BUDGET (3 MB) - image + mask + multipart overhead, i.e. 1.5 MB under the platform limit
//   * an image that is already small enough (<= MAX_DIM and <= budget) is sent UNTOUCHED: no needless lossy pass
//
// Alignment contract (Magic Eraser): the mask is always rasterised to EXACTLY the width/height of the encoded image
// (encodePair), never to an independently rounded size, and the caller scales ref_size by the same factor. The
// result the server returns has the encoded size, so callers map it back with the returned `scale`.
//
// Pure helpers (fitSize, nextAttempt, scaledRefSize) take plain numbers and are unit-tested from Node.
// Global: window.YoUploadCompress.
(function (root) {
  'use strict';

  var MAX_DIM = 2048;
  var QUALITY = 0.85;
  var MIN_QUALITY = 0.6;
  var QUALITY_STEP = 0.07;
  var DIM_SHRINK = 0.85;                       // each failed round shrinks the longest side by 15 %
  var MIN_DIM = 640;                           // never go below this (the models work on 512 / 518 px anyway)
  var TOTAL_BUDGET = 3 * 1024 * 1024;          // whole multipart body
  var OVERHEAD = 64 * 1024;                    // multipart boundaries, field names, small form fields
  var MASK_THRESHOLD = 63;                     // downscaled mask: >= 25 % coverage counts as painted (never drops thin strokes)

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  // Target size for a w x h image whose longest side may not exceed maxDim. scale is exactly 1 when nothing changes.
  function fitSize(w, h, maxDim) {
    if (!isNum(w) || !isNum(h) || w <= 0 || h <= 0) return { w: 0, h: 0, scale: 1 };
    var longest = Math.max(w, h);
    if (!isNum(maxDim) || maxDim <= 0 || longest <= maxDim) return { w: Math.round(w), h: Math.round(h), scale: 1 };
    var s = maxDim / longest;
    return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)), scale: s };
  }

  // What to try after an encode that was still over budget: lower the JPEG quality first, then shrink the picture and
  // start again from the top quality. PNG (lossless) has no quality, so it only shrinks. null = nothing left to try.
  function nextAttempt(state, lossy) {
    if (lossy && state.quality - QUALITY_STEP >= MIN_QUALITY - 1e-9) {
      return { quality: Math.round((state.quality - QUALITY_STEP) * 100) / 100, maxDim: state.maxDim };
    }
    var smaller = Math.floor(state.maxDim * DIM_SHRINK);
    if (smaller < MIN_DIM) return null;
    return { quality: lossy ? state.startQuality : state.quality, maxDim: smaller };
  }

  // ref_size (longest side of the full photo, see backend /api/erase) expressed in the scaled window's pixels. Never
  // smaller than the scaled window itself, always an integer >= 1.
  function scaledRefSize(refSize, scale, sentW, sentH) {
    var r = Math.round((isNum(refSize) ? refSize : 0) * (isNum(scale) && scale > 0 ? scale : 1));
    return Math.max(1, r, Math.round(sentW || 0), Math.round(sentH || 0));
  }

  // ---------------------------------------------------------------- canvas helpers
  function newCanvas(w, h) {
    var c = root.document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }
  function release(c) { if (c) { c.width = 0; c.height = 0; } }

  // The app's one canvas -> Blob wrapper (js/ui/save-image.js); this file never calls canvas.toBlob() itself.
  function toBlob(canvas, mime, quality) {
    if (typeof root.canvasToBlob !== 'function') return Promise.reject(new Error('Could not encode the image'));
    return root.canvasToBlob(canvas, mime, quality);
  }

  // `source` canvas drawn into a fresh w x h canvas (high-quality resampling). Returns the SAME canvas when no resize is needed.
  function drawScaled(source, w, h) {
    if (source.width === w && source.height === h) return source;
    var c = newCanvas(w, h);
    var cx = c.getContext('2d');
    cx.imageSmoothingEnabled = true;
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(source, 0, 0, source.width, source.height, 0, 0, w, h);
    return c;
  }

  // ---------------------------------------------------------------- encoders
  // Encode a canvas under `budget` bytes. opts: { maxDim, quality, budget, hasAlpha }.
  // Resolves { blob, width, height, scale, srcW, srcH, mime, name }. scale = sent size / source size (1 = same pixels).
  async function encodeCanvas(canvas, opts) {
    opts = opts || {};
    var budget = opts.budget > 0 ? opts.budget : TOTAL_BUDGET - OVERHEAD;
    var lossy = !opts.hasAlpha;
    var mime = lossy ? 'image/jpeg' : 'image/png';
    var state = { maxDim: opts.maxDim > 0 ? opts.maxDim : MAX_DIM, quality: opts.quality > 0 ? opts.quality : QUALITY };
    state.startQuality = state.quality;
    var srcW = canvas.width, srcH = canvas.height;
    var last = null;
    for (var guard = 0; guard < 24; guard++) {
      var size = fitSize(srcW, srcH, state.maxDim);
      var work = drawScaled(canvas, size.w, size.h);
      try {
        var blob = await toBlob(work, mime, lossy ? state.quality : undefined);
        last = { blob: blob, width: size.w, height: size.h, scale: size.w / srcW, srcW: srcW, srcH: srcH, mime: mime,
                 name: lossy ? 'image.jpg' : 'image.png' };
      } finally {
        if (work !== canvas) release(work);
      }
      if (last.blob.size <= budget) return last;
      var next = nextAttempt(state, lossy);
      if (!next) break;
      state.quality = next.quality; state.maxDim = next.maxDim;
    }
    var tooBig = new Error('The photo could not be compressed enough to upload');
    tooBig.code = 'TOO_LARGE';
    throw tooBig;
  }

  // Binary mask (0 / 255, opaque) rasterised to EXACTLY w x h, as PNG. A mask that is already w x h is only re-encoded.
  async function encodeMask(maskCanvas, w, h) {
    var work = drawScaled(maskCanvas, w, h);
    var scaled = work !== maskCanvas;
    try {
      if (scaled) {                              // resampling blurs the edges: back to hard 0 / 255
        var cx = work.getContext('2d');
        var img = cx.getImageData(0, 0, w, h);
        var d = img.data;
        for (var i = 0; i < d.length; i += 4) {
          var v = d[i] > MASK_THRESHOLD ? 255 : 0;
          d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
        }
        cx.putImageData(img, 0, 0);
      }
      return await toBlob(work, 'image/png');
    } finally {
      if (scaled) release(work);
    }
  }

  // Image + mask for /api/erase as one unit under TOTAL_BUDGET. The mask has the encoded image's exact size; if the
  // pair is still too big the picture shrinks one more step and BOTH are rebuilt together (they can never drift apart).
  async function encodePair(imageCanvas, maskCanvas, opts) {
    opts = opts || {};
    var total = opts.totalBudget > 0 ? opts.totalBudget : TOTAL_BUDGET;
    var maxDim = opts.maxDim > 0 ? opts.maxDim : MAX_DIM;
    for (var attempt = 0; attempt < 6; attempt++) {
      var img = await encodeCanvas(imageCanvas, { maxDim: maxDim, quality: opts.quality, budget: total - OVERHEAD - 128 * 1024 });
      var mask = await encodeMask(maskCanvas, img.width, img.height);
      if (img.blob.size + mask.size + OVERHEAD <= total) {
        img.maskBlob = mask;
        return img;
      }
      maxDim = Math.floor(Math.max(img.width, img.height) * DIM_SHRINK);
      if (maxDim < MIN_DIM) break;
    }
    var tooBig = new Error('The photo could not be compressed enough to upload');
    tooBig.code = 'TOO_LARGE';
    throw tooBig;
  }

  // ---------------------------------------------------------------- Blob entry point (portrait)
  function decodeBlob(blob) {
    if (typeof root.createImageBitmap === 'function') {
      return root.createImageBitmap(blob).catch(function () { return decodeWithImage(blob); });
    }
    return decodeWithImage(blob);
  }
  function decodeWithImage(blob) {
    return new Promise(function (resolve, reject) {
      var url = root.URL.createObjectURL(blob);
      var im = new root.Image();
      im.onload = function () { root.URL.revokeObjectURL(url); resolve(im); };
      im.onerror = function () { root.URL.revokeObjectURL(url); reject(new Error('Could not read the image')); };
      im.src = url;
    });
  }

  // Blob of a photo -> upload-ready blob. Resolves the encodeCanvas() shape; when the original already fits it is
  // returned as-is (scale 1, same blob, same type) so small photos are never degraded.
  //   opts: { maxDim, quality, budget, hasAlpha }
  async function prepareImageBlob(blob, opts) {
    opts = opts || {};
    var budget = opts.budget > 0 ? opts.budget : TOTAL_BUDGET - OVERHEAD;
    var maxDim = opts.maxDim > 0 ? opts.maxDim : MAX_DIM;
    var bmp = await decodeBlob(blob);
    var w = bmp.width, h = bmp.height;
    try {
      if (blob.size <= budget && Math.max(w, h) <= maxDim) {
        var t = blob.type === 'image/jpeg' ? 'image/jpeg' : 'image/png';
        return { blob: blob, width: w, height: h, scale: 1, srcW: w, srcH: h, mime: t, name: t === 'image/jpeg' ? 'image.jpg' : 'image.png' };
      }
      var c = newCanvas(w, h);
      try {
        c.getContext('2d').drawImage(bmp, 0, 0);
        return await encodeCanvas(c, opts);
      } finally { release(c); }
    } finally {
      try { if (bmp.close) bmp.close(); } catch (e) { /* ImageBitmap only */ }
    }
  }

  // A server result (blob) -> PNG blob of exactly w x h. Used to bring a portrait result back to the pixel size of the
  // working image when the upload was downscaled, so crop box / frame / overlays / history keep their geometry.
  async function resizeBlobTo(blob, w, h) {
    var bmp = await decodeBlob(blob);
    try {
      if (bmp.width === w && bmp.height === h) return blob;
      var c = newCanvas(w, h);
      try {
        var cx = c.getContext('2d');
        cx.imageSmoothingEnabled = true;
        cx.imageSmoothingQuality = 'high';
        cx.drawImage(bmp, 0, 0, bmp.width, bmp.height, 0, 0, w, h);
        return await toBlob(c, 'image/png');
      } finally { release(c); }
    } finally {
      try { if (bmp.close) bmp.close(); } catch (e) { /* ImageBitmap only */ }
    }
  }

  var api = {
    MAX_DIM: MAX_DIM, QUALITY: QUALITY, TOTAL_BUDGET: TOTAL_BUDGET, OVERHEAD: OVERHEAD,
    fitSize: fitSize, nextAttempt: nextAttempt, scaledRefSize: scaledRefSize,
    encodeCanvas: encodeCanvas, encodeMask: encodeMask, encodePair: encodePair,
    prepareImageBlob: prepareImageBlob, resizeBlobTo: resizeBlobTo
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.YoUploadCompress = api;
})(typeof window !== 'undefined' ? window : globalThis);
