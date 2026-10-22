// ============================================================
// magic-eraser.js — Magic Eraser (LaMa inpainting via FastAPI)
// ------------------------------------------------------------
// Single-canvas design: there is NO secondary preview/stage under the
// workspace. Everything happens on the main photo area at the top:
//
//   * the brush paints a translucent mask layer (#me-paint-layer) that is
//     pinned to the *rendered photo rectangle* of #mainImg, so strokes can
//     never land in the black letterbox margins;
//   * "Remove with AI" posts the photo + mask to /api/erase and commits the
//     result straight into the main editor (history / undo keep working);
//   * "Hold to Compare" temporarily shows the pre-removal photo on the main
//     canvas while the button is pressed;
//   * the controls live in one compact row directly below the canvas.
//
// Stroke state is kept as a vector list (not pixel snapshots), so undo /
// redo / clear are exact, tiny in memory and survive layer resizes.
// ============================================================

window.MagicEraser = (function () {
  'use strict';

  // ==== CONFIG ====
  // Relative path: js/config/api-client.js resolves it against the page origin, so the backend
  // port never matters, whatever uvicorn is started with. Timeout + friendly errors live there too.
  const INPAINT_PATH = '/api/erase';
  const BRUSH_DEFAULT = 40;        // brush diameter in on-screen CSS px
  const MASK_TINT = 'rgb(99,102,241)';

  // ==== DOM ====
  let mainImg = null, container = null, layer = null, cursorEl = null;

  // ==== State ====
  let ctxOv = null;                 // visible tinted mask (the paint layer)
  let maskCanvas = null;            // off-screen binary mask (white = remove)
  let ctxMask = null;
  let img = null;                   // decoded source image
  let origCanvas = null;            // full-resolution source, uploaded to the server
  let sourceUrl = null;             // URL currently loaded as the source
  let pendingUrl = null;            // URL being decoded right now
  let compareBeforeUrl = null;      // image as it was before the latest removal
  let bakedKey = null, bakedUrl = null, baking = false;   // overlay-baked source (blob URL) cache
  let brushSize = BRUSH_DEFAULT;    // CSS px on screen
  let active = false;               // eraser tab visible + photo present
  let busy = false;

  // One ordered action log drives undo / redo / clear:
  //   { type: 'stroke', d: canvas-px diameter, pts: [x,y,x,y,...] }  |  { type: 'clear' }
  let actions = [];
  let redoActions = [];
  let current = null;               // stroke being drawn
  let comparing = false;
  let cursorTimer = null;

  const $ = id => document.getElementById(id);
  const tr = (key, fallback) => {
    if (typeof window.t !== 'function') return fallback;
    const v = window.t(key);
    return (v && v !== key) ? v : fallback;
  };

  // ============ INIT ============
  function init() {
    mainImg   = $('mainImg');
    container = $('imgContainer');
    layer     = $('me-paint-layer');
    cursorEl  = $('me-cursor');
    if (!mainImg || !container || !layer) return;

    maskCanvas = document.createElement('canvas');
    ctxMask = maskCanvas.getContext('2d', { willReadFrequently: true });
    ctxOv = layer.getContext('2d');

    const bs = $('me-brush-size');
    if (bs) {
      bs.value = String(brushSize);
      bs.addEventListener('input', e => {
        brushSize = parseInt(e.target.value, 10) || BRUSH_DEFAULT;
        updateCursorSize();
        previewBrushAtCentre();
      });
    }

    layer.addEventListener('pointerdown', onDown);
    layer.addEventListener('pointermove', onMove);
    layer.addEventListener('pointerup', onUp);
    layer.addEventListener('pointercancel', onUp);
    layer.addEventListener('pointerleave', () => { if (!current) hideCursor(); });
    layer.addEventListener('contextmenu', e => e.preventDefault());

    // Hold-to-compare: press & hold shows the "before" photo on the main canvas.
    const cmp = $('me-compare-hold');
    if (cmp) {
      cmp.addEventListener('pointerdown', e => { e.preventDefault(); try { cmp.setPointerCapture(e.pointerId); } catch (x) {} compareStart(); });
      ['pointerup', 'pointercancel', 'lostpointercapture', 'pointerleave'].forEach(ev => cmp.addEventListener(ev, compareEnd));
      cmp.addEventListener('contextmenu', e => e.preventDefault());
      cmp.addEventListener('keydown', e => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); compareStart(); } });
      cmp.addEventListener('keyup', e => { if (e.key === ' ' || e.key === 'Enter') compareEnd(); });
      cmp.addEventListener('blur', compareEnd);
    }
    window.addEventListener('blur', compareEnd);

    // Keep the layer glued to the rendered photo whenever the photo changes.
    mainImg.addEventListener('load', refreshActive);
    if (typeof ResizeObserver === 'function') {
      const ro = new ResizeObserver(() => { if (active) syncLayer(); });
      ro.observe(mainImg); ro.observe(container);
    }
    const onResize = (window.Perf ? Perf.rafThrottle : (f => f))(() => { if (active) syncLayer(); });
    window.addEventListener('resize', onResize, { passive: true });

    // Capture phase: while the eraser has strokes, Ctrl+Z / Ctrl+Y act on the strokes
    // and never reach the global image-history handler.
    document.addEventListener('keydown', onKeyDown, true);

    updateButtons();
  }

  // ============ ACTIVATION ============
  function panelVisible() {
    const p = $('p-eraser');
    return !!(p && p.style.display !== 'none' && p.classList.contains('active'));
  }
  function hasPhoto() {
    return !!(mainImg && mainImg.classList.contains('active') && mainImg.src && mainImg.naturalWidth);
  }
  // Called by navigation.js after every tab change.
  function refreshActive() {
    if (!layer) return;
    const was = active;
    active = panelVisible() && hasPhoto();
    layer.classList.toggle('on', active);
    if (document.body) document.body.classList.toggle('eraser-active', active);   // Y14: CSS hook (replaces :has(.me-paint-layer.on))
    if (!active) { hideCursor(); compareEnd(); }
    if (active) {
      syncImage();
      syncLayer();
    }
    updateButtons();
  }

  // ============ LOAD / SYNC ============
  function getCurrentEditorDataUrl() {
    // R12: the eraser always works on the FULL-RESOLUTION working image with every colour edit baked in - never on
    // the <=1200 px on-screen preview. V24/G8: overlays are NOT baked in any more - the shared commit keeps them as live
    // overlays above the photo, so baking them here would draw them twice. With no edits that is simply
    // originalImage.src. Otherwise the bake is asynchronous (blob URL, no base64): serve a cached result
    // when it matches, else start one bake and let it call loadFromUrl() itself when ready.
    try {
      if (typeof originalImage !== 'undefined' && originalImage && originalImage.src) {
        const w = originalImage.naturalWidth || originalImage.width;
        const h = originalImage.naturalHeight || originalImage.height;
        if (!w || !h) return originalImage.src;
        const edited = typeof YoPipeline !== 'undefined' && YoPipeline.needsBake();
        if (!edited) return originalImage.src;
        const key = originalImage.src + '|' + w + 'x' + h + '|' +
          (typeof YoPipeline !== 'undefined' ? YoPipeline.editSignature() : '');
        if (bakedUrl && bakedKey === key) return bakedUrl;
        if (!baking) {
          baking = true;
          const c = document.createElement('canvas');
          YoPipeline.renderEditedCanvas(originalImage, w, h, c);
          const spec = YoPipeline.encodeSpec();             // V24/G9: lossless working buffer
          canvasToBlobUrl(c, spec.mime, spec.quality).then(res => {
            c.width = 0; c.height = 0;
            const old = bakedUrl;
            bakedUrl = res.url; bakedKey = key;
            loadFromUrl(bakedUrl);
            if (old && old !== bakedUrl && old !== sourceUrl && old !== pendingUrl && old !== compareBeforeUrl) {
              try { URL.revokeObjectURL(old); } catch (e) {}
            }
          }).catch(() => {}).then(() => { baking = false; });
        }
        return null;
      }
    } catch (e) { /* fall through to the on-screen image */ }
    const currentSrc = mainImg && mainImg.src ? mainImg.src : '';
    return currentSrc || null;
  }

  // Make sure the eraser works on the photo that is on screen right now.
  function syncImage() {
    if (!hasPhoto()) return false;
    const url = getCurrentEditorDataUrl();
    if (!url) return false;
    if (url === sourceUrl || url === pendingUrl) return !!img;
    loadFromUrl(url);
    return !!img;
  }
  function ensureCurrentImage() {
    refreshActive();
    return !!img;
  }

  // Y18: `opts.keepStrokes === false` discards the strokes (the mask was consumed by a removal, or a new photo
  // was opened). Otherwise strokes are kept whenever the new source has the SAME pixel size as the one they
  // were painted on - a colour edit, a tab switch or a re-bake changes the URL but never the geometry, and the
  // strokes are stored as vectors in source pixels, so they stay exactly where the user drew them.
  function loadFromUrl(url, opts) {
    const keepRequested = !(opts && opts.keepStrokes === false);
    pendingUrl = url;
    const im = new Image();
    im.crossOrigin = 'anonymous';
    im.onload = () => {
      if (pendingUrl !== url) return;           // superseded
      pendingUrl = null;
      const sameGeometry = !!img && maskCanvas.width === im.naturalWidth && maskCanvas.height === im.naturalHeight;
      const keepStrokes = keepRequested && sameGeometry && (actions.length > 0 || redoActions.length > 0);
      img = im;
      sourceUrl = url;
      // V23/G4: reuse the full-size source canvas instead of orphaning one (~17 MB) per reload. While an
      // inpainting job runs (busy) the job still holds the old canvas as `job.base`, so a fresh one is made.
      if (!origCanvas || busy) origCanvas = document.createElement('canvas');
      origCanvas.width = im.naturalWidth;
      origCanvas.height = im.naturalHeight;
      origCanvas.getContext('2d').drawImage(im, 0, 0);
      // Mask + paint layer are exactly the source resolution (1 mask px == 1 photo px),
      // so what is painted is what the server inpaints - no resampling anywhere.
      maskCanvas.width = layer.width = im.naturalWidth;      // (re)sizing clears both canvases ...
      maskCanvas.height = layer.height = im.naturalHeight;
      if (!keepStrokes) resetStrokes();
      repaint();                                              // ... and the vector strokes are drawn again
      syncLayer();
      updateButtons();
    };
    im.onerror = () => { if (pendingUrl === url) pendingUrl = null; toast(tr('toastLoadFailed', 'Couldn’t load the photo. Please try again.'), 'error'); };
    im.src = url;
  }

  // ============ GEOMETRY ============
  // Rendered picture rectangle inside #mainImg (object-fit: contain), relative to #imgContainer.
  function photoRect() {
    if (!mainImg || !container) return null;
    const box = mainImg.getBoundingClientRect();
    const cbox = container.getBoundingClientRect();
    const iw = mainImg.naturalWidth, ih = mainImg.naturalHeight;
    if (!box.width || !box.height || !iw || !ih) return null;
    const s = Math.min(box.width / iw, box.height / ih);
    const w = iw * s, h = ih * s;
    const left = box.left + (box.width - w) / 2;
    const top = box.top + (box.height - h) / 2;
    return { left: left - cbox.left, top: top - cbox.top, width: w, height: h };
  }
  // Pin the paint layer to exactly the photo area (this is what clips strokes to the photo).
  function syncLayer() {
    if (!layer || !img) return;
    const r = photoRect();
    if (!r) return;
    layer.style.left = r.left + 'px';
    layer.style.top = r.top + 'px';
    layer.style.width = r.width + 'px';
    layer.style.height = r.height + 'px';
    updateCursorSize();
  }
  function canvasPerCssPx() {
    const r = layer.getBoundingClientRect();
    return r.width ? layer.width / r.width : 1;
  }

  // ============ PAINT ============
  function getPoint(e) {
    const r = layer.getBoundingClientRect();
    const sx = layer.width / (r.width || 1), sy = layer.height / (r.height || 1);
    return {
      x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy,
      cx: e.clientX - container.getBoundingClientRect().left,
      cy: e.clientY - container.getBoundingClientRect().top
    };
  }
  function onDown(e) {
    if (!active || !img || busy) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try { layer.setPointerCapture(e.pointerId); } catch (x) {}
    const p = getPoint(e);
    current = { d: brushSize * canvasPerCssPx(), pts: [p.x, p.y] };
    dot(p.x, p.y, current.d);
    moveCursor(p, e.pointerType);
    e.preventDefault();
  }
  function onMove(e) {
    if (!active || !img) return;
    const evs = (current && typeof e.getCoalescedEvents === 'function') ? e.getCoalescedEvents() : null;
    const list = evs && evs.length ? evs : [e];
    for (let i = 0; i < list.length; i++) {
      const p = getPoint(list[i]);
      if (current) {
        const n = current.pts.length;
        segment(current.pts[n - 2], current.pts[n - 1], p.x, p.y, current.d);
        current.pts.push(p.x, p.y);
      }
      if (i === list.length - 1) moveCursor(p, e.pointerType);
    }
  }
  function onUp() {
    if (!current) return;
    actions.push({ type: 'stroke', d: current.d, pts: current.pts });
    redoActions = [];
    current = null;
    updateButtons();
  }
  function dot(x, y, d) {
    [[ctxMask, '#fff'], [ctxOv, MASK_TINT]].forEach(([c, col]) => {
      c.fillStyle = col; c.beginPath(); c.arc(x, y, d / 2, 0, Math.PI * 2); c.fill();
    });
  }
  function segment(x0, y0, x1, y1, d) {
    [[ctxMask, '#fff'], [ctxOv, MASK_TINT]].forEach(([c, col]) => {
      c.strokeStyle = col; c.lineWidth = d; c.lineCap = 'round'; c.lineJoin = 'round';
      c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    });
  }

  // ============ CURSOR ============
  function moveCursor(p, pointerType) {
    if (!cursorEl) return;
    if (pointerType === 'touch') { hideCursor(); return; }
    cursorEl.style.display = 'block';
    cursorEl.style.left = p.cx + 'px';
    cursorEl.style.top = p.cy + 'px';
  }
  function hideCursor() { if (cursorEl) cursorEl.style.display = 'none'; }
  function updateCursorSize() {
    if (!cursorEl) return;
    cursorEl.style.width = brushSize + 'px';
    cursorEl.style.height = brushSize + 'px';
  }
  // Slider feedback: flash the brush ring in the middle of the photo.
  function previewBrushAtCentre() {
    if (!cursorEl || !active) return;
    const r = photoRect();
    if (!r) return;
    cursorEl.style.display = 'block';
    cursorEl.style.left = (r.left + r.width / 2) + 'px';
    cursorEl.style.top = (r.top + r.height / 2) + 'px';
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(hideCursor, 900);
  }

  // ============ HISTORY (vector strokes) ============
  function resetStrokes() { actions = []; redoActions = []; current = null; }
  // Strokes painted since the last Clear (what is currently on the mask).
  function visibleStrokes() {
    const out = [];
    for (let i = actions.length - 1; i >= 0 && actions[i].type !== 'clear'; i--) out.push(actions[i]);
    return out.reverse();
  }
  function repaint() {
    if (!maskCanvas) return;
    ctxMask.fillStyle = '#000';
    ctxMask.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    ctxOv.clearRect(0, 0, layer.width, layer.height);
    visibleStrokes().forEach(st => {
      dot(st.pts[0], st.pts[1], st.d);
      for (let i = 2; i < st.pts.length; i += 2) segment(st.pts[i - 2], st.pts[i - 1], st.pts[i], st.pts[i + 1], st.d);
    });
    if (current && current.pts.length >= 2) {           // a reload in the middle of a stroke must not cut it off
      dot(current.pts[0], current.pts[1], current.d);
      for (let i = 2; i < current.pts.length; i += 2) segment(current.pts[i - 2], current.pts[i - 1], current.pts[i], current.pts[i + 1], current.d);
    }
  }
  function undo() {
    if (current || !actions.length) return;
    redoActions.push(actions.pop());
    repaint(); updateButtons();
  }
  function redo() {
    if (current || !redoActions.length) return;
    actions.push(redoActions.pop());
    repaint(); updateButtons();
  }
  function clearMask() {
    if (!hasAnyMask()) return;
    actions.push({ type: 'clear' });
    redoActions = [];
    repaint(); updateButtons();
  }
  function hasAnyMask() { return !!img && visibleStrokes().length > 0; }
  function canUndo() { return actions.length > 0; }
  function canRedo() { return redoActions.length > 0; }

  function onKeyDown(e) {
    if (!active || busy) return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' && e.target.type !== 'range') return;
    if (e.key === '[') { setBrushSize(Math.max(8, brushSize - 4)); return; }
    if (e.key === ']') { setBrushSize(Math.min(140, brushSize + 4)); return; }
    if (!(e.ctrlKey || e.metaKey)) return;
    const k = (e.key || '').toLowerCase();
    if (k === 'z' && !e.shiftKey && canUndo()) { e.preventDefault(); e.stopImmediatePropagation(); undo(); }
    else if ((k === 'y' || (k === 'z' && e.shiftKey)) && canRedo()) { e.preventDefault(); e.stopImmediatePropagation(); redo(); }
  }
  function setBrushSize(v) {
    brushSize = v;
    const s = $('me-brush-size'); if (s) s.value = String(v);
    updateCursorSize(); previewBrushAtCentre();
  }

  // ============ BUTTON STATE ============
  function updateButtons() {
    const photo = !!img && active;
    const set = (id, disabled) => { const b = $(id); if (b) b.disabled = !!disabled; };
    set('me-undo', !photo || busy || !canUndo());
    set('me-redo', !photo || busy || !canRedo());
    set('me-clear', !photo || busy || !hasAnyMask());
    set('me-apply', !photo || busy || !hasAnyMask());
    set('me-compare-hold', !photo || busy);
    set('me-download', !photo || busy);
    const bs = $('me-brush-size'); if (bs) bs.disabled = !photo || busy;
  }

  // ============ ETA (local, learned from this device's own runs) ============
  // localStorage 'yo_erase_ms' keeps the durations of the last 5 successful erases. Once there are at
  // least 2 samples the island shows " · usually ~N s" (median, so one slow outlier does not skew it).
  const ETA_KEY = 'yo_erase_ms';
  const ETA_KEEP = 5;
  const ETA_MIN_SAMPLES = 2;

  function loadDurations() {
    try {
      const raw = JSON.parse(localStorage.getItem(ETA_KEY) || '[]');
      return Array.isArray(raw) ? raw.filter(n => typeof n === 'number' && isFinite(n) && n > 0).slice(-ETA_KEEP) : [];
    } catch (e) { return []; }                       // private mode / blocked storage / corrupt JSON
  }
  function recordDuration(ms) {
    if (!(ms > 0) || !isFinite(ms)) return;
    try {
      const list = loadDurations();
      list.push(Math.round(ms));
      localStorage.setItem(ETA_KEY, JSON.stringify(list.slice(-ETA_KEEP)));
    } catch (e) { /* storage full or unavailable: ETA is a nicety, never an error */ }
  }
  function medianOf(list) {
    if (!list.length) return 0;
    const a = list.slice().sort((x, y) => x - y);
    const mid = a.length >> 1;
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }
  function etaSuffix() {
    const list = loadDurations();
    if (list.length < ETA_MIN_SAMPLES) return '';
    const secs = Math.max(1, Math.round(medianOf(list) / 1000));
    return ' \u00B7 ' + tr('etaUsually', 'Usually about {s} s').replace('{s}', String(secs));
  }
  // Stage line under the island title: current step + learned ETA.
  function setStage(text) {
    if (typeof updateLoadingSub === 'function') updateLoadingSub(text + etaSuffix());
  }

  // ============ APPLY (LaMa inpainting server) ============
  let eraseCtl = null;                 // AbortController of the erase in flight (null when idle)

  function abortError() { const e = new Error('Aborted'); e.name = 'AbortError'; return e; }

  // User pressed Cancel in the island: stop the request, free the UI right now, tell the user.
  // The canvas, the main editor and the history are untouched - nothing was committed yet.
  function cancelErase() {
    const ctl = eraseCtl;
    if (!ctl || ctl.signal.aborted) return;
    eraseCtl = null;                   // late results compare against this and are dropped
    try { ctl.abort(); } catch (e) {}
    busy = false;
    updateButtons();   // the user asked for this: the modal closing is the feedback, no toast
  }

  async function apply() {
    if (!img || !hasAnyMask() || busy) return;
    busy = true; updateButtons();
    const ctl = new AbortController();
    eraseCtl = ctl;
    // True once this run was cancelled or superseded: every await below re-checks it so a response that
    // arrives late can never touch the canvas, the editor or the history.
    const stale = () => ctl.signal.aborted || eraseCtl !== ctl;
    const guard = () => { if (stale()) throw abortError(); };
    const t0 = performance.now();
    let out = null;
    const loadJob = showBusy(true, cancelErase);     // AILoadingManager ticket: complete() on success, hide() otherwise
    try {
      setStage(tr('stagePreparing', 'Preparing…'));
      const job = await buildInpaintFormData();
      guard();

      setStage(tr('stageWaiting', 'Waiting for the AI server…'));
      const response = await apiRequest(INPAINT_PATH, { method: 'POST', body: job.formData, signal: ctl.signal });
      guard();

      if (!response.ok) throw await apiHttpError(response);

      setStage(tr('stageReceiving', 'Receiving the result…'));
      const contentType = response.headers.get('content-type') || '';
      if (contentType.indexOf('image/') === 0) {
        const resultBlob = await response.blob();
        guard();
        if (job.win.w !== job.fullW || job.win.h !== job.fullH || job.scale !== 1) {
          setStage(tr('stageBlending', 'Blending the result into your photo…'));
          out = await compositeWindow(resultBlob, job);
        } else {
          out = registerBlobUrl(resultBlob);       // server PNG used as-is: no base64 round trip
        }
      } else {
        let payload = null;
        try { payload = await response.json(); } catch (parseError) {
          throw new Error('The inpainting server returned an invalid response');
        }
        guard();
        if (!payload || payload.success !== true || typeof payload.result_image !== 'string' || !payload.result_image) {
          throw new Error(payload && (payload.error || payload.message)
            ? (payload.error || payload.message)
            : 'The inpainting server did not return a result image');
        }
        out = payload.result_image;
        if (out.indexOf('data:') === 0) {        // legacy JSON/base64 reply -> blob URL
          try { out = registerBlobUrl(await (await fetch(out)).blob()); } catch (e) { /* keep data URL */ }
        }
      }
      guard();                                   // last checkpoint before anything is committed

      const elapsedMs = performance.now() - t0;   // server round trip + local encode/blend, excl. committing
      setStage(tr('stageApplying', 'Finishing up…'));
      compareBeforeUrl = sourceUrl;              // what "Hold to Compare" will show
      const committed = out;
      out = null;                                // from here the history owns the blob
      await applyResultToMainEditor(committed);  // result lands on the main canvas
      resetStrokes(); repaint();                 // Y18: the mask was consumed - even a racing reload must not bring it back
      loadFromUrl(getCurrentEditorDataUrl() || committed, { keepStrokes: false });   // fresh source, empty mask, ready for the next pass
      recordDuration(elapsedMs);                 // only runs that really landed feed the next ETA
      if (window.aiLoader) window.aiLoader.complete(loadJob);   // snap to 100 %, pop, hold, fade
    } catch (err) {
      if (err && err.name === 'AbortError') {
        // Cancelled on purpose: not an error, nothing to announce.
      } else if (stale()) {
        // failed after the user cancelled (e.g. the socket died with the abort): stay quiet
      } else if (err && err.code === 'EMPTY_MASK') {            // nothing painted inside the photo: no request was made
        toast(err.message, 'error');
      } else {
        console.error('[MagicEraser] erase failed', err);
        toast(describeError(err), 'error');
      }
    } finally {
      // A result that was produced but never committed (cancelled meanwhile) must not leak its blob.
      if (out && out.indexOf('blob:') === 0) {
        if (typeof releaseTransientImageUrl === 'function') releaseTransientImageUrl(out);   // revoke + drop size record
        else { try { URL.revokeObjectURL(out); } catch (e) {} }
      }
      if (eraseCtl === ctl) {                    // still the active run: normal teardown
        eraseCtl = null;
        busy = false; showBusy(false, null, loadJob); updateButtons();   // no-op after complete(); closes at once on error
      }
      // else: cancelErase() already closed the island and released busy - never hide a NEWER run's island.
    }
  }

  // ---- network helpers (js/config/api-client.js; plain relative fetch if it failed to load) ----
  function apiRequest(path, opts) {
    return window.YoApi ? window.YoApi.request(path, opts) : fetch(path, opts);
  }
  async function apiHttpError(response) {
    if (window.YoApi) return window.YoApi.httpError(response);
    return new Error('The inpainting server returned an error (' + response.status + ')');
  }
  function describeError(err) {
    if (window.YoApi) return window.YoApi.describe(err);
    return (err && err.message) || String(err);
  }

  // The centred loading modal of the main canvas is the only busy indicator. js/ui/ai-loading-manager.js (aiLoader) owns it;
  // the plain showLoading / hideLoading path below only runs if that script failed to load.
  //   showBusy(true, onCancel)        -> returns the manager's ticket (undefined without the manager)
  //   showBusy(false, null, ticket)   -> closes it at once (error / cancel); a no-op after aiLoader.complete(ticket)
  function showBusy(on, onCancel, ticket) {
    if (on) {
      // onCancel makes the modal show its Cancel button. The counter is paced by the median of the last runs (the same
      // numbers as the "usually ~N s" hint); 0 = not enough runs yet.
      const runs = loadDurations();
      const expectedMs = runs.length >= ETA_MIN_SAMPLES ? medianOf(runs) : 0;
      if (window.aiLoader) return window.aiLoader.show('removeObject', { onCancel: onCancel, expectedMs: expectedMs });
      if (typeof showLoading === 'function') showLoading(tr('removingObject', 'Removing object…'), { onCancel: onCancel, progress: { expectedMs: expectedMs } });
      return undefined;
    }
    if (window.aiLoader) window.aiLoader.hide(ticket);
    else if (typeof hideLoading === 'function') hideLoading();
    return undefined;
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read the inpainting result'));
      reader.readAsDataURL(blob);
    });
  }
  // Uploads only the window the server really needs (painted area + context, see erase-window.js),
  // not the whole photo: a 12 MP lossless PNG is ~24 MB each way and ~3 s to encode.
  async function buildInpaintFormData() {
    if (!origCanvas || !maskCanvas) throw new Error('Image and mask are not ready');
    const w = origCanvas.width, h = origCanvas.height;
    if (maskCanvas.width !== w || maskCanvas.height !== h) throw new Error('Mask and image sizes differ');
    let full = ctxMask.getImageData(0, 0, w, h);
    const EW = window.EraseWindow;
    const bb = EW ? EW.findBBox(full.data, w, h) : null;
    if (EW && !bb) {
      // Zero painted pixels (strokes fully outside the photo, or too small to rasterise): never upload.
      const e = new Error(window.t('toastNothingToErase'));
      e.code = 'EMPTY_MASK';
      throw e;
    }
    let win = bb ? EW.windowFromBBox(bb, w, h) : null;
    // Off-bounds / malformed window -> fall back to the whole photo, which is always valid.
    if (!win || !EW.isValidWindow(win, w, h)) win = { x: 0, y: 0, w: w, h: h, refSize: Math.max(w, h) };
    // Hard 0/255 mask at the exact source size (anti-aliased brush edges become binary), window only.
    const src = win.w === w && win.h === h ? full : ctxMask.getImageData(win.x, win.y, win.w, win.h);
    if (src !== full) full = null;                 // V23/G4: the full-size copy is not needed any more
    const d = src.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i] > 127 ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
    }
    const maskOut = document.createElement('canvas');
    maskOut.width = win.w; maskOut.height = win.h;
    maskOut.getContext('2d').putImageData(src, 0, 0);
    let imageSrc = origCanvas;
    if (win.w !== w || win.h !== h) {
      imageSrc = document.createElement('canvas');
      imageSrc.width = win.w; imageSrc.height = win.h;
      imageSrc.getContext('2d').drawImage(origCanvas, win.x, win.y, win.w, win.h, 0, 0, win.w, win.h);
    }
    // Vercel rejects request bodies over 4.5 MB: the window is compressed for transport only (js/core/upload-compress.js,
    // quality 0.92 - it is pasted back into the photo, so it stays above the 0.85 default) and the mask is rasterised
    // to the EXACT size of the encoded image.
    // The window is <= ~1060 px a side, so in practice scale is 1 (same pixels, same coordinates); if a fallback
    // full-photo window ever exceeds 2048 px, `scale` < 1 and ref_size / the pasted result are mapped back below.
    let imageBlob, maskBlob, imageName = 'image.png', scale = 1, sentW = win.w, sentH = win.h;
    try {
      const C = window.YoUploadCompress;
      if (C) {
        const enc = await C.encodePair(imageSrc, maskOut, { quality: 0.92 });
        imageBlob = enc.blob; maskBlob = enc.maskBlob; imageName = enc.name;
        scale = enc.scale; sentW = enc.width; sentH = enc.height;
      } else {
        [imageBlob, maskBlob] = await Promise.all([
          window.canvasToBlob(imageSrc, 'image/png'),
          window.canvasToBlob(maskOut, 'image/png')
        ]);
      }
    } finally {
      // V23/G4: the window canvases are only needed for encoding - free them now, also when encoding fails.
      maskOut.width = 0; maskOut.height = 0;
      if (imageSrc !== origCanvas) { imageSrc.width = 0; imageSrc.height = 0; }
    }
    if (!imageBlob || !imageBlob.size || !maskBlob || !maskBlob.size) throw new Error('Could not prepare the image for upload');
    const formData = new FormData();
    formData.append('image', imageBlob, imageName);
    formData.append('mask', maskBlob, 'mask.png');
    // keeps hole growth identical to a full-photo request (in the uploaded pixels when the window was downscaled)
    const refSize = scale === 1 ? win.refSize : window.YoUploadCompress.scaledRefSize(win.refSize, scale, sentW, sentH);
    formData.append('ref_size', String(refSize));
    return { formData, win, fullW: w, fullH: h, base: origCanvas, scale };
  }

  // The server returns just the window; paste it back over the untouched full-resolution photo.
  async function compositeWindow(patchBlob, job) {
    const url = URL.createObjectURL(patchBlob);
    try {
      const patch = await window.YoAiCommon.loadImage(url).catch(() => { throw new Error('Could not read the inpainting result'); });
      const c = document.createElement('canvas');
      c.width = job.fullW; c.height = job.fullH;
      try {
        const cx = c.getContext('2d');
        cx.drawImage(job.base, 0, 0);
        cx.drawImage(patch, job.win.x, job.win.y, job.win.w, job.win.h);   // dest size = window: also maps a downscaled result back
        const res = await canvasToBlobUrl(c, 'image/png');
        return res.url;
      } finally {
        c.width = 0; c.height = 0;                 // released on success AND failure
      }
    } finally { URL.revokeObjectURL(url); }
  }

  // ============ HOLD TO COMPARE (on the main canvas) ============
  function compareStart() {
    if (comparing || !active || busy) return;
    // Y15: the pre-removal photo (same geometry as the canvas) or, failing that, the upload cropped like the canvas
    const url = compareBeforeUrl || (typeof getOriginalPreviewSrc === 'function' ? getOriginalPreviewSrc() : '') ||
      (typeof originalForCompare !== 'undefined' && originalForCompare ? originalForCompare.src : null);
    const ci = $('compareImg');
    if (!url || !ci) return;
    comparing = true;
    ci.src = url;
    ci.style.clipPath = 'none';
    if (typeof positionCompareElements === 'function') positionCompareElements();
    ci.classList.add('active');
    layer.classList.add('comparing');
    const b = $('me-compare-hold'); if (b) b.classList.add('holding');
  }
  function compareEnd() {
    if (!comparing) return;
    comparing = false;
    const ci = $('compareImg');
    if (ci) { ci.classList.remove('active'); ci.style.clipPath = ''; }
    if (layer) layer.classList.remove('comparing');
    const b = $('me-compare-hold'); if (b) b.classList.remove('holding');
  }

  // ============ DOWNLOAD HD ============
  // Y18: lossless PNG at the FULL working resolution of the photo (naturalWidth x naturalHeight of the working
  // image - never the <=1200 px on-screen preview), rendered from the working image with the SAME engine as the
  // main export (sliders, curves, Color Mix, filter, overlays). Before, this saved whatever the eraser had
  // loaded last: with a re-bake still pending that was a stale picture without the newest edits.
  async function download() {
    if (typeof originalImage === 'undefined' || !originalImage) { if (!origCanvas) return; }
    let c = null;
    try {
      let source = origCanvas;
      if (typeof originalImage !== 'undefined' && originalImage && typeof YoPipeline !== 'undefined') {
        const w = originalImage.naturalWidth || originalImage.width;
        const h = originalImage.naturalHeight || originalImage.height;
        if (w && h) {
          c = document.createElement('canvas');
          YoPipeline.renderEditedCanvas(originalImage, w, h, c);
          if (typeof overlayManager !== 'undefined' && overlayManager && overlayManager.items && overlayManager.items.length &&
              typeof overlayManager.drawToCanvas === 'function') {
            overlayManager.drawToCanvas(c.getContext('2d'), w, h);
          }
          source = c;
        }
      }
      // window.* = the shared helpers in js/ui/save-image.js.
      const blob = await window.canvasToBlob(source, 'image/png');
      if (c) { c.width = 0; c.height = 0; c = null; }      // release the full-size backing store before the share sheet opens
      const result = await window.saveBlob(blob, 'magic-eraser-' + Date.now() + '.png');
      if (result === 'cancelled') return;
      toast(window.t(result === 'shared' ? 'toastPngShared' : 'toastPngSaved', { size: window.blobSizeMB(blob) }), 'success');
    } catch (err) {
      console.error(err);
      toast(window.t('toastDownloadFailed'), 'error');
    } finally {
      if (c) { c.width = 0; c.height = 0; }
    }
  }

  // ============ COMMIT RESULT TO THE MAIN EDITOR ============
  // V24/G8: the shared commit (YoPipeline.commitImageUrl). Colour edits were baked into the source the server saw, so they
  // are cleared; the Frame, the overlays and the crop ratio stay exactly as the user left them.
  function applyResultToMainEditor(url) {
    if (!url) return Promise.resolve();
    return window.YoPipeline.commitImageUrl(url).then(() => {
      const compareBtn = $('compareBtn'); if (compareBtn) compareBtn.classList.add('active');
    });
  }

  function toast(msg, type) {
    if (typeof window.showToast === 'function') { window.showToast(msg, type || 'success'); return; }
    console.error('[MagicEraser]', msg);
  }

  return {
    init, apply, undo, redo,
    clear: clearMask,
    cancel: cancelErase,
    download, refreshActive, ensureCurrentImage,
    // Y18: a new photo always starts with an empty mask, even when it happens to have the same pixel size
    discardStrokes: function () { resetStrokes(); if (img) repaint(); updateButtons(); },
    // blob URLs the eraser still needs (history must not revoke these)
    getHeldUrls: () => [sourceUrl, pendingUrl, compareBeforeUrl, bakedUrl].filter(Boolean),
    compareStart, compareEnd,
    // read-only helpers (used by tests / other modules)
    hasAnyMask, hasLoadedImage: () => !!img,
    _eta: { loadDurations, recordDuration, medianOf, etaSuffix },
    _debug: () => ({ actions: actions.length, redo: redoActions.length, layerSize: layer ? [layer.width, layer.height] : null })
  };
})();

// Inline onclick hooks used by index.html
window.magicEraserApply    = () => window.MagicEraser.apply();
window.magicEraserUndo     = () => window.MagicEraser.undo();
window.magicEraserRedo     = () => window.MagicEraser.redo();
window.magicEraserClear    = () => window.MagicEraser.clear();
window.magicEraserDownload = () => window.MagicEraser.download();

document.addEventListener('DOMContentLoaded', () => window.MagicEraser.init());
