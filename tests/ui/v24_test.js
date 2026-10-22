// V24 G6-G10: curves drag cost, no monkey-patching, one commit routine + shared helpers, lossless working buffer
// (JPEG only at export), HTML-sink hardening.
// Plain Node + vm, no browser: behaviour is checked through stub DOM objects; static guards pin the structure so a
// regression (a new override, a private commit copy, a JPEG intermediate, an innerHTML sink) fails here.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test, run, assert, createContext } = require('./_harness');

const root = path.join(__dirname, '../..');
const read = (f) => fs.readFileSync(path.join(root, f), 'utf8');
const jsFiles = [];
(function walk(d) {
  for (const x of fs.readdirSync(path.join(root, d), { withFileTypes: true })) {
    const p = path.join(d, x.name);
    if (x.isDirectory()) walk(p); else if (x.name.endsWith('.js')) jsFiles.push(p.split(path.sep).join('/'));
  }
})('js');
const nonI18n = jsFiles.filter((f) => !f.endsWith('i18n/i18n.js'));
// comments never count as code in the static guards
const code = (f) => read(f).replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"\\])\/\/.*$/gm, '$1');

// ======================================================= G6 - curves
function loadCurves() {
  const sandbox = { console, Math, Uint8Array, Object, Array, document: { getElementById: () => null },
    applyAllEdits() {}, requestAnimationFrame() {} };
  createContext(sandbox);
  vm.runInContext(`
    function defaults(){ return [{x:0,y:0},{x:64,y:64},{x:128,y:128},{x:192,y:192},{x:255,y:255}]; }
    var curvesData = { rgb: defaults(), red: defaults(), green: defaults(), blue: defaults() };
    var curvesLUT = {}; var activeCurveChannel = 'rgb';
    var __sorts = 0; var __origSort = Array.prototype.sort;
    Array.prototype.sort = function () { __sorts++; return __origSort.apply(this, arguments); };`, sandbox);
  vm.runInContext(read('js/editing/curves.js'), sandbox);
  return { sandbox, run: (c) => vm.runInContext(c, sandbox) };
}
// the V23 algorithm, kept here as the reference the optimised one must reproduce exactly
function referenceLut(points) {
  const f = (x) => {
    const s = points.slice().sort((a, b) => a.x - b.x);
    let i = 0;
    for (; i < s.length - 1; i++) if (x >= s[i].x && x <= s[i + 1].x) break;
    if (i >= s.length - 1) return s[s.length - 1].y;
    if (x <= s[0].x) return s[0].y;
    const p0 = s[Math.max(0, i - 1)], p1 = s[i], p2 = s[i + 1], p3 = s[Math.min(s.length - 1, i + 2)];
    const t = (x - p1.x) / (p2.x - p1.x), t2 = t * t, t3 = t2 * t;
    return Math.max(0, Math.min(255, 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)));
  };
  return Array.from({ length: 256 }, (_, i) => Math.round(f(i)));
}

test('G6: sorted control points are cached - building all four LUTs sorts at most once per channel, a repeat build never sorts', () => {
  const e = loadCurves();
  e.run('__sorts = 0; buildCurvesLUTs();');
  assert.ok(e.run('__sorts') <= 4, 'first build: ' + e.run('__sorts') + ' sorts (V23: 1024)');
  e.run('__sorts = 0; buildCurvesLUTs(); drawCurves && 0;');
  assert.equal(e.run('__sorts'), 0);
});

test('G6: the active-channel rebuild produces the same LUT as V23 and touches ONLY that channel (fresh array, others keep identity)', () => {
  const e = loadCurves();
  e.run('buildCurvesLUTs(); var __before = { rgb: curvesLUT.rgb, red: curvesLUT.red, green: curvesLUT.green, blue: curvesLUT.blue };');
  e.run('curvesData.red[2].y = 170; curvesData.red[1].x = 40; invalidateCurveCache("red"); __sorts = 0; buildCurveLUT("red");');
  assert.equal(e.run('__sorts'), 0, 'no sort while rebuilding from the cache');
  assert.equal(e.run('curvesLUT.red !== __before.red'), true, 'NEW array - webgl-renderer compares references to decide the texture re-upload');
  for (const ch of ['rgb', 'green', 'blue']) assert.equal(e.run(`curvesLUT.${ch} === __before.${ch}`), true, ch + ' untouched');
  const got = Array.from(e.run('curvesLUT.red'));
  assert.deepEqual(got, referenceLut(JSON.parse(e.run('JSON.stringify(curvesData.red)'))));
});

test('G6: replacing curvesData[channel] (reset / undo / new photo) can never serve a stale cache', () => {
  const e = loadCurves();
  e.run('buildCurvesLUTs();');
  e.run('curvesData.rgb = [{x:0,y:30},{x:64,y:90},{x:128,y:150},{x:192,y:200},{x:255,y:255}]; buildCurvesLUTs();');
  assert.deepEqual(Array.from(e.run('curvesLUT.rgb')), referenceLut(JSON.parse(e.run('JSON.stringify(curvesData.rgb)'))));
  // a re-ordered in-place edit is caught by the cheap order check, not by a sort on every call
  e.run('curvesData.green[1].x = 150; buildCurveLUT("green");');
  assert.deepEqual(Array.from(e.run('curvesLUT.green')), referenceLut(JSON.parse(e.run('JSON.stringify(curvesData.green)'))));
});

test('G6: the drag handler has no nested requestAnimationFrame, rebuilds only the active LUT and uses the live GPU path', () => {
  const t = code('js/editing/curves.js');
  assert.ok(!/requestAnimationFrame/.test(t), 'no rAF wrapper left in curves.js (applyAllEdits coalesces by itself)');
  const move = t.slice(t.indexOf('const handleMove'), t.indexOf('const handleEnd'));
  assert.ok(/buildCurveLUT\(activeCurveChannel\)/.test(move) && !/buildCurvesLUTs\(\)/.test(move));
  assert.ok(/applyAllEdits\(true\)/.test(move), 'live (no per-move encode) path');
  assert.ok(/const handleEnd[\s\S]*saveToHistory/.test(t));
});

test('G6: curve drags reach the GPU as live frames - applyAllEdits(true) is what schedules the single settle encode', () => {
  const s = code('js/editing/sliders.js');
  assert.ok(/function applyAllEdits\(live\)/.test(s) || /applyAllEdits\s*=\s*function\s*\(live\)|function applyAllEdits\(\s*live/.test(s), 'applyAllEdits takes the live flag');
  assert.ok(/_scheduleLiveSettle|settle/i.test(s));
});

// ======================================================= G7 - no monkey patching
test('G7: gestures.js no longer overrides handleFileSelect (or anything else)', () => {
  const t = code('js/ui/gestures.js');
  assert.ok(!/originalHandleFileSelect|handleFileSelect\s*=/.test(t));
  assert.ok(!/OVERRIDES/.test(read('js/ui/gestures.js')));
});

test('G7: no declared top-level function is reassigned anywhere, and no function name is declared twice', () => {
  const decl = new Map();
  for (const f of nonI18n) for (const m of code(f).matchAll(/^function\s+([A-Za-z_$][\w$]*)/gm)) decl.set(m[1], (decl.get(m[1]) || []).concat(f));
  const dups = [...decl].filter(([, fs_]) => fs_.length > 1).map(([n]) => n);
  assert.deepEqual(dups, [], 'duplicate declarations silently override each other');
  const reassigned = [];
  for (const f of nonI18n) for (const name of decl.keys()) {
    if (new RegExp('^\\s*(?:window\\.)?' + name.replace(/\$/g, '\\$') + '\\s*=[^=]', 'm').test(code(f))) reassigned.push(f + ':' + name);
  }
  assert.deepEqual(reassigned, []);
});

test('G7: #mainImg load handling uses addEventListener once - no `.onload =` on the shared element', () => {
  const fh = code('js/core/file-handler.js');
  assert.ok(!/mainImg\.onload\s*=/.test(fh));
  for (const f of nonI18n) assert.ok(!/\bmainImg\.(onload|onerror)\s*=/.test(code(f)), f);
  const sandbox = { requestAnimationFrame: (f) => f(), document: { getElementById: () => null } };
  createContext(sandbox);
  vm.runInContext(read('js/core/file-handler.js') + '\nvar __fits = 0; fitImageToWorkspace = function () { __fits++; };', sandbox);
  const listeners = [];
  const img = { addEventListener: (t, fn) => listeners.push([t, fn]) };
  vm.runInContext('_bindMainImgLoadOnce', sandbox)(img);
  vm.runInContext('_bindMainImgLoadOnce', sandbox)(img);       // second upload: must NOT stack another listener
  assert.equal(listeners.length, 1);
  assert.equal(listeners[0][0], 'load');
  listeners[0][1]();
  assert.equal(vm.runInContext('__fits', sandbox), 2, 'fit now + fit on the next frame (as before)');
});

test('G7: the Compare button is revealed by file-handler itself (the removed override used to do it)', () => {
  const fh = code('js/core/file-handler.js');
  assert.ok(/getElementById\('compareBtn'\)[\s\S]{0,80}style\.display\s*=\s*'block'/.test(fh));
});

// ======================================================= G8 - one commit routine, shared helpers
function loadPipeline(over) {
  const calls = [];
  const mainImg = { style: {}, className: '', src: '' };
  const sandbox = Object.assign({ console, Math, JSON, Uint8Array, Uint8ClampedArray, Float32Array, Object, Array, Promise, setTimeout,
    document: { createElement() { return { getContext() { return null; } }; }, getElementById: (id) => (id === 'mainImg' ? mainImg : null) },
    Image: function () { const im = this; Object.defineProperty(im, 'src', { set(v) { im._s = v; Promise.resolve().then(() => (v.indexOf('bad') >= 0 ? im.onerror() : im.onload())); }, get() { return im._s; } }); },
    resetColorEdits: () => calls.push('resetColorEdits'), resetAllValues: () => calls.push('resetAllValues'),
    resetFrameState: () => calls.push('resetFrameState'),
    fitImageToWorkspace: () => calls.push('fit'), saveToHistory: () => calls.push('saveToHistory'),
    applyAllEdits: () => calls.push('applyAllEdits'), releaseTransientImageUrl: (u) => calls.push('release:' + u),
    overlayManager: { items: [{ id: 'keep-me' }], resizeCanvas: () => calls.push('overlayResize') },
    canvasToBlobUrl: (c, type, q) => { calls.push('encode:' + type + ':' + q); return Promise.resolve({ url: 'blob:enc' }); },
  }, over || {});
  createContext(sandbox);
  vm.runInContext(`
    const sliders = { bright:0, contrast:0, exposure:0, shadows:0, whites:0, blacks:0, sat:0, vibrance:0, warmth:0, tint:0, sharp:0, clarity:0, vignette:0, grain:0 };
    let chosenPreset = 'none'; let colorMixData = [0,1,2,3,4,5,6,7].map(() => ({ hue:0, sat:0, lum:0 }));
    let curvesData = {}; const curvesLUT = {}; const allPresets = [{ id:'none', style:'' }];
    var originalImage = null; var previewCanvas = null; var previewCtx = null;`, sandbox);
  for (const f of ['js/editing/curves.js', 'js/editing/presets.js', 'js/editing/sliders.js', 'js/editing/effects.js',
                   'js/editing/color-mix-apply.js', 'js/core/render-pipeline.js']) vm.runInContext(read(f), sandbox);
  sandbox.__spyApply = () => calls.push('applyAllEdits');
  vm.runInContext('applyAllEdits = __spyApply;', sandbox);      // sliders.js declared the real one; the commit tests only need the call
  return { sandbox, calls, mainImg };
}

test('G8: commitImageUrl - clears colour edits, KEEPS frame/overlays/ratio, sets the alpha cache, writes exactly one history entry', async () => {
  const { sandbox, calls, mainImg } = loadPipeline();
  const img = await sandbox.YoPipeline.commitImageUrl('blob:new', { hasAlpha: true });
  assert.equal(mainImg.src, 'blob:new');
  assert.equal(img.__yfAlpha, true);
  assert.equal(vm.runInContext('originalImage', sandbox), img);
  assert.ok(calls.includes('resetColorEdits'));
  assert.ok(!calls.includes('resetAllValues') && !calls.includes('resetFrameState'), 'Frame / overlays / ratio are never reset');
  assert.equal(sandbox.overlayManager.items.length, 1);
  assert.equal(calls.filter((c) => c === 'saveToHistory').length, 1);
  assert.ok(calls.includes('overlayResize') && calls.includes('fit'));
  assert.ok(!calls.includes('applyAllEdits'), 'baked edits: nothing to re-render');
});

test('G8: commitImageUrl with keepColorEdits (crop / portrait) re-renders instead of clearing; beforeHistory runs before the history entry', async () => {
  const { sandbox, calls, mainImg } = loadPipeline();
  await sandbox.YoPipeline.commitImageUrl('blob:crop', { keepColorEdits: true, resetImageBox: true, beforeHistory: () => calls.push('before') });
  assert.ok(!calls.includes('resetColorEdits'));
  assert.ok(calls.indexOf('before') < calls.indexOf('applyAllEdits') && calls.indexOf('applyAllEdits') < calls.indexOf('saveToHistory'));
  assert.equal(mainImg.style.objectFit, 'contain'); assert.equal(mainImg.style.aspectRatio, 'auto');
});

test('G8: commitImageUrl on an undecodable URL releases it, writes no history and rejects', async () => {
  const { sandbox, calls } = loadPipeline();
  await assert.rejects(sandbox.YoPipeline.commitImageUrl('blob:bad'), /could not sync/);
  assert.ok(calls.includes('release:blob:bad'));
  assert.ok(!calls.includes('saveToHistory'));
});

test('G8: commitCanvas encodes through the shared canvasToBlobUrl, lossless, and can release the canvas right after the encode', async () => {
  const { sandbox, calls } = loadPipeline();
  const canvas = { width: 100, height: 50 };
  await sandbox.YoPipeline.commitCanvas(canvas, { hasAlpha: false, releaseCanvas: true });
  assert.ok(calls.includes('encode:image/png:undefined'));
  assert.equal(canvas.width, 0); assert.equal(canvas.height, 0);
  await sandbox.YoPipeline.commitCanvasAsOriginal({ width: 4, height: 4, getContext: () => null }).catch(() => {});   // legacy entry point still exported
  assert.equal(typeof sandbox.YoPipeline.commitCanvasAsOriginal, 'function');
});

test('G8: every tool commits through YoPipeline - no private copy of the commit steps is left', () => {
  const tools = { 'js/editing/crop.js': 'commitCanvas', 'js/editing/ai-common.js': 'commitImageUrl', 'js/editing/ai-mask.js': 'commitCanvasAsOriginal',
    'js/editing/magic-eraser.js': 'commitImageUrl', 'js/portrait/portrait-mode.js': 'commitImageUrl' };
  for (const [f, fn] of Object.entries(tools)) {
    const t = code(f);
    assert.ok(new RegExp('\\.' + fn + '\\(').test(t), f + ' must call ' + fn);
    assert.ok(!/resetAllValues\(|resetColorEdits\(|syncPreviewCanvas\(|saveToHistory\(/.test(t), f + ' must not repeat the commit steps itself');
    assert.ok(!/originalImage\s*=[^=]/.test(t), f + ' must not swap originalImage itself');
  }
  // the only places allowed to assign originalImage: the shared commit, the upload, history restore
  const owners = nonI18n.filter((f) => /\boriginalImage\s*=[^=]/.test(code(f))).sort();
  assert.deepEqual(owners, ['js/core/file-handler.js', 'js/core/history.js', 'js/core/render-pipeline.js', 'js/core/state.js']);   // state.js = the declaration
});

test('G8: one canvas.toBlob() wrapper - save-image.js; history.js, eraser, portrait, compare, upload all use it', () => {
  const direct = nonI18n.filter((f) => /\.toBlob\(/.test(code(f))).sort();
  assert.deepEqual(direct, ['js/ui/modals-export.js', 'js/ui/save-image.js'], 'modals-export keeps only the Cordova storage path');
  assert.ok(/canvasToBlob\(canvas, type, quality\)\.then/.test(code('js/core/history.js')));
  assert.ok(!/function canvasToBlob/.test(code('js/editing/magic-eraser.js')));
  for (const f of ['js/ui/compare.js', 'js/portrait/portrait-mode.js']) assert.ok(/canvasToBlob\(/.test(code(f)), f);
  assert.ok(/canvasToBlobUrl\(cv, 'image\/png'\)/.test(code('js/core/file-handler.js')));
});

test('G8: backend errors are read by the shared YoApi.httpError in every tool that calls the server', () => {
  for (const f of ['js/editing/magic-eraser.js', 'js/portrait/portrait-mode.js']) assert.ok(/YoApi\.httpError\(/.test(code(f)), f);
  const callers = nonI18n.filter((f) => /YoApi\.request\(/.test(code(f))).sort();
  assert.deepEqual(callers, ['js/editing/magic-eraser.js', 'js/portrait/portrait-mode.js']);
  for (const f of callers) assert.ok(!/response\.json\(\)/.test(code(f)) || /httpError/.test(code(f)), f);
  assert.ok(!/await response\.json\(\)/.test(code('js/portrait/portrait-mode.js')), 'portrait no longer parses error bodies itself');
});

test('G8: the eraser no longer bakes overlays into its source (they stay live above the photo and survive the commit)', () => {
  const t = code('js/editing/magic-eraser.js');
  const bake = t.slice(t.indexOf('getCurrentEditorDataUrl'), t.indexOf('getCurrentEditorDataUrl') + 1800);
  assert.ok(!/drawToCanvas|hasOverlays|overlayManager/.test(bake), 'the working-image bake contains colour edits only');
  assert.ok(!/hasOverlays/.test(t));
  // the eraser's own Download is an export: it still composites the overlays into the saved file
  assert.ok(/download[\s\S]*drawToCanvas/.test(t));
});

test('G8: the AI-tool wrapper reports commit failures with a toast and never rejects', async () => {
  const toasts = [];
  const sandbox = { console, Math, JSON, Promise, Object, Array, document: { createElement() { return {}; }, getElementById: () => null },
    URL: { revokeObjectURL() {} }, IS_MOBILE_DEVICE: false, requestAnimationFrame: (f) => f(), setTimeout: (f) => { f(); return 0; },
    showLoading() {}, hideLoading() {}, updateLoadingSub() {}, getCurrentEditedImageSrc: () => Promise.resolve('blob:s'), releaseTransientImageUrl() {} };
  sandbox.window = sandbox; sandbox.t = (k) => k;
  createContext(sandbox);
  vm.runInContext(read('js/editing/ai-common.js'), sandbox);
  sandbox.YoPipeline = { commitImageUrl: () => Promise.reject(new Error('x')) };
  assert.equal(await sandbox.YoAiCommon.applyResultToEditor('blob:u', (m, t) => toasts.push([m, t])), false);
  assert.deepEqual(toasts, [['toastSyncFailed', 'error']]);
  sandbox.YoPipeline = { commitImageUrl: () => Promise.resolve() };
  assert.equal(await sandbox.YoAiCommon.applyResultToEditor('blob:u', () => {}), true);
});

// ======================================================= G9 - lossless pipeline
test('G9: the working-buffer spec is PNG for every input (alpha or not, any quality asked for)', () => {
  const { sandbox } = loadPipeline();
  for (const args of [[], [false], [true], [false, 0.5], [true, 0.95]]) {
    const spec = sandbox.YoPipeline.encodeSpec(...args);
    assert.equal(spec.mime, 'image/png'); assert.equal(spec.quality, undefined);
  }
});

test('G9: JPEG exists in exactly two places - the final export and the display-only preview / thumbnails (+ the transport-only upload copy)', () => {
  const jpeg = nonI18n.filter((f) => /image\/jpeg|image\/jpg/.test(code(f))).sort();
  assert.deepEqual(jpeg, ['js/core/file-handler.js', 'js/core/render-pipeline.js', 'js/core/upload-compress.js', 'js/editing/presets.js', 'js/portrait/portrait-mode.js', 'js/ui/modals-export.js'],
    'file-handler: only the "is the file a JPEG" alpha shortcut; render-pipeline: preview + type lookup; presets: filter thumbnails; ' +
    'portrait-mode: picks the download extension from the real blob type; upload-compress: the copy sent to the Vercel backend (<= 4.5 MB body limit) - ' +
    'transport only, never becomes the working image or the export');
  assert.ok(!/toDataURL|toBlob|canvasToBlob/.test(code('js/core/file-handler.js').replace(/canvasToBlobUrl\(cv, 'image\/png'\)/, '')), 'upload path has no other encoder');
  assert.ok(/image\/jpeg/.test(code('js/ui/modals-export.js')), 'the one lossy encode happens at export');
  // no intermediate step passes a lossy quality, and none but portrait-mode's download-extension lookup even names JPEG
  const lossy = /(canvasToBlob(Url)?|toBlob|toDataURL)\([^)]*,\s*0?\.\d+\s*\)/;
  for (const f of ['js/editing/crop.js', 'js/editing/ai-mask.js', 'js/editing/portrait-enhance.js', 'js/editing/bg-removal.js', 'js/editing/magic-eraser.js', 'js/portrait/portrait-mode.js', 'js/core/history.js'])
    assert.ok(!lossy.test(code(f)), f + ' has no lossy intermediate');
  for (const f of ['js/editing/crop.js', 'js/editing/ai-mask.js', 'js/editing/portrait-enhance.js', 'js/editing/bg-removal.js', 'js/editing/magic-eraser.js', 'js/core/history.js'])
    assert.ok(!/jpeg/i.test(code(f)), f + ' never names JPEG');
});

test('G9: uploads are re-encoded as PNG (never JPEG), at most once, with a bounded fallback', async () => {
  const types = [];
  const el = () => ({ style: {}, classList: { add() {}, remove() {}, toggle() {}, contains: () => true }, addEventListener() {}, getContext: () => ({ drawImage() {}, imageSmoothingEnabled: true }), width: 0, height: 0 });
  const sandbox = { console, Math, Promise, setTimeout, requestAnimationFrame: (f) => f(),
    document: { createElement: el, getElementById: () => el() },
    window: { t: (k) => k }, showLoading() {}, hideLoading() {}, showToast() {},
    YoPipeline: { canvasHasAlpha: () => false },
    createImageBitmap: () => Promise.resolve({ width: 4000, height: 3000, close() {} }),
    canvasToBlobUrl: (c, type) => { types.push([type, c.width, c.height]); return new Promise(() => {}); } };
  createContext(sandbox);
  vm.runInContext(read('js/core/file-handler.js'), sandbox);
  vm.runInContext('handleFileSelect', sandbox)({ target: { files: [{ type: 'image/jpeg' }], value: 'x' } });
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(types.length, 1);
  assert.equal(types[0][0], 'image/png');
  assert.deepEqual([types[0][1], types[0][2]], [2560, 1920], 'downscaled to the editing cap, then encoded losslessly');
  // a failing encoder falls back to FileReader once and then reports - no endless retry loop
  assert.ok(/usedFallbackReader/.test(code('js/core/file-handler.js')));
});

test('G9: history keeps the lossless PNG blobs - budget raised to 150 MB (low-end / unknown) and 200 MB (>4 GB devices)', () => {
  const t = code('js/core/history.js');
  assert.ok(/mem && mem > 4 \? 200 : 150\) \* 1024 \* 1024/.test(t));
  assert.ok(/MIN_HISTORY_STATES/.test(t));
});

test('G9: portrait baseline is uploaded as PNG and the server answers with PNG', () => {
  const t = code('js/portrait/portrait-mode.js');
  assert.ok(/portrait-source\.png/.test(t) && !/portrait-source\.jpg/.test(t));
  assert.ok(/canvasToBlob\(c, 'image\/png'\)/.test(t));
  const py = read('backend/app/portrait.py');
  assert.ok(!/format="JPEG"/.test(py) && /format="PNG"/.test(py) && !/"image\/jpeg"/.test(py));
});

// ======================================================= G10 - HTML sinks
test('G10: overlay-manager.js and ai-mask.js contain no HTML-string sink at all', () => {
  for (const f of ['js/overlays/overlay-manager.js', 'js/editing/ai-mask.js'])
    assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment/.test(code(f)), f);
});

test('G10: no innerHTML/outerHTML anywhere assigns anything but an empty string - except the constant focus-ring SVG', () => {
  const sinks = [];
  for (const f of nonI18n) for (const m of code(f).matchAll(/\.(innerHTML|outerHTML)\s*=\s*([^;\n]+)/g)) {
    if (!/^(''|"")$/.test(m[2].trim())) sinks.push(f + ' ' + m[1] + ' = ' + m[2].trim());
  }
  assert.deepEqual(sinks, ['js/portrait/portrait-mode.js innerHTML = RING_SVG']);
  assert.ok(/var RING_SVG =\s*'<span/.test(read('js/portrait/portrait-mode.js')), 'RING_SVG is a literal constant');
  assert.ok(!nonI18n.some((f) => /insertAdjacentHTML/.test(code(f))));
});

test('G10: ai-mask setStatus renders hostile text as text (no element, no attribute, no handler) and still shows the spinner', () => {
  const src = read('js/editing/ai-mask.js');
  const fnSrc = src.slice(src.indexOf('function setStatus'), src.indexOf('// V23: loader wrappers'));
  const makeEl = (tag) => {
    const el = { tag, children: [], style: {}, className: '', _text: '' };
    Object.defineProperty(el, 'innerHTML', { set() { throw new Error('innerHTML must not be used'); }, get() { return ''; } });
    Object.defineProperty(el, 'textContent', { get: () => el._text, set: (v) => { el._text = String(v); if (v === '') el.children = []; } });
    el.appendChild = (c) => { el.children.push(c); return c; };
    return el;
  };
  const status = makeEl('div');
  const sandbox = { document: { getElementById: () => status, createElement: makeEl }, $: null };
  createContext(sandbox);
  vm.runInContext('function $(id) { return document.getElementById(id); }\n' + fnSrc, sandbox);
  const evil = '<img src=x onerror=alert(1)><script>alert(2)</script>';
  vm.runInContext('setStatus', sandbox)(evil, true);
  assert.equal(status.style.display, 'flex');
  assert.deepEqual(status.children.map((c) => c.tag), ['span', 'span']);
  assert.equal(status.children[0].className, 'spin');
  assert.equal(status.children[1].textContent, evil, 'the markup is carried verbatim as TEXT');
  assert.equal(status.children[1].children.length, 0);
  vm.runInContext('setStatus', sandbox)('', false);
  assert.equal(status.style.display, 'none'); assert.equal(status.children.length, 0);
  vm.runInContext('setStatus', sandbox)('plain', false);
  assert.deepEqual(status.children.map((c) => c.tag), ['span']);
});

// ======================================================= packaging / docs
test('V24: docs and package are in place', () => {
  assert.ok(fs.existsSync(path.join(root, 'docs/V24_CHANGES.md')));
  assert.ok(fs.existsSync(path.join(root, 'docs/V23_CHANGES.md')), 'V23 notes kept');
  assert.ok(/G6[\s\S]*G7[\s\S]*G8[\s\S]*G9[\s\S]*G10/.test(read('docs/V24_CHANGES.md')));
});

run('v24_test');
