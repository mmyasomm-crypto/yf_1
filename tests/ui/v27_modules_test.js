// V27 - module structure guard. Plain Node, no dependencies.
// Locks in: the registry loads first, every module's manifest is accurate, private helpers stay private, the live-binding
// semantics the old global-script world relied on (reads see updates, reassignment reaches the module) still hold.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test, run, assert } = require('./_harness');

const root = path.resolve(__dirname, '../..');
const read = (rel) => fs.readFileSync(path.join(root, rel), 'utf8');
const html = read('index.html');
const loaded = [...html.matchAll(/<script\b[^>]*\bsrc="\.\/(js\/[^"]+)"[^>]*>/g)].map((m) => m[1]);

function walk(dir, out) {
  for (const f of fs.readdirSync(dir)) {
    const p = path.join(dir, f);
    if (fs.statSync(p).isDirectory()) walk(p, out); else if (p.endsWith('.js')) out.push(path.relative(root, p).split(path.sep).join('/'));
  }
  return out;
}
const appFiles = walk(path.join(root, 'js'), []).filter((f) => !f.startsWith('js/workers/'));

// Files that are intentionally NOT IIFE modules. A new file must either register via YF.expose or be added here on purpose.
const NOT_WRAPPED = {
  'js/core/namespace.js': 'the registry itself',
  'js/core/state.js': '30 bindings are mutated from many files - shared state stays global (V28 candidate: YF.state accessors)',
  'js/config/app-config.js': 'stamped by scripts/sync_version.py (regex rewrite) - config layer, intentionally global',
  'js/config/api-client.js': 'already self-contained (UMD-style, no top-level declarations)',
  'js/config/presets-data.js': 'pure data constant',
  'js/core/edge-model.js': 'Edge AI - already a single namespaced object (YoEdgeModel); kept byte-identical',
  'js/core/segmenter-service.js': 'Edge AI / segmentation - already namespaced (YoSegmenter); kept byte-identical',
  'js/core/render-pipeline.js': 'already namespaced objects (YoFilterMath / YoCompat / YoPipeline)',
  'js/core/webgl-renderer.js': 'already a single namespaced object (GPU)',
  'js/core/perf-utils.js': 'no top-level declarations (window assignments only)',
  'js/core/startup-lang.js': 'no top-level declarations',
  'js/portrait/edge-portrait.js': 'Edge AI - already namespaced (YoEdgePortrait); kept byte-identical',
  'js/portrait/portrait-math.js': 'Edge AI - already namespaced (YoPortraitMath); kept byte-identical',
  'js/ui/edge-model-ui.js': 'Edge AI UI - already namespaced (YoEdgeModelUI); kept byte-identical',
  'js/editing/ai-common.js': 'no top-level declarations (already scoped)',
  'js/editing/ai-mask.js': 'no top-level declarations (already scoped)',
  'js/editing/bg-removal.js': 'no top-level declarations (already scoped)',
  'js/editing/erase-window.js': 'no top-level declarations (already scoped)',
  'js/core/upload-compress.js': 'no top-level declarations (already scoped; Node-testable UMD-style like erase-window.js)',
  'js/editing/magic-eraser.js': 'no top-level declarations (already scoped)',
  'js/editing/portrait-enhance.js': 'no top-level declarations (already scoped)',
  'js/i18n/i18n.js': 'no top-level declarations (already scoped)',
  'js/ui/a11y.js': 'no top-level declarations (already scoped)',
  'js/ui/progress-counter.js': 'no top-level declarations - already a single namespaced object (YoProgress), shared by loading-toast.js and the AILoadingManager',
  'js/ui/ai-loading-manager.js': 'no top-level declarations - one closed IIFE publishing AILoadingManager + the aiLoader instance, shared by magic-eraser.js and portrait-mode.js',
};

// ---- manifest parsing -------------------------------------------------------------------------------------------------
function parseModule(rel) {
  const src = read(rel);
  const m = /^YF\.expose\('([^']+)', \{\n([\s\S]*?)\n\}\);$/m.exec(src);
  if (!m) return null;
  const entries = [...m[2].matchAll(/^  ([A-Za-z_$][\w$]*): \[\(\) => ([A-Za-z_$][\w$]*)(?:, \(v\) => \{ ([A-Za-z_$][\w$]*) = v; \})?\],?$/gm)]
    .map((e) => ({ name: e[1], getter: e[2], setterTarget: e[3] || null }));
  const rows = m[2].split('\n').filter((l) => l.trim());
  return { rel, src, mod: m[1], entries, rows };
}
const modules = appFiles.map(parseModule).filter(Boolean);
const topLevelDecls = (src) => {
  const out = new Map();   // module bodies are not re-indented, so top-level declarations start in column 0
  for (const m of src.matchAll(/^(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/gm)) out.set(m[1], 'fn');
  for (const m of src.matchAll(/^(let|const|var)\s+([A-Za-z_$][\w$]*)/gm)) out.set(m[2], m[1]);
  for (const m of src.matchAll(/^class\s+([A-Za-z_$][\w$]*)/gm)) out.set(m[1], 'class');
  return out;
};

test('V27: namespace.js is the first app script; every script exists once; every app file is loaded', () => {
  assert.equal(loaded[0], 'js/core/namespace.js', 'the registry must load before any module');
  assert.equal(new Set(loaded).size, loaded.length, 'a script tag appears twice');
  for (const f of loaded) assert.ok(fs.existsSync(path.join(root, f)), f + ' is referenced but missing');
  for (const f of appFiles) assert.ok(loaded.includes(f), f + ' exists under js/ but is not loaded by index.html');
  assert.equal(loaded[loaded.length - 1], 'js/core/startup-lang.js', 'startup-lang.js stays the LAST script');
});

test('V27: every app file is either a registered module or consciously listed as not wrapped', () => {
  const registered = new Set(modules.map((m) => m.rel));
  for (const f of appFiles) assert.ok(registered.has(f) || NOT_WRAPPED[f], f + ': neither calls YF.expose nor appears in NOT_WRAPPED');
  for (const f of Object.keys(NOT_WRAPPED)) { assert.ok(appFiles.includes(f), 'NOT_WRAPPED lists a missing file: ' + f); assert.ok(!registered.has(f), f + ' is wrapped but still listed as not wrapped'); }
  assert.equal(modules.length, 23, 'registered module count');
});

test('V27: each module is a closed IIFE named after its file, manifest at the top, sorted, no duplicate names', () => {
  for (const m of modules) {
    assert.equal(m.mod, path.basename(m.rel, '.js'), m.rel + ': registry name must equal the file name');
    assert.ok(/^\(function \(g\) \{$/m.test(m.src), m.rel + ': IIFE opener');
    assert.ok(m.src.trimEnd().endsWith('})(globalThis);'), m.rel + ': IIFE closer');
    assert.equal(m.entries.length, m.rows.length, m.rel + ': every manifest row must match the canonical shape (' + m.entries.length + '/' + m.rows.length + ')');
    const names = m.entries.map((e) => e.name);
    assert.deepEqual(names, [...names].sort(), m.rel + ': manifest is sorted');
    assert.equal(new Set(names).size, names.length, m.rel + ': duplicate export');
    for (const e of m.entries) assert.equal(e.name, e.getter, m.rel + ': ' + e.name + ' getter points at a different variable');
  }
});

test('V27: manifest honesty - every export is a real top-level declaration; const/class are read-only, the rest are live read+write', () => {
  for (const m of modules) {
    const decls = topLevelDecls(m.src);
    assert.ok(m.entries.length > 0, m.rel + ': manifest parsed to zero entries');
    for (const e of m.entries) {
      assert.ok(decls.has(e.name), m.rel + ': exports "' + e.name + '" but declares no such top-level binding');
      const kind = decls.get(e.name);
      if (kind === 'const' || kind === 'class') assert.equal(e.setterTarget, null, m.rel + ': ' + e.name + ' is ' + kind + ' - it must not have a setter');
      else assert.equal(e.setterTarget, e.name, m.rel + ': ' + e.name + ' (' + kind + ') needs a live setter onto itself');
    }
  }
});

test('V27: no two modules publish the same global name', () => {
  const owner = new Map();
  for (const m of modules) for (const e of m.entries) { assert.ok(!owner.has(e.name), e.name + ' exported by both ' + owner.get(e.name) + ' and ' + m.rel); owner.set(e.name, m.rel); }
});

test('V27: private stays private - a non-exported top-level helper is not referenced by any other file or by index.html', () => {
  // Crude on purpose (no parser in the zero-dependency suite): drop comments, then ignore .member access, `name(...) {` method
  // definitions and `name:` object keys - those are not references to a top-level binding.
  const strip = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"\\])\/\/.*$/gm, '$1');
  const refs = (src, name) => {
    const re = new RegExp('(^|[^A-Za-z0-9_$.])' + name.replace(/\$/g, '\\$') + '(?![A-Za-z0-9_$])', 'g');
    for (const m of src.matchAll(re)) {
      const lineStart = src.lastIndexOf('\n', m.index + m[1].length) + 1;
      const lineEnd = src.indexOf('\n', m.index);
      const line = src.slice(lineStart, lineEnd === -1 ? undefined : lineEnd);
      const after = src.slice(m.index + m[0].length);
      if (new RegExp('^\\s+(?:async\\s+|static\\s+|get\\s+|set\\s+)?' + name.replace(/\$/g, '\\$') + '\\s*\\([^)]*\\)\\s*\\{\\s*$').test(line)) continue;   // method definition
      if (/^:(?!:)/.test(after)) continue;                                                                                                              // object key
      return true;
    }
    return false;
  };
  const sources = Object.fromEntries(appFiles.map((f) => [f, strip(read(f))]));
  const htmlNoComments = html.replace(/<!--[\s\S]*?-->/g, '');
  for (const m of modules) {
    const exported = new Set(m.entries.map((e) => e.name));
    for (const name of topLevelDecls(m.src).keys()) {
      if (exported.has(name)) continue;
      for (const f of appFiles) if (f !== m.rel) assert.ok(!refs(sources[f], name), m.rel + ': "' + name + '" is private but ' + f + ' uses it - add it to the manifest (or stop using it)');
      assert.ok(!refs(htmlNoComments, name), m.rel + ': "' + name + '" is private but index.html uses it - add it to the manifest');
    }
  }
});

test('V27: Edge AI files and state.js are untouched by the module wrap (no YF.expose, still their original shape)', () => {
  for (const f of ['js/core/edge-model.js', 'js/portrait/edge-portrait.js', 'js/portrait/portrait-math.js', 'js/core/segmenter-service.js', 'js/ui/edge-model-ui.js', 'js/core/state.js'])
    assert.ok(!/YF\.expose/.test(read(f)), f + ' must not be wrapped');
  assert.ok(/^\s*(import|importScripts)\b/m.test(read('js/workers/depth-worker.js')), 'the depth worker keeps its own module/import loading');
});

// ---- registry semantics (the contract the old global scripts provided) -------------------------------------------------
function registry() {
  const sb = vm.createContext({});
  vm.runInContext(read('js/core/namespace.js'), sb, { filename: 'namespace.js' });
  return sb;
}

test('V27: YF.expose publishes LIVE bindings - reads see the module updating its own variable', () => {
  const sb = registry();
  vm.runInContext(`(function (g) {
    YF.expose('demo-mod', { counter: [() => counter, (v) => { counter = v; }], bump: [() => bump, (v) => { bump = v; }], LIMIT: [() => LIMIT] });
    var counter = 1; const LIMIT = 9; function bump() { counter++; return counter; }
  })(globalThis);`, sb);
  assert.equal(vm.runInContext('counter', sb), 1);
  vm.runInContext('bump()', sb);
  assert.equal(vm.runInContext('counter', sb), 2, 'a bare-identifier read in another script sees the update');
  assert.equal(vm.runInContext('YF.demoMod.counter', sb), 2, 'the YF namespace sees the same live value');
  assert.equal(vm.runInContext('LIMIT', sb), 9);
});

test('V27: reassigning an exported function from outside reaches the module itself (tests / patches keep working)', () => {
  const sb = registry();
  vm.runInContext(`(function (g) {
    YF.expose('patchy', { render: [() => render, (v) => { render = v; }], caller: [() => caller] });
    function render() { return 'real'; }
    function caller() { return render(); }       // an in-module call must see the patch, exactly like a global function did
  })(globalThis);`, sb);
  assert.equal(vm.runInContext('caller()', sb), 'real');
  vm.runInContext('render = function () { return "stubbed"; }', sb);
  assert.equal(vm.runInContext('caller()', sb), 'stubbed');
  assert.equal(vm.runInContext('YF.patchy.render()', sb), 'stubbed');
});

test('V27: exported const / class have no setter; YF.modules() lists what each module exports; names are camel-cased on YF', () => {
  const sb = registry();
  vm.runInContext(`(function (g) {
    YF.expose('fit-frame-x', { MAX: [() => MAX], Thing: [() => Thing], zed: [() => zed, (v) => { zed = v; }] });
    const MAX = 5; class Thing {} let zed = 0;
  })(globalThis);`, sb);
  const d = Object.getOwnPropertyDescriptor(sb, 'MAX');
  assert.equal(typeof d.get, 'function'); assert.equal(d.set, undefined, 'const must stay read-only');
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(YF.modules())', sb)), { 'fit-frame-x': ['MAX', 'Thing', 'zed'] });
  assert.equal(vm.runInContext('typeof YF.fitFrameX.Thing', sb), 'function');
});

test('V27: a name clash with a non-configurable global fails loudly instead of silently losing the export', () => {
  const sb = registry();
  vm.runInContext("Object.defineProperty(globalThis, 'taken', { value: 1, configurable: false });", sb);
  assert.throws(() => vm.runInContext(`(function () { YF.expose('clash', { taken: [() => taken] }); var taken = 2; })();`, sb), /cannot publish global "taken"/);
});

run('v27_modules_test');
