// ============================================================
// portrait-mode.js
// Portrait / Bokeh tool — server-side depth blur.
//
// The photo is POSTed to /api/portrait (Depth Anything V2 Small, ONNX) together
// with an integer blur_level (1 = sharp .. 30 = widest bokeh), an integer skin_smooth (0 = off .. 100 =
// strongest natural skin smoothing) plus a normalised focus_x / focus_y (tap to focus). The server returns
// the finished image (smoothed skin + depth-of-field bokeh in one pass), which is
// committed as the new working image exactly like a crop is: originalImage is
// replaced, the canvas pipeline re-renders (slider/filter edits stay live) and
// one history entry is saved, so undo/redo keep working.
// ============================================================

(function (g) {
// V27 module "portrait-mode": private helpers stay inside this closure; only the manifest below is shared. Body is intentionally not re-indented.
YF.expose('portrait-mode', {
  getCurrentEditedImageSrc: [() => getCurrentEditedImageSrc, (v) => { getCurrentEditedImageSrc = v; }],
  initializePortraitModel: [() => initializePortraitModel, (v) => { initializePortraitModel = v; }],
  releaseTransientImageUrl: [() => releaseTransientImageUrl, (v) => { releaseTransientImageUrl = v; }],
});
(function () {
    'use strict';

    // Relative path, resolved by js/config/api-client.js (same origin as the page, so any backend port works).
    var PORTRAIT_PATH = '/api/portrait';
    var REFINE_DEBOUNCE_MS = 400;
    var BLUR_MIN = 1;                  // slider floor: level 1 = completely sharp
    var BLUR_MAX = 30;                 // slider ceiling: widest f/1.4-style bokeh
    var SKIN_MIN = 0;                  // skin slider floor: smoothing off
    var SKIN_MAX = 100;                // skin slider ceiling: strongest (still texture-preserving) smoothing
    var _refineTimer = null;

    // ---------------------------------------------------------- small helpers
    function $(id) { return document.getElementById(id); }
    function toast(msg, type) {
        if (typeof showToast === 'function') showToast(msg, type || 'success');
        else console.error('[Portrait]', msg);
    }
    function show(id, on) { var el = $(id); if (el) el.style.display = on ? 'block' : 'none'; }

    function showSections(which) {
        show('portrait-init-section',    which === 'init');
        show('portrait-result-section',  which === 'result');
        show('portrait-loading-section', which === 'loading');
        show('portrait-error-section',   which === 'error');
        // While loading/refining keep the Apply entry hidden but the slider usable.
        if (which === 'error') show('portrait-init-section', true);
    }

    // Compact orb on the canvas panel: js/ui/ai-loading-manager.js (aiLoader) owns it. It carries the 1 -> 100 counter and the
    // status line but, unlike the Eraser's modal, never blocks the canvas or the blur slider: Portrait is refined live, and
    // every new request restarts the count (the newer request replaces the older one).
    //   aiLoader.show('portrait')  right before the work starts;  complete(ticket) on success;  hide(ticket) otherwise.
    function loaderShow() { return window.aiLoader ? window.aiLoader.show('portrait') : undefined; }
    function loaderComplete(ticket) { if (window.aiLoader) window.aiLoader.complete(ticket); }
    function loaderHide(ticket) { if (window.aiLoader) window.aiLoader.hide(ticket); }

    // Blob -> object URL (size registered for the history budget). Replaces the old
    // FileReader.readAsDataURL round trip, which put a base64 copy on the JS heap.
    function blobToObjectUrl(blob) {
        return Promise.resolve(registerBlobUrl(blob));
    }

    // V24/G9: the portrait baseline is sent to the server as PNG (lossless) - the working image is already PNG, so this
    // fallback only runs when its blob cannot be fetched.
    function imageToBlob(img) {
        var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
        var c = document.createElement('canvas');
        c.width = w; c.height = h;
        c.getContext('2d').drawImage(img, 0, 0, w, h);
        return canvasToBlob(c, 'image/png').then(function (b) {
            c.width = 0; c.height = 0;
            return b;
        }, function (err) {
            c.width = 0; c.height = 0;
            throw err;
        });
    }

    // Blob of the current working image (crop/portrait baseline, before slider edits).
    function currentImageBlob() {
        var src = originalImage && originalImage.src;
        if (src && (src.indexOf('data:') === 0 || src.indexOf('blob:') === 0)) {
            return fetch(src).then(function (r) { return r.blob(); })
                .catch(function () { return imageToBlob(originalImage); });
        }
        return imageToBlob(originalImage);
    }

    // Upload-ready copy of the baseline for POST /api/portrait (see js/core/upload-compress.js). Without the module
    // (should never happen: it is loaded before this file) the original blob is sent exactly as before.
    function prepareServerUpload(blob) {
        var C = window.YoUploadCompress;
        if (!C) return Promise.resolve({ blob: blob, name: 'portrait-source.png', scale: 1, srcW: 0, srcH: 0 });
        var alpha = false;
        try { alpha = !!(window.YoPipeline && YoPipeline.imageHasAlpha && YoPipeline.imageHasAlpha(originalImage)); } catch (e) { alpha = false; }
        return C.prepareImageBlob(blob, { maxDim: C.MAX_DIM, quality: C.QUALITY, hasAlpha: alpha });
    }

    // True while originalImage is still the portrait result we produced
    // (false after undo, crop, eraser, a new upload, ...).
    function isOnPortraitResult() {
        return !!(portraitResultSrc && originalImage && originalImage.src === portraitResultSrc);
    }

    // ------------------------------------------------------------ commit result
    // V24/G8: the shared commit routine (YoPipeline.commitImageUrl). The colour edits stay on top of the portrait
    // pixels (the baseline sent to the server never contains them); Frame / overlays / ratio are untouched.
    function commitImage(imageUrl, opts) {
        opts = opts || {};
        var previousResultSrc = portraitResultSrc;
        return YoPipeline.commitImageUrl(imageUrl, {
            hasAlpha: YoPipeline.imageHasAlpha(originalImage),      // the portrait pass never adds or removes transparency
            keepColorEdits: true,
            resetImageBox: true,
            beforeHistory: function () {
                // A slider refinement replaces the previous portrait entry instead of stacking one undo step per tweak.
                if (opts.replacePrevious && previousResultSrc &&
                    historyIndex === historyStack.length - 1 && historyIndex > 0 &&
                    historyStack[historyIndex].originalImageSrc === previousResultSrc) {
                    _dropHistoryTop();          // releases the replaced result's blob
                }
            }
        }).then(function () {}, function () {
            throw new Error(window.t('errProcessedLoad'));
        });
    }

    // ------------------------------------------------------------------ request
    function friendlyError(err, status, detail) {
        if (err && err.name === 'AbortError') return null;           // superseded / cancelled
        if (err && err.yoApi && err.kind !== 'http') return window.YoApi.describe(err);   // unreachable / timeout
        if (status === 503) return window.t('errPortraitModel');
        if (status === 413) return window.t('errImageTooLarge');
        if (status) return detail || window.t('errServerStatus', { status: status });
        return detail ? String(detail) : window.YoApi.describe(err);
    }

    async function runPortraitRequest() {
        if (!originalImage) { toast(window.t('toastLoadImage'), 'error'); return; }

        var refine = isOnPortraitResult() && !!portraitBaseBlob;
        var seq = ++portraitRequestSeq;
        if (portraitAbortCtl) { try { portraitAbortCtl.abort(); } catch (e) {} }
        var ctl = new AbortController();
        portraitAbortCtl = ctl;

        portraitBusy = true;
        var applyBtn = $('portraitApplyBtn');
        if (applyBtn) applyBtn.disabled = true;
        showSections('loading');
        var loaderTicket = loaderShow();                          // just before the inference (on-device or server) starts

        var status = 0, detail = '';
        try {
            // Always blur the pre-portrait baseline so repeated runs never stack blur.
            if (!refine) portraitBaseBlob = await currentImageBlob();
            if (seq !== portraitRequestSeq) return;

            var level = Math.round(portraitCurrentBlur);
            var skin = Math.round(portraitCurrentSkin);
            var dataUrl, sharpOnly = level <= BLUR_MIN && skin <= SKIN_MIN;
            if (sharpOnly) {
                // Level 1 with no skin smoothing is "completely untouched": restore the baseline
                // locally - no network round trip, no re-encode, no blur.
                dataUrl = await blobToObjectUrl(portraitBaseBlob);
            } else {
                // Hybrid: on-device first (model cached in the Cache API, depth in a Web Worker), the server otherwise.
                // YoEdgePortrait.run() answers null whenever the local path cannot or should not handle the request
                // (not configured, skin smoothing, low memory, download cancelled/failed, local error) - then the
                // unchanged server request below runs.
                var localBlob = null;
                if (window.YoEdgePortrait && window.YoEdgePortrait.canHandle({ skin: skin })) {
                    localBlob = await window.YoEdgePortrait.run(portraitBaseBlob, {
                        level: level, focusX: portraitFocus.x, focusY: portraitFocus.y,
                        isStale: function () { return seq !== portraitRequestSeq; }
                    });
                    if (seq !== portraitRequestSeq) return;
                }
                if (localBlob) {
                    dataUrl = await blobToObjectUrl(localBlob);
                } else {
                    // Vercel rejects request bodies over 4.5 MB: resize (<= 2048 px) and JPEG-encode (q 0.85) the baseline
                    // first. A photo that already fits (and transparent photos, which must stay PNG) are handled in
                    // upload-compress.js. `up.scale` < 1 means the server result has to be brought back to full size below.
                    var up = await prepareServerUpload(portraitBaseBlob);
                    if (seq !== portraitRequestSeq) return;
                    var form = new FormData();
                    form.append('image', up.blob, up.name);
                    form.append('blur_level', String(level));
                    form.append('skin_smooth', String(skin));
                    form.append('focus_x', portraitFocus.x.toFixed(4));
                    form.append('focus_y', portraitFocus.y.toFixed(4));

                    var response = await window.YoApi.request(PORTRAIT_PATH, { method: 'POST', body: form, signal: ctl.signal });
                    status = response.status;
                    if (!response.ok) {
                        var httpErr = await window.YoApi.httpError(response);     // shared error-body reader (api-client.js)
                        detail = httpErr.detail;
                        throw new Error(detail || ('Server error (' + status + ')'));
                    }
                    var blob = await response.blob();
                    if (!blob.size || (blob.type && blob.type.indexOf('image/') !== 0)) {
                        throw new Error(window.t('errNotImage'));
                    }
                    // The result has the UPLOADED size (or, as a last resort for a huge result, a size the server shrank to
                    // stay under the platform's 4.5 MB response limit): restore the working image's pixel size so crop box,
                    // frame, overlays and history keep their geometry (focus_x / focus_y are normalised, so they never
                    // change). resizeBlobTo() hands the blob back untouched when the size already matches.
                    if (window.YoUploadCompress && up.srcW > 0 && up.srcH > 0) blob = await window.YoUploadCompress.resizeBlobTo(blob, up.srcW, up.srcH);
                    dataUrl = await blobToObjectUrl(blob);
                }
            }
            if (seq !== portraitRequestSeq) return;                   // a newer request won

            await commitImage(dataUrl, { replacePrevious: refine });
            portraitResultSrc = originalImage.src;
            showSections('result');
            loaderComplete(loaderTicket);                             // success: snap to 100 %, pop, hold, fade
            if (!sharpOnly) hideFocusRing();                          // sharp restore keeps the tap ring visible
        } catch (err) {
            if (seq !== portraitRequestSeq) return;
            var message = friendlyError(err, status, detail || (err && err.message));
            if (message === null) { showSections(isOnPortraitResult() ? 'result' : 'init'); return; }
            console.error('Portrait error:', err);
            showSections(isOnPortraitResult() ? 'result' : 'error');
            toast(message, 'error');
        } finally {
            if (seq === portraitRequestSeq) {
                if (Math.round(portraitCurrentBlur) > BLUR_MIN) hideFocusRing();   // endpoint answered (or failed)
                portraitBusy = false;
                portraitAbortCtl = null;
                loaderHide(loaderTicket);                                 // error / cancel: close at once (no-op after complete)
                if (applyBtn) applyBtn.disabled = false;
            }
        }
    }

    // Keep the panel truthful when the image changes underneath it (undo, crop,
    // eraser...): once originalImage is no longer our portrait result, offer Apply again.
    (function watchImageChanges() {
        var main = $('mainImg');
        if (!main) return;
        main.addEventListener('load', function () {
            var resultVisible = $('portrait-result-section') &&
                                $('portrait-result-section').style.display === 'block';
            if (resultVisible && !portraitBusy && !isOnPortraitResult()) showSections('init');
        });
    })();

    // ------------------------------------------------------- tap to focus
    // Clicking/tapping the photo while the Portrait tab is open picks the focal
    // point. Coordinates are normalised to the image (0..1), so they are
    // independent of zoom, fit and letterboxing.
    var FOCUS_DEFAULT_BLUR = 12;       // blur level used when focus is set while still at the sharp floor
    var FOCUS_RING_MS = 1500;          // ring auto-hides after this long ...
    var FOCUS_RING_FADE_MS = 280;      // ... fading out over this long
    var _ringTimer = null, _ringFadeTimer = null;

    // Manual focus mode: tapping the photo does NOTHING until the user presses
    // "Set Focus". The next single tap on the photo sets the focal point, then the
    // mode switches itself off so normal gestures stay uninterrupted.
    var _focusMode = false;
    function setFocusMode(on) {
        _focusMode = !!on;
        var btn = $('portrait-set-focus');
        if (btn) { btn.classList.toggle('active', _focusMode); btn.setAttribute('aria-pressed', _focusMode ? 'true' : 'false'); }
        var vp = $('viewport');
        if (vp) vp.classList.toggle('portrait-focus-cursor', _focusMode);
    }

    var RING_SVG =
        '<span class="pf-pulse"></span><span class="pf-ring"></span>' +
        '<svg class="pf-cross" viewBox="0 0 72 72" width="72" height="72" aria-hidden="true">' +
        '<path d="M36 7v11M36 54v11M7 36h11M54 36h11" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/>' +
        '<circle cx="36" cy="36" r="2.4" fill="#fff"/></svg>';

    function portraitTabActive() {
        var sec = $('p-portrait');
        return !!(sec && sec.style.display !== 'none' && sec.classList.contains('active'));
    }

    // Rendered rectangle of the picture inside <img> (honours object-fit: contain).
    function renderedImageRect() {
        var img = $('mainImg');
        if (!img || !originalImage) return null;
        var box = img.getBoundingClientRect();
        var iw = originalImage.naturalWidth || originalImage.width;
        var ih = originalImage.naturalHeight || originalImage.height;
        if (!box.width || !box.height || !iw || !ih) return null;
        var scale = Math.min(box.width / iw, box.height / ih);
        var w = iw * scale, h = ih * scale;
        return { left: box.left + (box.width - w) / 2, top: box.top + (box.height - h) / 2, width: w, height: h };
    }

    function getFocusRing() {
        var viewport = $('viewport');
        if (!viewport) return null;
        var ring = $('portraitFocusRing');
        if (!ring) {
            ring = document.createElement('div');
            ring.id = 'portraitFocusRing';
            ring.className = 'portrait-focus-ring';
            ring.setAttribute('aria-hidden', 'true');
            ring.innerHTML = RING_SVG;
            viewport.appendChild(ring);      // above the busy-spinner layer
        }
        return ring;
    }

    // Smooth fade-out; safe to call any time (also when nothing is showing).
    function hideFocusRing() {
        if (_ringTimer) { clearTimeout(_ringTimer); _ringTimer = null; }
        var ring = $('portraitFocusRing');
        if (!ring || !ring.classList.contains('on')) return;
        ring.classList.remove('on');
        ring.classList.add('out');
        if (_ringFadeTimer) clearTimeout(_ringFadeTimer);
        _ringFadeTimer = setTimeout(function () { ring.classList.remove('out'); }, FOCUS_RING_FADE_MS + 40);
    }

    function showFocusRing(clientX, clientY) {
        var ring = getFocusRing();
        if (!ring) return;
        var vr = $('viewport').getBoundingClientRect();
        ring.style.left = (clientX - vr.left) + 'px';       // exact tap point
        ring.style.top  = (clientY - vr.top)  + 'px';
        if (_ringFadeTimer) { clearTimeout(_ringFadeTimer); _ringFadeTimer = null; }
        ring.classList.remove('on', 'out');
        void ring.offsetWidth;                              // restart the animation
        ring.classList.add('on');
        if (_ringTimer) clearTimeout(_ringTimer);
        _ringTimer = setTimeout(hideFocusRing, FOCUS_RING_MS);
    }

    function onViewportTap(ev) {
        if (!_focusMode) return;                              // plain taps never set focus
        if (!portraitTabActive() || !originalImage) return;
        if (ev.target && ev.target.closest &&
            ev.target.closest('.crop-overlay, .crop-actions, .loading-overlay, label, button')) return;
        var main = $('mainImg');
        if (!main || getComputedStyle(main).display === 'none') return;
        var rect = renderedImageRect();
        if (!rect) return;
        var nx = (ev.clientX - rect.left) / rect.width;
        var ny = (ev.clientY - rect.top)  / rect.height;
        if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;      // tapped the letterbox, not the photo

        portraitFocus = { x: nx, y: ny, set: true };
        setFocusMode(false);                                  // one tap, then back to normal gestures
        showFocusRing(ev.clientX, ev.clientY);
        // Setting the focal point applies the bokeh depth effect right away. At the sharp
        // floor (level 1) there would be nothing to see, so lift it to a pleasant default.
        if (Math.round(portraitCurrentBlur) <= BLUR_MIN) setBlur(FOCUS_DEFAULT_BLUR);
        if (_refineTimer) clearTimeout(_refineTimer);
        _refineTimer = setTimeout(runPortraitRequest, 120);
    }

    (function bindTapToFocus() {
        var viewport = $('viewport');
        if (!viewport) return;
        viewport.addEventListener('click', onViewportTap);
    })();

    // ------------------------------------------------------------ public handlers
    window.togglePortraitFocusMode = function () {
        if (!originalImage) { toast(window.t('toastLoadImage'), 'error'); return; }
        setFocusMode(!_focusMode);
    };
    // G13: read / leave "Set Focus" mode without toggling (used by the Back button layer stack).
    window.isPortraitFocusModeActive = function () { return _focusMode; };
    window.exitPortraitFocusMode = function () { setFocusMode(false); };
    // Side rail shows only on the Portrait tab with a photo loaded.
    function syncSideRail() {
        var rail = $('portraitSideRail');
        if (!rail) return;
        var show = portraitTabActive() && !!originalImage;
        if (rail.hidden === show) rail.hidden = !show;
    }
    (function bindSideRail() {
        var rail = $('portraitSideRail');
        if (!rail) return;
        // The rail sits on top of the viewport: keep pan/zoom/double-tap gestures from seeing its touches.
        ['touchstart', 'pointerdown', 'mousedown', 'dblclick'].forEach(function (t) {
            rail.addEventListener(t, function (e) { e.stopPropagation(); }, { passive: true });
        });
        var sec = $('p-portrait'), img = $('mainImg');
        if (window.MutationObserver) {
            var mo = new MutationObserver(syncSideRail);
            if (sec) mo.observe(sec, { attributes: true, attributeFilter: ['class', 'style'] });
            if (img) mo.observe(img, { attributes: true, attributeFilter: ['src', 'style'] });
        }
        if (img) img.addEventListener('load', syncSideRail);
        syncSideRail();
    })();
    // Called after tab changes: focus mode never outlives the Portrait tab.
    window.syncPortraitFocusMode = function () {
        syncSideRail();
        if (_focusMode && !portraitTabActive()) setFocusMode(false);
    };

    window.applyPortraitMode = function () {
        runPortraitRequest();
    };

    window.reapplyPortraitMode = function () {
        runPortraitRequest();
    };

    function setBlur(val) {
        val = parseInt(val, 10);
        if (!isFinite(val)) val = BLUR_MIN;
        val = Math.max(BLUR_MIN, Math.min(BLUR_MAX, val));
        portraitCurrentBlur = val;
        var slider = $('portrait-blur-slider');
        if (slider) slider.value = val;
        var label = $('v-portrait-blur');
        if (label) label.innerText = val;
    }

    // Live label while dragging (no network traffic).
    window.onPortraitBlurInput = function () {
        setBlur($('portrait-blur-slider').value);
    };

    function setSkin(val) {
        val = parseInt(val, 10);
        if (!isFinite(val)) val = SKIN_MIN;
        val = Math.max(SKIN_MIN, Math.min(SKIN_MAX, val));
        portraitCurrentSkin = val;
        var slider = $('portrait-skin-slider');
        if (slider) slider.value = val;
        var label = $('v-portrait-skin');
        if (label) label.innerText = val;
    }

    // Live label while dragging (no network traffic).
    window.onPortraitSkinInput = function () {
        setSkin($('portrait-skin-slider').value);
    };

    // Released: re-render from the pristine baseline so smoothing never stacks.
    window.onPortraitSkinChange = function () {
        setSkin($('portrait-skin-slider').value);
        if (sliderShouldRender()) runPortraitRequest();
    };

    window.adjustPortraitSkin = function (delta) {
        setSkin(portraitCurrentSkin + delta);
        if (!sliderShouldRender()) return;
        if (_refineTimer) clearTimeout(_refineTimer);
        _refineTimer = setTimeout(runPortraitRequest, REFINE_DEBOUNCE_MS);
    };

    // True when moving the slider should (re)render: a portrait result is active,
    // or the user has lifted the blur above the sharp floor on a loaded photo.
    function sliderShouldRender() {
        if (!originalImage) return false;
        return isOnPortraitResult() || portraitCurrentBlur > BLUR_MIN || portraitCurrentSkin > SKIN_MIN || portraitBusy;
    }

    // Slider released: render from the pristine baseline so blur never stacks.
    window.onPortraitBlurChange = function () {
        setBlur($('portrait-blur-slider').value);
        if (sliderShouldRender()) runPortraitRequest();
    };

    window.adjustPortraitBlur = function (delta) {
        setBlur(portraitCurrentBlur + delta);
        if (!sliderShouldRender()) return;
        if (_refineTimer) clearTimeout(_refineTimer);
        _refineTimer = setTimeout(runPortraitRequest, REFINE_DEBOUNCE_MS);
    };

    // Entering the Portrait tool starts completely sharp (level 1), unless a
    // portrait result is already active - then the user's level is kept.
    window.onPortraitTabEnter = function () {
        if (isOnPortraitResult() || portraitBusy) return;
        setBlur(BLUR_MIN);
        setSkin(SKIN_MIN);
    };

    // The result is already an encoded image (blob: URL), so its bytes are saved as-is: re-drawing it through a
    // canvas would re-compress the JPEG and change pixels. fetch() on a blob: URL hands back the same Blob, no copy.
    window.downloadPortraitResult = function () {
        var mainImg = $('mainImg');
        if (!mainImg || !mainImg.src) { toast(window.t('toastNoResultDownload'), 'error'); return; }
        var src = mainImg.src;
        fetch(src)
            .then(function (r) {
                if (!r.ok) throw new Error('Could not read the result');
                return r.blob();
            })
            .then(function (blob) {
                var isJpeg = (blob.type || getImageUrlType(src)) === 'image/jpeg';
                return saveBlob(blob, 'yofilter-portrait-' + Date.now() + (isJpeg ? '.jpg' : '.png')).then(function (result) {
                    if (result === 'cancelled') return;
                    toast(window.t(result === 'shared' ? 'toastPortraitShared' : 'toastPortraitSaved', { size: blobSizeMB(blob) }), 'success');
                });
            })
            .catch(function (err) {
                console.error(err);
                toast(window.t('toastDownloadFailed'), 'error');
            });
    };

    // Cancel: step back to the pre-portrait state through the normal history.
    window.resetPortraitMode = function () {
        setFocusMode(false);
        hideFocusRing();
        if (_refineTimer) clearTimeout(_refineTimer);
        portraitRequestSeq++;                                         // drop any in-flight response
        if (portraitAbortCtl) { try { portraitAbortCtl.abort(); } catch (e) {} portraitAbortCtl = null; }
        portraitBusy = false;
        loaderHide();
        var applyBtn = $('portraitApplyBtn');
        if (applyBtn) applyBtn.disabled = false;

        var wasOnResult = isOnPortraitResult();
        var topIsResult = historyIndex > 0 && historyStack[historyIndex] &&
                          historyStack[historyIndex].originalImageSrc === portraitResultSrc;
        if (wasOnResult && topIsResult) {
            undoEdit();                                               // restores image + all edit params
        } else if (wasOnResult && portraitBaseBlob) {
            blobToObjectUrl(portraitBaseBlob).then(function (url) { return commitImage(url); })
                .catch(function () { toast(window.t('toastRestoreFailed'), 'error'); });
        }
        portraitResultSrc = null;
        portraitBaseBlob = null;
        portraitFocus = { x: 0.5, y: 0.5, set: false };
        setBlur(BLUR_MIN);
        setSkin(SKIN_MIN);
        showSections('init');
    };

    // Called when a new image is loaded.
    window.resetPortraitState = function () {
        setFocusMode(false);
        hideFocusRing();
        if (_refineTimer) clearTimeout(_refineTimer);
        portraitRequestSeq++;
        if (portraitAbortCtl) { try { portraitAbortCtl.abort(); } catch (e) {} portraitAbortCtl = null; }
        portraitBusy = false;
        portraitBaseBlob = null;
        portraitResultSrc = null;
        portraitFocus = { x: 0.5, y: 0.5, set: false };
        setBlur(BLUR_MIN);
        setSkin(SKIN_MIN);
        loaderHide();
        var applyBtn = $('portraitApplyBtn');
        if (applyBtn) applyBtn.disabled = false;
    };
})();

// ============================================================
// Shared MediaPipe person segmenter (R17 / R18)
// ============================================================
// The loader, the request queue, the local-first asset handling and the offline/timeout messages now
// live in js/core/segmenter-service.js (YoSegmenter). bg-removal.js and ai-mask.js call
// YoSegmenter.segment() and get their own result back through a Promise - no shared global result
// handler any more, so two tools can no longer receive each other's masks.
// initializePortraitModel() stays as a thin warm-up hook for old call sites.
function initializePortraitModel() {
    if (typeof YoSegmenter === 'undefined') return;
    YoSegmenter.load().catch(function () { /* the caller that needs it shows the message */ });
}

// Shared helper (used by ai-mask.js, bg-removal.js and portrait-enhance.js): resolves to a URL of
// the current image with canvas/CSS edits baked in. Async because baking now encodes with
// canvas.toBlob() into a blob: URL instead of a base64 data: URL (a 12 MP base64 string is ~20 MB
// of heap). Callers that may receive a freshly created blob must pass it to
// releaseTransientImageUrl() when done; that call is a no-op for URLs owned by history/the screen.
function getCurrentEditedImageSrc() {
    // R12: AI tools always read the working image at FULL working resolution (originalImage), with the
    // current colour edits rendered into it by the shared pipeline (same engine as preview/export).
    // They never use the <=1200 px on-screen preview, so results are no longer silently downscaled.
    return YoPipeline.getEditedWorkingSrc();
}

// Revoke a blob: URL returned by getCurrentEditedImageSrc() once it has been consumed,
// unless history or the screen still uses it.
function releaseTransientImageUrl(url) {
    if (!url || url.indexOf('blob:') !== 0) return;
    if (typeof _imageUrlRefs !== 'undefined' && _imageUrlRefs.has(url)) return;
    if (typeof _isImageUrlInUse === 'function' && _isImageUrlInUse(url)) return;
    try { URL.revokeObjectURL(url); } catch (e) {}
    if (typeof _imageUrlBytes !== 'undefined') { _imageUrlBytes.delete(url); _imageUrlTypes.delete(url); }
}
})(globalThis);
