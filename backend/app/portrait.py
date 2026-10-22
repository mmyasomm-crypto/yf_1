"""Depth-Anything-V2-Small portrait (background blur) pipeline.

Pipeline
    1. Preprocess  - resize to 518x518, scale to 0..1, ImageNet mean/std, NCHW.
    2. Inference   - ONNX Runtime (CPU) produces a relative inverse-depth map
                     (larger value = closer to the camera).
    3. Resize      - depth map is resized back to the original image size.
    4. Depth clean - cv2.bilateralFilter smooths depth noise without crossing
                     depth edges, then a colour guided filter re-aligns the
                     depth edges to the real image edges (hair, shoulders,
                     fingers) so no blur halo is drawn around the subject.
    5. Focus       - the depth under the user's tap (focus_x, focus_y; default
                     image centre) is the focal plane. Blur grows gradually
                     with depth distance from it (natural DSLR-style falloff);
                     the connected focus region (the subject) is held 100%
                     sharp.
    6. Edge guard  - two guided-filter passes (coarse + fine) snap the mask to
                     image edges.
    7. Bokeh       - masked (normalised) convolution with a disc kernel at
                     4-6 blur layers (more layers at higher levels), blended
                     per pixel by depth. Sharp pixels never feed the
                     background and bright highlights bloom into soft
                     lens-like discs that grow with the blur level.
    Scale          - blur_level runs 1..30. Level 1 is completely sharp (the
                     photo is returned untouched); 2..30 ramp the aperture up
                     to a wide f/1.4-style look at 30.
    8. Skin        - optional (skin_smooth 1..100): natural, texture-preserving skin
                     smoothing (backend/app/skin.py) on skin/face areas only, using the
                     subject mask from steps 4-6. Eyes, brows, lips, hair and edges stay
                     sharp. It runs BEFORE the bokeh pass, so the smoothed subject is
                     exactly what the depth-of-field blend keeps sharp.
    9. Encode      - always returned as PNG (V24: lossless). The result becomes the editor's working image, so a
                     JPEG here would add one more lossy generation to every portrait pass; the only lossy encode
                     left in the whole pipeline is the user's final export in the browser.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.ort_utils import CPU, build_session, check_model_file, effective_cpu_count, provider_preference
from backend.app.encoding import encode_result
from backend.app.config import (
    PNG_MAX_RESPONSE_BYTES,
    RESULT_JPEG_QUALITY,
    PORTRAIT_INPUT_SIZE,
    PORTRAIT_MAX_PIXELS,
    PORTRAIT_MAX_BLUR_LEVEL,
    PORTRAIT_MAX_SKIN_SMOOTH,
    PORTRAIT_MIN_BLUR_LEVEL,
    PORTRAIT_MIN_SKIN_SMOOTH,
    PORTRAIT_MODEL_PATH,
)
from backend.app.skin import smooth_skin

PORTRAIT_PNG_COMPRESS_LEVEL = 1

logger = logging.getLogger("yofilter.portrait")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


WORK_SIZE = 1600.0   # longest side used for mask analysis and bokeh layers


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class DepthPortrait:
    """Loads one ONNX session and serializes inference requests."""

    def __init__(self, model_path: Path = PORTRAIT_MODEL_PATH) -> None:
        self.model_path = check_model_file(Path(model_path), "Portrait depth")
        # CPU by default (the depth net has dynamic-shape ops that not every DirectML driver likes);
        # PORTRAIT_PROVIDER=auto|dml opts in to DirectML, with automatic CPU fallback.
        self.session, self.provider = build_session(
            self.model_path, label="portrait", preference=provider_preference("PORTRAIT_PROVIDER") if "PORTRAIT_PROVIDER" in os.environ else "cpu",
            threads=max(2, effective_cpu_count()),
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        self.lock = threading.Lock()
        # Small LRU of depth maps so re-tapping the focus point or moving the
        # blur slider on the same photo never re-runs the network.
        self._depth_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._depth_cache_size = 3
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ decode
    @staticmethod
    def _decode(data: bytes) -> Tuple[np.ndarray, np.ndarray | None]:
        """Return (rgb uint8 HxWx3, alpha uint8 HxW or None)."""
        if not data:
            raise ValueError("image is empty")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > PORTRAIT_MAX_PIXELS:
                    raise ValueError(
                        f"image is too large ({source.width}x{source.height}); "
                        f"the limit is {PORTRAIT_MAX_PIXELS // 1_000_000} megapixels"
                    )
                source.load()
                source = ImageOps.exif_transpose(source)
                has_alpha = source.mode in ("RGBA", "LA") or (
                    source.mode == "P" and "transparency" in source.info
                )
                if has_alpha:
                    rgba = source.convert("RGBA")
                    arr = np.asarray(rgba, dtype=np.uint8)
                    return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3])
                return np.asarray(source.convert("RGB"), dtype=np.uint8), None
        except ValueError:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("image is not a valid image") from exc

    # --------------------------------------------------------- 1. preprocessing
    @staticmethod
    def _preprocess(rgb: np.ndarray) -> np.ndarray:
        size = PORTRAIT_INPUT_SIZE
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...], dtype=np.float32)

    # ------------------------------------------------- 2 + 3. depth + resize back
    def estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Return a float32 HxW depth map in 0..1 where 1 is nearest."""
        height, width = rgb.shape[:2]
        tensor = self._preprocess(rgb)
        with self.lock:
            try:
                outputs = self.session.run([self.output_name], {self.input_name: tensor})
            except Exception:
                if self.provider == CPU:
                    raise
                # GPU run failed: rebuild on CPU once and retry instead of failing the request.
                self.session, self.provider = build_session(self.model_path, label="portrait", preference="cpu")
                outputs = self.session.run([self.output_name], {self.input_name: tensor})
        if not outputs or outputs[0] is None:
            raise RuntimeError("Depth model returned no output")

        depth = np.asarray(outputs[0], dtype=np.float32).reshape(
            outputs[0].shape[-2], outputs[0].shape[-1]
        )
        if not np.isfinite(depth).all():
            raise RuntimeError("Depth model returned non-finite values")

        # Robust normalisation (ignore the extreme 1% on each side).
        low, high = np.percentile(depth, (1.0, 99.0))
        depth = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        resized = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        return resized.clip(0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------ depth cache
    def _cached_depth(self, image_bytes: bytes, rgb: np.ndarray) -> np.ndarray:
        key = hashlib.sha1(image_bytes).hexdigest()
        with self._cache_lock:
            hit = self._depth_cache.get(key)
            if hit is not None:
                self._depth_cache.move_to_end(key)
                return hit
        depth = self.estimate_depth(rgb)
        with self._cache_lock:
            self._depth_cache[key] = depth
            while len(self._depth_cache) > self._depth_cache_size:
                self._depth_cache.popitem(last=False)
        return depth

    # ------------------------------------------------------- 4. focal plane
    @staticmethod
    def _focus_depth(depth: np.ndarray, focus_x: float, focus_y: float) -> float:
        """Median depth in a small window around the tap (robust to depth noise)."""
        height, width = depth.shape
        cx = int(round(min(max(focus_x, 0.0), 1.0) * (width - 1)))
        cy = int(round(min(max(focus_y, 0.0), 1.0) * (height - 1)))
        radius = max(2, int(min(width, height) * 0.012))
        patch = depth[
            max(0, cy - radius): min(height, cy + radius + 1),
            max(0, cx - radius): min(width, cx + radius + 1),
        ]
        return float(np.median(patch))

    @staticmethod
    def _focus_region(delta: np.ndarray, focus_x: float, focus_y: float, tol: float) -> np.ndarray:
        """Binary (0/1) region around the tap whose depth stays within ``tol``.

        Connectivity is solved on a small copy (fast, robust to noise); the final
        edge comes from thresholding the FULL-resolution refined depth, so the
        boundary stays as precise as the image itself.
        """
        height, width = delta.shape
        scale = min(1.0, 512.0 / max(height, width))
        if scale < 1.0:
            small = cv2.resize(delta, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = delta
        sh, sw = small.shape
        candidate = (small < tol).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sx = min(sw - 1, max(0, int(round(focus_x * (sw - 1)))))
        sy = min(sh - 1, max(0, int(round(focus_y * (sh - 1)))))
        if not candidate[sy, sx]:
            # Tap landed on a speck the opening removed - snap to the nearest set pixel.
            ys, xs = np.nonzero(candidate)
            if ys.size == 0:
                return (delta < tol).astype(np.float32) * 0.0
            idx = int(np.argmin((ys - sy) ** 2 + (xs - sx) ** 2))
            sy, sx = int(ys[idx]), int(xs[idx])
        _, labels = cv2.connectedComponents(candidate, connectivity=4)
        gate = (labels == labels[sy, sx]).astype(np.uint8)
        if scale < 1.0:
            gate = cv2.resize(gate, (width, height), interpolation=cv2.INTER_NEAREST)
            grow = int(np.ceil(1.0 / scale)) | 1          # absorb the low-res quantisation
            gate = cv2.dilate(gate, np.ones((grow, grow), np.uint8))
        return ((delta < tol) & (gate > 0)).astype(np.float32)

    # ------------------------------------------------------------ 4. depth clean
    @staticmethod
    def _sharpen_depth_edges(depth: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Shock-filter style step: depth models blur object borders over a ramp, which
        would leave mid-depth values (= partial blur) on the subject's rim. Pixels on
        a steep ramp snap to the nearer of the local min / max plateau, so each border
        becomes a clean step at the ramp midpoint; the guided filter then aligns that
        step to the true image edge. Gentle slopes (floors, walls) are left untouched.
        """
        height, width = depth.shape
        win = max(3, int(max(height, width) / 48.0) | 1)
        kernel = np.ones((win, win), np.uint8)
        out = depth
        for _ in range(iterations):
            hi = cv2.dilate(out, kernel)
            lo = cv2.erode(out, kernel)
            steep = (hi - lo) > 0.12
            pick = np.where((hi - out) < (out - lo), hi, lo)
            out = np.where(steep, pick, out).astype(np.float32)
        return out

    def _refine_depth(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Edge-preserving depth smoothing: bilateral filter + image-guided snap."""
        height, width = depth.shape
        scale = min(1.0, 640.0 / max(height, width))
        if scale < 1.0:
            small = cv2.resize(
                depth, (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = depth
        small = np.ascontiguousarray(small, dtype=np.float32)
        # Two light passes beat one heavy pass: noise goes, depth edges stay.
        small = cv2.bilateralFilter(small, 9, 0.08, 7)
        small = cv2.bilateralFilter(small, 7, 0.05, 4)
        small = self._sharpen_depth_edges(small)
        smooth = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR) \
            if scale < 1.0 else small
        guide = rgb.astype(np.float32) / 255.0
        radius = max(3, int(min(width, height) / 140.0))
        refined = self._guided_filter(guide, smooth.astype(np.float32), radius, eps=1e-3)
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------- 5. edge guard
    @staticmethod
    def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """Colour guided filter (He et al.) for a single-channel ``src``.

        a/b coefficients are solved on a reduced grid and upsampled (Fast Guided
        Filter) while the final linear model is evaluated against the FULL
        resolution guide, so hair and cloth edges stay pixel-accurate.
        """
        height, width = src.shape
        scale = min(1.0, 1024.0 / max(height, width))
        if scale < 1.0:
            sw, sh = max(1, int(width * scale)), max(1, int(height * scale))
            g = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
            p = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
            r = max(1, int(round(radius * scale)))
        else:
            g, p, r = guide, src, max(1, radius)
        ksize = (2 * r + 1, 2 * r + 1)

        def mean(x: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

        gr, gg, gb = g[..., 0], g[..., 1], g[..., 2]
        mg = [mean(c) for c in (gr, gg, gb)]
        mp = mean(p)
        cov_gp = [mean(c * p) - m * mp for c, m in zip((gr, gg, gb), mg)]
        chans = (gr, gg, gb)
        v = [[mean(chans[i] * chans[j]) - mg[i] * mg[j] for j in range(3)] for i in range(3)]
        for i in range(3):
            v[i][i] = v[i][i] + eps

        # Closed-form inverse of the symmetric 3x3 covariance matrix, per pixel.
        a00, a01, a02 = v[0][0], v[0][1], v[0][2]
        a11, a12, a22 = v[1][1], v[1][2], v[2][2]
        c00 = a11 * a22 - a12 * a12
        c01 = a02 * a12 - a01 * a22
        c02 = a01 * a12 - a02 * a11
        c11 = a00 * a22 - a02 * a02
        c12 = a01 * a02 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        det = a00 * c00 + a01 * c01 + a02 * c02
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        inv = ((c00, c01, c02), (c01, c11, c12), (c02, c12, c22))
        coef = [
            (inv[k][0] * cov_gp[0] + inv[k][1] * cov_gp[1] + inv[k][2] * cov_gp[2]) / det
            for k in range(3)
        ]
        b = mp - coef[0] * mg[0] - coef[1] * mg[1] - coef[2] * mg[2]
        coef = [mean(c) for c in coef]
        b = mean(b)

        if scale < 1.0:
            coef = [cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR) for c in coef]
            b = cv2.resize(b, (width, height), interpolation=cv2.INTER_LINEAR)
        out = coef[0] * guide[..., 0] + coef[1] * guide[..., 1] + coef[2] * guide[..., 2] + b
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------- 6. blur mask
    def _sharpness_mask(
        self, depth: np.ndarray, rgb: np.ndarray, focus_x: float, focus_y: float
    ) -> np.ndarray:
        """Full-resolution sharpness mask (1.0 = sharp, 0.0 = fully blurred).

        Large photos are analysed at a working resolution (<= 1600 px) for speed and
        memory; one final guided-filter pass against the FULL-resolution image then
        puts the mask edge back on the real pixel edges (hair, fingers, clothing).
        """
        height, width = depth.shape
        scale = min(1.0, WORK_SIZE / max(height, width))
        if scale >= 1.0:
            return self._mask_at_working_res(depth, rgb, focus_x, focus_y)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        small = self._mask_at_working_res(
            cv2.resize(depth, size, interpolation=cv2.INTER_AREA),
            cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
            focus_x, focus_y,
        )
        upsampled = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        guide = rgb.astype(np.float32) / 255.0
        mask = self._guided_filter(guide, upsampled, max(2, int(min(width, height) / 300.0)), eps=2e-4)
        firm = cv2.erode((upsampled > 0.999).astype(np.float32), np.ones((3, 3), np.uint8))
        return np.clip(np.maximum(mask, firm), 0.0, 1.0).astype(np.float32)

    def _mask_at_working_res(
        self, depth: np.ndarray, rgb: np.ndarray, focus_x: float, focus_y: float
    ) -> np.ndarray:
        """1.0 = keep sharp (focal plane / subject), 0.0 = fully blurred.

        Blur grows gradually with the *absolute* depth distance from the tapped
        plane (nearer and farther content both defocus). The connected region
        around the tap is held perfectly sharp, and the mask edge is snapped to
        real image edges so hair and clothing never pick up a halo.
        """
        height, width = depth.shape
        depth = self._refine_depth(depth, rgb)
        focus = self._focus_depth(depth, focus_x, focus_y)
        delta = np.abs(depth - focus)

        # Flat in-focus band, then a long smooth S-curve: a natural depth gradient
        # instead of an on/off cut-out.
        coc = _smoothstep(0.05, 0.60, delta) ** 0.85
        mask = 1.0 - coc

        # Subject protection: the connected focus region stays 100% sharp.
        short = min(width, height)
        core = self._focus_region(delta, focus_x, focus_y, tol=0.20)
        close_k = max(3, int(short / 200.0) | 1)
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        core_soft = cv2.GaussianBlur(core, (0, 0), max(0.8, short / 700.0))
        mask = np.maximum(mask, core_soft).astype(np.float32)

        # Edge-aware feathering: a coarse pass for body shape, a fine pass for hair.
        guide = rgb.astype(np.float32) / 255.0
        radius = max(4, int(short / 90.0))
        mask = self._guided_filter(guide, mask, radius, eps=1e-3)
        mask = self._guided_filter(guide, mask, max(2, radius // 3), eps=2e-4)

        # Re-assert the subject interior so filtering can never soften it. The core edge
        # already follows the refined (image-aligned) depth, so only a hair's-width
        # erosion is needed; the guided filter owns the transition outside it.
        erode_px = max(1, int(round(short / 500.0)))
        firm = cv2.erode(core, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
        mask = np.maximum(mask, cv2.GaussianBlur(firm, (0, 0), 0.7))

        # Tiny final smoothing (the guided output is already edge-accurate).
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.5, short / 1500.0))
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------- 7. bokeh
    @staticmethod
    def _disc_blur(image: np.ndarray, radius: float) -> np.ndarray:
        """Lens-like blur: anti-aliased disc kernel on a downscaled copy, softened slightly."""
        if radius < 0.8:
            return image
        if radius < 3.0:                      # too small for a visible disc: plain (fast) Gaussian
            return cv2.GaussianBlur(image, (0, 0), radius * 0.58)
        height, width = image.shape[:2]
        # Work on a copy shrunk so the kernel radius is <= ~3.5 px: bokeh is
        # low-frequency, so this is visually lossless and much faster.
        factor = max(1.0, radius / 3.5)
        if factor > 1.0:
            small = cv2.resize(
                image, (max(1, int(round(width / factor))), max(1, int(round(height / factor)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image
        r = radius / factor
        k = int(np.ceil(r)) + 1
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1].astype(np.float32)
        kernel = np.clip(r + 0.5 - np.sqrt(xx * xx + yy * yy), 0.0, 1.0)
        kernel /= kernel.sum()
        out = cv2.filter2D(small, -1, kernel, borderType=cv2.BORDER_REFLECT)
        out = cv2.GaussianBlur(out, (0, 0), max(0.6, r * 0.22))  # creamy disc edge
        if factor > 1.0:
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
        return out

    def _composite(self, rgb: np.ndarray, sharp: np.ndarray, blur_level: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        # Level 1 = completely sharp; 2..30 map linearly onto 1/29..30 "sigma units"
        # so level 30 keeps the full wide-aperture look.
        level = int(min(max(blur_level, PORTRAIT_MIN_BLUR_LEVEL), PORTRAIT_MAX_BLUR_LEVEL))
        strength = (level - PORTRAIT_MIN_BLUR_LEVEL) * PORTRAIT_MAX_BLUR_LEVEL / float(
            PORTRAIT_MAX_BLUR_LEVEL - PORTRAIT_MIN_BLUR_LEVEL
        )
        t = strength / float(PORTRAIT_MAX_BLUR_LEVEL)               # 0..1 aperture openness
        sigma = strength * max(width, height) / 1600.0
        max_radius = sigma * 1.8                      # disc radius with the same spread as sigma
        amount = np.clip(1.0 - sharp, 0.0, 1.0).astype(np.float32)   # 0 = sharp, 1 = fully blurred
        if max_radius < 0.8 or not amount.any():
            return rgb
        source = rgb.astype(np.float32)

        # Only pixels that are being blurred feed the bokeh (no subject colour
        # bleeding), and bright highlights are weighted up so they bloom into discs.
        luminance = rgb.max(axis=2).astype(np.float32) / 255.0
        highlight = 1.0 + (2.0 + 3.0 * t) * _smoothstep(0.78, 1.0, luminance) ** 2   # wider aperture -> bigger bloom
        weight = (np.clip((amount - 0.02) / 0.98, 0.0, 1.0) ** 1.5 * highlight).astype(np.float32)
        del luminance, highlight
        scale = min(1.0, WORK_SIZE / max(height, width))
        premultiplied = source * weight[..., None]
        if scale < 1.0:
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            packed = np.concatenate([
                cv2.resize(premultiplied, size, interpolation=cv2.INTER_AREA),
                cv2.resize(weight, size, interpolation=cv2.INTER_AREA)[..., None],
            ], axis=2)
            source_small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
            amount_small = cv2.resize(amount, size, interpolation=cv2.INTER_AREA)
        else:
            packed = np.concatenate([premultiplied, weight[..., None]], axis=2)
            source_small, amount_small = source, amount
        del premultiplied, weight
        packed = np.ascontiguousarray(packed)

        # 4-6 blur radii (1/n .. full) interpolated per pixel by depth distance: a
        # continuous depth-of-field gradient rather than one flat blur. Wider
        # apertures get more layers so the falloff stays smooth, not banded.
        levels = 4 if level <= 10 else (5 if level <= 20 else 6)
        position = np.clip(amount_small * levels, 1.0, float(levels))
        bokeh = np.zeros_like(source_small)
        for k in range(1, levels + 1):
            blend = np.clip(1.0 - np.abs(position - k), 0.0, 1.0)
            if not blend.any():
                continue
            blurred = self._disc_blur(packed, max_radius * scale * k / levels)
            den = blurred[..., 3:4]
            colour = np.where(den > 1e-3, blurred[..., :3] / np.maximum(den, 1e-4), source_small)
            bokeh += colour * blend[..., None]
            del blurred, colour, den
        if scale < 1.0:
            bokeh = cv2.resize(bokeh, (width, height), interpolation=cv2.INTER_LINEAR)

        # Full-resolution mix: 0 = untouched source pixel, 1 = bokeh.
        mix = np.clip(amount * levels, 0.0, 1.0)[..., None]
        result = source + (bokeh - source) * mix
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- 9. public
    def blur_background(
        self,
        image_bytes: bytes,
        blur_level: int = PORTRAIT_MIN_BLUR_LEVEL,
        focus_x: float = 0.5,
        focus_y: float = 0.5,
        skin_smooth: int = 0,
    ) -> Tuple[bytes, str]:
        rgb, alpha = self._decode(image_bytes)
        focus_x = 0.5 if not np.isfinite(focus_x) else float(min(max(focus_x, 0.0), 1.0))
        focus_y = 0.5 if not np.isfinite(focus_y) else float(min(max(focus_y, 0.0), 1.0))
        blur_level = int(min(max(blur_level, PORTRAIT_MIN_BLUR_LEVEL), PORTRAIT_MAX_BLUR_LEVEL))
        skin_smooth = int(min(max(skin_smooth, PORTRAIT_MIN_SKIN_SMOOTH), PORTRAIT_MAX_SKIN_SMOOTH))
        if blur_level <= PORTRAIT_MIN_BLUR_LEVEL and skin_smooth <= PORTRAIT_MIN_SKIN_SMOOTH:
            result = rgb                                  # level 1, no smoothing: completely untouched, no depth pass
        else:
            depth = self._cached_depth(image_bytes, rgb)
            mask = self._sharpness_mask(depth, rgb, focus_x, focus_y)
            subject = rgb
            if skin_smooth > PORTRAIT_MIN_SKIN_SMOOTH:
                try:
                    subject = smooth_skin(rgb, mask, skin_smooth)
                except Exception:
                    # Beauty smoothing is an enhancement: never fail the whole portrait because of it.
                    logger.exception("[portrait] skin smoothing failed; continuing without it")
                    subject = rgb
            result = subject if blur_level <= PORTRAIT_MIN_BLUR_LEVEL else self._composite(subject, mask, blur_level)

        output = io.BytesIO()
        if alpha is not None:
            image = Image.fromarray(np.dstack([result, alpha]), mode="RGBA")
        else:
            image = Image.fromarray(result, mode="RGB")
        # Lossless PNG whenever it fits under PNG_MAX_RESPONSE_BYTES (~3.5 MB; Vercel rejects responses over 4.5 MB with a
        # plain-text 413 the page cannot read). Level 1 = fast deflate: encode time stays small next to the depth pass.
        # A bigger result falls back to JPEG q92 (alpha photos stay PNG and shrink instead) - see encoding.encode_result.
        if PNG_MAX_RESPONSE_BYTES <= 0:
            image.save(output, format="PNG", compress_level=PORTRAIT_PNG_COMPRESS_LEVEL)
            return output.getvalue(), "image/png"
        return encode_result(image, png_level=PORTRAIT_PNG_COMPRESS_LEVEL, png_max_bytes=PNG_MAX_RESPONSE_BYTES,
                             jpeg_quality=RESULT_JPEG_QUALITY)
