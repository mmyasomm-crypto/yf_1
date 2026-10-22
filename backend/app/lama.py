"""LaMa ONNX model loading and image inpainting helpers."""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.config import LAMA_WARMUP_PASSES, MODEL_PATH, MODEL_SIZE, PNG_MAX_RESPONSE_BYTES, RESULT_JPEG_QUALITY
from backend.app.encoding import encode_result
from backend.app.ort_utils import CPU, build_session, check_model_file, effective_cpu_count, provider_preference

logger = logging.getLogger("yofilter.lama")
# Per-request timing report on the console is opt-in (LAMA_PROFILE=1); the Server-Timing header is always sent.
PROFILE = os.environ.get("LAMA_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}

# --- crop-window geometry (MUST stay in sync with js/editing/erase-window.js) ---------------------
# The network sees a square window: hole span + context on every side. Context per side is
# max(CONTEXT_MIN_PX, 60% of the hole span); the square is never smaller than MIN_CROP_SIDE (so tiny
# holes still see plenty of texture) and never larger than 2x the model input (bigger only blurs).
CONTEXT_MIN_PX = 48
MIN_CROP_SIDE = 256


def crop_side(span: int) -> int:
    context = max(CONTEXT_MIN_PX, (span * 3 + 4) // 5)
    return min(max(span + 2 * context, MIN_CROP_SIDE), MODEL_SIZE * 2)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class StepTimer:
    """High-precision per-step timer (time.perf_counter). ``mark(name)`` records the time since the previous mark."""

    __slots__ = ("t0", "last", "steps")

    def __init__(self) -> None:
        self.t0 = self.last = time.perf_counter()
        self.steps: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.steps.append((name, now - self.last))
        self.last = now

    @property
    def total(self) -> float:
        return time.perf_counter() - self.t0

    def report(self, header: str) -> str:
        total = max(self.total, 1e-9)
        lines = [f"[erase-profile] {header} | total {total * 1000:8.1f} ms"]
        for name, secs in self.steps:
            lines.append(f"[erase-profile]   {name:<44s} {secs * 1000:8.1f} ms  {secs / total * 100:5.1f}%")
        return "\n".join(lines)

    def server_timing(self) -> str:
        return ", ".join(f"{n.split(' ')[0].lower()};dur={s * 1000:.1f}" for n, s in self.steps)


class LamaInpainter:
    """Loads one ONNX session and safely serializes inference requests."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = check_model_file(model_path, "LaMa ONNX")

        self.threads = effective_cpu_count()
        # DirectML (GPU) first when onnxruntime-directml is installed, else the tuned CPU provider.
        self.session, self.provider = build_session(
            self.model_path, label="lama", preference=provider_preference("LAMA_PROVIDER"), threads=self.threads,
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        required = {"image", "mask"}
        if not required.issubset(self.input_names):
            raise RuntimeError(
                f"Unexpected model inputs {sorted(self.input_names)}; "
                "expected image and mask."
            )
        self.lock = threading.Lock()
        self.warmed_up = False

    def warm_up(self, passes: int = LAMA_WARMUP_PASSES) -> float:
        """Run dummy inference passes so graph optimisation, memory arenas and
        kernel selection happen at server start instead of on the first user
        request. Returns the wall-clock seconds spent."""
        started = time.perf_counter()
        if passes <= 0:                      # LAMA_WARMUP_PASSES=0: skip, model stays cold
            return 0.0
        image = np.full((1, 3, MODEL_SIZE, MODEL_SIZE), 0.5, dtype=np.float32)
        mask = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        centre, half = MODEL_SIZE // 2, MODEL_SIZE // 8
        mask[..., centre - half:centre + half, centre - half:centre + half] = 1.0
        for _ in range(passes):
            self._run(image, mask)
        self.warmed_up = True
        return time.perf_counter() - started

    def _run(self, image_tensor: np.ndarray, mask_tensor: np.ndarray) -> list:
        """Run inference; if a GPU (DirectML) run fails, rebuild a CPU session once and retry."""
        feeds = {"image": image_tensor, "mask": mask_tensor}
        with self.lock:
            try:
                return self.session.run(None, feeds)
            except Exception:
                if self.provider == CPU:
                    raise
                logger.exception("LaMa %s run failed; switching to the CPU provider", self.provider)
                self.session, self.provider = build_session(self.model_path, label="lama", preference="cpu", threads=self.threads)
                return self.session.run(None, feeds)

    @staticmethod
    def _decode_image(data: bytes, field_name: str, mode: str) -> Image.Image:
        if not data:
            raise ValueError(f"{field_name} image is empty")
        try:
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                return source.convert(mode)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"{field_name} is not a valid image") from exc

    def erase_profiled(self, image_bytes: bytes, mask_bytes: bytes, ref_size: int = 0) -> tuple[bytes, StepTimer]:
        """Inpaint the white area of ``mask`` and return a PNG of the ORIGINAL size.

        The 512x512 network only ever sees a square crop around the masked area
        (with context), so the object is repaired at the best resolution the photo
        offers instead of squeezing the whole picture to 512 px and stretching it
        back. Only the masked pixels (plus a thin feather) are replaced: everything
        else stays byte-identical to the upload.

        ``ref_size`` (optional): longest side of the *full* photo when the client uploads only a
        crop of it. It keeps the hole-growth / feather radius identical to a full-photo request,
        so a client-side crop gives exactly the same pixels as uploading everything.
        """
        timer = StepTimer()
        image = np.array(self._decode_image(image_bytes, "image", "RGB"), dtype=np.uint8)   # writable copy, edited in place
        mask_img = self._decode_image(mask_bytes, "mask", "L")
        height, width = image.shape[:2]
        if mask_img.size != (width, height):
            # The client sends a 1:1 mask; tolerate other sizes by mapping them onto the image.
            mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        raw_mask = np.asarray(mask_img, dtype=np.uint8)
        timer.mark("A decode image+mask (PIL->numpy)")

        # Bounding box of the painted area via one C pass (no full-frame nonzero/dilate/copy).
        bx, by, bw, bh = cv2.boundingRect((raw_mask > 127).astype(np.uint8))
        if bw == 0 or bh == 0:
            png = self._encode_png(image)
            timer.mark("F png encode (empty mask)")
            return png, timer

        # Grow the hole a little so stroke edges / anti-aliasing never survive.
        grow = max(2, int(round(max(width, height, int(ref_size or 0)) / 400.0)))
        # Everything below runs on this small window only: hole bbox + grow margin.
        wx0, wy0 = max(0, bx - grow), max(0, by - grow)
        wx1, wy1 = min(width, bx + bw + grow), min(height, by + bh + grow)
        hole_win = (raw_mask[wy0:wy1, wx0:wx1] > 127).astype(np.uint8)
        hole_win = cv2.dilate(hole_win, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        x0, x1, y0, y1 = wx0, wx1, wy0, wy1

        # Square crop = hole + generous context (see crop_side). Centred on the hole so a client-side
        # crop (js/editing/erase-window.js) reproduces exactly the same window.
        span = max(x1 - x0, y1 - y0)
        side = crop_side(span)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left, top = cx - side // 2, cy - side // 2
        right, bottom = left + side, top + side

        # Where the square leaves the photo, mirror the photo (texture-preserving) - and mirror the
        # hole mask the same way so a reflected copy of the object is never mistaken for clean context.
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - width), max(0, bottom - height)
        cl, ct, cr, cb = max(0, left), max(0, top), min(width, right), min(height, bottom)
        crop = image[ct:cb, cl:cr]
        # Full-size hole inside the crop window (hole_win pasted into a zero canvas of crop size).
        crop_mask = np.zeros((cb - ct, cr - cl), dtype=np.uint8)
        ix0, iy0 = max(x0, cl), max(y0, ct)
        ix1, iy1 = min(x1, cr), min(y1, cb)
        if ix1 > ix0 and iy1 > iy0:
            crop_mask[iy0 - ct:iy1 - ct, ix0 - cl:ix1 - cl] = hole_win[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        if pad_l or pad_t or pad_r or pad_b:
            ch, cw = crop.shape[:2]
            # Reflection needs pad < size; very small photos fall back to edge replication.
            mirror = max(pad_l, pad_r) < cw and max(pad_t, pad_b) < ch
            border = cv2.BORDER_REFLECT_101 if mirror else cv2.BORDER_REPLICATE
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, border)
            crop_mask_padded = cv2.copyMakeBorder(crop_mask, pad_t, pad_b, pad_l, pad_r, border)
        else:
            crop_mask_padded = crop_mask
        timer.mark("B bbox + dilate + crop slicing")

        if side != MODEL_SIZE:
            shrink = side > MODEL_SIZE
            model_image = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE),
                                     interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
            # AREA keeps thin strokes alive when shrinking; any coverage counts as hole.
            model_mask = (cv2.resize(crop_mask_padded.astype(np.float32), (MODEL_SIZE, MODEL_SIZE),
                                     interpolation=cv2.INTER_AREA) > 0.05).astype(np.float32)
        else:
            model_image = crop
            model_mask = crop_mask_padded.astype(np.float32)

        image_tensor = np.ascontiguousarray(model_image.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        image_tensor *= 1.0 / 255.0
        mask_tensor = np.ascontiguousarray(model_mask[None, None, ...], dtype=np.float32)
        timer.mark("C resize crop -> 512x512 + tensors")
        outputs = self._run(image_tensor, mask_tensor)
        if not outputs or outputs[0] is None:
            raise RuntimeError("LaMa returned no output")
        timer.mark("D onnx session.run (inference)")
        patch = np.asarray(self._to_rgb_image(outputs[0]), dtype=np.uint8)
        timer.mark("  output tensor -> uint8 patch")

        # Back to crop scale (only the crop is ever resized), then keep just the photo area of it.
        if side != MODEL_SIZE:
            patch = cv2.resize(patch, (side, side),
                               interpolation=cv2.INTER_CUBIC if side < MODEL_SIZE else cv2.INTER_AREA)
        patch = patch[pad_t:side - pad_b, pad_l:side - pad_r]

        self._blend(image, patch, crop_mask, ct, cl, grow)
        timer.mark("E resize back + feather + alpha paste")
        png = self._encode_png(image)
        timer.mark("F png encode")
        if PROFILE:
            logger.info(timer.report(f"image {width}x{height} in={len(image_bytes) / 1e6:.2f}MB mask={len(mask_bytes) / 1e6:.2f}MB "
                                     f"crop={side}px threads={self.threads}"))
        return png, timer

    @staticmethod
    def _blend(image: np.ndarray, patch: np.ndarray, hole: np.ndarray, ct: int, cl: int, grow: int) -> None:
        """Paste ``patch`` into ``image`` (in place) over the crop whose top-left is (ct, cl).

        1. Seam colour match: the network's low-frequency colour/brightness can differ slightly from the
           photo. The (patch - photo) difference is measured on the ring of KNOWN pixels around the hole
           (normalised Gaussian convolution, near + far scale) and added to the patch, so the seam has
           no step - a light-weight stand-in for Poisson blending.
        2. Feather: alpha is 1 inside the hole and falls off over ``feather`` px OUTSIDE it along the true
           distance to the hole (smoothstep), so edges stay perfectly sharp beyond the band and the band
           itself blends two valid, matching images instead of smearing the hole border.
        """
        ys, xs = np.nonzero(hole)
        if ys.size == 0:
            return
        feather = int(min(14, max(3, round(grow * 1.5))))
        ring_w = max(8, feather * 2)
        far_sigma = max(24.0, 0.12 * max(hole.shape))
        margin = int(max(ring_w * 3, far_sigma * 2)) + 1
        h, w = hole.shape
        fy0, fy1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin)
        fx0, fx1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin)
        sub = hole[fy0:fy1, fx0:fx1]
        # Photo pixels of this window (the "known" side) and the model's patch for the same pixels.
        region = image[ct + fy0:ct + fy1, cl + fx0:cl + fx1].astype(np.float32)
        new = patch[fy0:fy1, fx0:fx1].astype(np.float32)

        dist = cv2.distanceTransform((sub == 0).astype(np.uint8), cv2.DIST_L2, 3)   # px to nearest hole pixel
        alpha = np.where(sub > 0, 1.0, 1.0 - _smoothstep(dist / float(feather + 1))).astype(np.float32)

        ring = ((sub == 0) & (dist <= ring_w)).astype(np.float32)
        if ring.any():
            diff = (region - new) * ring[..., None]
            near_sigma = float(max(3.0, ring_w * 0.75))
            num_n = cv2.GaussianBlur(diff, (0, 0), near_sigma)
            den_n = cv2.GaussianBlur(ring, (0, 0), near_sigma)[..., None]
            num_f = cv2.GaussianBlur(diff, (0, 0), far_sigma)
            den_f = cv2.GaussianBlur(ring, (0, 0), far_sigma)[..., None]
            off_n = num_n / np.maximum(den_n, 1e-4)
            off_f = num_f / np.maximum(den_f, 1e-4)
            trust = np.clip(den_n / 0.25, 0.0, 1.0)            # near-field estimate where the ring is close
            offset = np.clip(trust * off_n + (1.0 - trust) * off_f, -32.0, 32.0)
            new = new + offset * alpha[..., None]               # correction fades out with the feather

        blended = region + (new - region) * alpha[..., None]
        image[ct + fy0:ct + fy1, cl + fx0:cl + fx1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    @staticmethod
    def _encode_png(pixels: np.ndarray) -> bytes:
        image = Image.fromarray(pixels, mode="RGB")
        if PNG_MAX_RESPONSE_BYTES <= 0:
            output = io.BytesIO()
            image.save(output, format="PNG", compress_level=0)     # raw/stored deflate: fastest; size is not critical on LAN/localhost
            return output.getvalue()
        # Stored deflate is ~3 bytes/pixel: a big window would break a 4.5 MB response limit. Level 1 keeps the encode fast
        # and the result lossless for every window the client normally sends (<= ~1060 px); only an oversized result
        # falls back to JPEG (callers read the media type from the first bytes, see main.erase).
        return encode_result(image, png_level=1, png_max_bytes=PNG_MAX_RESPONSE_BYTES, jpeg_quality=RESULT_JPEG_QUALITY)[0]

    @staticmethod
    def _to_rgb_image(raw_output: np.ndarray) -> Image.Image:
        output = np.asarray(raw_output, dtype=np.float32)
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
            raise RuntimeError(f"Unexpected LaMa output shape: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("LaMa returned non-finite pixel values")

        output = np.transpose(output[0], (1, 2, 0))
        value_min = float(output.min())
        value_max = float(output.max())
        if value_min >= -1e-4 and value_max <= 1.0001:
            output = output * 255.0
        elif value_min >= -1.0001 and value_max <= 1.0001:
            output = (output + 1.0) * 127.5
        elif value_min >= -1e-4 and value_max <= 255.0001:
            # This bundled export returns values in the 0..255 range.
            pass
        else:
            raise RuntimeError(
                f"Unsupported LaMa output value range [{value_min:.3f}, {value_max:.3f}]"
            )

        pixels = np.clip(np.rint(output), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")
