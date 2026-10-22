"""Result encoding that keeps every API response under the serverless body limit.

Vercel rejects (413 FUNCTION_PAYLOAD_TOO_LARGE) any function response over 4.5 MB, and that rejection never reaches
our code or the page as JSON. A lossless PNG of a detailed 2048 px photo is routinely 6-10 MB, so results are encoded
here with a ladder:

    1. PNG (lossless) - used whenever it fits under ``png_max_bytes`` (default ~3.5 MB, 1 MB of headroom).
    2. Photos without transparency: JPEG, quality 92 (4:4:4 chroma), then 88/84/80/75 if still too big.
    3. Still too big (or the image has an alpha channel, which JPEG cannot carry): shrink the picture by 15 % per round
       (PNG for alpha, JPEG q80 otherwise) until it fits or the longest side would drop below ``MIN_SIDE``.

Callers that need the original pixel size (the frontend does) resize the answer back; the media type is always returned
so the HTTP layer can label it. Pure Pillow - no FastAPI / ONNX imports, so it is unit-tested in isolation.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

logger = logging.getLogger("yofilter.encoding")

DEFAULT_PNG_MAX_BYTES = 3_500_000     # 4.5 MB platform limit - 1 MB headroom for headers / multipart-free body
DEFAULT_JPEG_QUALITY = 92
JPEG_LADDER = (92, 88, 84, 80, 75)
SHRINK_STEP = 0.85
MIN_SIDE = 640
MAX_SHRINK_ROUNDS = 12

PNG = "image/png"
JPEG = "image/jpeg"


def has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def _png(image: Image.Image, level: int) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", compress_level=level)
    return out.getvalue()


def _jpeg(image: Image.Image, quality: int) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=quality, subsampling=0)   # no optimize=True: Pillow fails with "broken data stream" on big noisy frames
    return out.getvalue()


def _shrunk(image: Image.Image, factor: float) -> Image.Image | None:
    w, h = image.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    if max(nw, nh) < MIN_SIDE:
        return None
    return image.resize((nw, nh), Image.LANCZOS)


def encode_result(image: Image.Image, *, png_level: int = 1, png_max_bytes: int = DEFAULT_PNG_MAX_BYTES,
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> tuple[bytes, str]:
    """Encode ``image`` and return ``(bytes, media_type)`` with ``len(bytes) <= png_max_bytes`` whenever that is possible."""
    limit = max(1, int(png_max_bytes))
    png = _png(image, png_level)
    if len(png) <= limit:
        return png, PNG

    alpha = has_alpha(image)
    if not alpha:
        ladder = (jpeg_quality,) + tuple(q for q in JPEG_LADDER if q < jpeg_quality)
        for quality in ladder:
            data = _jpeg(image, quality)
            if len(data) <= limit:
                logger.info("result PNG %.1f MB > %.1f MB: sent as JPEG q%d (%.1f MB)",
                            len(png) / 1e6, limit / 1e6, quality, len(data) / 1e6)
                return data, JPEG
    else:
        data = _png(image, 9)                                  # transparency must stay PNG: try the smallest lossless encode
        if len(data) <= limit:
            return data, PNG

    # Last resort: fewer pixels. The caller restores the size; a result that fits beats a platform-level 413.
    current = image
    best = (png, PNG) if alpha else (_jpeg(image, JPEG_LADDER[-1]), JPEG)
    for _ in range(MAX_SHRINK_ROUNDS):
        smaller = _shrunk(current, SHRINK_STEP)
        if smaller is None:
            break
        current = smaller
        data, media = (_png(current, 9), PNG) if alpha else (_jpeg(current, 80), JPEG)
        best = (data, media)
        if len(data) <= limit:
            logger.warning("result needed downscaling to %dx%d to fit %.1f MB", current.width, current.height, limit / 1e6)
            return best
    logger.error("result could not be brought under %.1f MB (best %.1f MB at %dx%d)",
                 limit / 1e6, len(best[0]) / 1e6, current.width, current.height)
    return best
