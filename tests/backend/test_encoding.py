"""Response-size guard (Vercel rejects function responses over 4.5 MB): PNG while it fits, else JPEG q92, never over the cap.

Pure Pillow + NumPy - needs no ONNX model, so it always runs (also in CI without the model files)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from backend.app.encoding import DEFAULT_PNG_MAX_BYTES, JPEG, PNG, encode_result

RNG = np.random.default_rng(7)


def _photo(w: int, h: int) -> Image.Image:
    """Smooth gradient + mild noise: compresses like a real photo (PNG far bigger than JPEG)."""
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)], -1).astype(np.int16)
    return Image.fromarray(np.clip(base + RNG.integers(-18, 18, (h, w, 3)), 0, 255).astype(np.uint8))


def test_small_result_stays_lossless_png():
    data, media = encode_result(_photo(600, 400))
    assert media == PNG and Image.open(io.BytesIO(data)).format == "PNG"


def test_big_photo_falls_back_to_jpeg_q92_under_the_cap_at_full_size():
    image = _photo(2048, 1536)
    png_size = len(encode_result(image, png_max_bytes=10**9)[0])
    assert png_size > DEFAULT_PNG_MAX_BYTES, "fixture must be a case where PNG is too big"
    data, media = encode_result(image)
    assert media == JPEG and len(data) <= DEFAULT_PNG_MAX_BYTES
    decoded = Image.open(io.BytesIO(data))
    assert decoded.format == "JPEG" and decoded.size == image.size            # no downscale needed


def test_transparent_photo_is_never_flattened_to_jpeg():
    rgba = _photo(1800, 1200).convert("RGBA")
    rgba.putalpha(Image.fromarray(np.tile(np.linspace(0, 255, 1800, dtype=np.uint8), (1200, 1))))
    data, media = encode_result(rgba, png_max_bytes=1_500_000)
    assert media == PNG
    decoded = Image.open(io.BytesIO(data))
    assert decoded.mode == "RGBA" and len(data) <= 1_500_000                  # shrunk, alpha kept


def test_hopeless_noise_shrinks_until_it_fits_instead_of_returning_an_oversized_body():
    noise = Image.fromarray(RNG.integers(0, 255, (2048, 2048, 3), dtype=np.uint8))
    data, media = encode_result(noise)
    assert media == JPEG and len(data) <= DEFAULT_PNG_MAX_BYTES
    assert max(Image.open(io.BytesIO(data)).size) < 2048


def test_custom_limits_are_honoured():
    data, media = encode_result(_photo(1000, 800), png_max_bytes=400_000, jpeg_quality=85)
    assert media == JPEG and len(data) <= 400_000
