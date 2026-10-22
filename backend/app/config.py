"""Application settings and absolute model path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "inpainting_lama.onnx"

# --- Serverless (Vercel / AWS Lambda) --------------------------------------------------------------------------
# The deployment bundle is read-only; /tmp is the only writable place (and it is per instance, wiped on cold start).
# There the ONNX models are NOT bundled: they are downloaded on first use into MODEL_CACHE_DIR (see main.py).
#   YOFILTER_MODEL_DIR=/some/dir   force a model cache directory anywhere (also useful locally)
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def _model_cache_dir() -> Path | None:
    raw = os.environ.get("YOFILTER_MODEL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/yofilter-models") if IS_SERVERLESS else None


MODEL_CACHE_DIR = _model_cache_dir()
if IS_SERVERLESS:
    # huggingface_hub writes its own cache / metadata next to the HOME directory, which is read-only here.
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _load_app_meta() -> dict[str, str]:
    """Single source of truth for the product name / version: ``version.json`` in the project root.

    ``scripts/sync_version.py`` stamps the same value into config.xml, js/i18n/i18n.js and index.html, and a test
    fails if any copy drifts. A missing / unreadable file never stops the server: it falls back to neutral values.
    """
    meta = {"name": "YoFilter", "api_title": "YoFilter API", "version": "0.0.0"}
    try:
        data = json.loads((PROJECT_DIR / "version.json").read_text(encoding="utf-8"))
        for key in meta:
            if isinstance(data.get(key), str) and data[key].strip():
                meta[key] = data[key].strip()
    except (OSError, ValueError, AttributeError):
        pass
    return meta


APP_META = _load_app_meta()
APP_VERSION = APP_META["version"]
APP_TITLE = APP_META["api_title"]
MODEL_SIZE = 512
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# --- Portrait (Depth Anything V2 Small) ---------------------------------------
DEFAULT_PORTRAIT_MODEL_PATH = BACKEND_DIR / "models" / "portrait_depth.onnx"


def _resolve_model_path(env_name: str, default: Path, filename: str) -> Path:
    """Absolute model path that survives any working directory.

    Order: $ENV override -> model cache dir (/tmp/yofilter-models on serverless) -> backend/models (next to this
    source file) -> <project>/models -> ./backend/models and ./models relative to the cwd. The first candidate that
    exists wins (a stale override never hides a real file). If none exists, the path the model will be DOWNLOADED to
    is returned: the override, else the cache dir (serverless), else the default - so error messages name a sensible
    location and ``ensure_model_file`` knows where to write.
    """
    candidates: list[Path] = []
    override = os.environ.get(env_name, "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    cached = MODEL_CACHE_DIR / filename if MODEL_CACHE_DIR else None
    if cached:
        candidates.append(cached)
    candidates += [default, PROJECT_DIR / "models" / filename,
                   Path.cwd() / "backend" / "models" / filename, Path.cwd() / "models" / filename]
    for candidate in candidates:
        try:
            if candidate.resolve().is_file():
                return candidate.resolve()
        except OSError:
            continue
    if override:
        return Path(override).expanduser().resolve()
    return (cached or default).resolve()


MODEL_PATH = _resolve_model_path("LAMA_MODEL_PATH", DEFAULT_MODEL_PATH, "inpainting_lama.onnx")


PORTRAIT_MODEL_PATH = _resolve_model_path("PORTRAIT_MODEL_PATH", DEFAULT_PORTRAIT_MODEL_PATH, "portrait_depth.onnx")
PORTRAIT_INPUT_SIZE = 518  # Depth-Anything-V2 ViT patch size 14 * 37
PORTRAIT_MIN_BLUR_LEVEL = 1    # 1 = completely sharp (no blur applied)
PORTRAIT_MAX_BLUR_LEVEL = 30   # 30 = widest f/1.4-style bokeh
PORTRAIT_DEFAULT_BLUR_LEVEL = 1
PORTRAIT_MIN_SKIN_SMOOTH = 0      # 0 = skin smoothing off (photo skin untouched)
PORTRAIT_MAX_SKIN_SMOOTH = 100    # 100 = strongest (still texture-preserving) beauty smoothing
PORTRAIT_DEFAULT_SKIN_SMOOTH = 0
PORTRAIT_MAX_PIXELS = 64_000_000   # decoded-size guard (decompression bombs / OOM): larger photos answer 400


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


PORTRAIT_TIMEOUT_S = _float_env("PORTRAIT_TIMEOUT_S", 120.0)       # server-side cap per request -> 504 (with CORS)
PORTRAIT_MAX_CONCURRENCY = max(1, int(_float_env("PORTRAIT_MAX_CONCURRENCY", 1)))   # parallel blur jobs (memory)


# --- Public-traffic hardening (P1.3) -------------------------------------------
def _int_env(name: str, default: int, minimum: int = 0) -> int:
    """Integer env var; garbage or empty falls back to the default, never below ``minimum``."""
    try:
        return max(minimum, int(float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Origin of a packaged Android build (Cordova's default https://localhost scheme/host). It is the ONLY cross-origin
# caller a production deployment needs: the browser build is served by this same backend (same origin, no CORS).
NATIVE_APP_ORIGINS = ["https://localhost"]


def resolve_allowed_origins(raw: str | None, production: bool) -> list[str]:
    """CORS allow-list from the ``ALLOWED_ORIGINS`` value.

    * set            -> exactly that list (whitespace / trailing slashes stripped); an explicit ``*`` is honoured
                        (the operator asked for it; production start-up logs a warning).
    * unset, dev     -> ``["*"]`` (local development from any port).
    * unset, prod    -> ``NATIVE_APP_ORIGINS`` only: cross-origin is DENIED for every web origin.
    """
    origins = [part.strip().rstrip("/") for part in (raw or "").split(",") if part.strip()]
    if not origins:
        return list(NATIVE_APP_ORIGINS) if production else ["*"]
    return ["*"] if "*" in origins else origins


# --- Lazy model loading + retries (serverless) ---------------------------------------------------------------
# Lazy (default ON on serverless, OFF elsewhere so local dev / the test-suite keep the eager boot): nothing is
# downloaded or loaded at start-up. The first /api/erase (LaMa) or /api/portrait (depth) request downloads its OWN model
# into MODEL_CACHE_DIR and loads it; the other model is never touched by that request. A failed attempt is retried
# MODEL_LOAD_RETRIES times (linear back-off MODEL_RETRY_DELAY_S * attempt); after the last failure the endpoint answers
# 503 immediately for MODEL_FAILURE_COOLDOWN_S seconds instead of re-downloading ~100 MB on every request.
LAZY_MODEL_LOAD = _bool_env("YOFILTER_LAZY_MODELS", "1" if IS_SERVERLESS else "0")
MODEL_LOAD_RETRIES = _int_env("MODEL_LOAD_RETRIES", 3, minimum=1)
MODEL_RETRY_DELAY_S = float(_int_env("MODEL_RETRY_DELAY_S", 2))
MODEL_FAILURE_COOLDOWN_S = float(_int_env("MODEL_FAILURE_COOLDOWN_S", 20))

# --- Response size guard (Vercel rejects function responses over 4.5 MB) ---------------------------------------
# Results are PNG while they fit under PNG_MAX_RESPONSE_BYTES, otherwise JPEG q92 (see encoding.py). 0 disables the guard.
PNG_MAX_RESPONSE_BYTES = _int_env("PNG_MAX_RESPONSE_BYTES", 3_500_000)
RESULT_JPEG_QUALITY = min(100, max(50, _int_env("RESULT_JPEG_QUALITY", 92)))

ERASE_MAX_CONCURRENCY = _int_env("ERASE_MAX_CONCURRENCY", 1, minimum=1)   # parallel LaMa jobs (CPU/memory)
ERASE_MAX_QUEUE = _int_env("ERASE_MAX_QUEUE", 4)                          # requests allowed to WAIT for a slot; more -> 503
ERASE_MAX_PIXELS = _int_env("ERASE_MAX_PIXELS", 64_000_000, minimum=1)    # decoded-size guard for /api/erase uploads
RATE_LIMIT_PER_MIN = _int_env("RATE_LIMIT_PER_MIN", 30)                   # POSTs per client per minute on /api/*; 0 = off
TRUST_PROXY = _bool_env("TRUST_PROXY", "0")                               # honour X-Forwarded-For (only behind YOUR proxy)
# Set YOFILTER_ENV=production on public deployments: hides /docs, /redoc, /openapi.json, trims /health, adds the
# security headers and flips the CORS default to "native app origin only" (see resolve_allowed_origins).
IS_PRODUCTION = os.environ.get("YOFILTER_ENV", os.environ.get("ENV", "")).strip().lower() in {"prod", "production"}
ALLOWED_ORIGINS = resolve_allowed_origins(os.environ.get("ALLOWED_ORIGINS"), IS_PRODUCTION)   # "*" = any origin

# LaMa warm-up: dummy inference passes run once at server start so the first
# real /api/erase request never pays the ONNX Runtime cold-start cost.
# Default is 1 pass (a second pass froze boot for ~75 s on low-core CPUs).
# Set LAMA_WARMUP_PASSES=0 to skip warm-up entirely (fastest boot, first erase is slower).
def _warmup_passes() -> int:
    try:
        return max(0, int(os.environ.get("LAMA_WARMUP_PASSES", "1")))
    except ValueError:
        return 1


LAMA_WARMUP_PASSES = _warmup_passes()
