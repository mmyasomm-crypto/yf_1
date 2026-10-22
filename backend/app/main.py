"""YoFilter standalone FastAPI service."""

from __future__ import annotations

import asyncio
import io
import mimetypes
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
from PIL import Image
from starlette.concurrency import run_in_threadpool

from backend.app.config import (
    ALLOWED_ORIGINS,
    APP_TITLE,
    APP_VERSION,
    ERASE_MAX_CONCURRENCY,
    ERASE_MAX_PIXELS,
    ERASE_MAX_QUEUE,
    IS_PRODUCTION,
    IS_SERVERLESS,
    LAMA_WARMUP_PASSES,
    LAZY_MODEL_LOAD,
    MAX_UPLOAD_BYTES,
    MODEL_FAILURE_COOLDOWN_S,
    MODEL_LOAD_RETRIES,
    MODEL_PATH,
    MODEL_RETRY_DELAY_S,
    PORTRAIT_DEFAULT_BLUR_LEVEL,
    PORTRAIT_DEFAULT_SKIN_SMOOTH,
    PORTRAIT_MAX_BLUR_LEVEL,
    PORTRAIT_MAX_CONCURRENCY,
    PORTRAIT_MAX_SKIN_SMOOTH,
    PORTRAIT_MIN_BLUR_LEVEL,
    PORTRAIT_MIN_SKIN_SMOOTH,
    PORTRAIT_MODEL_PATH,
    PORTRAIT_TIMEOUT_S,
    PROJECT_DIR,
    RATE_LIMIT_PER_MIN,
    TRUST_PROXY,
)
from backend.app.lama import PROFILE, LamaInpainter
from backend.app.limits import rate_limit
from backend.app.slots import run_in_owned_slot
from backend.app.log import configure_logging
from backend.app.ort_utils import MIN_MODEL_BYTES
from backend.app.portrait import DepthPortrait
from backend.app.security import (
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    PORTRAIT_UNAVAILABLE_MESSAGE,
    health_payload,
    public_model_error,
    security_headers,
)


logger = configure_logging()


EXPOSED_HEADERS = ["Server-Timing", "Content-Disposition"]


def cors_headers_for(request: Request) -> dict[str, str]:
    """CORS headers for responses produced OUTSIDE CORSMiddleware (Starlette's last-resort 500 handler).
    Same rule as the middleware: the Origin is echoed only if it is on ALLOWED_ORIGINS - never a blanket '*'
    on a restricted deployment."""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    if "*" in ALLOWED_ORIGINS:
        allow = "*"
    elif origin.rstrip("/") in ALLOWED_ORIGINS:
        allow = origin
    else:
        return {}
    return {"Access-Control-Allow-Origin": allow, "Access-Control-Expose-Headers": ", ".join(EXPOSED_HEADERS),
            **({"Vary": "Origin"} if allow != "*" else {})}


def internal_error(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback under a short reference id and answer JSON (with CORS headers) that quotes it."""
    error_id = uuid.uuid4().hex[:8]
    logger.error("[%s %s] unhandled exception (ref %s) from %s", request.method, request.url.path, error_id,
                 request.client.host if request.client else "?", exc_info=(type(exc), exc, exc.__traceback__))
    return error_response(500, f"Internal server error (ref {error_id}); see the server log.",
                          cors_headers_for(request))


def error_response(status: int, detail: str, extra: dict[str, str] | None = None) -> JSONResponse:
    """JSON error body. CORS headers are NOT added here: CORSMiddleware is outermost and stamps every
    response (2xx, 4xx, 429, 503, 5xx) with the headers for the requesting origin. Hard-coding '*' here
    would leak access to origins that ALLOWED_ORIGINS excludes."""
    return JSONResponse(status_code=status, content={"detail": detail}, headers=extra or None)


HF_MODELS_REPO = "ertegfg/yofilter-models"            # Hugging Face repo that hosts the ONNX models
LAMA_MODEL_FILENAME = "inpainting_lama.onnx"
PORTRAIT_MODEL_FILENAME = "portrait_depth.onnx"


def _model_is_usable(path: Path) -> bool:
    """True for a real model file. A git-lfs pointer (~130 B) or a truncated file counts as missing, so a
    checkout without LFS content is repaired by the download instead of failing later inside ONNX Runtime."""
    try:
        return path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES
    except OSError:
        return False


def ensure_model_file(path: Path, filename: str, label: str) -> bool:
    """Make sure ``path`` holds the model: use the local file when it is there, otherwise download
    ``filename`` from ``HF_MODELS_REPO`` with huggingface_hub. Blocking - call it from a worker thread.

    Never raises: on failure it logs the reason and returns False, so the server still starts and only the
    endpoint that needs this model answers 503. Private repo: set the HF_TOKEN environment variable
    (huggingface_hub reads it automatically)."""
    if _model_is_usable(path):
        logger.info("%s model found locally: %s", label, path)
        return True
    logger.info("%s model not found locally (%s) - downloading %s from Hugging Face repo %s",
                label, path, filename, HF_MODELS_REPO)
    try:
        from huggingface_hub import hf_hub_download         # imported lazily: only needed when a download happens

        path.parent.mkdir(parents=True, exist_ok=True)
        # local_dir puts a real file (no symlink into ~/.cache) next to where the app expects it; the download goes
        # to a temporary ".incomplete" file first, so an interrupted deploy never leaves a half-written model.
        downloaded = Path(hf_hub_download(repo_id=HF_MODELS_REPO, filename=filename, local_dir=str(path.parent)))
        if downloaded.resolve() != path.resolve():          # LAMA_MODEL_PATH / PORTRAIT_MODEL_PATH uses another file name
            shutil.copyfile(downloaded, path)
        if not _model_is_usable(path):
            raise RuntimeError(f"downloaded file is smaller than {MIN_MODEL_BYTES} bytes")
        logger.info("%s model ready: %s (%.1f MB)", label, path, path.stat().st_size / 1e6)
        return True
    except Exception:
        logger.exception("Could not download the %s model from %s - its endpoint will answer 503", label, HF_MODELS_REPO)
        return False


def load_portrait(app: FastAPI) -> bool:
    """(Re)load the portrait depth model into app.state. Never raises; the reason is kept for 503s."""
    try:
        app.state.portrait = DepthPortrait()
        app.state.portrait_error = None
        return True
    except Exception as exc:
        app.state.portrait = None
        app.state.portrait_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Portrait depth model failed to load from %s", PORTRAIT_MODEL_PATH)
        return False


# ---------------------------------------------------------------------------------------------------------------
# Application state + lazy model loading.
#
# Serverless platforms (Vercel) may start the app without running the ASGI lifespan, and a cold start must stay
# cheap: nothing heavy happens at boot. init_app_state() therefore creates every piece of shared state on demand
# (idempotent), and get_inpainter() / get_portrait() download + load a model the first time an endpoint needs it.
# Outside serverless (LAZY_MODEL_LOAD off) the original eager boot in lifespan() is unchanged.
# ---------------------------------------------------------------------------------------------------------------
def init_app_state(app: FastAPI, *, force: bool = False) -> None:
    state = app.state
    if getattr(state, "_yf_state_ready", False) and not force:
        return
    state.inpainter = None
    state.portrait = None
    state.portrait_error = None
    state.lama_error = None
    state.lama_retry_after = 0.0              # time.monotonic() before which a failed LaMa load is not retried
    state.portrait_retry_after = 0.0
    state.lama_load_lock = asyncio.Lock()
    state.portrait_load_lock = asyncio.Lock()
    state.portrait_slots = asyncio.Semaphore(PORTRAIT_MAX_CONCURRENCY)
    state.erase_slots = asyncio.Semaphore(ERASE_MAX_CONCURRENCY)   # parallel LaMa jobs
    state.erase_waiting = 0                                        # requests queued for a slot
    state._yf_state_ready = True


def _build_lama() -> LamaInpainter:
    """Blocking (worker thread): make sure the model file is on disk (download into the cache dir if needed), then load it."""
    if not ensure_model_file(MODEL_PATH, LAMA_MODEL_FILENAME, "LaMa"):
        raise RuntimeError("LaMa model file is not available (download failed)")
    return LamaInpainter()


def _load_portrait_with_download(app: FastAPI) -> bool:
    """Blocking (worker thread). Lazy mode downloads the file first; eager mode already did that at boot."""
    if LAZY_MODEL_LOAD and not ensure_model_file(PORTRAIT_MODEL_PATH, PORTRAIT_MODEL_FILENAME, "Portrait depth"):
        app.state.portrait = None
        app.state.portrait_error = "RuntimeError: portrait model file is not available (download failed)"
        return False
    return load_portrait(app)


async def get_inpainter(app: FastAPI) -> LamaInpainter | None:
    """The LaMa session, or None (-> 503). In lazy mode the first call downloads + loads it (retried with back-off); after
    the last failed attempt further calls answer None immediately for MODEL_FAILURE_COOLDOWN_S seconds. When lazy
    loading is off a missing session stays missing (the eager boot already tried)."""
    init_app_state(app)
    state = app.state
    if state.inpainter is not None or not LAZY_MODEL_LOAD:
        return state.inpainter
    async with state.lama_load_lock:                        # concurrent first requests share ONE download
        if state.inpainter is not None:
            return state.inpainter
        if time.monotonic() < state.lama_retry_after:
            return None
        for attempt in range(1, MODEL_LOAD_RETRIES + 1):
            try:
                state.inpainter = await run_in_threadpool(_build_lama)
                state.lama_error = None
                logger.info("LaMa model ready (attempt %d/%d) on %s", attempt, MODEL_LOAD_RETRIES, state.inpainter.provider)
                return state.inpainter
            except Exception as exc:
                state.lama_error = f"{type(exc).__name__}: {exc}"
                logger.exception("LaMa model load failed (attempt %d/%d)", attempt, MODEL_LOAD_RETRIES)
                if attempt < MODEL_LOAD_RETRIES:
                    await asyncio.sleep(MODEL_RETRY_DELAY_S * attempt)
        state.lama_retry_after = time.monotonic() + MODEL_FAILURE_COOLDOWN_S
        return None


async def get_portrait(app: FastAPI) -> DepthPortrait | None:
    """The depth model, or None (-> 503). Always retries once on demand (a model copied in later, a transient error);
    lazy mode downloads it first and retries MODEL_LOAD_RETRIES times with back-off, then cools down like get_inpainter."""
    init_app_state(app)
    state = app.state
    if state.portrait is not None:
        return state.portrait
    async with state.portrait_load_lock:
        if state.portrait is not None:
            return state.portrait
        if LAZY_MODEL_LOAD and time.monotonic() < state.portrait_retry_after:
            return None
        attempts = MODEL_LOAD_RETRIES if LAZY_MODEL_LOAD else 1
        for attempt in range(1, attempts + 1):
            if await run_in_threadpool(_load_portrait_with_download, app):
                logger.info("Portrait depth model ready (attempt %d/%d)", attempt, attempts)
                return state.portrait
            if attempt < attempts:
                await asyncio.sleep(MODEL_RETRY_DELAY_S * attempt)
        if LAZY_MODEL_LOAD:
            state.portrait_retry_after = time.monotonic() + MODEL_FAILURE_COOLDOWN_S
        return state.portrait


async def _warm_up_lama(app: FastAPI) -> None:
    """Silent background warm-up: the server answers immediately instead of waiting for it."""
    inpainter = app.state.inpainter
    if LAMA_WARMUP_PASSES <= 0 or inpainter is None:
        return
    try:
        await run_in_threadpool(inpainter.warm_up)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("LaMa warm-up failed; the first erase will be slower")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_app_state(app, force=True)
    if LAZY_MODEL_LOAD:
        # Serverless: no download, no model load, no warm-up at start-up (a cold start must answer fast and stay far
        # below the platform timeout). Each model is fetched by the first request that needs it.
        logger.info("Lazy model loading: models are fetched into %s on first use (serverless=%s)", MODEL_PATH.parent, IS_SERVERLESS)
    else:
        # Each model is optional at boot: a missing/broken file disables only ITS endpoint (503 with the
        # reason) and never prevents the server from starting. Models are loaded in worker threads.
        # Render (and any fresh container) ships without the ~190 MB ONNX files: fetch whatever is missing from
        # Hugging Face first (both downloads in parallel; a no-op when the files are already on disk).
        await asyncio.gather(
            run_in_threadpool(ensure_model_file, MODEL_PATH, LAMA_MODEL_FILENAME, "LaMa"),
            run_in_threadpool(ensure_model_file, PORTRAIT_MODEL_PATH, PORTRAIT_MODEL_FILENAME, "Portrait depth"),
        )
        try:
            app.state.inpainter = await run_in_threadpool(LamaInpainter)
        except Exception:
            logger.exception("LaMa model failed to load from %s - /api/erase will answer 503", MODEL_PATH)
        await run_in_threadpool(load_portrait, app)       # before the (slow) LaMa warm-up
    if "*" in ALLOWED_ORIGINS:
        (logger.warning if IS_PRODUCTION else logger.info)(
            "CORS allows ANY origin (ALLOWED_ORIGINS=*)%s", " in production - set ALLOWED_ORIGINS to your site(s)" if IS_PRODUCTION else "")
    elif IS_PRODUCTION:
        logger.info("CORS allow-list: %s", ", ".join(ALLOWED_ORIGINS) or "(none)")
    logger.info("Hardening: rate limit %s/min per client (trust proxy: %s) | erase: %d concurrent, queue %d, max %d MP",
                RATE_LIMIT_PER_MIN or "off", TRUST_PROXY, ERASE_MAX_CONCURRENCY, ERASE_MAX_QUEUE, ERASE_MAX_PIXELS // 1_000_000)
    if not LAZY_MODEL_LOAD:
        logger.info("YoFilter ready - eraser: %s | portrait: %s",
                    app.state.inpainter.provider if app.state.inpainter else "UNAVAILABLE",
                    app.state.portrait.provider if app.state.portrait else "UNAVAILABLE")
    warm_task = asyncio.create_task(_warm_up_lama(app))
    try:
        yield
    finally:
        warm_task.cancel()
        app.state.inpainter = None
        app.state.portrait = None


app = FastAPI(
    title=APP_TITLE,                                    # "YoFilter API" - from version.json (single source)
    version=APP_VERSION,
    description="YoFilter image-editing API: LaMa object erase and depth-based portrait blur.",
    lifespan=lifespan,
    # Interactive docs and the machine-readable schema expose every route / parameter: off in production.
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)


@app.middleware("http")
async def never_drop_the_connection(request: Request, call_next):
    """Turn ANY escaped exception into a JSON 500. Added before CORSMiddleware, so CORS wraps it and the
    error reaches the page with headers instead of a silently dropped connection."""
    try:
        return await call_next(request)
    except asyncio.CancelledError:
        raise                                   # client went away: let Starlette clean up
    except Exception as exc:
        return internal_error(request, exc)


# Middleware order (add_middleware inserts OUTSIDE everything added before it):
#   never_drop_the_connection (innermost)  ->  rate_limit  ->  CORSMiddleware (outermost, added LAST)
# so a 429 from the limiter, a 503 "busy" and a JSON 500 all leave through CORS with proper headers.
app.middleware("http")(rate_limit)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
    expose_headers=EXPOSED_HEADERS,
    max_age=86400,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Outermost layer (added after CORSMiddleware): every response that leaves the app - 2xx, 4xx, 429, 503 -
    carries nosniff, and in production the CSP / frame-ancestors / X-Frame-Options / Referrer-Policy / HSTS set.
    The production flag is read at call time (module global) so tests can flip it."""
    response = await call_next(request)
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower() if TRUST_PROXY else ""
    https = request.url.scheme == "https" or forwarded == "https"
    for name, value in security_headers(request.url.path, IS_PRODUCTION, https).items():
        response.headers.setdefault(name, value)
    return response


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log WHY a request was rejected (422), then answer exactly like FastAPI would."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg')}" for err in exc.errors()
    )
    logger.warning(
        "[%s %s] 422 validation failed from %s (content-type=%s, content-length=%s): %s",
        request.method, request.url.path, request.client.host if request.client else "?",
        request.headers.get("content-type", "-").split(";")[0], request.headers.get("content-length", "-"), problems,
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def log_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: log the traceback and answer with JSON that still carries CORS headers.

    Starlette runs this handler *outside* the CORS middleware, so without the explicit header a
    cross-origin page would only see an opaque "Failed to fetch" instead of the server's message."""
    return internal_error(request, exc)    # attaches CORS headers itself (this handler sits outside the middleware)


def installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def _health_details(request: Request) -> dict[str, object]:
    """Diagnostics for development only: absolute model paths, provider list, dependency versions, load errors."""
    inpainter = getattr(request.app.state, "inpainter", None)
    ort_version = installed_version("onnxruntime")
    dependencies = {
        "fastapi": installed_version("fastapi"),
        "uvicorn": installed_version("uvicorn"),
        "onnxruntime": ort_version if ort_version != "missing" else installed_version("onnxruntime-directml"),
        "pillow": installed_version("Pillow"),
        "numpy": installed_version("numpy"),
        "opencv": installed_version("opencv-python-headless"),
    }
    return {
        "model_warmed_up": bool(getattr(inpainter, "warmed_up", False)),
        "model_path": str(MODEL_PATH),
        "lama_provider": getattr(inpainter, "provider", None),
        "portrait_model_path": str(PORTRAIT_MODEL_PATH),
        "portrait_model_exists": PORTRAIT_MODEL_PATH.is_file(),
        "portrait_load_error": getattr(request.app.state, "portrait_error", None),
        "erase_queue": getattr(request.app.state, "erase_waiting", 0),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "onnx_providers": ort.get_available_providers(),
        "dependencies": dependencies,
    }


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Production profile (YOFILTER_ENV=production): ONLY status + the two model flags. Development: full diagnostics.
    With lazy loading the flags are False until the first erase / portrait request has loaded that model."""
    init_app_state(request.app)
    return health_payload(
        IS_PRODUCTION,
        model_loaded=getattr(request.app.state, "inpainter", None) is not None,
        portrait_loaded=getattr(request.app.state, "portrait", None) is not None,
        detailed=lambda: _health_details(request),
    )


def _reject_if_huge(data: bytes, limit: int, what: str) -> None:
    """400 if the image's DECODED size would exceed ``limit`` pixels (decompression bombs / OOM).

    PIL reads only the header in Image.open(), so this costs microseconds and never allocates the pixels.
    Undecodable data is left alone: the real decoder rejects it later with its own 'not a valid image'."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            pixels = probe.width * probe.height
            size = f"{probe.width}x{probe.height}"
    except Image.DecompressionBombError:
        raise HTTPException(status_code=400, detail=f"{what} is too large to process (limit {limit // 1_000_000} megapixels)")
    except Exception:
        return
    if pixels > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{what} is {pixels / 1e6:.1f} megapixels ({size}); the limit is {limit / 1e6:g} megapixels",
        )


@app.post("/api/erase")
async def erase(
    request: Request,
    image: UploadFile = File(..., description="Source image to inpaint"),
    mask: UploadFile = File(..., description="White pixels mark areas to remove"),
    ref_size: int = Form(0, ge=0, le=65535, description="Longest side of the full photo when only a crop is uploaded"),
) -> Response:
    t_req = time.perf_counter()
    req_id = uuid.uuid4().hex[:8]
    client = request.client.host if request.client else "?"
    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    mask_bytes = await mask.read(MAX_UPLOAD_BYTES + 1)
    t_read = time.perf_counter() - t_req            # reading the multipart upload from the spooled temp files
    # One context string reused by every log line of this request, so a failure can be traced back.
    context = (f"req={req_id} client={client} image={image.filename!r}/{image.content_type} {len(image_bytes)} B "
               f"mask={mask.filename!r}/{mask.content_type} {len(mask_bytes)} B ref_size={ref_size}")
    if len(image_bytes) > MAX_UPLOAD_BYTES or len(mask_bytes) > MAX_UPLOAD_BYTES:
        logger.warning("[erase] 413 upload over %d MB limit | %s", MAX_UPLOAD_BYTES // (1024 * 1024), context)
        raise HTTPException(status_code=413, detail="Each upload must be 25 MB or smaller")
    if not image_bytes or not mask_bytes:
        logger.warning("[erase] 400 empty upload | %s", context)
        raise HTTPException(status_code=400, detail="Both image and mask must be non-empty")

    inpainter = await get_inpainter(request.app)     # lazy mode: downloads + loads on the first request (with retries)
    if inpainter is None:
        logger.error("[erase] 503 LaMa model is not loaded (%s) | %s", getattr(request.app.state, "lama_error", None), context)
        raise HTTPException(status_code=503, detail="LaMa model is not loaded")

    # Cheap header-only size check BEFORE anything is queued or decoded.
    _reject_if_huge(image_bytes, ERASE_MAX_PIXELS, "Image")
    _reject_if_huge(mask_bytes, ERASE_MAX_PIXELS, "Mask")

    state = request.app.state
    if state.erase_slots.locked() and state.erase_waiting >= ERASE_MAX_QUEUE:
        logger.warning("[erase] 503 busy: %d already queued | %s", state.erase_waiting, context)
        raise HTTPException(status_code=503, detail="The eraser is busy; please retry in a few seconds",
                            headers={"Retry-After": "5"})

    state.erase_waiting += 1                    # no await between the check above and this line: race-free
    queued = True
    try:
        await state.erase_slots.acquire()       # at most ERASE_MAX_CONCURRENCY LaMa jobs at once
        state.erase_waiting -= 1
        queued = False
        if await request.is_disconnected():
            state.erase_slots.release()
            logger.info("[erase] 499 client left while queued | %s", context)
            return Response(status_code=499)
        try:
            # The slot is released by the worker thread's completion (slots.py), not by this request.
            result, timer = await run_in_owned_slot(state.erase_slots, inpainter.erase_profiled,
                                                    image_bytes, mask_bytes, ref_size)
        except ValueError as exc:
            logger.warning("[erase] 400 rejected input: %s | %s", exc, context)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            # Full traceback + the request context goes to the server console / log.
            logger.exception("[erase] 500 LaMa inpainting failed | %s", context)
            raise HTTPException(
                status_code=500, detail=f"LaMa inpainting failed (ref {req_id}); see the server log.",
            ) from exc
    finally:
        if queued:                              # cancelled / failed while still waiting for a slot
            state.erase_waiting -= 1

    total = time.perf_counter() - t_req
    if PROFILE:
        logger.info("[erase-profile] endpoint: upload read %.1f ms | pipeline %.1f ms | request total (excl. network) %.1f ms | response %.2f MB",
                    t_read * 1000, timer.total * 1000, total * 1000, len(result) / 1e6)
    # lama.py answers PNG, or JPEG when a (very large) result would break the platform's response-size limit.
    erase_jpeg = result[:2] == b"\xff\xd8"
    return Response(
        content=result,
        media_type="image/jpeg" if erase_jpeg else "image/png",
        headers={
            "Content-Disposition": f'inline; filename="result.{"jpg" if erase_jpeg else "png"}"',
            "Server-Timing": f"upload;dur={t_read * 1000:.1f}, {timer.server_timing()}, total;dur={total * 1000:.1f}",
        },
    )


@app.post("/api/portrait")
async def portrait(
    request: Request,
    image: UploadFile = File(..., description="Photo whose background should be blurred"),
    blur_level: int = Form(
        PORTRAIT_DEFAULT_BLUR_LEVEL,
        ge=PORTRAIT_MIN_BLUR_LEVEL,
        le=PORTRAIT_MAX_BLUR_LEVEL,
        description="Background blur strength, 1 (sharp) to 30 (widest bokeh)",
    ),
    focus_x: float = Form(
        0.5, ge=0.0, le=1.0, description="Tap-to-focus X, normalised 0..1 (default centre)"
    ),
    focus_y: float = Form(
        0.5, ge=0.0, le=1.0, description="Tap-to-focus Y, normalised 0..1 (default centre)"
    ),
    skin_smooth: int = Form(
        PORTRAIT_DEFAULT_SKIN_SMOOTH,
        ge=PORTRAIT_MIN_SKIN_SMOOTH,
        le=PORTRAIT_MAX_SKIN_SMOOTH,
        description="Natural skin smoothing, 0 (off) to 100 (strongest); combines with the bokeh blur",
    ),
) -> Response:
    """Depth-based background blur plus optional natural skin smoothing. Every failure path answers JSON *with CORS headers*:
    400 bad input, 413 too big, 503 model missing, 504 timeout, 500 anything else."""
    t_req = time.perf_counter()
    req_id = uuid.uuid4().hex[:8]
    client = request.client.host if request.client else "?"
    context = f"req={req_id} client={client} blur={blur_level} skin={skin_smooth} focus=({focus_x:.2f},{focus_y:.2f})"
    try:
        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        context += f" image={image.filename!r}/{image.content_type} {len(image_bytes)} B"
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            logger.warning("[portrait] 413 upload over limit | %s", context)
            return error_response(413, "Upload must be 25 MB or smaller")
        if not image_bytes:
            logger.warning("[portrait] 400 empty upload | %s", context)
            return error_response(400, "Image must be non-empty")

        # The model may have failed at boot (file copied later, transient error) or not be loaded yet (lazy mode):
        # get_portrait() downloads / loads it on demand, with retries in lazy mode.
        depth_portrait = await get_portrait(request.app)
        if depth_portrait is None:
            reason = getattr(request.app.state, "portrait_error", None) or "unknown error"
            logger.error("[portrait] 503 model not loaded (%s) | %s", reason, context)      # full reason: server log only
            # S1 (V26): the reason can contain absolute paths / raw exception text - the client only gets it in development.
            return error_response(503, public_model_error(IS_PRODUCTION, PORTRAIT_UNAVAILABLE_MESSAGE, reason))

        async def run_job() -> tuple[bytes, str]:
            slots = request.app.state.portrait_slots            # bound memory: N photos at a time
            await slots.acquire()
            # The slot is handed back only when the worker THREAD finishes (see slots.py): a timed-out
            # job keeps counting against the limit until it is really gone.
            return await run_in_owned_slot(slots, depth_portrait.blur_background,
                                           image_bytes, blur_level, focus_x, focus_y, skin_smooth)

        try:
            result, media_type = await asyncio.wait_for(run_job(), timeout=PORTRAIT_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error("[portrait] 504 timed out after %.0fs | %s", PORTRAIT_TIMEOUT_S, context)
            return error_response(504, f"Portrait processing timed out after {PORTRAIT_TIMEOUT_S:.0f}s (ref {req_id}); try a smaller photo.")
        except ValueError as exc:
            logger.warning("[portrait] 400 rejected input: %s | %s", exc, context)
            return error_response(400, str(exc))
        except MemoryError:
            logger.error("[portrait] 500 out of memory | %s", context)
            return error_response(500, f"Not enough memory to process this photo (ref {req_id}); try a smaller one.")
        except Exception:
            logger.exception("[portrait] 500 depth blur failed | %s", context)
            return error_response(500, f"Portrait processing failed (ref {req_id}); see the server log.")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("[portrait] 500 unexpected failure | %s", context)
        return error_response(500, f"Portrait request failed (ref {req_id}); see the server log.")

    total = time.perf_counter() - t_req
    extension = "png" if media_type == "image/png" else "jpg"
    return Response(
        content=result,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="portrait.{extension}"',
            "Server-Timing": f"total;dur={total * 1000:.1f}",
        },
    )


# Self-hosted fonts: pin the MIME type so it never depends on the host OS registry (Windows can lack .woff2).
mimetypes.add_type("font/woff2", ".woff2")
# Locally bundled MediaPipe segmenter (vendor/mediapipe): module JS, WASM and the model need exact types
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".tflite")

if (PROJECT_DIR / "index.html").is_file():
    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(PROJECT_DIR / "index.html", media_type="text/html")

    for asset_directory in ("css", "js", "fonts", "vendor"):
        asset_path = PROJECT_DIR / asset_directory
        if asset_path.is_dir():
            app.mount(
                f"/{asset_directory}",
                StaticFiles(directory=asset_path),
                name=f"frontend-{asset_directory}",
            )

    @app.get("/{filename}", include_in_schema=False)
    async def frontend_root_asset(filename: str) -> FileResponse:
        # config.xml (Cordova build config) is deliberately NOT served.
        if filename not in {"favicon.ico", "icon.png", "robots.txt"}:
            raise HTTPException(status_code=404, detail="Not found")
        asset_path = PROJECT_DIR / filename
        if not asset_path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(asset_path)
