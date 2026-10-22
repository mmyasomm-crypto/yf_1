# Deploying YoFilter to Vercel (Hobby, free)

One project serves both the web app (static files through FastAPI) and the API (`/api/erase`, `/api/portrait`, `/health`).

## What the repo contains for Vercel

| File | Role |
|---|---|
| `app.py` | Entrypoint Vercel auto-detects; re-exports `backend.app.main:app`. |
| `vercel.json` | `maxDuration` 300 s, `excludeFiles` (tests, docs, scripts, `*.onnx`), `no-store` on `/api/*`. |
| `requirements.txt` | Runtime dependencies (installed by Vercel). |
| `.python-version` | Python 3.12. |
| `.vercelignore` | Keeps `vercel deploy` uploads small (CLI deploys only). |

## Behaviour on Vercel

* **No bundled models.** Detected through the `VERCEL` environment variable, the app uses `/tmp/yofilter-models` and
  downloads `inpainting_lama.onnx` / `portrait_depth.onnx` from the Hugging Face repo `ertegfg/yofilter-models`.
* **Lazy loading.** Cold start does nothing heavy. The first `/api/erase` downloads + loads LaMa (~92 MB); the first
  `/api/portrait` downloads + loads the depth model (~99 MB). Concurrent first requests share one download.
* **Retries.** A failed download / load is retried 3 times (2 s, 4 s back-off). After the last failure the endpoint
  answers `503` immediately for 20 s instead of re-downloading on every request.
* **Warm instances keep `/tmp`.** A new instance (cold start after idle time) downloads again - expect the first erase /
  portrait after idle time to take noticeably longer.
* **Response guard.** Results are lossless PNG while they are under ~3.5 MB, otherwise JPEG quality 92 (transparent
  photos stay PNG and are shrunk if needed). The frontend restores the working image's pixel size.

## Environment variables (Project -> Settings -> Environment Variables)

| Name | Value | Why |
|---|---|---|
| `YOFILTER_ENV` | `production` | Hides `/docs`, trims `/health`, adds security headers, CORS = native app origin only. |
| `TRUST_PROXY` | `1` | Rate limiting uses the client IP from Vercel's `X-Forwarded-For`. |
| `ALLOWED_ORIGINS` | your site(s), comma-separated | Only if the web page is hosted on a different origin than the API. |
| `HF_TOKEN` | token | Only if the Hugging Face model repo is private. |

Optional tuning: `MODEL_LOAD_RETRIES` (3), `MODEL_RETRY_DELAY_S` (2), `MODEL_FAILURE_COOLDOWN_S` (20),
`PNG_MAX_RESPONSE_BYTES` (3500000; `0` = always PNG), `RESULT_JPEG_QUALITY` (92), `YOFILTER_LAZY_MODELS` (`0` forces eager
boot), `YOFILTER_MODEL_DIR` (another cache directory).

Project settings: Framework Preset **Other** (FastAPI is detected from `app.py`), **Fluid compute enabled** (needed for the
300 s duration on Hobby).

## Deploy

```
git add -A && git commit -m "Vercel backend + upload compression" && git push     # Git integration, or:
npx vercel --prod
```

Smoke test after the deploy:

```
curl https://<project>.vercel.app/health          # {"status":"degraded", ...}  -> no model loaded yet (lazy), that is expected
# in the app: Magic Eraser once, Portrait once (first call of each downloads its model)
curl https://<project>.vercel.app/health          # both model flags true on that warm instance
```

## Limits to keep in mind (Vercel Hobby)

* Request **and** response bodies: 4.5 MB each (handled by `js/core/upload-compress.js` + `backend/app/encoding.py`).
* Function memory 2 GB, duration 300 s (with Fluid compute), Python bundle 500 MB uncompressed.
* Rate limiting and the erase queue are per instance (in memory), not global.
