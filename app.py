"""Vercel entrypoint: Vercel auto-detects ``app.py`` at the project root and serves its ASGI ``app`` variable.

All the logic lives in ``backend.app.main`` (the same module ``uvicorn backend.app.main:app`` runs locally), so there
is exactly one application. Importing it is cheap: no model is downloaded or loaded until the first /api/erase or
/api/portrait request (see LAZY_MODEL_LOAD in backend/app/config.py).
"""

import sys
from pathlib import Path

# The project root must be importable as the parent of the ``backend`` package, whatever the working directory is.
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.app.main import app  # noqa: E402,F401  (the name Vercel looks for)

__all__ = ["app"]
