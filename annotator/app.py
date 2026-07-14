"""Local annotation viewer/editor web app (untracked, self-contained).

Run from the project root:

    ./annotator/run.sh

or directly:

    uv run --with fastapi --with "uvicorn[standard]" \
        python -m uvicorn annotator.app:app --reload --port 8000

The app reads/writes ``data/annotations.csv`` in place (creating a one-time
``data/annotations.csv.bak`` backup) and serves images from
``data/combined_dataset/``.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the canonical label schema so the UI always matches training.
from pxmodel.labels import LABEL_NAMES

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root = parent of this file's dir).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "annotations.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "annotations.csv.bak"
IMAGES_DIR = PROJECT_ROOT / "data" / "combined_dataset"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _read_csv() -> tuple[list[str], list[dict[str, str]]]:
    """Return (header, rows) preserving the on-disk column order."""
    if not CSV_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"CSV not found: {CSV_PATH}")
    with CSV_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return header, rows


def _write_csv(header: list[str], rows: list[dict[str, str]]) -> None:
    """Atomically write rows back to the CSV, backing up once."""
    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
    fd, tmp_name = tempfile.mkstemp(dir=str(CSV_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, CSV_PATH)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="pxModel Annotation Editor")


class LabelUpdate(BaseModel):
    labels: dict[str, int]


@app.get("/api/labels")
def get_labels() -> dict[str, list[str]]:
    return {"labels": LABEL_NAMES}


@app.get("/api/annotations")
def get_annotations(request: Request, page: int = 1, page_size: int = 60) -> JSONResponse:
    """Return paginated, optionally filtered annotation rows.

    Per-label filters are read from the query string, e.g. ``?damaged=1&open=0``.
    A value of ``1`` requires positive, ``0`` requires negative, absent = any.
    """
    filters: dict[str, str] = {}
    for name in LABEL_NAMES:
        want = request.query_params.get(name)
        if want in ("0", "1"):
            filters[name] = want
    header, rows = _read_csv()
    return _build_response(header, rows, page, page_size, filters=filters)


def _build_response(
    header: list[str],
    rows: list[dict[str, str]],
    page: int,
    page_size: int,
    filters: dict[str, str],
) -> JSONResponse:
    if filters:
        filtered = []
        for r in rows:
            keep = True
            for name, want in filters.items():
                if str(r.get(name, "0")).strip() not in ("0", "1"):
                    val = "0"
                else:
                    val = str(r.get(name, "0")).strip()
                if val != want:
                    keep = False
                    break
            if keep:
                filtered.append(r)
        rows = filtered

    total = len(rows)
    page = max(1, page)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]

    items = []
    for r in page_rows:
        fname = r.get("filename", "")
        items.append(
            {
                "filename": fname,
                "labels": {name: int(str(r.get(name, "0")).strip() or "0") for name in LABEL_NAMES},
                "exists": (IMAGES_DIR / fname).is_file() if fname else False,
            }
        )

    return JSONResponse(
        {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, (total + page_size - 1) // page_size),
            "items": items,
        }
    )


@app.post("/api/annotations/{filename}")
def update_annotation(filename: str, update: LabelUpdate) -> dict[str, object]:
    with _write_lock:
        header, rows = _read_csv()
        target = None
        for r in rows:
            if r.get("filename") == filename:
                target = r
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Unknown filename: {filename}")

        for name, value in update.labels.items():
            if name not in LABEL_NAMES:
                raise HTTPException(status_code=400, detail=f"Unknown label: {name}")
            if value not in (0, 1):
                raise HTTPException(status_code=400, detail="Label values must be 0 or 1")
            target[name] = str(value)

        _write_csv(header, rows)

    return {"ok": True, "filename": filename, "labels": update.labels}


@app.get("/images/{filename}")
def get_image(filename: str) -> FileResponse:
    # Prevent path traversal; only serve files directly inside IMAGES_DIR.
    safe = (IMAGES_DIR / filename).resolve()
    if IMAGES_DIR.resolve() not in safe.parents or not safe.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(safe)


# Mount the SPA last so API routes take precedence.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
