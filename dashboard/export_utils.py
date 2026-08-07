"""Download/export helpers for dashboard artifacts."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename token."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "dashboard"


def timestamp_token() -> str:
    """Return a compact local timestamp token."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Encode a DataFrame as CSV bytes."""

    return frame.to_csv(index=False).encode("utf-8")


def json_bytes(payload: dict[str, Any]) -> bytes:
    """Encode JSON payload bytes."""

    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def zip_dataframes(frames: dict[str, pd.DataFrame], metadata: dict[str, Any] | None = None) -> bytes:
    """Create a zip containing CSV tables and optional metadata JSON."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in frames.items():
            archive.writestr(f"{safe_filename(name)}.csv", frame.to_csv(index=False))
        if metadata is not None:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
    return buffer.getvalue()

