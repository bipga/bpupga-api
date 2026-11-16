from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not slug:
        raise ValueError(f"Cannot derive a safe path component from {value!r}")
    return slug


def dataset_path_slug(root: str | Path) -> str:
    """Include the parent name so roots such as FB-TW/rho_0.8 stay unique."""
    path = Path(root).expanduser().resolve()
    parts = path.parts[-2:] if len(path.parts) >= 2 else path.parts
    return "__".join(slugify(part) for part in parts)


def utc_run_id() -> str:
    """Return a sortable, path-safe UTC timestamp for one experiment batch."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def validate_run_id(value: str) -> str:
    """Validate a user-supplied run identifier without silently rewriting it."""

    normalized = slugify(value)
    if normalized != value:
        raise ValueError(
            f"Invalid run ID {value!r}; use only letters, numbers, '.', '_' or '-'."
        )
    return normalized
