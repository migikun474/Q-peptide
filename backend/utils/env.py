"""Minimal .env loader.

Credentials live in .env (gitignored) rather than the shell profile, so every entry point
that may touch IBM Quantum loads it. Deliberately dependency-free and non-overriding: a
value already exported in the environment always wins over the file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE pairs from ``path``. Returns the names actually set."""
    loaded: list[str] = []
    p = Path(path)
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded
