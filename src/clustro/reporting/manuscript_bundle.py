"""Manuscript bundle export helpers."""

from __future__ import annotations

from pathlib import Path

from clustro.utils.io import ensure_directory


def create_manuscript_bundle(root: Path) -> Path:
    bundle = ensure_directory(root / "manuscript_bundle")
    ensure_directory(bundle / "figures")
    ensure_directory(bundle / "tables")
    ensure_directory(bundle / "supplementary")
    ensure_directory(bundle / "methods")
    return bundle
