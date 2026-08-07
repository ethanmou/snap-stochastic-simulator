"""Compatibility wrapper for the full-vs-light validation script."""

from __future__ import annotations

from scripts.validation.compare_full_vs_light import *  # noqa: F401,F403
from scripts.validation.compare_full_vs_light import main


if __name__ == "__main__":
    main()
