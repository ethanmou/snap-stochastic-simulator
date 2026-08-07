"""Compatibility wrapper for the light-vs-fluid validation script."""

from __future__ import annotations

from scripts.validation.compare_light_vs_fluid import *  # noqa: F401,F403
from scripts.validation.compare_light_vs_fluid import main


if __name__ == "__main__":
    main()
