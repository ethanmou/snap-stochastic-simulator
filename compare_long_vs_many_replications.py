"""Compatibility wrapper for the long-vs-many validation script."""

from __future__ import annotations

from scripts.validation.compare_long_vs_many_replications import *  # noqa: F401,F403
from scripts.validation.compare_long_vs_many_replications import main


if __name__ == "__main__":
    main()
