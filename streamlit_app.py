"""Compatibility entry point for the Streamlit dashboard.

The main dashboard implementation lives in ``apps/streamlit_app.py``. Keeping
this wrapper preserves the historical ``streamlit run streamlit_app.py`` command.
"""

from __future__ import annotations

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "apps" / "streamlit_app.py"


if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")
