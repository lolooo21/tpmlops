from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> Path:
    # Streamlit pages run from different entry points, so each page adds the
    # project root explicitly before importing shared project modules.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
