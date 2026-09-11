"""Small compatibility helper shared by the V13 Raven SR diagnostics.

The pre-V13 staged Micro trainer was retired with the geometry/profile/seam training
curriculum. V13/V13.1 only need the cross-platform result opener from that module, so
keep that utility here while removing the dead staged training implementation.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _open_result(path: Path) -> None:
    """Open a generated diagnostic image with the platform default application."""
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass
