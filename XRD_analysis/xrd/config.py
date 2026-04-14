import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CIF_DIR = DATA_DIR / "cif"
PHASE5_PREFERRED_PYTHON = "/opt/homebrew/bin/python3.14"
PHASE5_GSASII_SOURCE_HINT = str(Path.home() / "GSAS-II-src" / "backcompat")
_GSASII_DEFAULT_PATH = PHASE5_GSASII_SOURCE_HINT  # hardcoded installation


def ensure_results() -> Path:
    """Create the results directory if needed and return it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def ensure_cif_dir() -> Path:
    """Create the data/cif directory if needed and return it."""
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    return CIF_DIR


# ---------------------------------------------------------------------------
# Phase 5: GSAS-II path configuration
# ---------------------------------------------------------------------------

GSASII_PATH = os.environ.get("GSASII_PATH") or _GSASII_DEFAULT_PATH
if GSASII_PATH and GSASII_PATH not in sys.path:
    sys.path.insert(0, GSASII_PATH)
# The backcompat shim does "from GSASII.GSASIIscriptable import *", so the
# parent directory (which contains the GSASII package) must also be on sys.path.
_gsasii_parent = str(Path(GSASII_PATH).parent)
if _gsasii_parent not in sys.path:
    sys.path.insert(0, _gsasii_parent)

# Instrument parameter file (Cu Ka Bragg-Brentano, bundled in repo)
INSTPRM_FILE = DATA_DIR / "TaRe_Cu_Ka.instprm"

# GSAS-II project file written before refinement for crash recovery
GPX_FILE = RESULTS_DIR / "TaRe_refinement.gpx"

# Raw histogram input
RAW_FILE = DATA_DIR / "TaRe_Full_Oxide.raw"

# Phase 3 CSV fallback if Bruker .raw v3 offset bug triggers
RAW_CSV_FALLBACK = RESULTS_DIR / "TaRe_Full_Oxide.csv"


def phase5_runtime_guidance() -> str:
    """Return the documented interpreter/env contract for GSAS-II runs."""
    return (
        "Phase 5 GSAS-II runtime contract:\n"
        f"- Preferred interpreter: {PHASE5_PREFERRED_PYTHON}\n"
        '- Export GSASII_PATH="$HOME/GSAS-II-src/backcompat" or your equivalent '
        "GSAS-II source path before importing GSASIIscriptable\n"
        "- Keep the D-01/D-02 bootstrap order: sys.path.insert(0, GSASII_PATH) "
        "then import GSASIIscriptable as G2sc\n"
        f"- Supported instrument template: {INSTPRM_FILE.name}\n"
        "- Histogram input policy: try the Bruker .raw first and fall back to the "
        "Phase 3 CSV only after an explicit GSAS-II read/smoke-test failure"
    )


def require_gsasii_path() -> str:
    """Bootstrap GSAS-II onto sys.path, or confirm it's already importable.

    Priority: GSASII_PATH env var → already a site-package → hardcoded default.
    Returns the resolved path string (empty string if already a site-package).
    """
    import importlib.util
    gsasii_path = os.environ.get("GSASII_PATH")
    if gsasii_path:
        if gsasii_path not in sys.path:
            sys.path.insert(0, gsasii_path)
        _parent = str(Path(gsasii_path).parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        return gsasii_path
    if importlib.util.find_spec("GSASIIscriptable") is not None:
        return ""
    if _GSASII_DEFAULT_PATH not in sys.path:
        sys.path.insert(0, _GSASII_DEFAULT_PATH)
    _parent = str(Path(_GSASII_DEFAULT_PATH).parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    return _GSASII_DEFAULT_PATH
