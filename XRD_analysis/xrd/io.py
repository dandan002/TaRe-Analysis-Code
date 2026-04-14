from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

try:
    import xylib  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    _XYLIB_IMPORT_ERROR = exc

    def _missing_xylib(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "xylib is required to read Bruker .raw files. Install the Phase 3 "
            "dependencies from requirements.txt, including xylib-py."
        ) from _XYLIB_IMPORT_ERROR

    xylib = SimpleNamespace(load_file=_missing_xylib)
else:
    _XYLIB_IMPORT_ERROR = None


CU_KA_AVG = 1.5406
WAVELENGTH_TOL = 0.001


def load_raw(path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Bruker .raw file into 2theta and intensity arrays."""
    dataset = xylib.load_file(str(path), "")
    block = dataset.get_block(0)
    nrow = block.get_point_count()
    two_theta = np.array([block.get_column(1).get_value(j) for j in range(nrow)])
    intensity = np.array([block.get_column(2).get_value(j) for j in range(nrow)])
    zero_mask = intensity == 0
    if zero_mask.any():
        warnings.warn(
            f"Replacing {int(zero_mask.sum())} zero-count intensity values with epsilon.",
            RuntimeWarning,
            stacklevel=2,
        )
        intensity = intensity.copy()
        intensity[zero_mask] = np.finfo(float).eps
    return two_theta, intensity


def get_wavelength(path) -> float | None:
    """Return the USED_LAMBDA header value in angstroms when available."""
    dataset = xylib.load_file(str(path), "")
    block = dataset.get_block(0)
    try:
        if hasattr(block, "get_meta"):
            value = block.get_meta("USED_LAMBDA")
        else:
            value = block.meta.get("USED_LAMBDA")
        if value:
            return float(value)
    except Exception as exc:
        warnings.warn(
            f"USED_LAMBDA metadata unavailable for {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    warnings.warn(
        f"USED_LAMBDA metadata unavailable for {path}: empty value",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def validate(
    two_theta: np.ndarray, intensity: np.ndarray, wavelength: float | None = None
) -> None:
    """Raise ValueError with a diagnostic message when quality checks fail."""
    span = float(two_theta[-1]) - float(two_theta[0])
    if span < 60.0:
        raise ValueError(
            f"2θ span is {span:.1f}° — expected ≥ 60° (covering 20–80°). "
            "Data may be truncated."
        )

    if not (intensity > 0).all():
        n_bad = int((intensity <= 0).sum())
        raise ValueError(f"{n_bad} intensity values ≤ 0 — possible corrupt read.")

    if wavelength is not None and abs(wavelength - CU_KA_AVG) > WAVELENGTH_TOL:
        raise ValueError(
            f"Header wavelength {wavelength:.5f} Å deviates from "
            f"Cu Kα ({CU_KA_AVG} Å) by > {WAVELENGTH_TOL} Å. "
            "Check X-ray source setting."
        )


def export_csv(two_theta: np.ndarray, intensity: np.ndarray, out_path) -> None:
    """Write two_theta and intensity arrays to a CSV file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"two_theta": two_theta, "intensity": intensity}).to_csv(
        out_path, index=False
    )


__all__ = ["load_raw", "get_wavelength", "validate", "export_csv"]
