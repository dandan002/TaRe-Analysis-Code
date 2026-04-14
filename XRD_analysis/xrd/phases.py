"""Phase 4: Materials Project phase search, ranking, and CIF download."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.signal import find_peaks
from tabulate import tabulate

from xrd.config import CIF_DIR, RESULTS_DIR, ensure_cif_dir, ensure_results


TOP_N_CIFS = 3
TARE_FALLBACK_CIF = CIF_DIR / "fallback" / "TaRe_bcc.cif"

PHASE_LABEL_MAP: dict[str, str] = {
    "Ta2O5": "Ta₂O₅",
    "ReO2": "ReO₂",
    "ReO3": "ReO₃",
    "TaRe": "TaRe bcc",
    "Al2O3": "Al₂O₃",
}
PHASE_ORDER = ["Ta2O5", "ReO2", "ReO3", "TaRe", "Al2O3"]
_FIELDS = [
    "material_id",
    "formula_pretty",
    "symmetry",
    "energy_above_hull",
    "is_stable",
]
_XRD = XRDCalculator(wavelength="CuKa")
PEAK_MATCH_TOL_DEG = 0.5


def _mpr() -> Any:
    from mp_api.client import MPRester

    return MPRester


def _get_api_key() -> str:
    """Return the Materials Project API key or raise a helpful error."""
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError(
            "MP_API_KEY environment variable not set.\n"
            "Register at https://materialsproject.org and set MP_API_KEY in your shell profile."
        )
    return key


def _sort_key(doc: Any) -> tuple[int, float]:
    energy = getattr(doc, "energy_above_hull", None)
    if energy is None:
        return (1, float("inf"))
    return (0, float(energy))


def _space_group(doc: Any) -> str:
    symmetry = getattr(doc, "symmetry", None)
    if not symmetry:
        return "Unknown"
    symbol = getattr(symmetry, "symbol", None) or "Unknown"
    number = getattr(symmetry, "number", None)
    return f"{symbol} ({number})" if number else symbol


def find_observed_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    prominence_frac: float = 0.01,
    min_distance_deg: float = 0.5,
) -> np.ndarray:
    """Return 2theta positions of observed scan peaks above the prominence threshold."""
    step = float(two_theta[1] - two_theta[0])
    distance_pts = max(1, int(min_distance_deg / step))
    peak_indices, _ = find_peaks(
        intensity,
        prominence=prominence_frac * float(intensity.max()),
        distance=distance_pts,
    )
    return two_theta[peak_indices]


def filter_peaks_to_observed(
    cif_peaks: list[float],
    observed_peaks: np.ndarray,
    tol_deg: float = PEAK_MATCH_TOL_DEG,
) -> list[float]:
    """Keep only theoretical peaks that correspond to observed scan peaks."""
    if len(observed_peaks) == 0:
        return []
    return [peak for peak in cif_peaks if np.any(np.abs(observed_peaks - peak) <= tol_deg)]


def _search_single_phase(mpr: Any, phase: str) -> list[Any]:
    if phase == "TaRe":
        docs = list(
            mpr.materials.summary.search(
                formula=phase,
                crystal_system="Cubic",
                fields=_FIELDS,
            )
        )
        bcc_docs = [
            doc
            for doc in docs
            if getattr(getattr(doc, "symmetry", None), "number", None) == 229
        ]
        if not bcc_docs:
            print(
                "WARNING: No TaRe Im-3m (#229) candidates found in Materials Project. "
                "Using bundled fallback CIF (see data/cif/fallback/PROVENANCE.md)."
            )
            return [{"fallback": True, "cif_path": TARE_FALLBACK_CIF}]
        return sorted(bcc_docs, key=_sort_key)

    docs = list(
        mpr.materials.summary.search(
            formula=phase,
            fields=_FIELDS,
        )
    )
    return sorted(docs, key=_sort_key)


def search_all_phases(api_key: str) -> dict[str, list[Any]]:
    """Search Materials Project for all target phases."""
    with _mpr()(api_key) as mpr:
        return {phase: _search_single_phase(mpr, phase) for phase in PHASE_ORDER}


def build_candidates_df(results: dict[str, list[Any]]) -> pd.DataFrame:
    """Convert search results into the ranked candidate table."""
    rows = []
    for phase in PHASE_ORDER:
        docs = sorted(results.get(phase, []), key=_sort_key)
        for doc in docs:
            if isinstance(doc, dict) and doc.get("fallback"):
                rows.append(
                    {
                        "target_phase": phase,
                        "material_id": "TaRe_fallback",
                        "formula": "TaRe",
                        "space_group": "Im-3m (#229) fallback",
                        "energy_above_hull": float("nan"),
                        "is_stable": False,
                    }
                )
                continue
            rows.append(
                {
                    "target_phase": phase,
                    "material_id": str(getattr(doc, "material_id", "")),
                    "formula": getattr(doc, "formula_pretty", ""),
                    "space_group": _space_group(doc),
                    "energy_above_hull": getattr(doc, "energy_above_hull", None),
                    "is_stable": bool(getattr(doc, "is_stable", False)),
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "target_phase",
            "material_id",
            "formula",
            "space_group",
            "energy_above_hull",
            "is_stable",
        ],
    )
    if frame.empty:
        return frame

    phase_rank = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    frame["target_phase"] = pd.Categorical(
        frame["target_phase"], categories=PHASE_ORDER, ordered=True
    )
    frame = frame.sort_values(
        by=["target_phase", "energy_above_hull"],
        key=lambda column: column.map(phase_rank)
        if column.name == "target_phase"
        else column.fillna(float("inf")),
        kind="stable",
    ).reset_index(drop=True)
    frame["target_phase"] = frame["target_phase"].astype(str)
    return frame


def print_candidates_table(frame: pd.DataFrame) -> None:
    """Print the ranked candidate table."""
    if frame.empty:
        print("No candidate phases found.")
        return
    print(tabulate(frame, headers="keys", tablefmt="github", showindex=False))


def save_candidates_csv(frame: pd.DataFrame) -> Path:
    """Save the ranked candidate list to results/candidates.csv."""
    ensure_results()
    out_path = RESULTS_DIR / "candidates.csv"
    frame.to_csv(out_path, index=False)
    return out_path


def _fetch_structures(
    material_ids: list[str], api_key: str
) -> dict[str, Any]:
    if not material_ids:
        return {}
    with _mpr()(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=material_ids,
            fields=["material_id", "structure"],
        )
    return {str(doc.material_id): doc for doc in docs}


def download_top_cifs(results: dict[str, list[Any]], api_key: str) -> dict[str, list[Path]]:
    """Download or reuse the top-N CIFs per phase."""
    cif_dir = ensure_cif_dir()
    downloaded: dict[str, list[Path]] = {}

    for phase in PHASE_ORDER:
        docs = sorted(results.get(phase, []), key=_sort_key)[:TOP_N_CIFS]
        if not docs:
            downloaded[phase] = []
            continue

        real_docs = [doc for doc in docs if not (isinstance(doc, dict) and doc.get("fallback"))]
        docs_by_id = {
            str(getattr(doc, "material_id", "")): doc
            for doc in real_docs
            if getattr(doc, "structure", None) is not None
        }
        missing_ids = [
            str(getattr(doc, "material_id", ""))
            for doc in real_docs
            if getattr(doc, "structure", None) is None
        ]
        docs_by_id.update(_fetch_structures(missing_ids, api_key))

        phase_paths: list[Path] = []
        for rank, doc in enumerate(docs, start=1):
            if isinstance(doc, dict) and doc.get("fallback"):
                dest = cif_dir / f"{phase}_{rank}_fallback.cif"
                if not dest.exists():
                    shutil.copy2(doc["cif_path"], dest)
                phase_paths.append(dest)
                continue

            material_id = str(getattr(doc, "material_id", "unknown"))
            out_path = cif_dir / f"{phase}_{rank}_{material_id}.cif"
            if not out_path.exists():
                structure_doc = docs_by_id.get(material_id, doc)
                structure = getattr(structure_doc, "structure", None)
                if structure is None:
                    continue
                structure.to(filename=out_path)
            phase_paths.append(out_path)
        downloaded[phase] = phase_paths

    return downloaded


def peaks_from_cif(path: Path | str) -> list[float]:
    """Return Cu K-alpha 2theta peaks computed from a CIF file."""
    structure = Structure.from_file(path)
    pattern = _XRD.get_pattern(structure, two_theta_range=(10.0, 100.0))
    return [float(value) for value in pattern.x]


def build_phase_peaks_dict(
    downloaded: dict[str, list[Path]],
    observed_peaks: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Map plot labels to peak positions derived from the top CIF per phase."""
    phase_peaks: dict[str, list[float]] = {}
    for phase in PHASE_ORDER:
        paths = downloaded.get(phase, [])
        if not paths:
            continue
        raw_peaks = peaks_from_cif(paths[0])
        peaks = (
            filter_peaks_to_observed(raw_peaks, observed_peaks)
            if observed_peaks is not None
            else raw_peaks
        )
        phase_peaks[PHASE_LABEL_MAP[phase]] = peaks
    return phase_peaks


def identify_candidate_phases() -> None:
    """Run the Phase 4 search, report, and CIF download pipeline."""
    api_key = _get_api_key()
    results = search_all_phases(api_key)
    frame = build_candidates_df(results)
    print_candidates_table(frame)
    save_candidates_csv(frame)
    download_top_cifs(results, api_key)


__all__ = [
    "PEAK_MATCH_TOL_DEG",
    "build_candidates_df",
    "build_phase_peaks_dict",
    "download_top_cifs",
    "filter_peaks_to_observed",
    "find_observed_peaks",
    "identify_candidate_phases",
    "peaks_from_cif",
    "print_candidates_table",
    "save_candidates_csv",
    "search_all_phases",
]


__all__ = [
    "TOP_N_CIFS",
    "TARE_FALLBACK_CIF",
    "PHASE_LABEL_MAP",
    "PHASE_ORDER",
    "_get_api_key",
    "search_all_phases",
    "build_candidates_df",
    "print_candidates_table",
    "save_candidates_csv",
    "download_top_cifs",
    "peaks_from_cif",
    "build_phase_peaks_dict",
    "identify_candidate_phases",
]
