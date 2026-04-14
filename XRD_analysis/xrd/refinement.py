"""Phase 5: GSAS-II Rietveld refinement via GSASIIscriptable."""

from __future__ import annotations

import json
import math
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from xrd.config import GSASII_PATH, phase5_runtime_guidance

try:
    import GSASIIscriptable as G2sc  # type: ignore[import-untyped]
except ImportError as exc:
    G2sc = None  # type: ignore[assignment]
    GSASII_IMPORT_ERROR = exc
else:
    GSASII_IMPORT_ERROR = None


REFINEMENT_STAGES = [
    {
        "set": {
            "Background": {"type": "chebyschev-1", "no. coeffs": 6, "refine": True},
            "Sample Parameters": ["Scale"],
        },
        "maxCycles": 10,
    },
    {
        "set": {"Scale": True},
        "maxCycles": 10,
    },
    {
        "set": {"Cell": True},
        "maxCycles": 10,
    },
    {
        "set": {"Instrument Parameters": ["U", "V", "W"]},
        "clear": {"Cell": True},
        "maxCycles": 10,
    },
]


@dataclass
class RefinementConvergenceError(RuntimeError):
    """Structured failure raised when staged refinement is not physically usable."""

    stage: int
    reason: str
    rwp_history: list[float]

    def __str__(self) -> str:
        return f"Stage {self.stage} failed to converge: {self.reason}"


def setup_project(
    selected_cifs: dict[str, Path],
    raw_file: Path,
    instprm_file: Path,
    gpx_file: Path,
) -> tuple[Any, Any]:
    """Create a project, add the histogram, link phases, and save the GPX."""
    if G2sc is None:
        raise RuntimeError(
            "GSASIIscriptable is not importable with the current Phase 5 runtime.\n"
            f"GSASII_PATH={GSASII_PATH}\n"
            f"Import error: {GSASII_IMPORT_ERROR}\n"
            f"{phase5_runtime_guidance()}"
        )

    gpx = G2sc.G2Project(newgpx=str(gpx_file))
    format_hint = histogram_format_hint(raw_file)
    hist = gpx.add_powder_histogram(
        str(raw_file),
        str(instprm_file),
        fmthint=format_hint,
    )

    for phase_name, cif_path in selected_cifs.items():
        gpx.add_phase(
            str(cif_path),
            phasename=phase_name,
            fmthint="CIF",
            histograms=[hist],
        )

    gpx.save()
    return gpx, hist


def histogram_format_hint(raw_file: Path) -> str:
    """Return the GSAS-II reader hint for the requested histogram input."""
    if raw_file.suffix.lower() == ".csv":
        return "comma/tab/semicolon separated"
    return "Bruker RAW"


def probe_histogram_input(
    selected_cifs: dict[str, Path],
    raw_file: Path,
    instprm_file: Path,
) -> str:
    """Exercise the real setup path against a candidate histogram input."""
    with tempfile.TemporaryDirectory(prefix="phase5-probe-") as tmp_dir:
        probe_gpx = Path(tmp_dir) / "probe.gpx"
        gpx, hist = setup_project(
            selected_cifs=selected_cifs,
            raw_file=raw_file,
            instprm_file=instprm_file,
            gpx_file=probe_gpx,
        )
        if hist is None:
            raise RuntimeError(f"{raw_file.name} returned an unusable histogram")
        save = getattr(gpx, "save", None)
        if callable(save):
            save()
    return histogram_format_hint(raw_file)


_SUBSTRATE_PHASE_PRIOR = 0.70
_FILM_PHASE_PRIOR = 0.075
_SUBSTRATE_PHASES = frozenset({"Al2O3"})


def _substrate_aware_prior(phase_name: str, selected_cifs: dict[str, Path]) -> float:
    """Return the initial HAP scale prior for a phase.

    Al2O3 dominates the diffractogram as a thick crystalline substrate.
    All other phases share a small equal-weight residual.
    """
    if phase_name in _SUBSTRATE_PHASES:
        return _SUBSTRATE_PHASE_PRIOR
    return _FILM_PHASE_PRIOR


def _set_phase_scale(gpx: Any, hist: Any, phase_name: str, scale_value: float) -> None:
    """Write a single HAP scale value for phase_name."""
    phase = gpx.phase(phase_name)
    hap = phase.getHAPvalues(hist)
    scale_entry = hap.get("Scale")
    if isinstance(scale_entry, list) and len(scale_entry) >= 1:
        scale_entry[0] = scale_value
    elif isinstance(scale_entry, (int, float)):
        hap["Scale"] = scale_value
    phase.setHAPvalues(hap, [hist])


def _initialize_phase_scales(gpx: Any, hist: Any, selected_cifs: dict[str, Path]) -> None:
    """Set substrate-aware HAP scale priors so no phase starts at zero.

    Al2O3 receives a 0.70 prior (dominant substrate signal).
    All other phases receive 0.075 (equal-share residual).

    Prevents divide-by-zero in GSAS-II Hessian normalization when any phase
    has scale=0.0 at the start of Stage 2 scale refinement.
    """
    for phase_name in selected_cifs:
        prior = _substrate_aware_prior(phase_name, selected_cifs)
        _set_phase_scale(gpx, hist, phase_name, prior)


def _read_phase_scales(
    gpx: Any, hist: Any, selected_cifs: dict[str, Path]
) -> dict[str, float]:
    """Read current HAP scale values for all phases."""
    scales: dict[str, float] = {}
    for phase_name in selected_cifs:
        phase = gpx.phase(phase_name)
        hap = phase.getHAPvalues(hist)
        try:
            scales[phase_name] = _extract_scale_value(hap)
        except (ValueError, TypeError):
            scales[phase_name] = math.nan
    return scales


def _has_invalid_scales(scales: dict[str, float]) -> bool:
    """Return True if any phase scale is zero, negative, or NaN."""
    return any(not math.isfinite(v) or v <= 0 for v in scales.values())


def run_staged_refinement(
    gpx: Any,
    hist: Any,
    selected_cifs: dict[str, Path] | None = None,
    diagnostics_path: Path | None = None,
    histogram_input: Path | None = None,
    histogram_format_hint: str | None = None,
) -> list[float]:
    """Run the four planned stages and halt on non-convergent trajectories.

    Args:
        gpx: GSAS-II project object.
        hist: GSAS-II powder histogram object.
        selected_cifs: Phase name → CIF path mapping used for scale initialisation.
        diagnostics_path: If given, write a JSON diagnostics file on failure.
        histogram_input: Path of the histogram file used (recorded in diagnostics).
        histogram_format_hint: Reader hint string (recorded in diagnostics).
    """
    if selected_cifs is not None:
        _initialize_phase_scales(gpx, hist, selected_cifs)
    rwp_history: list[float] = []
    stage_history: list[dict[str, Any]] = []

    def _record_stage(stage_index: int) -> None:
        entry: dict[str, Any] = {"stage": stage_index}
        if selected_cifs is not None:
            scales = _read_phase_scales(gpx, hist, selected_cifs)
            entry["phase_scales"] = scales
        stage_history.append(entry)

    def _write_diagnostics(failure: RefinementConvergenceError | None) -> None:
        if diagnostics_path is None:
            return
        gpx_filename = getattr(gpx, "filename", None) or ""
        lst_path = str(Path(gpx_filename).with_suffix(".lst")) if gpx_filename else ""
        payload: dict[str, Any] = {
            "gpx_path": gpx_filename,
            "lst_path": lst_path,
            "rwp_history": rwp_history,
            "stage_history": stage_history,
            "selected_cifs": (
                {name: str(path.name) for name, path in selected_cifs.items()}
                if selected_cifs is not None
                else {}
            ),
            "histogram_input": str(histogram_input) if histogram_input is not None else None,
            "histogram_format_hint": histogram_format_hint,
        }
        if failure is not None:
            payload["failure"] = {
                "stage": failure.stage,
                "reason": failure.reason,
            }
        else:
            payload["failure"] = None
        diagnostics_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    STAGE2_INDEX = 2

    for index, stage_dict in enumerate(REFINEMENT_STAGES, start=1):
        _record_stage(index)

        try:
            stage_output = _run_refinement_stage(gpx, stage_dict)
            first_exc: Exception | None = None
        except Exception as exc:
            first_exc = exc
            stage_output = ""

        # Stage 2: attempt a single retry on exception OR invalid scales.
        # This prevents divide-by-zero failures in Stage 3 cell refinement.
        if index == STAGE2_INDEX and selected_cifs is not None:
            needs_retry = first_exc is not None
            if not needs_retry:
                scales = _read_phase_scales(gpx, hist, selected_cifs)
                needs_retry = _has_invalid_scales(scales)

            if needs_retry:
                rwp = _normalized_rwp(hist)
                rwp_history.append(rwp)
                print(
                    f"  Stage {index}: {'exception' if first_exc else 'invalid scales'} — "
                    "re-initialising and retrying."
                )
                _initialize_phase_scales(gpx, hist, selected_cifs)
                _record_stage(index)
                try:
                    stage_output = _run_refinement_stage(gpx, stage_dict)
                    first_exc = None
                except Exception as exc:
                    first_exc = exc
                    stage_output = ""
                rwp = _normalized_rwp(hist)
                print(f"  Stage {index} retry Rwp: {rwp:.4f}%")
                rwp_history.append(rwp)
                if first_exc is not None:
                    error = RefinementConvergenceError(
                        stage=index,
                        reason=str(first_exc),
                        rwp_history=rwp_history,
                    )
                    _write_diagnostics(error)
                    raise error from first_exc
                # Continue to post-run checks below using the retry's rwp.
                rwp = rwp_history[-1]
                refinement_signal = _read_refinement_signal(stage_output, gpx)
                if refinement_signal:
                    error = RefinementConvergenceError(
                        stage=index,
                        reason=refinement_signal,
                        rwp_history=rwp_history,
                    )
                    _write_diagnostics(error)
                    raise error
                if _is_hard_rwp_failure(rwp):
                    error = RefinementConvergenceError(
                        stage=index,
                        reason=f"Rwp remained pinned near 100% ({rwp:.4f}%)",
                        rwp_history=rwp_history,
                    )
                    _write_diagnostics(error)
                    raise error
                if len(rwp_history) >= 2 and rwp > rwp_history[-2]:
                    error = RefinementConvergenceError(
                        stage=index,
                        reason=(
                            f"Rwp rose from {rwp_history[-2]:.4f}% to {rwp:.4f}% "
                            f"after Stage {index} retry"
                        ),
                        rwp_history=rwp_history,
                    )
                    _write_diagnostics(error)
                    raise error
                continue  # Stage 2 retry succeeded; move to Stage 3.

        # Non-Stage-2 exception (or Stage 2 without retry needed): propagate.
        if first_exc is not None:
            rwp = _normalized_rwp(hist)
            rwp_history.append(rwp)
            error = RefinementConvergenceError(
                stage=index,
                reason=str(first_exc),
                rwp_history=rwp_history,
            )
            _write_diagnostics(error)
            raise error from first_exc

        rwp = _normalized_rwp(hist)
        print(f"  Stage {index} Rwp: {rwp:.4f}%")
        rwp_history.append(rwp)

        refinement_signal = _read_refinement_signal(stage_output, gpx)
        if refinement_signal:
            error = RefinementConvergenceError(
                stage=index,
                reason=refinement_signal,
                rwp_history=rwp_history,
            )
            _write_diagnostics(error)
            raise error

        if _is_hard_rwp_failure(rwp):
            error = RefinementConvergenceError(
                stage=index,
                reason=f"Rwp remained pinned near 100% ({rwp:.4f}%)",
                rwp_history=rwp_history,
            )
            _write_diagnostics(error)
            raise error

        if len(rwp_history) >= 2 and rwp > rwp_history[-2]:
            error = RefinementConvergenceError(
                stage=index,
                reason=(
                    f"Rwp rose from {rwp_history[-2]:.4f}% to {rwp:.4f}% "
                    f"after Stage {index}"
                ),
                rwp_history=rwp_history,
            )
            _write_diagnostics(error)
            raise error

    return rwp_history


def _safe_get_cell_esd(phase: Any) -> list[float]:
    """Return a/b/c ESDs or NaNs when the GSAS-II helper is unavailable."""
    try:
        _cell, esd = phase.get_cell_and_esd()
        if isinstance(esd, dict):
            return [
                float(esd.get("length_a", math.nan)),
                float(esd.get("length_b", math.nan)),
                float(esd.get("length_c", math.nan)),
            ]
        return [float(esd[0]), float(esd[1]), float(esd[2])]
    except (AttributeError, IndexError, TypeError, KeyError):
        return [math.nan, math.nan, math.nan]


def _safe_get_cell_values(phase: Any) -> tuple[float, float, float, float, float, float]:
    """Return a/b/c/alpha/beta/gamma across list-like and dict GSAS-II payloads."""
    cell = phase.get_cell()
    if isinstance(cell, dict):
        return (
            float(cell["length_a"]),
            float(cell["length_b"]),
            float(cell["length_c"]),
            float(cell["angle_alpha"]),
            float(cell["angle_beta"]),
            float(cell["angle_gamma"]),
        )
    return (
        float(cell[0]),
        float(cell[1]),
        float(cell[2]),
        float(cell[3]),
        float(cell[4]),
        float(cell[5]),
    )


def _normalized_rwp(hist: Any) -> float:
    """Normalize histogram Rwp to a percentage value."""
    rwp = float(hist.get_wR())
    if rwp < 1.0:
        rwp *= 100
    return rwp


def _is_hard_rwp_failure(rwp: float) -> bool:
    """Treat trajectories pinned near 100% as non-converged."""
    return math.isfinite(rwp) and rwp >= 99.0


def _run_refinement_stage(gpx: Any, stage_dict: dict[str, Any]) -> str:
    """Capture GSAS-II stage output so failure strings can be parsed reliably."""
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        gpx.do_refinements([stage_dict])
    stage_output = output.getvalue()
    if stage_output:
        print(stage_output, end="")
    return stage_output


def _read_refinement_signal(stage_output: str, gpx: Any) -> str | None:
    """Return a GSAS-II failure signal from stage output or fallback artifacts."""
    if "Note refinement problem:" in stage_output:
        detail = stage_output.split("Note refinement problem:", 1)[1].strip().splitlines()[0]
        return detail.strip()
    if "Warning: Soft (SVD) singularity in the Hessian" in stage_output:
        return "Soft (SVD) singularity in the Hessian"

    filename = getattr(gpx, "filename", None)
    if not filename:
        return None

    lst_path = Path(filename).with_suffix(".lst")
    if not lst_path.exists():
        return None

    lst_text = lst_path.read_text(encoding="utf-8", errors="ignore")
    if "**** ERROR: Refinement failed ****" in lst_text:
        if "Note refinement problem:" in lst_text:
            detail = lst_text.split("Note refinement problem:", 1)[1].strip().splitlines()[0]
            return detail.strip()
        return "GSAS-II reported a refinement failure"
    if "Warning: Soft (SVD) singularity in the Hessian" in lst_text:
        return "Soft (SVD) singularity in the Hessian"
    return None


def _extract_scale_value(hap: dict[str, Any]) -> float:
    """Read the real GSAS-II HAP Scale payload across legacy and runtime shapes."""
    scale_entry = hap.get("Scale")
    if isinstance(scale_entry, (int, float)):
        return float(scale_entry)
    if isinstance(scale_entry, (list, tuple)):
        if len(scale_entry) >= 2 and isinstance(scale_entry[1], bool):
            return float(scale_entry[0])
        if len(scale_entry) >= 2 and scale_entry[1] is not None:
            return float(scale_entry[1])
        if len(scale_entry) >= 1:
            return float(scale_entry[0])
    raise ValueError(f"Unsupported GSAS-II Scale payload: {scale_entry!r}")


def extract_results(gpx: Any, hist: Any, selected_cifs: dict[str, Path]) -> list[dict[str, Any]]:
    """Extract lattice parameters, weight fractions, and Rwp for each phase."""
    rwp = _normalized_rwp(hist)

    phase_rows: list[tuple[str, tuple[float, float, float, float, float, float], list[float], float]] = []
    total_scale = 0.0
    for phase_name in selected_cifs:
        phase = gpx.phase(phase_name)
        a_val, b_val, c_val, alpha_val, beta_val, gamma_val = _safe_get_cell_values(phase)
        a_esd, b_esd, c_esd = _safe_get_cell_esd(phase)
        hap = phase.getHAPvalues(hist)
        scale_value = _extract_scale_value(hap)
        total_scale += scale_value
        phase_rows.append(
            (
                phase_name,
                (a_val, b_val, c_val, alpha_val, beta_val, gamma_val),
                [a_esd, b_esd, c_esd],
                scale_value,
            )
        )

    if total_scale <= 0 or not math.isfinite(total_scale):
        raise ValueError("Weight fraction normalization requires a positive finite total scale")

    rows: list[dict[str, Any]] = []
    for phase_name, cell_values, cell_esds, scale_value in phase_rows:
        a_val, b_val, c_val, alpha_val, beta_val, gamma_val = cell_values
        a_esd, b_esd, c_esd = cell_esds
        weight_fraction = scale_value / total_scale
        rows.append(
            {
                "phase": phase_name,
                "a_Å": a_val,
                "b_Å": b_val,
                "c_Å": c_val,
                "alpha_deg": alpha_val,
                "beta_deg": beta_val,
                "gamma_deg": gamma_val,
                "a_esd": a_esd,
                "b_esd": b_esd,
                "c_esd": c_esd,
                "weight_fraction": float(weight_fraction),
                "wf_esd": math.nan,
                "Rwp": rwp,
            }
        )

    return rows
