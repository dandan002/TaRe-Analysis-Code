"""Phase 6: post-processing functions for Rietveld figure and XRD/XPS table."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root must be on sys.path before importing tare_analysis.
# postprocess.py lives at: XRD_analysis/xrd/postprocess.py
# Repo root is two levels up: XRD_analysis/../ = Thesis/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tare_analysis.plots import style  # noqa: E402 — sets rcParams at import time (dpi=300, 8pt)

from xrd.plot import GSASII_TO_PLOT_KEY, PHASE_COLORS  # noqa: E402

RWP_WARN_THRESHOLD = 20.0  # Rwp above this value = not converged → print warning


_GSASII_DEFAULT_PATH = str(Path.home() / "GSAS-II-src" / "backcompat")


def require_gsasii_path() -> str:
    """Bootstrap GSAS-II onto sys.path, or confirm it's already importable.

    Priority:
      1. GSASII_PATH env var (explicit override)
      2. Already importable as a site-package (python3.14 with GSASII installed)
      3. Hardcoded default installation path

    Returns the resolved path string (empty string if already a site-package).
    """
    gsasii_path = os.environ.get("GSASII_PATH")
    if gsasii_path:
        if gsasii_path not in sys.path:
            sys.path.insert(0, gsasii_path)
        return gsasii_path

    # Check if GSASIIscriptable is already importable (e.g. installed via pip/brew)
    import importlib.util
    if importlib.util.find_spec("GSASIIscriptable") is not None:
        return ""

    # Fall back to hardcoded installation
    if _GSASII_DEFAULT_PATH not in sys.path:
        sys.path.insert(0, _GSASII_DEFAULT_PATH)
    return _GSASII_DEFAULT_PATH


# ---------------------------------------------------------------------------
# XRD ↔ XPS mapping tables (hardcoded per D-05)
# ---------------------------------------------------------------------------

XRD_XPS_MAP = {
    "Ta2O5": {"xps_species": "Ta+5 (Ta2O5)",      "xps_element": "Ta4f"},
    "ReO2":  {"xps_species": "ReO2 (Re4+)",        "xps_element": "Re4f"},
    "ReO3":  {"xps_species": "ReO3 (Re6+)",        "xps_element": "Re4f"},
    "TaRe":  None,          # expanded to two sub-rows (TaRe-Ta, TaRe-Re) below
    "Al2O3": {"xps_species": "N/A (substrate)",    "xps_element": ""},
}

# Exact species names verified in be_shift_summary.csv (spin='7/2' rows).
# TaRe-Ta uses 'Ta alloy' (alloyed Ta signal, distinct from pure 'Ta metal').
XPS_SPECIES_NAMES = {
    "Ta2O5":   "Ta+5 (Ta2O5)",
    "ReO2":    "ReO2 (Re4+)",
    "ReO3":    "ReO3 (Re6+)",
    "TaRe-Ta": "Ta alloy",
    "TaRe-Re": "Re metal",
}


# ---------------------------------------------------------------------------
# rietveld_plot()
# ---------------------------------------------------------------------------

def rietveld_plot(gpx_file: Path, out_path: Path) -> None:
    """Produce a 2-panel Rietveld figure from a GSAS-II project file.

    Top panel: observed (black), calculated (orange/red), phase tick marks.
    Bottom panel: difference curve with zero-line reference.

    Prints a WARNING if Rwp > RWP_WARN_THRESHOLD (unconverged) but still saves
    the figure — useful for diagnostic output during Phase 5 debugging.

    Args:
        gpx_file: Path to TaRe_refinement.gpx (must exist before this call).
        out_path: Path to write the PNG (parent dirs created automatically).

    Raises:
        FileNotFoundError: if gpx_file does not exist.
        EnvironmentError: if GSASII_PATH is not set.
    """
    if not gpx_file.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_file}")

    # Bootstrap GSAS-II before import (per D-01 / Phase 5 bootstrap pattern).
    # Call our local require_gsasii_path() so this module stays importable
    # without GSASII_PATH being set at module load time.
    require_gsasii_path()
    import GSASIIscriptable as G2sc  # noqa: PLC0415

    gpx  = G2sc.G2Project(gpxfile=str(gpx_file))
    hist = gpx.histograms()[0]

    two_theta = np.asarray(hist.getdata("X"))
    yobs      = np.asarray(hist.getdata("Yobs"))
    ycalc     = np.asarray(hist.getdata("Ycalc"))
    ydiff     = np.asarray(hist.getdata("Residual"))

    # CRITICAL: hist.residuals is a @property returning a dict — NOT a method call.
    # Calling hist.residuals() raises TypeError: 'dict' object is not callable.
    rwp = hist.residuals.get("wR")

    if rwp is None or rwp > RWP_WARN_THRESHOLD:
        rwp_str = f"{rwp:.1f}%" if rwp is not None else "unknown"
        print(
            f"WARNING: Rwp={rwp_str} — refinement not converged. "
            "Figure shows diagnostic output. Run Phase 5 to convergence "
            "before producing final thesis figures."
        )

    # Per-phase tick mark 2theta positions.
    # Select the _MAX_TICKS strongest reflections by Fcalc² (col 9 of RefList),
    # merging degenerate reflections at the same 2θ before ranking.
    # Complex oxides have 300+ allowed reflections — only the strongest are shown.
    _MAX_TICKS = 20
    _DEGEN_TOL = 0.02  # degrees — merge reflections closer than this
    tick_dict: dict[str, np.ndarray] = {}
    for phase_name, phase_data in hist.reflections().items():
        ref_list = phase_data.get("RefList")
        if ref_list is None or len(ref_list) == 0:
            continue
        rl = np.asarray(ref_list)
        two_th = rl[:, 5]   # 2theta
        fcalc2 = rl[:, 9]   # Fcalc²

        # Merge degenerate positions: sort by 2theta, group within _DEGEN_TOL
        order = np.argsort(two_th)
        two_th_s, fcalc2_s = two_th[order], fcalc2[order]
        groups: list[tuple[float, float]] = []  # (2theta, summed_Fcalc2)
        g_pos, g_fc = two_th_s[0], fcalc2_s[0]
        for pos, fc in zip(two_th_s[1:], fcalc2_s[1:]):
            if pos - g_pos < _DEGEN_TOL:
                g_fc += fc  # accumulate intensity for this peak position
            else:
                groups.append((g_pos, g_fc))
                g_pos, g_fc = pos, fc
        groups.append((g_pos, g_fc))

        # Sort groups by total Fcalc² descending, take top _MAX_TICKS
        groups.sort(key=lambda x: x[1], reverse=True)
        top = sorted(g[0] for g in groups[:_MAX_TICKS])  # re-sort by 2theta for plotting
        tick_dict[phase_name] = np.array(top)

    # --- Figure: 2-panel GridSpec per D-02 ---
    # style import above already set rcParams (dpi=300, 8pt, no top/right spines).
    fig = plt.figure(figsize=(6.75, 4.5))  # slightly taller than FIG_DOUBLE_T for 2 panels
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Top panel: observed + calculated
    ax_main.plot(two_theta, yobs,  color="black",   lw=0.8, label="Observed")
    ax_main.plot(two_theta, ycalc, color="#D55E00", lw=0.8, label="Calculated")

    # Annotate Rwp in top-right corner
    rwp_label = f"Rwp = {rwp:.1f}%" if rwp is not None else "Rwp = n/a"
    ax_main.text(
        0.98, 0.97, rwp_label,
        transform=ax_main.transAxes,
        ha="right", va="top", fontsize=7,
    )

    ax_main.set_ylabel("Intensity (counts)")

    # Tick marks: one row per phase, stacked below the data curve (clip_on=False).
    # Use a fraction of the y-range so ticks are visible at any intensity scale.
    ax_main.autoscale_view()
    ymin_data, ymax_data = ax_main.get_ylim()
    y_range = ymax_data - ymin_data
    tick_row_height = 0.04 * y_range

    legend_handles = [
        plt.Line2D([], [], color="black",   lw=0.8, label="Observed"),
        plt.Line2D([], [], color="#D55E00", lw=0.8, label="Calculated"),
    ]

    for row_idx, (gsasii_name, plot_key) in enumerate(GSASII_TO_PLOT_KEY.items()):
        ticks = tick_dict.get(gsasii_name)
        if ticks is None or len(ticks) == 0:
            continue
        y_tick = ymin_data - (row_idx + 1) * tick_row_height * 1.2
        color = PHASE_COLORS[plot_key]
        ax_main.plot(
            ticks,
            np.full_like(ticks, y_tick),
            "|",
            color=color, ms=4, mew=0.8,
            clip_on=False,
        )
        legend_handles.append(
            plt.Line2D([], [], color=color, marker="|", ls="none", ms=4, label=plot_key)
        )

    ax_main.legend(handles=legend_handles, fontsize=7, frameon=False, ncol=3)

    # Bottom panel: difference curve
    ax_diff.plot(two_theta, ydiff, color="#56B4E9", lw=0.5)
    ax_diff.axhline(0, color="black", lw=0.5, ls="--")
    ax_diff.set_xlabel(r"2$\theta$ (degrees)")
    ax_diff.set_ylabel("Difference")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# build_xrd_xps_table()
# ---------------------------------------------------------------------------

def build_xrd_xps_table(
    rietveld_csv: Path,
    be_csv: Path,
) -> pd.DataFrame:
    """Build the XRD/XPS cross-reference DataFrame (D-04 column order).

    TaRe phase expands to two sub-rows (TaRe-Ta / Ta4f and TaRe-Re / Re4f).
    Al2O3 row has xps_species='N/A (substrate)' and NaN binding energies.
    Prints a WARNING if weight fractions are all 1.0 (unconverged placeholder).

    Columns (in order per D-04):
        xrd_phase, xps_species, xps_element, weight_fraction, wf_esd,
        be_literature_eV, be_measured_eV, Rwp

    Args:
        rietveld_csv: Path to rietveld_results.csv from Phase 5.
        be_csv: Path to be_shift_summary.csv from the XPS pipeline.

    Returns:
        pd.DataFrame with one row per XPS species (6 rows: 4 single + 2 TaRe sub-rows).
    """
    rdf = pd.read_csv(rietveld_csv)

    # Graceful warning if weight fractions are placeholders (all == 1.0)
    wf_values = rdf["weight_fraction"].astype(float)
    if (wf_values == 1.0).all():
        print(
            "WARNING: weight fractions are all 1.0 (not converged placeholder). "
            "Table shows diagnostic data — run Phase 5 to convergence for final output."
        )

    # Build BE lookup: species_name → {'expected_eV': float, 'mean_fitted_eV': float}
    be_df  = pd.read_csv(be_csv)
    be_7_2 = be_df[be_df["spin"] == "7/2"]
    be_lookup: dict[str, dict] = {}
    for sp in be_7_2["species"].unique():
        rows = be_7_2[be_7_2["species"] == sp]
        be_lookup[sp] = {
            "expected_eV":    float(rows["expected_eV"].iloc[0]),
            "mean_fitted_eV": float(rows["fitted_eV"].mean()),
        }

    output_rows: list[dict] = []

    for _, phase_row in rdf.iterrows():
        phase   = str(phase_row["phase"])
        wf      = float(phase_row["weight_fraction"])
        wf_esd_raw = phase_row["wf_esd"]
        wf_esd  = float(wf_esd_raw) if pd.notna(wf_esd_raw) else float("nan")
        rwp     = float(phase_row["Rwp"])

        if phase == "TaRe":
            # Two sub-rows: Ta alloy (alloyed Ta4f) and Re metal (Re4f)
            for xps_sp, xps_el in [("Ta alloy", "Ta4f"), ("Re metal", "Re4f")]:
                be_entry = be_lookup.get(xps_sp, {})
                output_rows.append({
                    "xrd_phase":        phase,
                    "xps_species":      xps_sp,
                    "xps_element":      xps_el,
                    "weight_fraction":  wf,
                    "wf_esd":           wf_esd,
                    "be_literature_eV": be_entry.get("expected_eV",    float("nan")),
                    "be_measured_eV":   be_entry.get("mean_fitted_eV", float("nan")),
                    "Rwp":              rwp,
                })
        elif phase == "Al2O3":
            output_rows.append({
                "xrd_phase":        phase,
                "xps_species":      "N/A (substrate)",
                "xps_element":      "",
                "weight_fraction":  wf,
                "wf_esd":           wf_esd,
                "be_literature_eV": float("nan"),
                "be_measured_eV":   float("nan"),
                "Rwp":              rwp,
            })
        else:
            mapping = XRD_XPS_MAP.get(phase, {"xps_species": phase, "xps_element": ""})
            if mapping is None:
                continue  # should not happen for non-TaRe phases
            xps_sp  = mapping["xps_species"]
            xps_el  = mapping["xps_element"]
            be_entry = be_lookup.get(xps_sp, {})
            output_rows.append({
                "xrd_phase":        phase,
                "xps_species":      xps_sp,
                "xps_element":      xps_el,
                "weight_fraction":  wf,
                "wf_esd":           wf_esd,
                "be_literature_eV": be_entry.get("expected_eV",    float("nan")),
                "be_measured_eV":   be_entry.get("mean_fitted_eV", float("nan")),
                "Rwp":              rwp,
            })

    col_order = [
        "xrd_phase", "xps_species", "xps_element",
        "weight_fraction", "wf_esd",
        "be_literature_eV", "be_measured_eV",
        "Rwp",
    ]
    return pd.DataFrame(output_rows, columns=col_order)
