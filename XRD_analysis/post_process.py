"""Phase 6 entrypoint: post-processing — Rietveld figure + XRD/XPS cross-reference table."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# ── path constants (defined here to avoid importing xrd.config which raises
#    EnvironmentError at load time when GSASII_PATH is unset) ────────────────
_XRD_ROOT   = Path(__file__).resolve().parent        # XRD_analysis/
_REPO_ROOT   = _XRD_ROOT.parent                       # Thesis/
RESULTS_DIR  = _XRD_ROOT / "results"
GPX_FILE     = RESULTS_DIR / "TaRe_refinement.gpx"

RIETVELD_CSV  = RESULTS_DIR / "rietveld_results.csv"
FIGURE_OUT    = RESULTS_DIR / "TaRe_rietveld.png"
TABLE_CSV_OUT = RESULTS_DIR / "xrd_xps_table.csv"
TABLE_TEX_OUT = RESULTS_DIR / "xrd_xps_table.tex"

# XPS BE data (from tare_analysis Phase results — fixed path for S8 run)
BE_CSV = (
    _REPO_ROOT
    / "tare_analysis"
    / "results"
    / "2026-04-09_22-43-47_S8"
    / "csv"
    / "be_shift_summary.csv"
)


def _ensure_results() -> None:
    """Create results/ directory if it does not yet exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: generate Rietveld figure and XRD/XPS cross-reference table"
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip Rietveld figure generation.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip cross-reference table generation.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive / headless execution (no prompts).",
    )
    args = parser.parse_args()

    load_dotenv()
    _ensure_results()

    # ── pre-flight checks ──────────────────────────────────────────────────
    if not args.no_figure and not GPX_FILE.exists():
        print(
            f"ERROR: {GPX_FILE} not found.\n"
            "Phase 5 must complete before post-processing.\n"
            "Run: python refine_phases.py --auto"
        )
        sys.exit(1)

    if not RIETVELD_CSV.exists():
        print(
            f"ERROR: {RIETVELD_CSV} not found.\n"
            "Phase 5 must complete and write rietveld_results.csv before post-processing."
        )
        sys.exit(1)

    if not BE_CSV.exists():
        print(f"ERROR: XPS BE data not found at {BE_CSV}")
        sys.exit(1)

    # ── hard-block gate: reject diagnostic/non-converged Phase 5 outputs ─────
    # D-04: Phase 6 must not emit thesis-facing artifacts from invalid inputs.
    # Warning-only behavior is insufficient — this check must abort the run.
    import csv as _csv  # noqa: PLC0415
    import math as _math  # noqa: PLC0415

    HARD_FAILURE_RWP = 50.0  # mirrors refine_phases.HARD_FAILURE_RWP

    with RIETVELD_CSV.open(newline="", encoding="utf-8") as _f:
        _reader = _csv.DictReader(_f)
        _rows = list(_reader)

    if not _rows:
        print("ERROR: rietveld_results.csv is empty — Phase 5 produced no output.")
        sys.exit(1)

    _rwps = [float(r["Rwp"]) for r in _rows if r.get("Rwp")]
    _wfs = [float(r["weight_fraction"]) for r in _rows if r.get("weight_fraction")]

    _bad_rwp = any(not _math.isfinite(v) or v >= HARD_FAILURE_RWP for v in _rwps)
    _placeholder_wf = (
        len(_wfs) > 1
        and len({round(v, 6) for v in _wfs}) == 1
        and round(_wfs[0], 6) == 1.0
    )

    if _bad_rwp or _placeholder_wf:
        if _bad_rwp:
            print(
                f"ERROR: rietveld_results.csv contains Rwp >= {HARD_FAILURE_RWP:.0f}% "
                f"({max(_rwps):.1f}%). Phase 5 refinement did not converge.\n"
                "Phase 6 is blocked until Phase 5 produces a valid convergence result.\n"
                "These outputs are diagnostic only and must not be used as thesis evidence."
            )
        if _placeholder_wf:
            print(
                "ERROR: rietveld_results.csv contains placeholder weight fractions "
                "(all 1.0). Phase 5 serialised non-converged output.\n"
                "Phase 6 is blocked until Phase 5 produces valid weight fractions."
            )
        sys.exit(1)

    # ── lazy import of xrd.postprocess (safe: no GSASII_PATH needed at import
    #    time — guard is deferred inside rietveld_plot() call) ────────────────
    from xrd.postprocess import build_xrd_xps_table, rietveld_plot  # noqa: PLC0415

    # ── figure ─────────────────────────────────────────────────────────────
    if not args.no_figure:
        print(f"\nGenerating Rietveld figure from {GPX_FILE.name} ...")
        try:
            rietveld_plot(GPX_FILE, FIGURE_OUT)
        except EnvironmentError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        print("Skipping figure (--no-figure).")

    # ── table ──────────────────────────────────────────────────────────────
    if not args.no_table:
        print(f"\nBuilding XRD/XPS cross-reference table ...")
        df = build_xrd_xps_table(RIETVELD_CSV, BE_CSV)

        df.to_csv(TABLE_CSV_OUT, index=False, float_format="%.4f")
        print(f"  CSV saved  → {TABLE_CSV_OUT}")

        latex_str = df.to_latex(
            index=False,
            float_format="%.3f",
            na_rep="—",
            caption="XRD phases and corresponding XPS species assignments.",
            label="tab:xrd_xps",
            column_format="llllrrrr",
        )
        TABLE_TEX_OUT.write_text(latex_str, encoding="utf-8")
        print(f"  LaTeX saved → {TABLE_TEX_OUT}")

        # Console preview
        print("\nCross-reference table:")
        preview_cols = ["xrd_phase", "xps_species", "xps_element", "weight_fraction", "Rwp"]
        # Filter to columns that exist (guard for unexpected schema)
        preview_cols = [c for c in preview_cols if c in df.columns]
        print(df[preview_cols].to_string(index=False))
    else:
        print("Skipping table (--no-table).")

    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()
