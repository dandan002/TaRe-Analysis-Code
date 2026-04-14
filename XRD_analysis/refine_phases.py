"""Phase 5 entrypoint: GSAS-II Rietveld refinement."""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from xrd.config import (
    CIF_DIR,
    DATA_DIR,
    GPX_FILE,
    GSASII_PATH,
    INSTPRM_FILE,
    RAW_CSV_FALLBACK,
    RAW_FILE,
    RESULTS_DIR,
    ensure_results,
    phase5_runtime_guidance,
)
from xrd.refinement import (
    RefinementConvergenceError,
    extract_results,
    histogram_format_hint,
    probe_histogram_input,
    run_staged_refinement,
    setup_project,
)


CANDIDATES_CSV = DATA_DIR / "candidates.csv"
OUTPUT_CSV = RESULTS_DIR / "rietveld_results.csv"
DIAGNOSTICS_JSON = RESULTS_DIR / "phase5_stage_diagnostics.json"
HARD_FAILURE_RWP = 50.0


def load_candidates(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load candidate rows grouped by target phase."""
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["target_phase"]].append(row)
    return dict(grouped)


def select_cifs(
    candidates: dict[str, list[dict[str, str]]], cif_dir: Path, auto: bool
) -> dict[str, Path]:
    """Select one CIF per target phase, interactively or by rank-1 default."""
    selected: dict[str, Path] = {}

    for target_phase, rows in candidates.items():
        cif_files = sorted(cif_dir.glob(f"{target_phase}_*.cif"))
        if not cif_files:
            print(f"WARNING: no CIF files found for {target_phase} — skipping")
            continue

        if auto or len(cif_files) == 1:
            chosen = cif_files[0]
            if len(cif_files) == 1:
                print(f"{target_phase}: only one option — auto-selected ({chosen.name})")
            else:
                print(f"{target_phase}: --auto selected rank-1 ({chosen.name})")
        else:
            print(f"\nSelect CIF for {target_phase}:")
            for index, cif_file in enumerate(cif_files, start=1):
                if index - 1 < len(rows):
                    row = rows[index - 1]
                    material_id = row["material_id"]
                    space_group = row["space_group"]
                    energy_above_hull = row["energy_above_hull"]
                else:
                    material_id = "?"
                    space_group = "?"
                    energy_above_hull = "?"
                print(
                    f"  {index}. {material_id}  {space_group}  "
                    f"eah={energy_above_hull}  [{cif_file.name}]"
                )
            raw = input("Choice [1]: ")
            try:
                idx = int(raw) - 1 if raw.strip() else 0
            except ValueError:
                idx = 0
            if idx < 0 or idx >= len(cif_files):
                idx = 0
            chosen = cif_files[idx]

        selected[target_phase] = chosen

    return selected


def save_results(rows: list[dict], output_path: Path) -> Path:
    _validate_results(rows)
    cols = [
        "phase",
        "a_Å",
        "b_Å",
        "c_Å",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "a_esd",
        "b_esd",
        "c_esd",
        "weight_fraction",
        "wf_esd",
        "Rwp",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_path, index=False, float_format="%.6g")
    return output_path


def _validate_results(rows: list[dict]) -> None:
    """Reject refinement output that is not physically usable downstream."""
    if not rows:
        raise ValueError("No refinement rows were produced.")

    weight_fractions = [float(row["weight_fraction"]) for row in rows]
    if any(not math.isfinite(value) for value in weight_fractions):
        raise ValueError("Weight fractions must be finite.")
    if any(value < 0 or value > 1 for value in weight_fractions):
        raise ValueError("Weight fractions must remain between 0 and 1.")
    if abs(sum(weight_fractions) - 1.0) > 0.05:
        raise ValueError("Weight fractions must normalize to approximately 1.")
    if len(rows) > 1 and len({round(value, 6) for value in weight_fractions}) == 1:
        raise ValueError("Weight fractions cannot all be identical for a multiphase refinement.")

    rwps = [float(row["Rwp"]) for row in rows]
    if any(not math.isfinite(value) for value in rwps):
        raise ValueError("Rwp values must be finite.")
    if any(value >= HARD_FAILURE_RWP for value in rwps):
        raise ValueError(f"Rwp must stay below the hard failure threshold of {HARD_FAILURE_RWP:.1f}%.")

    for lattice_key in ("a_Å", "b_Å", "c_Å"):
        if any(not math.isfinite(float(row[lattice_key])) for row in rows):
            raise ValueError(f"{lattice_key} must be finite for every phase.")

    by_phase = {str(row["phase"]): float(row["weight_fraction"]) for row in rows}
    if "Al2O3" in by_phase and by_phase["Al2O3"] < max(weight_fractions):
        raise ValueError("Al2O3 must remain the dominant phase fraction for this sample.")


def print_summary(rows: list[dict]) -> None:
    display_cols = ["phase", "a_Å", "b_Å", "c_Å", "weight_fraction", "Rwp"]
    table_data = [
        {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items() if k in display_cols}
        for row in rows
    ]
    print("\n" + tabulate(table_data, headers="keys", tablefmt="github"))


def choose_histogram_input(
    raw_file: Path,
    csv_fallback: Path,
    probe_reader,
) -> tuple[Path, str]:
    """Prefer the Bruker RAW input and fall back only after an explicit failure."""
    try:
        raw_probe = probe_reader(raw_file)
        if raw_probe is not None:
            suffix = raw_file.suffix.upper() if raw_file.suffix else "RAW"
            return raw_file, f"Using GSAS-II {suffix} reader for {raw_file.name}"
        raw_failure = f"{raw_file.name} returned an unusable histogram"
    except Exception as exc:  # pragma: no cover - exercised through tests
        raw_failure = str(exc)

    return (
        csv_fallback,
        f"Falling back to {csv_fallback.name} after explicit GSAS-II failure: {raw_failure}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: GSAS-II Rietveld refinement")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip interactive CIF selection; always pick rank-1 candidate.",
    )
    args = parser.parse_args()

    load_dotenv()
    ensure_results()

    if not CANDIDATES_CSV.exists():
        print(f"ERROR: {CANDIDATES_CSV} not found. Create data/candidates.csv with phase model or run search_phases.py first.")
        sys.exit(1)

    candidates = load_candidates(CANDIDATES_CSV)
    selected_cifs = select_cifs(candidates, CIF_DIR, auto=args.auto)

    if not selected_cifs:
        print("ERROR: No CIF files selected. Check data/cif/ directory.")
        sys.exit(1)

    print("\nSelected CIFs:")
    for phase, cif in selected_cifs.items():
        print(f"  {phase}: {cif.name}")

    print("\n" + phase5_runtime_guidance())
    print(f"Resolved GSASII_PATH: {GSASII_PATH}")

    if not RAW_FILE.exists() and not RAW_CSV_FALLBACK.exists():
        print(f"ERROR: neither {RAW_FILE} nor {RAW_CSV_FALLBACK} is available.")
        sys.exit(1)

    raw_input, selection_reason = choose_histogram_input(
        raw_file=RAW_FILE,
        csv_fallback=RAW_CSV_FALLBACK,
        probe_reader=lambda candidate: probe_histogram_input(
            selected_cifs=selected_cifs,
            raw_file=candidate,
            instprm_file=INSTPRM_FILE,
        ),
    )
    print(selection_reason)
    if raw_input == RAW_CSV_FALLBACK and not RAW_CSV_FALLBACK.exists():
        print(f"ERROR: CSV fallback selected but {RAW_CSV_FALLBACK} is missing.")
        sys.exit(1)

    print(f"\nCreating GSAS-II project → {GPX_FILE}")
    gpx, hist = setup_project(
        selected_cifs=selected_cifs,
        raw_file=raw_input,
        instprm_file=INSTPRM_FILE,
        gpx_file=GPX_FILE,
    )

    print("\nRunning staged refinement (4 stages, max 10 cycles each)...")
    try:
        rwp_history = run_staged_refinement(
            gpx,
            hist,
            selected_cifs=selected_cifs,
            diagnostics_path=DIAGNOSTICS_JSON,
            histogram_input=raw_input,
            histogram_format_hint=histogram_format_hint(raw_input),
        )
        print(f"Refinement complete. Rwp trace: {[f'{r:.3f}%' for r in rwp_history]}")

        rows = extract_results(gpx, hist, selected_cifs)
        csv_out = save_results(rows, OUTPUT_CSV)
    except (RefinementConvergenceError, ValueError) as exc:
        print(f"\nERROR: Phase 5 refinement failed: {exc}")
        print(f"Diagnostics → {DIAGNOSTICS_JSON}")
        sys.exit(1)

    print(f"\nSaved → {csv_out}")
    print(f"Diagnostics → {DIAGNOSTICS_JSON}")
    print_summary(rows)
    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
