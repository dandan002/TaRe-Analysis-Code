from xrd.config import DATA_DIR, RESULTS_DIR, ensure_results
from xrd.io import export_csv, get_wavelength, load_raw, validate
from xrd.plot import quicklook_plot


RAW_FILE = DATA_DIR / "TaRe_Full_Oxide.raw"
CSV_OUT = RESULTS_DIR / "TaRe_Full_Oxide.csv"
PLOT_OUT = RESULTS_DIR / "TaRe_Full_Oxide_quicklook.png"


def main() -> None:
    ensure_results()

    print(f"Loading {RAW_FILE} ...")
    two_theta, intensity = load_raw(RAW_FILE)
    print(f"  {len(two_theta)} points, 2θ = {two_theta[0]:.2f}–{two_theta[-1]:.2f}°")

    wavelength = get_wavelength(RAW_FILE)
    if wavelength is None:
        print("  Warning: USED_LAMBDA not found in header — skipping wavelength check")
    else:
        print(f"  Wavelength: {wavelength:.5f} Å")

    print("Validating ...")
    validate(two_theta, intensity, wavelength)
    print("  Validation passed.")

    print(f"Exporting CSV → {CSV_OUT}")
    export_csv(two_theta, intensity, CSV_OUT)

    print(f"Generating quick-look plot → {PLOT_OUT}")
    quicklook_plot(two_theta, intensity, PLOT_OUT, phase_peaks={})

    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
