"""Phase 4 entrypoint: Materials Project phase search and CIF download."""

import os

from dotenv import load_dotenv
from xrd.config import DATA_DIR, RESULTS_DIR, ensure_cif_dir, ensure_results
from xrd.io import get_wavelength, load_raw, validate
from xrd.phases import (
    _get_api_key,
    build_candidates_df,
    build_phase_peaks_dict,
    download_top_cifs,
    find_observed_peaks,
    print_candidates_table,
    save_candidates_csv,
    search_all_phases,
)
from xrd.plot import quicklook_plot


RAW_FILE = DATA_DIR / "TaRe_Full_Oxide.raw"
PLOT_OUT = RESULTS_DIR / "TaRe_Full_Oxide_quicklook.png"


def main() -> None:
    load_dotenv()
    ensure_results()
    ensure_cif_dir()

    print(f"Loading {RAW_FILE} ...")
    two_theta, intensity = load_raw(RAW_FILE)

    wavelength = get_wavelength(RAW_FILE)
    print("Validating ...")
    validate(two_theta, intensity, wavelength)

    MP_API_KEY = os.getenv("MP_API_KEY")
    if not MP_API_KEY:
        raise ValueError("MP_API_KEY not found in environment variables.")
    api_key = _get_api_key()
    print("Searching Materials Project ...")
    results = search_all_phases(api_key)
    frame = build_candidates_df(results)
    print_candidates_table(frame)
    csv_out = save_candidates_csv(frame)
    print(f"Saved candidates CSV → {csv_out}")

    downloaded = download_top_cifs(results, api_key)
    observed_peaks = find_observed_peaks(two_theta, intensity)
    phase_peaks = build_phase_peaks_dict(downloaded, observed_peaks=observed_peaks)

    print(f"Generating quick-look plot → {PLOT_OUT}")
    quicklook_plot(two_theta, intensity, PLOT_OUT, phase_peaks=phase_peaks)
    print("Phase 4 complete.")


if __name__ == "__main__":
    main()
