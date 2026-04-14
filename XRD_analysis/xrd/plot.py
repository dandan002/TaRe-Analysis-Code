from pathlib import Path

import matplotlib.pyplot as plt


PHASE_PEAKS = {
    "Ta₂O₅": [23.6, 28.3, 36.6, 46.5, 55.5],
    # BCC alloy reflections from a ≈ 3.061 A inferred from the observed
    # 110/220 peaks near 41.7° and 90.8° in the TaRe_Full_Oxide scan.
    "TaRe bcc": [41.7, 60.4, 76.1, 90.8],
    "ReO₂": [25.8, 33.1, 37.5, 45.0, 54.2],
    "ReO₃": [26.3, 37.5, 46.2, 53.7],
    "Al₂O₃": [25.6, 35.1, 37.8, 43.4, 52.5, 57.5, 66.5],
}

PHASE_COLORS = {
    "Ta₂O₅": "#D55E00",
    "TaRe bcc": "#0072B2",
    "ReO₂": "#E69F00",
    "ReO₃": "#CC79A7",
    "Al₂O₃": "#009E73",
}


# Mapping from GSAS-II ASCII phase name → PHASE_COLORS Unicode key.
# GSAS-II uses the phase names as typed when the CIF was loaded ('Ta2O5', 'TaRe', etc.)
# while PHASE_COLORS uses display-quality Unicode subscript keys ('Ta₂O₅', 'TaRe bcc', etc.).
GSASII_TO_PLOT_KEY = {
    "Ta2O5": "Ta₂O₅",
    "ReO2":  "ReO₂",
    "ReO3":  "ReO₃",
    "TaRe":  "TaRe bcc",
    "Al2O3": "Al₂O₃",
}


def quicklook_plot(two_theta, intensity, out_path, phase_peaks=None) -> None:
    """Render the observed scan with phase overlays.

    Args:
        two_theta: array of 2theta values.
        intensity: array of intensity counts.
        out_path: output PNG path.
        phase_peaks: optional dict mapping phase label to 2theta positions.
            If omitted, the hardcoded PHASE_PEAKS values are used.
    """
    peaks_to_use = phase_peaks if phase_peaks is not None else PHASE_PEAKS
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.75, 3.5))
    ax.plot(two_theta, intensity, color="#000000", lw=0.8, label="Observed")

    for phase, peaks in peaks_to_use.items():
        color = PHASE_COLORS[phase]
        for index, position in enumerate(peaks):
            ax.axvline(
                position,
                ymin=0.88,
                ymax=1.0,
                color=color,
                lw=1.2,
                label=phase if index == 0 else None,
            )

    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(float(two_theta[0]), float(two_theta[-1]))
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


__all__ = ["PHASE_COLORS", "PHASE_PEAKS", "GSASII_TO_PLOT_KEY", "quicklook_plot"]
