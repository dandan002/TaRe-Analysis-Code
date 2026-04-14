# tare_analysis/plots/spectra.py
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from plots.style import OVERLAY_CYCLE
from analysis.statistics import sample_display_label
from xps import peaks as xps_peaks
from xps.funcs import addPeakLabel


def plot_survey_overlays(scans, survey_inds, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in survey_inds:
        s = scans[idx]
        ax.plot(s.BE, s.intensity / np.max(s.intensity), label=sample_display_label(s.sample), lw=1)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Normalized Intensity")
    ax.invert_xaxis()
    ax.set_xlim(0, 500)
    fig.savefig(out_dir / "survey_overlays.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_element_overlays(scans, indDict, element, out_dir: Path, totalNorm=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    inds = indDict.get(element, [])
    if not inds:
        return
    norm = -np.inf
    if totalNorm:
        for i in inds:
            norm = max(norm, np.max(scans[i].intensity))
    for color, i in zip(OVERLAY_CYCLE, inds):
        s = scans[i]
        y = s.intensity / (norm if totalNorm else np.max(s.intensity))
        display_sample = sample_display_label(s.sample)
        label = f"{display_sample} (Lv{s.etchlevel})" if getattr(s, 'etchlevel', None) is not None else display_sample
        ax.plot(s.BE, y, label=label, lw=1, color=color)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Normalized Intensity")
    ax.invert_xaxis()
    safe = re.sub(r'[^\w]', '_', element)
    fig.savefig(out_dir / f"element_overlay_{safe}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def _overlay_week_label(sample_name: str) -> str:
    return sample_display_label(sample_name)


def _plot_overlay_subset(scans, inds, title: str, out_path: Path, totalNorm=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    norm = -np.inf
    if totalNorm:
        for i in inds:
            norm = max(norm, np.max(scans[i].intensity))
    for color, i in zip(OVERLAY_CYCLE, inds):
        scan = scans[i]
        y = scan.intensity / (norm if totalNorm else np.max(scan.intensity))
        ax.plot(scan.BE, y, label=_overlay_week_label(scan.sample), lw=1, color=color)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(title)
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Normalized Intensity")
    ax.invert_xaxis()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_postetch_core_overlays(scans, indDict, out_dir: Path, totalNorm=False):
    plot_specs = [
        ("Ta4f", "BOE", "Ta4f Spectra (post-etch)", "ta4f_postetch_overlay.png"),
        ("Re4f", "BOE", "Re4f Spectra (post-etch)", "re4f_postetch_overlay.png"),
        ("Ta4f", "Control", "Ta4f Spectra (Control)", "ta4f_control_overlay.png"),
        ("Re4f", "Control", "Re4f Spectra (Control)", "re4f_control_overlay.png"),
    ]
    for element, group_prefix, title, filename in plot_specs:
        inds = [
            i for i in indDict.get(element, [])
            if getattr(scans[i], "sample", "").startswith(f"{group_prefix}_")
        ]
        if not inds:
            continue
        _plot_overlay_subset(scans, inds, title, out_dir / filename, totalNorm=totalNorm)
