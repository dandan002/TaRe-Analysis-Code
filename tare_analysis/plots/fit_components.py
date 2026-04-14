# tare_analysis/plots/fit_components.py
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from plots.style import SPECIES_COLOR, SPECIES_LABEL, FIG_SINGLE_T


FIT_LINE_COLOR = "gray"


def _fit_display_title(sample_name, fallback_title=None):
    element_prefix = ""
    if fallback_title:
        prefix_match = re.match(r"^(Ta4f|Re4f)\s*-\s*", fallback_title)
        if prefix_match:
            element_prefix = f"{prefix_match.group(1)} "
    match = re.search(r"^(BOE|Control)_WK(\d+)$", sample_name)
    if not match:
        return fallback_title if fallback_title else f"{element_prefix}{sample_name}".strip()
    group, week_str = match.groups()
    display_week = int(week_str)
    if group == "Control":
        return f"{element_prefix}Control WK{display_week}".strip()
    return f"{element_prefix}{display_week}WK Post-Etch".strip()


def plot_ta4f_fit_components(data, result, modelDict, out_dir: Path, title=None,
                             species=None):
    x, y, s = data.BE, data.intensity, data.intensityErr
    tot = sum(mdl.eval(result.params, x=x) for mdl in modelDict.values())
    residual = (y - tot) / np.where(s > 0, s, 1.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SINGLE_T,
                                    gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
    ax1.errorbar(x, y, s, fmt='o', ms=2, capsize=3, label='Data', color=FIT_LINE_COLOR)
    _species = species if species is not None else ['metal', 'interface', 'alloy', 'Ta5', 'Ta1', 'Ta3']
    for sp in _species:
        col = SPECIES_COLOR.get(sp, 'gray')
        lbl = SPECIES_LABEL.get(sp, sp)
        y72 = modelDict[f'{sp}_7_2'].eval(result.params, x=x)
        y52 = modelDict[f'{sp}_5_2'].eval(result.params, x=x)
        ax1.fill_between(x, y72, alpha=0.35, color=col, label=f'{lbl} 7/2')
        ax1.fill_between(x, y52, alpha=0.35, color=col, hatch='//', label=f'{lbl} 5/2')
    ax1.plot(x, tot, color=FIT_LINE_COLOR, lw=1, label='fit')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.legend(bbox_to_anchor=(1, 1), fontsize=6)
    ax1.set_title(_fit_display_title(data.sample, title), fontsize=7)
    ax1.set_xlim(31, 20)
    ax2.plot(x, residual, lw=1)
    ax2.axhline(0, linestyle='--', lw=0.8)
    ymax = max(3, 1.2 * np.max(np.abs(residual)))
    ax2.set_ylim(-ymax, ymax)
    ax2.fill_between(x, -1, 1, alpha=0.15, color='gray')
    ax2.set_xlabel('Binding Energy (eV)')
    ax2.set_ylabel('Residual / σ')
    plt.tight_layout()
    fname = title if title else "ta4f_fit"
    safe = re.sub(r'[^\w\-_\. ]', '_', fname).strip().replace(' ', '_')
    fig.savefig(out_dir / f"{safe}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_re4f_fit_components(data, result, modelDict, out_dir: Path, title=None):
    x, y, s = data.BE, data.intensity, data.intensityErr
    tot = sum(mdl.eval(result.params, x=x) for mdl in modelDict.values())
    residual = (y - tot) / np.where(s > 0, s, 1.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SINGLE_T,
                                    gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
    ax1.errorbar(x, y, s, fmt='o', ms=2, capsize=3, label='Data', color=FIT_LINE_COLOR)
    for sp in ['Re_metal', 'ReO2', 'ReO3', 'Re2O7']:
        col = SPECIES_COLOR.get(sp, 'gray')
        lbl = SPECIES_LABEL.get(sp, sp)
        y72 = modelDict[f'{sp}_7_2'].eval(result.params, x=x)
        y52 = modelDict[f'{sp}_5_2'].eval(result.params, x=x)
        ax1.fill_between(x, y72, alpha=0.35, color=col, label=f'{lbl} 7/2')
        ax1.fill_between(x, y52, alpha=0.35, color=col, hatch='//', label=f'{lbl} 5/2')
    ax1.plot(x, tot, color=FIT_LINE_COLOR, lw=1, label='fit')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.legend(bbox_to_anchor=(1, 1), fontsize=6)
    ax1.set_title(_fit_display_title(data.sample, title), fontsize=7)
    ax1.set_xlim(52, 32)
    ax2.plot(x, residual, lw=1)
    ax2.axhline(0, linestyle='--', lw=0.8)
    ymax = max(3, 1.2 * np.max(np.abs(residual)))
    ax2.set_ylim(-ymax, ymax)
    ax2.fill_between(x, -1, 1, alpha=0.15, color='gray')
    ax2.set_xlabel('Binding Energy (eV)')
    ax2.set_ylabel('Residual / σ')
    plt.tight_layout()
    fname = title if title else "re4f_fit"
    safe = re.sub(r'[^\w\-_\. ]', '_', fname).strip().replace(' ', '_')
    fig.savefig(out_dir / f"{safe}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
