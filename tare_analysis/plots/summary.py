# tare_analysis/plots/summary.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots.style import SPECIES_COLOR, SPECIES_LABEL, GROUP_COLOR, FIG_DOUBLE, FIG_DOUBLE_T, FIG_SINGLE, FIG_SINGLE_T, FIG_SINGLE_TALL
from analysis.statistics import pearson_annot, get_shift_for, sample_display_label, sample_group, sample_week


ETCH_PANEL_SPECS = [
    ("Growth After BOE Etch", "BOE"),
    ("Control (6M Oxide Growth)", "Control"),
]


def _etch_sample_ordered(df):
    ordered = df.copy().dropna(subset=["etchlevel"]).copy()
    ordered["etchlevel"] = ordered["etchlevel"].astype(float)
    sort_cols = ["sample", "etchlevel"]
    if "etchtime" in ordered.columns:
        sort_cols.append("etchtime")
    return ordered.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _etch_grouped_rows(df):
    grouped = df.copy()
    grouped["group"] = grouped["sample"].map(sample_group)
    unknown_samples = grouped.loc[grouped["group"] == "Unknown", "sample"].unique()
    if len(unknown_samples):
        raise ValueError(f"Unknown sample group for sample: {unknown_samples[0]}")
    return grouped


def _etch_sample_label(row):
    display_sample = sample_display_label(str(row["sample"]))
    etchlevel = row.get("etchlevel")
    if pd.isna(etchlevel):
        return display_sample
    return f"{display_sample}\nLv{int(float(etchlevel))}"


def _apply_right_legend(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)


def _sample_label(sample_name, etchlevel=None):
    display_sample = sample_display_label(sample_name)
    if etchlevel is None or pd.isna(etchlevel):
        return display_sample
    return f"{display_sample}\nLv{int(float(etchlevel))}"


def _group_split_positions(labels):
    boe_count = sum(sample_group(label) == "BOE" for label in labels)
    control_count = sum(sample_group(label) == "Control" for label in labels)
    return list(range(boe_count)) + list(range(boe_count + 1, boe_count + 1 + control_count))


def _sort_plot_series(corrected_data, values, errors):
    rows = [
        {
            "sample": data.sample,
            "value": value,
            "error": error,
        }
        for data, value, error in zip(corrected_data, values, errors)
    ]
    ordered = pd.DataFrame(rows)
    ordered["group"] = ordered["sample"].map(sample_group)
    unknown_samples = ordered.loc[ordered["group"] == "Unknown", "sample"].unique()
    if len(unknown_samples):
        raise ValueError(f"Unknown sample group for sample: {unknown_samples[0]}")
    ordered["week"] = ordered["sample"].map(sample_week)
    return ordered.sort_values(["group", "week", "sample"], kind="stable").reset_index(drop=True)


def _plot_element_oxide_thickness(corrected_data, thickness, errors, *, title, color, marker, out_path: Path):
    ordered = _sort_plot_series(corrected_data, thickness, errors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for ax, panel_title, panel_group in [
        (ax1, ETCH_PANEL_SPECS[0][0], ETCH_PANEL_SPECS[0][1]),
        (ax2, ETCH_PANEL_SPECS[1][0], ETCH_PANEL_SPECS[1][1]),
    ]:
        panel_df = ordered[ordered["group"] == panel_group].reset_index(drop=True)
        x = np.arange(len(panel_df))
        ax.errorbar(
            x,
            panel_df["value"].to_numpy(),
            yerr=panel_df["error"].to_numpy(),
            fmt=f"{marker}-",
            capsize=5,
            capthick=2,
            markersize=8,
            linewidth=2,
            color=color,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([sample_display_label(sample) for sample in panel_df["sample"].tolist()], rotation=45, ha="right")
        ax.set_ylabel("Oxide Thickness (nm)")
        ax.set_title(panel_title)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_split_stacked_fractions(df, species, *, out_path: Path, title: str):
    grouped = _etch_grouped_rows(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for ax, panel_title, panel_group in [
        (ax1, ETCH_PANEL_SPECS[0][0], ETCH_PANEL_SPECS[0][1]),
        (ax2, ETCH_PANEL_SPECS[1][0], ETCH_PANEL_SPECS[1][1]),
    ]:
        panel_df = grouped[grouped["group"] == panel_group].reset_index(drop=True)
        x = np.arange(len(panel_df))
        bottom = np.zeros(len(panel_df))
        for sp in species:
            vals = panel_df[f"{sp}_frac"].to_numpy()
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.6,
                label=SPECIES_LABEL.get(sp, sp),
                color=SPECIES_COLOR.get(sp, None),
            )
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([_etch_sample_label(row) for _, row in panel_df.iterrows()], rotation=45, ha="right")
        ax.set_ylabel("Fraction")
        ax.set_title(panel_title)
        ax.set_ylim(0, 1.02)
    _apply_right_legend(ax2)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def stacked_fraction_plot_ta4f(df, species, out_dir: Path, title="Ta4f Component Fractions"):
    _plot_split_stacked_fractions(
        df,
        species,
        out_path=out_dir / "ta4f_stacked_fractions.png",
        title=title,
    )


def stacked_fraction_plot_re4f(df, species, out_dir: Path, title="Re4f Component Fractions"):
    _plot_split_stacked_fractions(
        df,
        species,
        out_path=out_dir / "re4f_stacked_fractions.png",
        title=title,
    )


def plot_oxide_thickness(ta_corrected, ta_thickness, ta_err,
                         re_corrected, re_thickness, re_err, out_dir: Path):
    _plot_element_oxide_thickness(
        ta_corrected,
        ta_thickness,
        ta_err,
        title="Tantalum Oxide Layer Thickness",
        color=GROUP_COLOR.get("BOE", "C1"),
        marker="o",
        out_path=out_dir / "ta_oxide_thickness_comparison.png",
    )
    _plot_element_oxide_thickness(
        re_corrected,
        re_thickness,
        re_err,
        title="Rhenium Oxide Layer Thickness",
        color=GROUP_COLOR.get("Control", "C0"),
        marker="s",
        out_path=out_dir / "re_oxide_thickness_comparison.png",
    )


def plot_timecourse_with_group_difference(summary_df, diff_df, title, ylabel, filename, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for group in ["BOE", "Control"]:
        group_df = summary_df[summary_df["group"] == group].sort_values("week")
        ax1.errorbar(
            group_df["week"],
            group_df["mean"],
            yerr=group_df["sem"],
            fmt="o-",
            capsize=4,
            label=group,
            color=GROUP_COLOR.get(group, "gray"),
        )

    ax1.set_xlabel("Week")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()

    diff_df = diff_df.sort_values("week")
    ax2.errorbar(
        diff_df["week"],
        diff_df["difference"],
        yerr=diff_df["difference_sem"],
        fmt="s-",
        capsize=4,
        color="black",
    )
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("BOE - Control")
    ax2.set_title("Common-week difference")

    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cabrera_mott_fit(summary_df, fit_result, *, element, filename, out_dir: Path):
    fig, ax = plt.subplots(figsize=FIG_SINGLE_T)

    boe_df = (
        summary_df[summary_df["group"] == "BOE"]
        .dropna(subset=["week", "mean"])
        .sort_values("week")
    )
    control_df = (
        summary_df[summary_df["group"] == "Control"]
        .dropna(subset=["week", "mean"])
        .sort_values("week")
    )

    if not control_df.empty:
        ax.errorbar(
            control_df["week"].to_numpy(),
            control_df["mean"].to_numpy(),
            yerr=control_df["sem"].to_numpy() if "sem" in control_df.columns else None,
            fmt="s--",
            capsize=4,
            markersize=4,
            linewidth=1.0,
            alpha=0.6,
            color=GROUP_COLOR.get("Control", "gray"),
            label="Control mean",
        )

    ax.errorbar(
        boe_df["week"].to_numpy(),
        boe_df["mean"].to_numpy(),
        yerr=boe_df["sem"].to_numpy() if "sem" in boe_df.columns else None,
        fmt="o",
        capsize=4,
        markersize=5,
        color=GROUP_COLOR.get("BOE", "black"),
        label="BOE mean",
    )
    ax.plot(
        fit_result["t_fit"],
        fit_result["x_fit"],
        color="black",
        linewidth=1.5,
        label="Cabrera-Mott fit",
    )

    # X-axis: integer ticks at data weeks, x range starts at 0 with small margin
    all_weeks = sorted(set(boe_df["week"].tolist() + control_df["week"].tolist()))
    ax.set_xticks(all_weeks)
    x_max = fit_result["t_fit"][-1]
    ax.set_xlim(-0.3, x_max + 0.3)

    ax.set_xlabel("Week")
    ax.set_ylabel("Oxide Thickness (nm)")
    ax.set_title(f"{element} Cabrera-Mott Growth Fit")
    ax.legend(fontsize=7)
    plt.tight_layout()
    # Place equation and stats to the right of the axes, outside the plot area
    ax_pos = ax.get_position()
    fig.text(
        ax_pos.x1 + 0.03,
        ax_pos.y1,
        (
            f"x(t) = x0 + k ln(1 + t/tau)\n"
            f"x0 = {fit_result['x0']:.2f} +/- {fit_result['x0_err']:.2f} nm\n"
            f"k = {fit_result['k']:.2f} +/- {fit_result['k_err']:.2f} nm\n"
            f"tau = {fit_result['tau']:.2f} +/- {fit_result['tau_err']:.2f} wk\n"
            f"red. chi2 = {fit_result['redchi']:.3g}"
        ),
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_profile_by_sample(df, value_col, error_col=None, *, ylabel, filename, out_dir: Path):
    ordered = df.sort_values(["sample", "depth_order"], kind="stable")
    samples = list(dict.fromkeys(ordered["sample"]))
    fig, axes = plt.subplots(
        len(samples),
        1,
        figsize=(FIG_SINGLE[0], max(FIG_SINGLE[1], 2.2 * len(samples))),
    )
    axes = np.atleast_1d(axes)

    for ax, (sample, sample_df) in zip(axes, ordered.groupby("sample", sort=False)):
        sample_df = sample_df.reset_index(drop=True)
        x = np.arange(len(sample_df))
        errorbar_kwargs = dict(fmt="o-", capsize=4, markersize=7, linewidth=1.5, color="black")
        if error_col and error_col in sample_df.columns:
            yerr = sample_df[error_col]
            if yerr.notna().any():
                errorbar_kwargs["yerr"] = yerr.to_numpy()
        ax.errorbar(x, sample_df[value_col].to_numpy(), **errorbar_kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Unetched"
                if pd.isna(level)
                else (
                    f"Lv{int(level)}\n({time:g}s)"
                    if pd.notna(time)
                    else f"Lv{int(level)}"
                )
                for level, time in zip(sample_df["etchlevel"], sample_df["etchtime"])
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(sample_display_label(sample))

    axes[-1].set_xlabel("Depth profile step")
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_etch_profile_per_sample(
    df,
    *,
    value_col,
    error_col=None,
    ylabel,
    filename_prefix,
    out_dir: Path,
):
    ordered = _etch_sample_ordered(df)
    for sample, sample_df in ordered.groupby("sample", sort=False):
        if "etchtime" in sample_df.columns:
            sample_df = sample_df.sort_values(["etchlevel", "etchtime"], kind="stable")
        else:
            sample_df = sample_df.sort_values(["etchlevel"], kind="stable")

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        yerr = None
        if error_col and error_col in sample_df.columns:
            yerr = sample_df[error_col].to_numpy()
        ax.errorbar(
            sample_df["etchlevel"].to_numpy(),
            sample_df[value_col].to_numpy(),
            yerr=yerr,
            fmt="o-",
            capsize=4,
            markersize=5,
            linewidth=1.5,
            color="black",
        )
        ax.set_xlabel("Etch level")
        ax.set_ylabel(ylabel)
        ax.set_title(sample_display_label(sample))
        plt.tight_layout()
        fig.savefig(out_dir / f"{filename_prefix}_{sample}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_etch_metric_evolution_by_sample(
    fit_results,
    corrected_data,
    species_order,
    element_name,
    colors_dict,
    *,
    metric,
    filename_prefix,
    out_dir: Path,
):
    if len(fit_results) != len(corrected_data):
        raise ValueError("fit_results and corrected_data must have the same length")

    if metric == "binding_energy":
        param_suffix = "center"
        ylabel = "Binding Energy (eV)"
        value_label = "Binding Energy"
    elif metric == "amplitude":
        param_suffix = "amplitude"
        ylabel = "Fitted Amplitude"
        value_label = "Amplitude"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    rows = []
    for data, result in zip(corrected_data, fit_results):
        if pd.isna(data.etchlevel):
            continue
        for species in species_order:
            param = result.params.get(f"{species}_7b2_{param_suffix}")
            if param is None:
                continue
            rows.append(
                {
                    "sample": data.sample,
                    "etchlevel": float(data.etchlevel),
                    "etchtime": getattr(data, "etchtime", None),
                    "species": species,
                    "value": param.value,
                    "stderr": param.stderr or 0.0,
                }
            )

    if not rows:
        return

    ordered = _etch_sample_ordered(pd.DataFrame(rows))
    for sample, sample_df in ordered.groupby("sample", sort=False):
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for species in species_order:
            species_df = sample_df[sample_df["species"] == species]
            if species_df.empty:
                continue
            species_df = species_df.sort_values(["etchlevel", "etchtime"], kind="stable")
            ax.errorbar(
                species_df["etchlevel"].to_numpy(),
                species_df["value"].to_numpy(),
                yerr=species_df["stderr"].to_numpy(),
                fmt="o-" if metric == "binding_energy" else "s-",
                capsize=4,
                markersize=5,
                linewidth=1.5,
                alpha=0.8,
                color=colors_dict.get(species, "gray"),
                label=species,
            )
        ax.set_xlabel("Etch level")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{element_name} - {value_label} Evolution\n{sample_display_label(sample)}")
        if not sample_df.empty:
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / f"{filename_prefix}_{sample}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_etch_oxide_thickness_by_sample(ta_df, re_df, *, out_dir: Path):
    ta_grouped = _etch_grouped_rows(_etch_sample_ordered(ta_df))
    re_grouped = _etch_grouped_rows(_etch_sample_ordered(re_df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    for ax, panel_title, panel_group in [
        (ax1, ETCH_PANEL_SPECS[0][0], ETCH_PANEL_SPECS[0][1]),
        (ax2, ETCH_PANEL_SPECS[1][0], ETCH_PANEL_SPECS[1][1]),
    ]:
        ta_panel = ta_grouped[ta_grouped["group"] == panel_group].reset_index(drop=True)
        re_panel = re_grouped[re_grouped["group"] == panel_group].reset_index(drop=True)
        x = np.arange(len(ta_panel))

        ax.errorbar(
            x,
            ta_panel["oxide_thickness_nm"].to_numpy(),
            yerr=ta_panel["thickness_err_nm"].to_numpy() if "thickness_err_nm" in ta_panel.columns else None,
            fmt="o-",
            capsize=4,
            markersize=5,
            linewidth=1.5,
            color=GROUP_COLOR.get("BOE", "black"),
            label="Ta",
        )
        ax.errorbar(
            x,
            re_panel["oxide_thickness_nm"].to_numpy(),
            yerr=re_panel["thickness_err_nm"].to_numpy() if "thickness_err_nm" in re_panel.columns else None,
            fmt="s-",
            capsize=4,
            markersize=5,
            linewidth=1.5,
            color=GROUP_COLOR.get("Control", "gray"),
            label="Re",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_etch_sample_label(row) for _, row in ta_panel.iterrows()], rotation=45, ha="right")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Oxide Thickness (nm)")
        ax.set_title(panel_title)
    _apply_right_legend(ax2)
    plt.tight_layout()
    fig.savefig(out_dir / "oxide_thickness_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_peak_deviation(comparisons_list, element_name, figprefix, out_dir: Path):
    from plots.style import OKABE_ITO
    for cd in comparisons_list:
        comps = cd['comparisons']
        if not comps:
            continue
        labels  = [f"{c['species']} {c['spin']}" for c in comps]
        deltas  = [c['delta']    for c in comps]
        expected = [c['expected'] for c in comps]
        fitted   = [c['fitted']   for c in comps]
        stderr   = [c['stderr']   for c in comps]
        y_pos = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE_T)
        ax1.errorbar(expected, fitted, yerr=stderr, fmt='o', capsize=5, markersize=8, alpha=0.7)
        lim = [min(min(expected), min(fitted)) - 0.5, max(max(expected), max(fitted)) + 0.5]
        ax1.plot(lim, lim, 'k--', alpha=0.5, label='Perfect agreement')
        ax1.set_xlabel('Expected BE (eV)')
        ax1.set_ylabel('Fitted BE (eV)')
        ax1.set_title(f'{element_name} Peak Positions\n{sample_display_label(cd["sample"])}')
        ax1.legend()
        bar_colors = [OKABE_ITO['vermillion'] if d > 0 else (OKABE_ITO['blue'] if d < 0 else 'gray')
                      for d in deltas]
        ax2.barh(y_pos, deltas, color=bar_colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Delta = Fitted - Expected (eV)')
        ax2.set_title('Peak Position Deviations')
        ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        safe  = re.sub(r'[^\w\-_\. ]', '_', cd['sample']).strip()
        etch  = f"_EtchLv{cd['etchlevel']}" if cd['etchlevel'] is not None else ""
        fig.savefig(out_dir / f"{figprefix}_peak_comparison_{safe}{etch}.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_be_shift_boe_vs_control(df_group, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    for ax, element in [(ax1, 'Ta4f'), (ax2, 'Re4f')]:
        sub = df_group[(df_group['element'] == element) & (df_group['spin'] == '7/2')]
        species_list = sub['species'].unique()
        x = np.arange(len(species_list))
        width = 0.35
        for group, offset in [('BOE', -width/2), ('Control', width/2)]:
            g = sub[sub['group'] == group].set_index('species')
            means = [g.loc[sp, 'weighted_mean_delta_eV'] if sp in g.index else 0.0 for sp in species_list]
            errs  = [g.loc[sp, 'weighted_stderr_eV']     if sp in g.index else 0.0 for sp in species_list]
            ax.bar(x + offset, means, width, label=group,
                   color=GROUP_COLOR[group], alpha=0.75,
                   yerr=errs, capsize=4, error_kw={'linewidth': 1.5})
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(species_list, rotation=45, ha='right')
        ax.set_ylabel('Weighted Mean Delta(BE) (eV)')
        ax.set_title(f'{element} — BOE vs. Control BE Shifts (7/2)')
        ax.legend()
        flagged = df_group[(df_group['element'] == element) &
                           (df_group['spin'] == '7/2') &
                           df_group['flagged']]['species'].unique()
        for sp in flagged:
            idx_sp = np.where(species_list == sp)[0]
            if len(idx_sp):
                ax.annotate('*', xy=(idx_sp[0], 0), xycoords=('data', 'axes fraction'),
                             xytext=(0, -5), textcoords='offset points',
                             ha='center', fontsize=14, color='red')
    plt.tight_layout()
    fig.savefig(out_dir / "be_shift_boe_vs_control.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ta_re_correlation(df_shifts, out_dir: Path):
    from analysis.statistics import sample_group
    ta_samples = set(df_shifts[df_shifts['element'] == 'Ta4f']['sample'])
    re_samples = set(df_shifts[df_shifts['element'] == 'Re4f']['sample'])
    common = sorted(ta_samples & re_samples)

    primary, secondary = [], []
    for sample in common:
        ta_m, _ = get_shift_for(df_shifts, sample, 'Ta4f', 'Ta metal')
        re_m, _ = get_shift_for(df_shifts, sample, 'Re4f', 'Re metal')
        ta_o, _ = get_shift_for(df_shifts, sample, 'Ta4f', 'Ta+5 (Ta2O5)')
        re_o, _ = get_shift_for(df_shifts, sample, 'Re4f', 'ReO3 (Re6+)')
        if not (np.isnan(ta_m) or np.isnan(re_m)):
            primary.append((ta_m, re_m, sample))
        if not (np.isnan(ta_o) or np.isnan(re_o)):
            secondary.append((ta_o, re_o, sample))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SINGLE_TALL)
    for ax, pairs, title in [
        (ax1, primary,   'Metal: Ta metal vs. Re metal (7/2)'),
        (ax2, secondary, 'Oxide: Ta+5 vs. ReO3 (7/2)'),
    ]:
        if not pairs:
            ax.set_title(f'{title}\n(no common samples)')
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        colors = [GROUP_COLOR.get(sample_group(lbl), 'gray') for lbl in labels]
        ax.scatter(xs, ys, c=colors, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(sample_display_label(lbl), (x, y), textcoords='offset points', xytext=(5, 5), fontsize=7)
        lim = [min(min(xs), min(ys)) - 0.1, max(max(xs), max(ys)) + 0.1]
        ax.plot(lim, lim, 'k--', alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5, linestyle=':')
        ax.axvline(0, color='gray', lw=0.5, linestyle=':')
        ax.set_xlabel('Ta Delta(BE) (eV)')
        ax.set_ylabel('Re Delta(BE) (eV)')
        ax.set_title(title)
        ax.text(0.03, 0.97, pearson_annot(xs, ys), transform=ax.transAxes,
                fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.tight_layout()
    fig.savefig(out_dir / "be_shift_ta_re_correlation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_peaks_vs_ev(fit_results, corrected_data, species_order, element_name,
                     colors_dict, out_dir: Path):
    sample_labels = []
    for data in corrected_data:
        etch_info = f"Lv{data.etchlevel}" if data.etchlevel is not None else ""
        sample_labels.append(_sample_label(data.sample, data.etchlevel if etch_info else None))

    for species in species_order:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
        x_positions, binding_energies, be_errors, amplitudes, amp_errors, valid_labels = [], [], [], [], [], []
        for idx, (data, result) in enumerate(zip(corrected_data, fit_results)):
            center_key = f'{species}_7b2_center'
            amp_key    = f'{species}_7b2_amplitude'
            if center_key in result.params and amp_key in result.params:
                x_positions.append(idx)
                binding_energies.append(result.params[center_key].value)
                be_errors.append(result.params[center_key].stderr or 0)
                amplitudes.append(result.params[amp_key].value)
                amp_errors.append(result.params[amp_key].stderr or 0)
                valid_labels.append(sample_labels[idx])
        if x_positions:
            color = colors_dict.get(species, 'gray')
            ax1.errorbar(x_positions, binding_energies, yerr=be_errors, fmt='o-', color=color,
                         markersize=8, linewidth=2, capsize=4, alpha=0.8)
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Binding Energy (eV)')
            ax1.set_title(f'{element_name} {species} - Binding Energy')
            ax2.errorbar(x_positions, amplitudes, yerr=amp_errors, fmt='s-', color=color,
                         markersize=8, linewidth=2, capsize=4, alpha=0.8)
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Fitted Amplitude')
            ax2.set_title(f'{element_name} {species} - Amplitude')
        plt.tight_layout()
        safe_species = species.replace(' ', '_').replace('/', '-')
        fig.savefig(out_dir / f"{element_name}_peaks_vs_samples_{safe_species}.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_peaks_vs_ev_merged(fit_results, corrected_data, species_order, element_name,
                             colors_dict, out_dir: Path):
    sample_labels = []
    for data in corrected_data:
        etch_info = f"Lv{data.etchlevel}" if data.etchlevel is not None else ""
        sample_labels.append(_sample_label(data.sample, data.etchlevel if etch_info else None))

    n_samples = len(corrected_data)
    x_all = np.arange(n_samples)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for species in species_order:
        x_positions, binding_energies, be_errors, amplitudes, amp_errors = [], [], [], [], []
        for idx, (data, result) in enumerate(zip(corrected_data, fit_results)):
            center_key = f'{species}_7b2_center'
            amp_key    = f'{species}_7b2_amplitude'
            if center_key in result.params and amp_key in result.params:
                x_positions.append(idx)
                binding_energies.append(result.params[center_key].value)
                be_errors.append(result.params[center_key].stderr or 0)
                amplitudes.append(result.params[amp_key].value)
                amp_errors.append(result.params[amp_key].stderr or 0)
        if x_positions:
            color = colors_dict.get(species, 'gray')
            ax1.errorbar(x_positions, binding_energies, yerr=be_errors, fmt='o-', color=color,
                         markersize=8, linewidth=2, capsize=4, alpha=0.8, label=species)
            ax2.errorbar(x_positions, amplitudes, yerr=amp_errors, fmt='s-', color=color,
                         markersize=8, linewidth=2, capsize=4, alpha=0.8, label=species)

    ax1.set_xticks(x_all)
    ax1.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Binding Energy (eV)')
    ax1.set_title(f'{element_name} - Binding Energy by Species')
    ax2.set_xticks(x_all)
    ax2.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Fitted Amplitude')
    ax2.set_title(f'{element_name} - Amplitude by Species')
    _apply_right_legend(ax2)
    plt.tight_layout()
    fig.savefig(out_dir / f"{element_name}_peaks_vs_samples_merged.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_metric_evolution_by_species(
    fit_results,
    corrected_data,
    species_order,
    element_name,
    colors_dict,
    *,
    metric,
    filename,
    out_dir: Path,
):
    from analysis.statistics import sample_group

    if len(fit_results) != len(corrected_data):
        raise ValueError("fit_results and corrected_data must have the same length")

    filtered_species_order = [species for species in species_order if species.lower() != "redp1"]

    if metric == "binding_energy":
        param_suffix = "center"
        ylabel = "Binding Energy (eV)"
        suptitle_metric = "Binding Energy"
    elif metric == "amplitude":
        param_suffix = "amplitude"
        ylabel = "Fitted Amplitude"
        suptitle_metric = "Amplitude"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    panel_specs = [
        (ax1, "Growth After BOE Etch", "BOE"),
        (ax2, "Control (6M Oxide Growth)", "Control"),
    ]

    for ax, panel_title, panel_group in panel_specs:
        panel_data = []
        for data, result in zip(corrected_data, fit_results):
            group = sample_group(data.sample)
            if group == "Unknown":
                raise ValueError(f"Unknown sample group for sample: {data.sample}")
            if group == panel_group:
                panel_data.append((data, result))
        x_positions = np.arange(len(panel_data))

        for species in filtered_species_order:
            y_values = []
            y_errors = []
            x_values = []
            param_key = f"{species}_7b2_{param_suffix}"

            for idx, (data, result) in enumerate(panel_data):
                param = result.params.get(param_key)
                if param is None:
                    continue
                x_values.append(idx)
                y_values.append(param.value)
                y_errors.append(param.stderr or 0)

            if x_values:
                color = colors_dict.get(species, "gray")
                marker = "o" if metric == "binding_energy" else "s"
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    fmt=f"{marker}-",
                    color=color,
                    markersize=8,
                    linewidth=2,
                    capsize=4,
                    alpha=0.8,
                    label=species,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([sample_display_label(data.sample) for data, _ in panel_data], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)

    _apply_right_legend(ax2)
    fig.suptitle(f"{element_name} - {suptitle_metric} Evolution by Species")
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_amplitude_by_species(
    area_fracs,
    area_errs,
    corrected_data,
    species_order,
    element_name,
    colors_dict,
    *,
    filename,
    out_dir: Path,
):
    from analysis.statistics import sample_group

    if len(area_fracs) != len(corrected_data) or len(area_errs) != len(corrected_data):
        raise ValueError("area_fracs, area_errs, and corrected_data must have the same length")

    filtered_species_order = [species for species in species_order if species.lower() != "redp1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    panel_specs = [
        (ax1, "Growth After BOE Etch", "BOE"),
        (ax2, "Control (6M Oxide Growth)", "Control"),
    ]

    for ax, panel_title, panel_group in panel_specs:
        panel_data = []
        for global_idx, data in enumerate(corrected_data):
            group = sample_group(data.sample)
            if group == "Unknown":
                raise ValueError(f"Unknown sample group for sample: {data.sample}")
            if group == panel_group:
                panel_data.append((global_idx, data))
        x_positions = np.arange(len(panel_data))

        for species in filtered_species_order:
            y_values = []
            y_errors = []
            x_values = []

            for idx, (global_idx, data) in enumerate(panel_data):
                frac = area_fracs[global_idx].get(species)
                err = area_errs[global_idx].get(species)
                if frac is None or err is None:
                    continue
                x_values.append(idx)
                y_values.append(frac)
                y_errors.append(err)

            if x_values:
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    fmt="s-",
                    color=colors_dict.get(species, "gray"),
                    markersize=8,
                    linewidth=2,
                    capsize=4,
                    alpha=0.8,
                    label=species,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([sample_display_label(data.sample) for _, data in panel_data], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Normalized Amplitude (fraction of total)")
        ax.set_title(panel_title)

    _apply_right_legend(ax2)
    fig.suptitle(f"{element_name} - Normalized Amplitude Evolution by Species")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
