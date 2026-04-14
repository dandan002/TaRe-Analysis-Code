# tare_analysis/analysis/statistics.py
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

STDERR_FLOOR = 0.1  # eV — avoid divide-by-zero for fixed parameters


def sample_group(sample_name):
    if 'BOE' in sample_name:
        return 'BOE'
    if 'Control' in sample_name:
        return 'Control'
    return 'Unknown'


def sample_week(sample_name):
    match = re.search(r"_WK(\d+)", sample_name)
    if not match:
        return np.nan
    return int(match.group(1))


def sample_display_label(sample_name):
    match = re.search(r"(WK\d+)", sample_name)
    if match:
        return match.group(1)
    return sample_name


def sample_timepoint_key(sample_name):
    return sample_group(sample_name), sample_week(sample_name)


def _with_group_week(df):
    out = df.copy()
    out["group"] = out["sample"].map(sample_group)
    out["week"] = out["sample"].map(sample_week)
    return out


def weighted_mean_shift(deltas, stderrs):
    """Weighted mean and uncertainty. stderrs floored at STDERR_FLOOR."""
    sigmas = np.array([max(s, STDERR_FLOOR) for s in stderrs])
    weights = 1.0 / sigmas ** 2
    mean = np.sum(weights * np.array(deltas)) / np.sum(weights)
    uncertainty = 1.0 / np.sqrt(np.sum(weights))
    return mean, uncertainty


def get_shift_for(df, sample, element, species, spin='7/2'):
    """Return (delta_eV, stderr_eV) for a specific sample/element/species/spin, or (nan, nan)."""
    mask = ((df['sample'] == sample) & (df['element'] == element) &
            (df['species'] == species) & (df['spin'] == spin))
    row = df[mask]
    if row.empty:
        return np.nan, np.nan
    return row.iloc[0]['delta_eV'], row.iloc[0]['stderr_eV']


def pearson_annot(xs, ys):
    if len(xs) < 3:
        return "r = N/A (n < 3)"
    r, p = pearsonr(xs, ys)
    sig = "significant" if p < 0.05 else "not significant at α=0.05"
    return f"r = {r:.2f}, p = {p:.3f} ({sig})"


def build_timecourse_summary(df, value_col):
    out = _with_group_week(df)
    grouped = (
        out.dropna(subset=["week", value_col])
        .groupby(["group", "week"], as_index=False)[value_col]
        .agg(mean="mean", std="std", count="count")
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def build_baseline_change(df, value_col):
    out = _with_group_week(df).dropna(subset=["week"]).copy()
    baselines = (
        out[out["week"] == 0]
        .groupby("group", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "baseline_value"})
    )
    out = out.merge(baselines, on="group", how="left")
    out["delta_from_wk0"] = out[value_col] - out["baseline_value"]
    return out


def build_group_week_differences(df, value_col):
    summary = build_timecourse_summary(df, value_col=value_col)
    boe = summary[summary["group"] == "BOE"][["week", "mean", "sem"]].rename(
        columns={"mean": "boe_mean", "sem": "boe_sem"}
    )
    control = summary[summary["group"] == "Control"][["week", "mean", "sem"]].rename(
        columns={"mean": "control_mean", "sem": "control_sem"}
    )
    merged = boe.merge(control, on="week", how="inner")
    merged["difference"] = merged["boe_mean"] - merged["control_mean"]
    merged["difference_sem"] = np.sqrt(merged["boe_sem"] ** 2 + merged["control_sem"] ** 2)
    return merged.sort_values("week").reset_index(drop=True)


def build_be_shift_timecourse(df_shifts):
    out = _with_group_week(df_shifts)
    grouped = (
        out.dropna(subset=["week", "delta_eV"])
        .groupby(["element", "species", "spin", "group", "week"], as_index=False)["delta_eV"]
        .agg(mean="mean", std="std", count="count")
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped.sort_values(["element", "species", "spin", "group", "week"]).reset_index(drop=True)


def etch_sort_key(etchlevel, etchtime):
    """Sort unetched rows first, then etched rows by level and time."""
    if pd.isna(etchlevel):
        return (0, float("-inf"), float("-inf"))
    if pd.isna(etchtime):
        return (1, etchlevel, float("inf"))
    return (1, etchlevel, etchtime)


def _add_depth_order(df, group_cols):
    out = df.copy()
    out["_etch_sort"] = [
        etch_sort_key(level, time)
        for level, time in zip(out["etchlevel"], out["etchtime"])
    ]
    out = out.sort_values(group_cols + ["_etch_sort"], kind="stable").reset_index(drop=True)
    if group_cols:
        out["depth_order"] = out.groupby(group_cols, sort=False).cumcount()
    else:
        out["depth_order"] = np.arange(len(out))
    return out.drop(columns=["_etch_sort"])


def build_depth_profile_summary(df, value_col, error_col):
    if error_col not in df.columns:
        raise KeyError(error_col)
    cols = ["sample", "etchlevel", "etchtime", value_col]
    cols.append(error_col)
    out = _add_depth_order(df.loc[:, cols], ["sample"])
    return out.loc[:, ["sample", "etchlevel", "etchtime", "depth_order"] + [c for c in [value_col, error_col] if c in out.columns]]


def build_be_shift_depth_profile(df_shifts):
    out = _add_depth_order(df_shifts, ["sample", "element", "species", "spin"])
    return out


def compute_be_shifts(corrected_data, fit_results, literature, species_map, element):
    """Build list of dicts with per-sample per-species shift data."""
    rows = []
    for data, result in zip(corrected_data, fit_results):
        group = sample_group(data.sample)
        for label, prefix in species_map.items():
            if label not in literature:
                continue
            for spin in ['7/2', '5/2']:
                spin_str = spin.replace('/', 'b')
                param_key = f'{prefix}_{spin_str}_center'
                if param_key not in result.params:
                    continue
                expected = literature[label][spin]
                fitted = result.params[param_key].value
                stderr = result.params[param_key].stderr or 0.0
                rows.append({
                    'sample': data.sample, 'group': group, 'element': element,
                    'species': label, 'spin': spin,
                    'expected_eV': expected, 'fitted_eV': fitted,
                    'stderr_eV': stderr, 'delta_eV': fitted - expected,
                })
    return rows


def build_be_shift_by_group(df_shifts):
    """Compute weighted mean shifts per (element, species, spin, group) and flag outliers."""
    group_rows = []
    for (element, species, spin), grp in df_shifts.groupby(['element', 'species', 'spin']):
        boe  = grp[grp['group'] == 'BOE']
        ctrl = grp[grp['group'] == 'Control']
        for subset, label in [(boe, 'BOE'), (ctrl, 'Control')]:
            if len(subset) == 0:
                continue
            mean, unc = weighted_mean_shift(subset['delta_eV'].values, subset['stderr_eV'].values)
            group_rows.append({
                'element': element, 'species': species, 'spin': spin, 'group': label,
                'weighted_mean_delta_eV': mean, 'weighted_stderr_eV': unc, 'flagged': False,
            })

    df_group = pd.DataFrame(group_rows)
    for (element, species, spin), pair in df_group.groupby(['element', 'species', 'spin']):
        if len(pair) < 2:
            continue
        boe_row  = pair[pair['group'] == 'BOE']
        ctrl_row = pair[pair['group'] == 'Control']
        if boe_row.empty or ctrl_row.empty:
            continue
        m_boe,  s_boe  = boe_row.iloc[0]['weighted_mean_delta_eV'],  boe_row.iloc[0]['weighted_stderr_eV']
        m_ctrl, s_ctrl = ctrl_row.iloc[0]['weighted_mean_delta_eV'], ctrl_row.iloc[0]['weighted_stderr_eV']
        if abs(m_boe - m_ctrl) > 2 * np.sqrt(s_boe ** 2 + s_ctrl ** 2):
            mask = ((df_group['element'] == element) & (df_group['species'] == species) &
                    (df_group['spin'] == spin))
            df_group.loc[mask, 'flagged'] = True
    return df_group
