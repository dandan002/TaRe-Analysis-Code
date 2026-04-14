import lmfit
import numpy as np
import pandas as pd

from config import QUBITS_DATA_DIR, make_run_dirs
from xps.import_ import import_ThermoAlpha
from analysis.models import (
    build_ta4f_models, build_re4f_models,
    ta4f_objective, re4f_objective,
    ta4f_expected_peaks, re4f_expected_peaks,
    ta4f_species_map, re4f_species_map,
    ta4f_expected_widths, re4f_expected_widths,
    ta4f_literature, re4f_literature,
)
from analysis.fitting import (
    correct_data, areaFractions, calculate_oxide_thickness,
    compare_peaks, _peak_rows,
    compare_ta4f_amplitudes, compare_re4f_amplitudes,
    compare_ta4f_widths, compare_re4f_widths,
)
from analysis.statistics import (
    compute_be_shifts,
    build_depth_profile_summary, build_be_shift_depth_profile,
)
from plots import spectra as plot_spectra
from plots import fit_components as plot_fits
from plots import summary as plot_summary


def _build_ta4f_initial_parameters():
    params = lmfit.Parameters()
    # Ta center and amplitude bounds here are fitting heuristics derived from the
    # local Ta workflow, not directly stated literature limits.
    params.add("Ta5_7b2_center", value=27.6, min=26.0, max=28, vary=True)
    params.add("Ta3_7b2_center", value=24.2, min=23.5, max=24.3, vary=True)
    params.add("Ta1_7b2_center", value=22.7, min=22.0, max=22.9, vary=True)
    params.add("metal_7b2_center", value=21.95, min=21.0, max=22.2, vary=True)
    params.add("interface_offset", value=0.44, min=0.2, max=0.7)
    params.add("alloy_offset", value=2.0, min=1.5, max=2.5, vary=True)
    params.add("_7b2_5b2_offset", value=29.24 - 27.35, min=1.0, max=3.0, vary=True)
    params.add("Ta5_7b2_sigma", value=0.16, min=0.0, max=1.0, vary=True)
    params.add("metal_7b2_sigma", value=0.08, min=0.0, max=0.2, vary=True)
    params.add("metal_7b2_skew", value=0.0, vary=False)
    params.add("Ta5_7b2_amplitude", value=0.16, min=0.0, max=6.0)
    params.add("Ta3_7b2_amplitude", value=0.015, min=0.0, max=0.3)
    params.add("Ta1_7b2_amplitude", value=0.008, min=0.0, max=0.3)
    params.add("metal_7b2_amplitude", value=0.25, min=0.05, max=6.0)
    params.add("interface_7b2_amplitude", value=0.03, min=0.01, max=0.06)
    params.add("alloy_7b2_amplitude", value=0.10, min=0.00, max=6.0)
    params.add("_7b2_5b2_ratio", value=1.4, min=1.0, max=3.0)
    params.add("interface_7b2_center", vary=False, expr="interface_offset+metal_7b2_center")
    params.add("alloy_7b2_center", vary=False, expr="alloy_offset+metal_7b2_center")
    params.add("Ta3_7b2_sigma", vary=False, expr="Ta5_7b2_sigma")
    params.add("Ta1_7b2_sigma", vary=False, expr="Ta5_7b2_sigma")
    params.add("interface_7b2_sigma", vary=False, expr="metal_7b2_sigma")
    params.add("alloy_7b2_sigma", vary=False, expr="metal_7b2_sigma")
    params.add("interface_7b2_skew", vary=False, expr="metal_7b2_skew")
    for species in ["Ta5", "Ta3", "Ta1", "metal", "interface", "alloy"]:
        params.add(
            f"{species}_5b2_center",
            vary=False,
            expr=f"{species}_7b2_center+_7b2_5b2_offset",
        )
        params.add(f"{species}_5b2_sigma", vary=False, expr=f"{species}_7b2_sigma")
        params.add(
            f"{species}_5b2_amplitude",
            vary=False,
            expr=f"{species}_7b2_amplitude/_7b2_5b2_ratio",
        )
    params.add("metal_5b2_skew", vary=False, expr="metal_7b2_skew")
    params.add("interface_5b2_skew", vary=False, expr="interface_7b2_skew")
    return params


def _build_re4f_initial_parameters():
    params = lmfit.Parameters()
    # Re references follow Greiner et al. (local paper set) with 2.42 eV
    # spin-orbit splitting; bounds remain moderately wide for robustness.
    params.add("Re_metal_7b2_center", value=40.35, min=39.8, max=41.0)
    params.add("ReO2_7b2_center", value=42.20, min=41.8, max=42.8)
    params.add("ReO3_7b2_center", value=43.10, min=42.6, max=44.0)
    params.add("Re2O7_7b2_center", value=45.5, min=45.0, max=46.7)
    params.add("Re_7b2_5b2_offset", value=2.42, min=2.35, max=2.50)
    params.add("Re_7b2_5b2_ratio", value=1.33, min=1.20, max=1.50)
    params.add("Re_metal_7b2_sigma", value=0.10, min=0.00, max=0.30)
    params.add("Re_metal_7b2_skew", value=0.00, vary=False)
    params.add("ReO2_7b2_sigma", value=0.18, min=0.05, max=0.60)
    params.add("ReO3_7b2_sigma", value=0.22, min=0.05, max=0.80)
    params.add("Re2O7_7b2_sigma", value=0.35, min=0.10, max=1.00)
    params.add("Re_metal_7b2_amplitude", value=0.10, min=0.00, max=6.00)
    params.add("ReO2_7b2_amplitude", value=0.020, min=0.00, max=2.00)
    params.add("ReO3_7b2_amplitude", value=0.015, min=0.00, max=2.00)
    params.add("Re2O7_7b2_amplitude", value=0.010, min=0.00, max=2.00)
    for species in ["Re_metal", "ReO2", "ReO3", "Re2O7"]:
        params.add(
            f"{species}_5b2_center",
            expr=f"{species}_7b2_center+Re_7b2_5b2_offset",
        )
        params.add(f"{species}_5b2_sigma", expr=f"{species}_7b2_sigma")
        params.add(
            f"{species}_5b2_amplitude",
            expr=f"{species}_7b2_amplitude/Re_7b2_5b2_ratio",
        )
    params.add("Re_metal_5b2_skew", expr="Re_metal_7b2_skew")
    return params


def _build_be_shift_depth_plot_frame(df_shift_depth_profile):
    mask = (
        (df_shift_depth_profile["spin"] == "7/2")
        & (df_shift_depth_profile["species"].isin(["Ta metal", "Re metal"]))
    )
    plot_df = df_shift_depth_profile.loc[mask].copy()
    plot_df["sample"] = (
        plot_df["sample"]
        + " - "
        + plot_df["species"]
    )
    return plot_df


def _etch_amplitude_reference_label():
    return "Control-derived reference"


def _etch_data_sort_key(data):
    etchlevel = data.etchlevel
    etchtime = data.etchtime
    if etchlevel is None or pd.isna(etchlevel):
        return (0, float("-inf"), float("-inf"))
    if etchtime is None or pd.isna(etchtime):
        return (1, etchlevel, float("inf"))
    return (1, etchlevel, etchtime)


def _dedupe_and_sort_etch_series(corrected_data):
    grouped = {}
    for data in corrected_data:
        grouped.setdefault(data.sample, []).append(data)

    filtered = []
    for sample in sorted(grouped):
        sample_rows = sorted(grouped[sample], key=_etch_data_sort_key)
        has_explicit_baseline = any(
            row.etchlevel is not None and not pd.isna(row.etchlevel) and float(row.etchlevel) == 0.0
            for row in sample_rows
        )
        for row in sample_rows:
            if has_explicit_baseline and (row.etchlevel is None or pd.isna(row.etchlevel)):
                continue
            filtered.append(row)
    return filtered


def _fit_corrected_series(corrected_data, *, objective, params_builder, model_dict):
    fit_results = []
    for data in corrected_data:
        params = params_builder()
        result = lmfit.minimize(
            objective,
            params,
            args=(data, model_dict),
            method="least_squares",
            nan_policy="omit",
        )
        fit_results.append(result)
    return fit_results


def main():
    figs_dir, csv_dir = make_run_dirs(run_suffix="ETCH")
    print(f"Output: {figs_dir.parent}")

    for sub in [
        "element_overlays",
        "fit_components",
        "area_fractions",
        "peak_comparisons",
        "peaks_vs_samples",
        "thickness",
        "be_shifts",
        "etch_profiles",
    ]:
        (figs_dir / sub).mkdir(parents=True, exist_ok=True)

    filepaths = [
        QUBITS_DATA_DIR / "ReTa_2023" / "12052023_ReTa03_composition.xlsx",
        QUBITS_DATA_DIR / "ReTa_2023" / "12052023_ReTa03.xlsx",
        QUBITS_DATA_DIR / "ReTa_2023" / "12062023_ReTa04.xlsx",
    ]
    missing = [str(filepath) for filepath in filepaths if not filepath.exists()]
    assert not missing, f"Missing files: {missing}"

    scans = []
    for filepath in filepaths:
        scans += import_ThermoAlpha(filepath)
    print(f"Imported {len(scans)} scans.")

    samples = sorted({scan.sample for scan in scans})
    elements = []
    for scan in scans:
        if scan.element != "XPS" and scan.element not in elements:
            elements.append(scan.element)

    ind_dict = {element: [] for element in elements}
    for index, scan in enumerate(scans):
        if scan.element in ind_dict:
            ind_dict[scan.element].append(index)

    survey_inds = [index for index, scan in enumerate(scans) if scan.element == "XPS"]
    print("Samples:", samples)
    print("Elements:", elements)

    plot_spectra.plot_survey_overlays(scans, survey_inds, figs_dir / "element_overlays")
    for element in elements:
        plot_spectra.plot_element_overlays(
            scans,
            ind_dict,
            element,
            out_dir=figs_dir / "element_overlays",
        )

    ta4f_inds = ind_dict.get("Ta4f", [])
    re4f_inds = ind_dict.get("Re4f", [])
    ta4f_corrected = _dedupe_and_sort_etch_series(correct_data(scans, ta4f_inds, be_max=30.5))
    re4f_corrected = _dedupe_and_sort_etch_series(correct_data(scans, re4f_inds, be_max=70.0))

    ta4f_model_dict = build_ta4f_models()
    re4f_model_dict = build_re4f_models()
    ta4f_fit_results = _fit_corrected_series(
        ta4f_corrected,
        objective=ta4f_objective,
        params_builder=_build_ta4f_initial_parameters,
        model_dict=ta4f_model_dict,
    )
    for index, (data, result) in enumerate(zip(ta4f_corrected, ta4f_fit_results)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        title = f"Ta4f - {data.sample}{etch}"
        print(f"[{index}] {data.sample}{etch}  ->  red. chi2 = {result.redchi:.6f}")
        plot_fits.plot_ta4f_fit_components(
            data,
            result,
            ta4f_model_dict,
            out_dir=figs_dir / "fit_components",
            title=title,
        )

    re4f_fit_results = _fit_corrected_series(
        re4f_corrected,
        objective=re4f_objective,
        params_builder=_build_re4f_initial_parameters,
        model_dict=re4f_model_dict,
    )
    for index, (data, result) in enumerate(zip(re4f_corrected, re4f_fit_results)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        title = f"Re4f - {data.sample}{etch}"
        print(f"[{index}] {data.sample}{etch}  ->  red. chi2 = {result.redchi:.6f}")
        plot_fits.plot_re4f_fit_components(
            data,
            result,
            re4f_model_dict,
            out_dir=figs_dir / "fit_components",
            title=title,
        )

    ta4f_species_order = ["metal", "Ta5", "Ta3", "Ta1", "interface", "alloy"]
    re4f_species_order = ["Re_metal", "ReO2", "ReO3", "Re2O7"]

    ta4f_area_frac = []
    ta4f_area_err = []
    for result in ta4f_fit_results:
        fractions, errors = areaFractions(result, ta4f_species_order)
        ta4f_area_frac.append(fractions)
        ta4f_area_err.append(errors)

    re4f_area_frac = []
    re4f_area_err = []
    for result in re4f_fit_results:
        fractions, errors = areaFractions(result, re4f_species_order)
        re4f_area_frac.append(fractions)
        re4f_area_err.append(errors)

    rows_ta = []
    for index, data in enumerate(ta4f_corrected):
        row = {
            "index": index,
            "sample": data.sample,
            "etchlevel": data.etchlevel,
            "etchtime": data.etchtime,
        }
        for species in ta4f_species_order:
            row[f"{species}_frac"] = ta4f_area_frac[index][species]
            row[f"{species}_err"] = ta4f_area_err[index][species]
        rows_ta.append(row)
    df_ta4f = pd.DataFrame(rows_ta).set_index("index")
    df_ta4f.to_csv(csv_dir / "ta4f_area_fractions.csv")

    rows_re = []
    for index, data in enumerate(re4f_corrected):
        row = {
            "index": index,
            "sample": data.sample,
            "etchlevel": data.etchlevel,
            "etchtime": data.etchtime,
        }
        for species in re4f_species_order:
            row[f"{species}_frac"] = re4f_area_frac[index][species]
            row[f"{species}_err"] = re4f_area_err[index][species]
        rows_re.append(row)
    df_re4f = pd.DataFrame(rows_re).set_index("index")
    df_re4f.to_csv(csv_dir / "re4f_area_fractions.csv")

    plot_summary.stacked_fraction_plot_ta4f(
        df_ta4f.reset_index(),
        ta4f_species_order,
        out_dir=figs_dir / "area_fractions",
    )
    plot_summary.stacked_fraction_plot_re4f(
        df_re4f.reset_index(),
        re4f_species_order,
        out_dir=figs_dir / "area_fractions",
    )

    ta_t = []
    ta_te = []
    for data, result in zip(ta4f_corrected, ta4f_fit_results):
        i_ox = (
            result.params["Ta5_7b2_amplitude"].value
            + result.params["Ta3_7b2_amplitude"].value
            + result.params["Ta1_7b2_amplitude"].value
        )
        i_metal = result.params["metal_7b2_amplitude"].value
        i_ox_err = np.sqrt(
            sum(
                (result.params[f"{species}_7b2_amplitude"].stderr or 0) ** 2
                for species in ["Ta5", "Ta3", "Ta1"]
            )
        )
        i_metal_err = result.params["metal_7b2_amplitude"].stderr or 0
        thickness, thickness_err = calculate_oxide_thickness(
            i_ox,
            i_metal,
            2.8,
            2.5,
            5.55e22,
            8.17e22,
            I_ox_err=i_ox_err,
            I_metal_err=i_metal_err,
        )
        ta_t.append(thickness)
        ta_te.append(thickness_err)

    re_t = []
    re_te = []
    for data, result in zip(re4f_corrected, re4f_fit_results):
        i_ox = (
            result.params["ReO2_7b2_amplitude"].value
            + result.params["ReO3_7b2_amplitude"].value
            + result.params["Re2O7_7b2_amplitude"].value
        )
        i_metal = result.params["Re_metal_7b2_amplitude"].value
        i_ox_err = np.sqrt(
            sum(
                (result.params[f"{species}_7b2_amplitude"].stderr or 0) ** 2
                for species in ["ReO2", "ReO3", "Re2O7"]
            )
        )
        i_metal_err = result.params["Re_metal_7b2_amplitude"].stderr or 0
        thickness, thickness_err = calculate_oxide_thickness(
            i_ox,
            i_metal,
            2.9,
            2.6,
            6.81e22,
            5.45e22,
            I_ox_err=i_ox_err,
            I_metal_err=i_metal_err,
        )
        re_t.append(thickness)
        re_te.append(thickness_err)

    ta_thick_rows = [
        {
            "index": index,
            "sample": data.sample,
            "etchlevel": data.etchlevel,
            "etchtime": data.etchtime,
            "oxide_thickness_nm": ta_t[index],
            "thickness_err_nm": ta_te[index],
        }
        for index, data in enumerate(ta4f_corrected)
    ]
    ta_thickness_df = pd.DataFrame(ta_thick_rows)
    ta_thickness_df.set_index("index").to_csv(csv_dir / "ta4f_oxide_thickness.csv")

    re_thick_rows = [
        {
            "index": index,
            "sample": data.sample,
            "etchlevel": data.etchlevel,
            "etchtime": data.etchtime,
            "oxide_thickness_nm": re_t[index],
            "thickness_err_nm": re_te[index],
        }
        for index, data in enumerate(re4f_corrected)
    ]
    re_thickness_df = pd.DataFrame(re_thick_rows)
    re_thickness_df.set_index("index").to_csv(csv_dir / "re4f_oxide_thickness.csv")

    ta_depth_profile = build_depth_profile_summary(
        ta_thickness_df,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
    )
    re_depth_profile = build_depth_profile_summary(
        re_thickness_df,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
    )
    ta_depth_profile.to_csv(csv_dir / "ta4f_thickness_depth_profile.csv", index=False)
    re_depth_profile.to_csv(csv_dir / "re4f_thickness_depth_profile.csv", index=False)
    plot_summary.plot_etch_profile_per_sample(
        ta_depth_profile,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
        ylabel="Ta oxide thickness (nm)",
        filename_prefix="ta4f_thickness_depth_profile",
        out_dir=figs_dir / "etch_profiles",
    )
    plot_summary.plot_etch_profile_per_sample(
        re_depth_profile,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
        ylabel="Re oxide thickness (nm)",
        filename_prefix="re4f_thickness_depth_profile",
        out_dir=figs_dir / "etch_profiles",
    )
    plot_summary.plot_etch_oxide_thickness_by_sample(
        ta_thickness_df,
        re_thickness_df,
        out_dir=figs_dir / "thickness",
    )

    ta4f_peak_comparisons = []
    re4f_peak_comparisons = []
    for index, (data, result) in enumerate(zip(ta4f_corrected, ta4f_fit_results)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        comparisons = compare_peaks(
            result,
            ta4f_expected_peaks,
            ta4f_species_map,
            sample_name=f"[{index}] {data.sample}{etch}",
            element="TA4F",
        )
        ta4f_peak_comparisons.append(
            {
                "index": index,
                "sample": data.sample,
                "etchlevel": data.etchlevel,
                "etchtime": data.etchtime,
                "comparisons": comparisons,
            }
        )
    for index, (data, result) in enumerate(zip(re4f_corrected, re4f_fit_results)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        comparisons = compare_peaks(
            result,
            re4f_expected_peaks,
            re4f_species_map,
            sample_name=f"[{index}] {data.sample}{etch}",
            element="RE4F",
        )
        re4f_peak_comparisons.append(
            {
                "index": index,
                "sample": data.sample,
                "etchlevel": data.etchlevel,
                "etchtime": data.etchtime,
                "comparisons": comparisons,
            }
        )

    pd.DataFrame(_peak_rows(ta4f_peak_comparisons, "Ta4f")).to_csv(
        csv_dir / "ta4f_peak_comparison.csv",
        index=False,
    )
    pd.DataFrame(_peak_rows(re4f_peak_comparisons, "Re4f")).to_csv(
        csv_dir / "re4f_peak_comparison.csv",
        index=False,
    )
    plot_summary.plot_peak_deviation(
        ta4f_peak_comparisons,
        "Ta4f",
        "Ta4f",
        out_dir=figs_dir / "peak_comparisons",
    )
    plot_summary.plot_peak_deviation(
        re4f_peak_comparisons,
        "Re4f",
        "Re4f",
        out_dir=figs_dir / "peak_comparisons",
    )

    ta4f_amp_rows = []
    re4f_amp_rows = []
    ta4f_width_rows = []
    re4f_width_rows = []
    amplitude_reference_type = "Control samples"
    amplitude_reference_label = _etch_amplitude_reference_label()
    for index, (data, result) in enumerate(zip(ta4f_corrected, ta4f_fit_results)):
        for comparison in compare_ta4f_amplitudes(result, data.sample, amplitude_reference_type):
            ta4f_amp_rows.append(
                {
                    "index": index,
                    "sample": data.sample,
                    "etchlevel": data.etchlevel,
                    "etchtime": data.etchtime,
                    "reference_profile": amplitude_reference_label,
                    **comparison,
                }
            )
        for comparison in compare_ta4f_widths(result, ta4f_expected_widths, data.sample):
            ta4f_width_rows.append(
                {
                    "index": index,
                    "sample": data.sample,
                    "etchlevel": data.etchlevel,
                    "etchtime": data.etchtime,
                    "species": comparison["species"],
                    "expected_sigma_eV": comparison["expected_sigma"],
                    "fitted_sigma_eV": comparison["fitted_sigma"],
                    "fitted_stderr": comparison["stderr"],
                    "delta_eV": comparison["delta"],
                }
            )

    for index, (data, result) in enumerate(zip(re4f_corrected, re4f_fit_results)):
        for comparison in compare_re4f_amplitudes(result, data.sample, amplitude_reference_type):
            re4f_amp_rows.append(
                {
                    "index": index,
                    "sample": data.sample,
                    "etchlevel": data.etchlevel,
                    "etchtime": data.etchtime,
                    "reference_profile": amplitude_reference_label,
                    **comparison,
                }
            )
        for comparison in compare_re4f_widths(result, re4f_expected_widths, data.sample):
            re4f_width_rows.append(
                {
                    "index": index,
                    "sample": data.sample,
                    "etchlevel": data.etchlevel,
                    "etchtime": data.etchtime,
                    "species": comparison["species"],
                    "expected_sigma_eV": comparison["expected_sigma"],
                    "fitted_sigma_eV": comparison["fitted_sigma"],
                    "fitted_stderr": comparison["stderr"],
                    "delta_eV": comparison["delta"],
                }
            )

    pd.DataFrame(ta4f_amp_rows).to_csv(csv_dir / "ta4f_amplitude_comparison.csv", index=False)
    pd.DataFrame(re4f_amp_rows).to_csv(csv_dir / "re4f_amplitude_comparison.csv", index=False)
    pd.DataFrame(ta4f_width_rows).to_csv(csv_dir / "ta4f_width_comparison.csv", index=False)
    pd.DataFrame(re4f_width_rows).to_csv(csv_dir / "re4f_width_comparison.csv", index=False)

    ta4f_colors = {
        "metal": "#1f77b4",
        "interface": "#ff7f0e",
        "alloy": "#2ca02c",
        "Ta5": "#d62728",
        "Ta3": "#9467bd",
        "Ta1": "#8c564b",
    }
    re4f_colors = {
        "Re_metal": "#1f77b4",
        "ReO2": "#ff7f0e",
        "ReO3": "#2ca02c",
        "Re2O7": "#d62728",
    }

    if ta4f_fit_results:
        plot_summary.plot_etch_metric_evolution_by_sample(
            ta4f_fit_results,
            ta4f_corrected,
            ta4f_species_order,
            "Ta4f",
            ta4f_colors,
            metric="binding_energy",
            filename_prefix="Ta4f_binding_energy_by_species",
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_etch_metric_evolution_by_sample(
            ta4f_fit_results,
            ta4f_corrected,
            ta4f_species_order,
            "Ta4f",
            ta4f_colors,
            metric="amplitude",
            filename_prefix="Ta4f_amplitude_by_species",
            out_dir=figs_dir / "peaks_vs_samples",
        )
    if re4f_fit_results:
        plot_summary.plot_etch_metric_evolution_by_sample(
            re4f_fit_results,
            re4f_corrected,
            re4f_species_order,
            "Re4f",
            re4f_colors,
            metric="binding_energy",
            filename_prefix="Re4f_binding_energy_by_species",
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_etch_metric_evolution_by_sample(
            re4f_fit_results,
            re4f_corrected,
            re4f_species_order,
            "Re4f",
            re4f_colors,
            metric="amplitude",
            filename_prefix="Re4f_amplitude_by_species",
            out_dir=figs_dir / "peaks_vs_samples",
        )

    shift_rows = []
    for data, result in zip(ta4f_corrected, ta4f_fit_results):
        for row in compute_be_shifts(
            [data],
            [result],
            ta4f_literature,
            ta4f_species_map,
            "Ta4f",
        ):
            row["etchlevel"] = data.etchlevel
            row["etchtime"] = data.etchtime
            shift_rows.append(row)
    for data, result in zip(re4f_corrected, re4f_fit_results):
        for row in compute_be_shifts(
            [data],
            [result],
            re4f_literature,
            re4f_species_map,
            "Re4f",
        ):
            row["etchlevel"] = data.etchlevel
            row["etchtime"] = data.etchtime
            shift_rows.append(row)

    df_shifts = pd.DataFrame(shift_rows)
    df_shifts.to_csv(csv_dir / "be_shift_summary.csv", index=False)
    df_shift_depth_profile = build_be_shift_depth_profile(df_shifts)
    df_shift_depth_profile.to_csv(csv_dir / "be_shift_depth_profile.csv", index=False)
    plot_summary.plot_etch_profile_per_sample(
        _build_be_shift_depth_plot_frame(df_shift_depth_profile),
        value_col="delta_eV",
        error_col="stderr_eV",
        ylabel="Binding-energy shift (eV)",
        filename_prefix="be_shift_depth_profile",
        out_dir=figs_dir / "etch_profiles",
    )

    plot_summary.plot_ta_re_correlation(df_shifts, out_dir=figs_dir / "be_shifts")

    print(f"\nDone. Results in {figs_dir.parent}")


if __name__ == "__main__":
    main()
