# tare_analysis/main.py
import numpy as np
import pandas as pd
import lmfit

from config import DATA_DIR, make_run_dirs
from xps.import_ import import_ThermoAlpha
from analysis.models import (
    build_ta4f_models, build_re4f_models,
    ta4f_objective, re4f_objective,
    ta4f_expected_peaks, re4f_expected_peaks,
    ta4f_species_map, re4f_species_map,
    ta4f_expected_widths, re4f_expected_widths,
    ta4f_literature, re4f_literature,
    re4f_redp_offsets,
)
from analysis.fitting import (
    correct_data, areaFractions, calculate_oxide_thickness,
    compare_peaks, _peak_rows,
    compare_ta4f_amplitudes, compare_re4f_amplitudes,
    compare_ta4f_widths, compare_re4f_widths,
    cabrera_mott_fit,
    check_ta5_center_constraint,
    safe_minimize,
)
from analysis.statistics import (
    compute_be_shifts, build_be_shift_by_group,
    build_timecourse_summary, build_baseline_change,
    build_group_week_differences, build_be_shift_timecourse,
)
from plots import spectra as plot_spectra
from plots import fit_components as plot_fits
from plots import summary as plot_summary


def main():
    figs_dir, csv_dir = make_run_dirs()
    print(f"Output: {figs_dir.parent}")

    # ── Sub-directories for figures ──────────────────────────────────────────
    for sub in ["element_overlays", "fit_components", "area_fractions",
                "peak_comparisons", "peaks_vs_samples", "thickness", "be_shifts",
                "timecourse"]:
        (figs_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Section 2: Data import & indexing ────────────────────────────────────
    BOE_files = [
        DATA_DIR / 'TaRe_1124' / 'BOE_1124.xlsx',
        DATA_DIR / 'TaRe_1201' / 'BOE_1201.xlsx',
        DATA_DIR / 'TaRe_1208' / 'BOE_Small_1208.xlsx',
        DATA_DIR / 'TaRe_1215' / 'BOE_Small_1215.xlsx',
    ]
    Control_files = [
        DATA_DIR / 'TaRe_1124' / 'Control_1124.xlsx',
        DATA_DIR / 'TaRe_1201' / 'Control_1201.xlsx',
        DATA_DIR / 'TaRe_1208' / 'Control_1208.xlsx',
        DATA_DIR / 'TaRe_1215' / 'Control_1215.xlsx',
    ]
    filepaths = BOE_files + Control_files
    missing = [str(f) for f in filepaths if not f.exists()]
    assert not missing, f'Missing files: {missing}'

    scans = []
    for fp in filepaths:
        scans += import_ThermoAlpha(fp)
    print(f'Imported {len(scans)} scans.')

    samples  = sorted({s.sample for s in scans})
    elements = []
    for s in scans:
        if s.element != "XPS" and s.element not in elements:
            elements.append(s.element)

    indDict = {el: [] for el in elements}
    for i, s in enumerate(scans):
        if s.element in indDict:
            indDict[s.element].append(i)

    survey_inds = [i for i, s in enumerate(scans) if s.element == "XPS"]
    print("Samples:", samples)
    print("Elements:", elements)

    # ── Section 3: Raw data plots ─────────────────────────────────────────────
    plot_spectra.plot_survey_overlays(scans, survey_inds, figs_dir / "element_overlays")
    plot_spectra.plot_postetch_core_overlays(
        scans,
        indDict,
        out_dir=figs_dir / "element_overlays",
    )

    # ── Section 4: Background correction ─────────────────────────────────────
    ta4f_inds = indDict.get("Ta4f", [])
    re4f_inds = indDict.get("Re4f", [])
    Ta4fCorrected = correct_data(scans, ta4f_inds, be_max=30.5)
    Re4fCorrected = correct_data(scans, re4f_inds, be_max=70.0)

    # ── Section 6: Initial parameters ────────────────────────────────────────
    ta4f_modelDict = build_ta4f_models()
    re4f_modelDict = build_re4f_models()

    p0_ta4f = lmfit.Parameters()
    p0_ta4f.add('Ta5_7b2_center',    value=26.8, min=25.5, max=28,   vary=True)
    p0_ta4f.add('Ta3_7b2_center',    value=24.2, min=23.5, max=24.3, vary=True)
    p0_ta4f.add('Ta1_7b2_center',    value=22.7, min=22.0, max=22.9, vary=True)
    p0_ta4f.add('metal_7b2_center',  value=22,   min=21.0, max=23,   vary=True)
    p0_ta4f.add('interface_offset',  value=0.44, min=0.2,  max=0.7)
    p0_ta4f.add('alloy_offset',      value=2.0,  min=1.5,  max=2.5,  vary=True)
    p0_ta4f.add('_7b2_5b2_offset',   value=1.91, min=1.0,  max=3.0,  vary=False)
    p0_ta4f.add('Ta5_7b2_sigma',     value=0.16, min=0.0,  max=0.75, vary=True)
    p0_ta4f.add('metal_7b2_sigma',   value=0.08, min=0.0,  max=0.2,  vary=True)
    # Metallic Ta requires asymmetric line shape; allow skew to float (D-S tail toward oxide region)
    p0_ta4f.add('metal_7b2_skew',    value=0.0,  min=0.0,  max=0.5,  vary=True)
    p0_ta4f.add('Ta5_7b2_amplitude',       value=0.16,  min=0.0, max=6.0)
    p0_ta4f.add('Ta3_7b2_amplitude',       value=0.015, min=0.0, max=0.3)
    p0_ta4f.add('Ta1_7b2_amplitude',       value=0.008, min=0.0, max=0.3)
    p0_ta4f.add('metal_7b2_amplitude',     value=0.3,   min=0.05, max=6.0)
    p0_ta4f.add('interface_7b2_amplitude', value=0.03,  min=0.01, max=0.06)
    p0_ta4f.add('alloy_7b2_amplitude',     value=0.10,  min=0.00, max=6.0)
    p0_ta4f.add('_7b2_5b2_ratio',    value=1.333, min=1.2,  max=1.5)
    p0_ta4f.add('interface_7b2_center', vary=False, expr='interface_offset+metal_7b2_center')
    p0_ta4f.add('alloy_7b2_center',     vary=False, expr='alloy_offset+metal_7b2_center')
    p0_ta4f.add('Ta3_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p0_ta4f.add('Ta1_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p0_ta4f.add('interface_7b2_sigma', vary=False, expr='metal_7b2_sigma')
    p0_ta4f.add('alloy_7b2_sigma',     vary=False, expr='metal_7b2_sigma')
    p0_ta4f.add('interface_7b2_skew',  vary=False, expr='metal_7b2_skew')
    p0_ta4f.add('alloy_7b2_skew',      vary=False, expr='metal_7b2_skew')
    for species in ['Ta5', 'Ta3', 'Ta1', 'metal', 'interface', 'alloy']:
        p0_ta4f.add(f'{species}_5b2_center',    vary=False, expr=f'{species}_7b2_center+_7b2_5b2_offset')
        p0_ta4f.add(f'{species}_5b2_sigma',     vary=False, expr=f'{species}_7b2_sigma')
        p0_ta4f.add(f'{species}_5b2_amplitude', vary=False, expr=f'{species}_7b2_amplitude/_7b2_5b2_ratio')
    p0_ta4f.add('metal_5b2_skew',     vary=False, expr='metal_7b2_skew')
    p0_ta4f.add('interface_5b2_skew', vary=False, expr='interface_7b2_skew')
    p0_ta4f.add('alloy_5b2_skew',     vary=False, expr='alloy_7b2_skew')

    p0_re4f = lmfit.Parameters()
    p0_re4f.add('Re_metal_7b2_center', value=40.4, min=39.8, max=41.0)
    p0_re4f.add('ReO2_7b2_center',     value=42.3, min=41.8, max=42.8)
    p0_re4f.add('ReO3_7b2_center',     value=43.2, min=42.6, max=44.0)
    p0_re4f.add('Re2O7_7b2_center',    value=45.5, min=45.0, max=46.7)
    p0_re4f.add('Re_7b2_5b2_offset',   value=2.42, min=2.35, max=2.50)
    p0_re4f.add('Re_7b2_5b2_ratio',    value=1.33, min=1.20, max=1.50)
    p0_re4f.add('Re_metal_7b2_sigma',  value=0.10, min=0.00, max=0.30)
    # Metallic Re requires asymmetric line shape; allow skew to float
    p0_re4f.add('Re_metal_7b2_skew',   value=0.00, min=0.00, max=0.50, vary=True)
    p0_re4f.add('ReO2_7b2_sigma',      value=0.18, min=0.05, max=0.60)
    p0_re4f.add('ReO3_7b2_sigma',      value=0.22, min=0.05, max=0.75)
    p0_re4f.add('Re2O7_7b2_sigma',     value=0.35, min=0.10, max=0.75)
    p0_re4f.add('Re_metal_7b2_amplitude', value=0.10,  min=0.00, max=6.00)
    # 'ReO' (Re²⁺) surface oxide — unique to Re metal surfaces, no bulk analog.
    # Center pinned near 41.45 eV per Greiner et al. 2014 Table 1; can shift ±0.05 eV.
    # Previously this signal was absorbed into the ReO2 component, pulling it ~0.4 eV low.
    p0_re4f.add('ReO_7b2_center',    value=41.45, min=41.20, max=41.70)
    p0_re4f.add('ReO_7b2_sigma',     value=0.18,  min=0.05,  max=0.50)
    p0_re4f.add('ReO_7b2_amplitude', value=0.005, min=0.00,  max=1.00)
    p0_re4f.add('ReO_7b2_skew',      value=0.00,  min=0.00,  max=0.30, vary=True)
    p0_re4f.add('ReO2_7b2_amplitude',     value=0.020, min=0.00, max=2.00)
    p0_re4f.add('ReO3_7b2_amplitude',     value=0.015, min=0.00, max=2.00)
    p0_re4f.add('Re2O7_7b2_amplitude',    value=0.010, min=0.00, max=2.00)
    # Reδ+ surface coordination shoulders: centers tied to Re⁰ + fixed offsets,
    # sigma and skew tied to Re metal, amplitude free (small).
    for sp, offset in re4f_redp_offsets.items():
        p0_re4f.add(f'{sp}_offset',        value=offset, vary=False)
        p0_re4f.add(f'{sp}_7b2_amplitude', value=0.005, min=0.0, max=0.5)
        p0_re4f.add(f'{sp}_7b2_center',    vary=False, expr=f'Re_metal_7b2_center+{sp}_offset')
        p0_re4f.add(f'{sp}_7b2_sigma',     vary=False, expr='Re_metal_7b2_sigma')
        p0_re4f.add(f'{sp}_7b2_skew',      vary=False, expr='Re_metal_7b2_skew')
        p0_re4f.add(f'{sp}_5b2_center',    vary=False, expr=f'{sp}_7b2_center+Re_7b2_5b2_offset')
        p0_re4f.add(f'{sp}_5b2_sigma',     vary=False, expr=f'{sp}_7b2_sigma')
        p0_re4f.add(f'{sp}_5b2_amplitude', vary=False, expr=f'{sp}_7b2_amplitude/Re_7b2_5b2_ratio')
        p0_re4f.add(f'{sp}_5b2_skew',      vary=False, expr=f'{sp}_7b2_skew')

    for species in ['Re_metal', 'ReO', 'ReO2', 'ReO3', 'Re2O7']:
        p0_re4f.add(f'{species}_5b2_center',    expr=f'{species}_7b2_center+Re_7b2_5b2_offset')
        p0_re4f.add(f'{species}_5b2_sigma',     expr=f'{species}_7b2_sigma')
        p0_re4f.add(f'{species}_5b2_amplitude', expr=f'{species}_7b2_amplitude/Re_7b2_5b2_ratio')
    p0_re4f.add('Re_metal_5b2_skew', expr='Re_metal_7b2_skew')
    p0_re4f.add('ReO_5b2_skew',      expr='ReO_7b2_skew')

    # ── Section 7: Fitting ────────────────────────────────────────────────────
    ta4f_fitResults = []
    for i, data in enumerate(Ta4fCorrected):
        res = safe_minimize(ta4f_objective, p0_ta4f, args=(data, ta4f_modelDict),
                            method='bfgs', nan_policy='omit')
        ta4f_fitResults.append(res)
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        title = f"Ta4f - {data.sample}{etch}"
        print(f"[{i}] {data.sample}{etch}  →  red. chi2 = {res.redchi:.6f}")
        plot_fits.plot_ta4f_fit_components(data, res, ta4f_modelDict,
                                           out_dir=figs_dir / "fit_components", title=title)

    # ── Section 7b: Ta5+ constraint diagnostic ───────────────────────────────
    boe_wk0_idx = next(
        (i for i, d in enumerate(Ta4fCorrected)
         if 'BOE' in d.sample and 'WK0' in d.sample),
        None,
    )
    if boe_wk0_idx is None:
        print("[WR] BOE WK0 not found in Ta4fCorrected — skipping Ta5+ constraint check.")
    else:
        check_ta5_center_constraint(
            Ta4fCorrected,
            boe_wk0_idx,
            ta4f_modelDict,
            p0_ta4f,
            ta4f_fitResults[boe_wk0_idx],
            figs_dir / "fit_components",
            csv_dir,
        )

    re4f_fitResults = []
    for i, data in enumerate(Re4fCorrected):
        res = safe_minimize(re4f_objective, p0_re4f, args=(data, re4f_modelDict),
                            method='bfgs', nan_policy='omit')
        re4f_fitResults.append(res)
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        title = f"Re4f - {data.sample}{etch}"
        print(f"[{i}] {data.sample}{etch}  →  red. chi2 = {res.redchi:.6f}")
        plot_fits.plot_re4f_fit_components(data, res, re4f_modelDict,
                                           out_dir=figs_dir / "fit_components", title=title)

    # ── Section 8: Area fractions ─────────────────────────────────────────────
    ta4f_speciesOrder = ['metal', 'Ta5', 'Ta3', 'Ta1', 'interface', 'alloy']
    re4f_speciesOrder = ['Re_metal', 'Redp1', 'Redp2', 'Redp3', 'ReO2', 'ReO3', 'Re2O7']

    ta4f_areaFrac, ta4f_areaErr = [], []
    for res in ta4f_fitResults:
        fr, er = areaFractions(res, ta4f_speciesOrder)
        ta4f_areaFrac.append(fr)
        ta4f_areaErr.append(er)

    re4f_areaFrac, re4f_areaErr = [], []
    for res in re4f_fitResults:
        fr, er = areaFractions(res, re4f_speciesOrder)
        re4f_areaFrac.append(fr)
        re4f_areaErr.append(er)

    # ── Section 9: CSV export ─────────────────────────────────────────────────
    rows_ta = []
    for i, data in enumerate(Ta4fCorrected):
        row = {"index": i, "sample": data.sample, "etchlevel": data.etchlevel,
               "etchtime": data.etchtime, "redchi": ta4f_fitResults[i].redchi}
        for sp in ta4f_speciesOrder:
            row[f"{sp}_frac"] = ta4f_areaFrac[i][sp]
            row[f"{sp}_err"]  = ta4f_areaErr[i][sp]
        rows_ta.append(row)
    df_ta4f = pd.DataFrame(rows_ta).set_index("index")
    df_ta4f.to_csv(csv_dir / "ta4f_area_fractions.csv")

    rows_ta_normalized = []
    for data, fractions, errors in zip(Ta4fCorrected, ta4f_areaFrac, ta4f_areaErr):
        for sp in ta4f_speciesOrder:
            rows_ta_normalized.append({
                "sample": data.sample,
                "species": sp,
                "normalized_amplitude": fractions[sp],
                "stderr_normalized": errors[sp],
            })
    pd.DataFrame(rows_ta_normalized).to_csv(csv_dir / "ta4f_normalized_amplitudes.csv", index=False)

    rows_re = []
    for i, data in enumerate(Re4fCorrected):
        row = {"index": i, "sample": data.sample, "etchlevel": data.etchlevel,
               "etchtime": data.etchtime, "redchi": re4f_fitResults[i].redchi}
        for sp in re4f_speciesOrder:
            row[f"{sp}_frac"] = re4f_areaFrac[i][sp]
            row[f"{sp}_err"]  = re4f_areaErr[i][sp]
        rows_re.append(row)
    df_re4f = pd.DataFrame(rows_re).set_index("index")
    df_re4f.to_csv(csv_dir / "re4f_area_fractions.csv")

    rows_re_normalized = []
    for data, fractions, errors in zip(Re4fCorrected, re4f_areaFrac, re4f_areaErr):
        for sp in re4f_speciesOrder:
            rows_re_normalized.append({
                "sample": data.sample,
                "species": sp,
                "normalized_amplitude": fractions[sp],
                "stderr_normalized": errors[sp],
            })
    pd.DataFrame(rows_re_normalized).to_csv(csv_dir / "re4f_normalized_amplitudes.csv", index=False)

    # ── Section 10: Stacked fraction plots ───────────────────────────────────
    plot_summary.stacked_fraction_plot_ta4f(df_ta4f.reset_index(), ta4f_speciesOrder,
                                            out_dir=figs_dir / "area_fractions")
    plot_summary.stacked_fraction_plot_re4f(df_re4f.reset_index(), re4f_speciesOrder,
                                            out_dir=figs_dir / "area_fractions")

    # ── Section 11: Oxide thickness ───────────────────────────────────────────
    ta_t, ta_te = [], []
    for data, result in zip(Ta4fCorrected, ta4f_fitResults):
        I_ox = (result.params['Ta5_7b2_amplitude'].value +
                result.params['Ta3_7b2_amplitude'].value +
                result.params['Ta1_7b2_amplitude'].value)
        I_m   = result.params['metal_7b2_amplitude'].value
        I_ox_err = np.sqrt(sum((result.params[f'{s}_7b2_amplitude'].stderr or 0) ** 2
                               for s in ['Ta5', 'Ta3', 'Ta1']))
        I_m_err  = result.params['metal_7b2_amplitude'].stderr or 0
        t, te = calculate_oxide_thickness(I_ox, I_m, 2.8, 2.5, 5.55e22, 8.17e22,
                                          I_ox_err=I_ox_err, I_metal_err=I_m_err)
        ta_t.append(t)
        ta_te.append(te)

    re_t, re_te = [], []
    for data, result in zip(Re4fCorrected, re4f_fitResults):
        I_ox = (result.params['ReO2_7b2_amplitude'].value +
                result.params['ReO3_7b2_amplitude'].value +
                result.params['Re2O7_7b2_amplitude'].value)
        I_m   = result.params['Re_metal_7b2_amplitude'].value
        I_ox_err = np.sqrt(sum((result.params[f'{s}_7b2_amplitude'].stderr or 0) ** 2
                               for s in ['ReO2', 'ReO3', 'Re2O7']))
        I_m_err  = result.params['Re_metal_7b2_amplitude'].stderr or 0
        t, te = calculate_oxide_thickness(I_ox, I_m, 2.9, 2.6, 6.81e22, 5.45e22,
                                          I_ox_err=I_ox_err, I_metal_err=I_m_err)
        re_t.append(t)
        re_te.append(te)

    # CSV: oxide thickness
    ta_thick_rows = [{"index": i, "sample": d.sample, "etchlevel": d.etchlevel,
                      "etchtime": d.etchtime, "oxide_thickness_nm": ta_t[i],
                      "thickness_err_nm": ta_te[i]}
                     for i, d in enumerate(Ta4fCorrected)]
    ta_thickness_df = pd.DataFrame(ta_thick_rows)
    ta_thickness_df.set_index("index").to_csv(csv_dir / "ta4f_oxide_thickness.csv")

    re_thick_rows = [{"index": i, "sample": d.sample, "etchlevel": d.etchlevel,
                      "etchtime": d.etchtime, "oxide_thickness_nm": re_t[i],
                      "thickness_err_nm": re_te[i]}
                     for i, d in enumerate(Re4fCorrected)]
    re_thickness_df = pd.DataFrame(re_thick_rows)
    re_thickness_df.set_index("index").to_csv(csv_dir / "re4f_oxide_thickness.csv")

    ta_timecourse = build_timecourse_summary(ta_thickness_df, value_col="oxide_thickness_nm")
    re_timecourse = build_timecourse_summary(re_thickness_df, value_col="oxide_thickness_nm")
    ta_baseline = build_baseline_change(ta_thickness_df, value_col="oxide_thickness_nm")
    re_baseline = build_baseline_change(re_thickness_df, value_col="oxide_thickness_nm")
    ta_group_diff = build_group_week_differences(ta_thickness_df, value_col="oxide_thickness_nm")
    re_group_diff = build_group_week_differences(re_thickness_df, value_col="oxide_thickness_nm")

    ta_timecourse.to_csv(csv_dir / "ta4f_thickness_timecourse.csv", index=False)
    re_timecourse.to_csv(csv_dir / "re4f_thickness_timecourse.csv", index=False)
    ta_baseline.to_csv(csv_dir / "ta4f_thickness_baseline_change.csv", index=False)
    re_baseline.to_csv(csv_dir / "re4f_thickness_baseline_change.csv", index=False)
    ta_group_diff.to_csv(csv_dir / "ta4f_thickness_group_differences.csv", index=False)
    re_group_diff.to_csv(csv_dir / "re4f_thickness_group_differences.csv", index=False)

    plot_summary.plot_timecourse_with_group_difference(
        ta_timecourse,
        ta_group_diff,
        title="Ta Oxide Thickness by Week",
        ylabel="Oxide Thickness (nm)",
        filename="ta4f_thickness_timecourse.png",
        out_dir=figs_dir / "timecourse",
    )

    plot_summary.plot_timecourse_with_group_difference(
        re_timecourse,
        re_group_diff,
        title="Re Oxide Thickness by Week",
        ylabel="Oxide Thickness (nm)",
        filename="re4f_thickness_timecourse.png",
        out_dir=figs_dir / "timecourse",
    )

    plot_summary.plot_oxide_thickness(Ta4fCorrected, ta_t, ta_te,
                                      Re4fCorrected, re_t, re_te,
                                      out_dir=figs_dir / "thickness")

    # ── Section 11b: Cabrera-Mott kinetic fit ────────────────────────────────
    # Fit x(t) = x0 + k * ln(1 + t/tau) to BOE weekly oxide thickness.
    # t=0 is immediately post-etch (WK0); time unit is weeks.
    cm_rows = []
    for element, timecourse_df in [('Ta', ta_timecourse), ('Re', re_timecourse)]:
        boe_rows = timecourse_df[timecourse_df['group'] == 'BOE'].dropna(subset=['week', 'mean'])
        if len(boe_rows) < 3:
            print(f"[CM fit] Insufficient BOE time points for {element} (n={len(boe_rows)}), skipping.")
            continue
        cm = cabrera_mott_fit(
            boe_rows['week'].values,
            boe_rows['mean'].values,
            boe_rows['sem'].values,
        )
        if cm is None:
            print(f"[CM fit] Fit failed for {element}.")
            continue
        print(f"[CM fit] {element}: x0={cm['x0']:.3f}±{cm['x0_err']:.3f} nm, "
              f"k={cm['k']:.3f}±{cm['k_err']:.3f} nm, "
              f"tau={cm['tau']:.2f}±{cm['tau_err']:.2f} weeks, "
              f"red.chi2={cm['redchi']:.4f}")
        plot_summary.plot_cabrera_mott_fit(
            timecourse_df,
            cm,
            element=element,
            filename=f"{element.lower()}_cabrera_mott_fit.png",
            out_dir=figs_dir / "timecourse",
        )
        cm_rows.append({'element': element, **{key: cm[key] for key in
                         ('x0', 'x0_err', 'k', 'k_err', 'tau', 'tau_err', 'redchi')}})
    if cm_rows:
        pd.DataFrame(cm_rows).to_csv(csv_dir / "cabrera_mott_fit.csv", index=False)

    # ── Section 12: Peak position vs literature ───────────────────────────────
    ta4f_peak_comparisons, re4f_peak_comparisons = [], []
    for i, (data, result) in enumerate(zip(Ta4fCorrected, ta4f_fitResults)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        comp = compare_peaks(result, ta4f_expected_peaks, ta4f_species_map,
                             sample_name=f"[{i}] {data.sample}{etch}", element="TA4F")
        ta4f_peak_comparisons.append({'index': i, 'sample': data.sample,
                                      'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                      'comparisons': comp})
    for i, (data, result) in enumerate(zip(Re4fCorrected, re4f_fitResults)):
        etch = f" | Etch Lv{data.etchlevel}" if data.etchlevel is not None else ""
        comp = compare_peaks(result, re4f_expected_peaks, re4f_species_map,
                             sample_name=f"[{i}] {data.sample}{etch}", element="RE4F")
        re4f_peak_comparisons.append({'index': i, 'sample': data.sample,
                                      'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                      'comparisons': comp})

    pd.DataFrame(_peak_rows(ta4f_peak_comparisons, 'Ta4f')).to_csv(
        csv_dir / "ta4f_peak_comparison.csv", index=False)
    pd.DataFrame(_peak_rows(re4f_peak_comparisons, 'Re4f')).to_csv(
        csv_dir / "re4f_peak_comparison.csv", index=False)
    plot_summary.plot_peak_deviation(ta4f_peak_comparisons, 'Ta4f', 'Ta4f',
                                     out_dir=figs_dir / "peak_comparisons")
    plot_summary.plot_peak_deviation(re4f_peak_comparisons, 'Re4f', 'Re4f',
                                     out_dir=figs_dir / "peak_comparisons")

    # ── Section 13: Amplitude & width comparisons ─────────────────────────────
    ta4f_amp_rows, re4f_amp_rows = [], []
    ta4f_width_rows, re4f_width_rows = [], []

    for i, (data, result) in enumerate(zip(Ta4fCorrected, ta4f_fitResults)):
        st = 'BOE samples' if 'BOE' in data.sample else 'Control samples'
        for comp in compare_ta4f_amplitudes(result, data.sample, st):
            ta4f_amp_rows.append({'index': i, 'sample': data.sample,
                                  'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                  'sample_type': st, **comp})
        for comp in compare_ta4f_widths(result, ta4f_expected_widths, data.sample):
            ta4f_width_rows.append({'index': i, 'sample': data.sample,
                                    'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                    'species': comp['species'],
                                    'expected_sigma_eV': comp['expected_sigma'],
                                    'fitted_sigma_eV': comp['fitted_sigma'],
                                    'fitted_stderr': comp['stderr'], 'delta_eV': comp['delta']})

    for i, (data, result) in enumerate(zip(Re4fCorrected, re4f_fitResults)):
        st = 'BOE samples' if 'BOE' in data.sample else 'Control samples'
        for comp in compare_re4f_amplitudes(result, data.sample, st):
            re4f_amp_rows.append({'index': i, 'sample': data.sample,
                                  'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                  'sample_type': st, **comp})
        for comp in compare_re4f_widths(result, re4f_expected_widths, data.sample):
            re4f_width_rows.append({'index': i, 'sample': data.sample,
                                    'etchlevel': data.etchlevel, 'etchtime': data.etchtime,
                                    'species': comp['species'],
                                    'expected_sigma_eV': comp['expected_sigma'],
                                    'fitted_sigma_eV': comp['fitted_sigma'],
                                    'fitted_stderr': comp['stderr'], 'delta_eV': comp['delta']})

    pd.DataFrame(ta4f_amp_rows).to_csv(csv_dir / "ta4f_amplitude_comparison.csv", index=False)
    pd.DataFrame(re4f_amp_rows).to_csv(csv_dir / "re4f_amplitude_comparison.csv", index=False)
    pd.DataFrame(ta4f_width_rows).to_csv(csv_dir / "ta4f_width_comparison.csv", index=False)
    pd.DataFrame(re4f_width_rows).to_csv(csv_dir / "re4f_width_comparison.csv", index=False)

    # ── Section 14: Peak properties vs sample ────────────────────────────────
    ta4f_colors = {'metal': '#1f77b4', 'interface': '#ff7f0e', 'alloy': '#2ca02c',
                   'Ta5': '#d62728', 'Ta3': '#9467bd', 'Ta1': '#8c564b'}
    re4f_colors = {'Re_metal': '#1f77b4', 'ReO2': '#ff7f0e',
                   'ReO3': '#2ca02c', 'Re2O7': '#d62728'}

    if ta4f_fitResults:
        plot_summary.plot_metric_evolution_by_species(
            ta4f_fitResults,
            Ta4fCorrected,
            ta4f_speciesOrder,
            'Ta4f',
            ta4f_colors,
            metric='binding_energy',
            filename='ta4f_binding_energy_evolution_by_species.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_metric_evolution_by_species(
            ta4f_fitResults,
            Ta4fCorrected,
            ta4f_speciesOrder,
            'Ta4f',
            ta4f_colors,
            metric='amplitude',
            filename='ta4f_amplitude_evolution_by_species.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_normalized_amplitude_by_species(
            ta4f_areaFrac,
            ta4f_areaErr,
            Ta4fCorrected,
            ta4f_speciesOrder,
            'Ta4f',
            ta4f_colors,
            filename='ta4f_amplitude_normalized.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )
    if re4f_fitResults:
        plot_summary.plot_metric_evolution_by_species(
            re4f_fitResults,
            Re4fCorrected,
            re4f_speciesOrder,
            'Re4f',
            re4f_colors,
            metric='binding_energy',
            filename='re4f_binding_energy_evolution_by_species.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_metric_evolution_by_species(
            re4f_fitResults,
            Re4fCorrected,
            re4f_speciesOrder,
            'Re4f',
            re4f_colors,
            metric='amplitude',
            filename='re4f_amplitude_evolution_by_species.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )
        plot_summary.plot_normalized_amplitude_by_species(
            re4f_areaFrac,
            re4f_areaErr,
            Re4fCorrected,
            re4f_speciesOrder,
            'Re4f',
            re4f_colors,
            filename='re4f_amplitude_normalized.png',
            out_dir=figs_dir / "peaks_vs_samples",
        )

    # ── Section 15a: BE shift summary ────────────────────────────────────────
    shift_rows = (
        compute_be_shifts(Ta4fCorrected, ta4f_fitResults, ta4f_literature, ta4f_species_map, 'Ta4f') +
        compute_be_shifts(Re4fCorrected, re4f_fitResults, re4f_literature, re4f_species_map, 'Re4f')
    )
    df_shifts = pd.DataFrame(shift_rows)
    df_shifts.to_csv(csv_dir / "be_shift_summary.csv", index=False)
    df_shift_timecourse = build_be_shift_timecourse(df_shifts)
    df_shift_timecourse.to_csv(csv_dir / "be_shift_timecourse.csv", index=False)

    # ── Section 15b: BOE vs Control shift comparison ──────────────────────────
    df_group = build_be_shift_by_group(df_shifts)
    df_group.to_csv(csv_dir / "be_shift_by_group.csv", index=False)
    plot_summary.plot_be_shift_boe_vs_control(df_group, out_dir=figs_dir / "be_shifts")

    # ── Section 15c: Ta↔Re cross-correlation ─────────────────────────────────
    plot_summary.plot_ta_re_correlation(df_shifts, out_dir=figs_dir / "be_shifts")

    print(f"\nDone. Results in {figs_dir.parent}")


if __name__ == "__main__":
    main()
