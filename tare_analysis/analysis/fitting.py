# tare_analysis/analysis/fitting.py
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import LinAlgWarning
from xps.funcs import backgroundCorrect
from analysis.models import (
    ta4f_expected_amplitudes, re4f_expected_amplitudes,
    ta4f_expected_widths, re4f_expected_widths,
)

try:
    from scipy.optimize import curve_fit as _curve_fit
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class CorrectedData:
    BE: np.ndarray
    intensity: np.ndarray
    intensityErr: np.ndarray
    sample: str
    etchlevel: int = None
    etchtime: float = None


def safe_minimize(fcn, params, **kwargs):
    """Run lmfit minimization and drop covariance if Hessian inversion is unstable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LinAlgWarning)
        result = lmfit.minimize(fcn, params, **kwargs)

    if any(isinstance(w.message, LinAlgWarning) for w in caught):
        result.covar = None
        for par in result.params.values():
            par.stderr = None
    return result


def correct_data(scans, inds, method_overrides=None, shirley_cutoff=0.01,
                 be_min=None, be_max=30.5):
    """Return list[CorrectedData] in same order as inds."""
    out = []
    method_overrides = method_overrides or {}
    for k, idx in enumerate(inds):
        s = scans[idx]
        method = method_overrides.get(k, "shirley").lower()
        kwargs = dict(BEWindow=0.2)
        if method == "flat":
            kwargs.update(func="flat")
        else:
            kwargs.update(shirleyCutoff=shirley_cutoff)
        be_min_actual = np.min(s.BE) if be_min is None else be_min
        c = backgroundCorrect(s, be_min_actual, be_max, **kwargs)
        area = np.abs(np.trapz(c.intensity, c.BE))
        area = area if area > 0 else 1.0
        out.append(CorrectedData(
            BE=c.BE.copy(),
            intensity=(c.intensity / area),
            intensityErr=(c.intensityErr / area),
            sample=s.sample,
            etchlevel=getattr(s, 'etchlevel', None),
            etchtime=getattr(s, 'etchtime', None),
        ))
    return out


def areaFractions(result, species_order):
    """Return (fracs, errs) dicts for given species, using 7b2 amplitudes."""
    amps = {sp: result.params[f'{sp}_7b2_amplitude'].value for sp in species_order}
    total = sum(amps.values()) or 1.0
    fracs = {sp: amps[sp] / total for sp in species_order}
    cov = result.covar
    idx = {name: i for i, name in enumerate(result.var_names)}
    errs = {}
    for sp in species_order:
        name = f'{sp}_7b2_amplitude'
        if cov is None or name not in idx:
            a = result.params[name]
            sa = (a.stderr or 0.0)
            errs[sp] = abs((1 / total - amps[sp] / (total * total)) * sa)
            continue
        d = np.zeros(len(result.var_names))
        for other in species_order:
            oname = f'{other}_7b2_amplitude'
            if oname not in idx:
                continue
            j = idx[oname]
            d[j] = (total - amps[sp]) / (total ** 2) if other == sp else -amps[sp] / (total ** 2)
        var = float(d @ cov @ d)
        errs[sp] = np.sqrt(var) if var > 0 else 0.0
    return fracs, errs


def calculate_oxide_thickness(I_ox, I_metal, lambda_ox, lambda_metal, N_metal, N_ox,
                               theta=90.0, I_ox_err=0, I_metal_err=0):
    """Strohmeier oxide thickness. Returns (thickness_nm, thickness_err_nm)."""
    if I_metal <= 0:
        return np.nan, np.nan
    sin_theta = np.sin(np.radians(theta))
    ratio = I_ox / I_metal
    lambda_ratio = lambda_metal / lambda_ox
    N_ratio = N_metal / N_ox
    argument = 1 + ratio * lambda_ratio * N_ratio
    if argument <= 0:
        return np.nan, np.nan
    thickness = lambda_ox * sin_theta * np.log(argument)
    if I_ox_err > 0 or I_metal_err > 0:
        prefactor = lambda_ox * sin_theta * lambda_ratio * N_ratio / argument
        thickness_err = np.sqrt((prefactor / I_metal * I_ox_err) ** 2 +
                                (-prefactor * I_ox / I_metal ** 2 * I_metal_err) ** 2)
    else:
        thickness_err = 0.0
    return thickness, thickness_err


def compare_peaks(result, expected_peaks, species_map, sample_name="", element=""):
    """Compare fitted vs expected peak centers. Returns list of dicts."""
    print(f"\n{'='*70}")
    print(f"{element} PEAK LOCATION COMPARISON: {sample_name}")
    print(f"{'='*70}")
    comparison_data = []
    for label, prefix in species_map.items():
        if label not in expected_peaks:
            continue
        for spin in ['7/2', '5/2']:
            spin_str = spin.replace('/', 'b')
            param_key = f'{prefix}_{spin_str}_center'
            if param_key not in result.params:
                continue
            expected = expected_peaks[label][spin]
            fitted = result.params[param_key].value
            stderr = result.params[param_key].stderr or 0.0
            delta = fitted - expected
            print(f"  {label:<20} {spin:<6} exp={expected:.2f}  fit={fitted:.2f}+/-{stderr:.3f}  delta={delta:+.3f}")
            comparison_data.append({'species': label, 'spin': spin,
                                    'expected': expected, 'fitted': fitted,
                                    'stderr': stderr, 'delta': delta})
    return comparison_data


def _peak_rows(comparisons_list, element):
    rows = []
    for cd in comparisons_list:
        for c in cd['comparisons']:
            rows.append({'index': cd['index'], 'sample': cd['sample'],
                         'etchlevel': cd['etchlevel'], 'etchtime': cd['etchtime'],
                         'element': element, 'species': c['species'], 'spin': c['spin'],
                         'expected_eV': c['expected'], 'fitted_eV': c['fitted'],
                         'fitted_stderr': c['stderr'], 'delta_eV': c['delta']})
    return rows


def compare_re4f_amplitudes(result, sample_name="", sample_type="BOE samples"):
    if sample_type not in re4f_expected_amplitudes:
        print(f"Unknown sample type: {sample_type}")
        return []
    expected_amps = re4f_expected_amplitudes[sample_type]
    species_mapping = {'Re metal': 'Re_metal', 'ReO2 (Re4+)': 'ReO2',
                       'ReO3 (Re6+)': 'ReO3', 'Re2O7 (Re7+)': 'Re2O7'}
    total_amp = sum(
        result.params[f'{p}_7b2_amplitude'].value
        for p in species_mapping.values()
        if f'{p}_7b2_amplitude' in result.params
    )
    comparison_data = []
    for species_label, param_name in species_mapping.items():
        if species_label in expected_amps:
            param_key = f'{param_name}_7b2_amplitude'
            if param_key in result.params:
                expected = expected_amps[species_label]
                fitted_raw = result.params[param_key].value
                fitted_norm = fitted_raw / total_amp if total_amp > 0 else 0
                delta = fitted_norm - expected
                stderr = result.params[param_key].stderr or 0.0
                comparison_data.append({'species': species_label, 'expected_amp': expected,
                                        'fitted_amp': fitted_norm, 'stderr': stderr / total_amp,
                                        'delta': delta})
    return comparison_data


def compare_ta4f_amplitudes(result, sample_name="", sample_type="BOE samples"):
    if sample_type not in ta4f_expected_amplitudes:
        print(f"Unknown sample type: {sample_type}")
        return []
    expected_amps = ta4f_expected_amplitudes[sample_type]
    species_mapping = {'Ta metal': 'metal', 'Ta interface': 'interface', 'Ta alloy': 'alloy',
                       'Ta+1': 'Ta1', 'Ta+3': 'Ta3', 'Ta+5 (Ta2O5)': 'Ta5'}
    total_amp = sum(
        result.params[f'{p}_7b2_amplitude'].value
        for p in species_mapping.values()
        if f'{p}_7b2_amplitude' in result.params
    )
    comparison_data = []
    for species_label, param_name in species_mapping.items():
        if species_label in expected_amps:
            param_key = f'{param_name}_7b2_amplitude'
            if param_key in result.params:
                expected = expected_amps[species_label]
                fitted_raw = result.params[param_key].value
                fitted_norm = fitted_raw / total_amp if total_amp > 0 else 0
                delta = fitted_norm - expected
                stderr = result.params[param_key].stderr or 0.0
                comparison_data.append({'species': species_label, 'expected_amp': expected,
                                        'fitted_amp': fitted_norm, 'stderr': stderr / total_amp,
                                        'delta': delta})
    return comparison_data


def cabrera_mott_fit(times, thicknesses, thickness_errs=None):
    """Fit Cabrera-Mott logarithmic growth: x(t) = x0 + k * ln(1 + t / tau).

    Args:
        times: array-like of time values (weeks; t=0 = immediately post-etch)
        thicknesses: array-like of oxide thickness in nm
        thickness_errs: optional array-like of 1-sigma thickness uncertainties (nm)

    Returns:
        dict with keys x0, k, tau (fitted parameters), x0_err, k_err, tau_err
        (1-sigma uncertainties), t_fit, x_fit (dense curve for plotting), and
        redchi (reduced chi-squared of the fit), or None if fit failed.
    """
    if not _SCIPY_AVAILABLE:
        return None

    times = np.asarray(times, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    mask = np.isfinite(times) & np.isfinite(thicknesses)
    t = times[mask]
    x = thicknesses[mask]
    if len(t) < 3:
        return None

    sigma = None
    if thickness_errs is not None:
        errs = np.asarray(thickness_errs, dtype=float)[mask]
        pos = errs[errs > 0]
        fallback = float(np.nanmean(pos)) if len(pos) > 0 else 1.0
        sigma = np.where(errs > 0, errs, fallback)

    def _model(t, x0, k, tau):
        return x0 + k * np.log(1.0 + t / tau)

    p0 = [max(x.min(), 0.0), max(x.max() - x.min(), 0.1), 1.0]
    bounds = ([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf])
    try:
        popt, pcov = _curve_fit(
            _model, t, x, p0=p0, bounds=bounds,
            sigma=sigma, absolute_sigma=(sigma is not None),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None

    perr = np.sqrt(np.diag(pcov))
    residuals = x - _model(t, *popt)
    dof = len(t) - 3
    redchi = float(np.sum(residuals ** 2 / (sigma ** 2 if sigma is not None else 1.0)) / dof) if dof > 0 else np.nan

    t_fit = np.linspace(0.0, max(t) * 1.1, 200)
    x_fit = _model(t_fit, *popt)

    return {
        'x0': popt[0], 'k': popt[1], 'tau': popt[2],
        'x0_err': perr[0], 'k_err': perr[1], 'tau_err': perr[2],
        't_fit': t_fit, 'x_fit': x_fit,
        'redchi': redchi,
    }


def compare_re4f_widths(result, expected_widths, sample_name=""):
    species_mapping = {'Re metal': 'Re_metal', 'ReO2 (Re4+)': 'ReO2',
                       'ReO3 (Re6+)': 'ReO3', 'Re2O7 (Re7+)': 'Re2O7'}
    comparison_data = []
    for species_label, param_name in species_mapping.items():
        if species_label in expected_widths:
            param_key = f'{param_name}_7b2_sigma'
            if param_key in result.params:
                expected = expected_widths[species_label]['sigma']
                fitted = result.params[param_key].value
                delta = fitted - expected
                stderr = result.params[param_key].stderr or 0.0
                comparison_data.append({'species': species_label, 'expected_sigma': expected,
                                        'fitted_sigma': fitted, 'stderr': stderr, 'delta': delta})
    return comparison_data


def compare_ta4f_widths(result, expected_widths, sample_name=""):
    species_mapping = {'Ta metal': 'metal', 'Ta interface': 'interface', 'Ta alloy': 'alloy',
                       'Ta+1': 'Ta1', 'Ta+3': 'Ta3', 'Ta+5 (Ta2O5)': 'Ta5'}
    comparison_data = []
    for species_label, param_name in species_mapping.items():
        if species_label in expected_widths:
            param_key = f'{param_name}_7b2_sigma'
            if param_key in result.params:
                expected = expected_widths[species_label]['sigma']
                fitted = result.params[param_key].value
                delta = fitted - expected
                stderr = result.params[param_key].stderr or 0.0
                comparison_data.append({'species': species_label, 'expected_sigma': expected,
                                        'fitted_sigma': fitted, 'stderr': stderr, 'delta': delta})
    return comparison_data


def check_ta5_center_constraint(
    corrected_data_list: list,
    boe_wk0_idx: int,
    ta4f_modelDict: dict,
    p0_ta4f,
    constrained_result,
    out_dir_figs,
    out_dir_csv,
) -> dict:
    """Refit BOE week0 Ta4f with Ta5_7b2_center unconstrained [24.0, 29.0] eV.

    Parameters
    ----------
    corrected_data_list : list of CorrectedData
        Output of correct_data() for Ta4f scans.
    boe_wk0_idx : int
        Index into corrected_data_list pointing to the BOE WK0 sample.
    ta4f_modelDict : dict
        Model dict from build_ta4f_models().
    p0_ta4f : lmfit.Parameters
        Shared starting parameters from main.py Section 6. NOT mutated.
    constrained_result : lmfit.MinimizerResult
        The main-loop fit result for BOE WK0, used to overlay constrained fit.
    out_dir_figs : Path
        Directory for figure output (fit_components/ subdir expected to exist).
    out_dir_csv : Path
        Directory for CSV output.

    Returns
    -------
    dict with keys: center_eV, stderr_eV, classification, redchi
    """
    from analysis.models import ta4f_objective  # local to avoid circular import risk
    from plots.style import SPECIES_COLOR, SPECIES_LABEL, FIG_SINGLE_T

    ARTEFACT_CENTER = 26.8   # eV — literature Ta2O5 7/2 position
    ARTEFACT_TOLERANCE = 0.3  # eV

    data = corrected_data_list[boe_wk0_idx]

    # Clone p0_ta4f — MUST NOT mutate the shared parameter set (D-10, pitfall 1)
    p_unc = p0_ta4f.copy()
    p_unc['Ta5_7b2_center'].set(min=24.0, max=29.0)
    # value stays at 26.8, all other params unchanged (D-09, D-10)

    result = lmfit.minimize(
        ta4f_objective, p_unc,
        args=(data, ta4f_modelDict),
        method='bfgs', nan_policy='omit',
    )

    center = result.params['Ta5_7b2_center'].value
    stderr = result.params['Ta5_7b2_center'].stderr or 0.0  # guard None (pitfall 2)
    classification = (
        "constraint artefact likely"
        if abs(center - ARTEFACT_CENTER) <= ARTEFACT_TOLERANCE
        else "real alloy shift"
    )
    # S2-04 — exact required print format
    print(f"Ta5+ center converged at {center:.2f} eV — {classification}")

    # ── Figure (D-12) ─────────────────────────────────────────────────────────
    out_dir_figs = Path(out_dir_figs)
    out_dir_figs.mkdir(parents=True, exist_ok=True)  # guard (pitfall 3)

    x, y, s = data.BE, data.intensity, data.intensityErr

    # Unconstrained fit total envelope
    tot_unc = sum(mdl.eval(result.params, x=x) for mdl in ta4f_modelDict.values())
    # Constrained fit total envelope (for overlay)
    tot_con = sum(mdl.eval(constrained_result.params, x=x) for mdl in ta4f_modelDict.values())

    residual = (y - tot_unc) / np.where(s > 0, s, 1.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=FIG_SINGLE_T,
        gridspec_kw={'height_ratios': [6, 1]}, sharex=True,
    )

    # Raw data
    ax1.errorbar(x, y, s, fmt='o', ms=2, capsize=3, label='Data', color='k')

    # ── CSV (D-13) — written before figure so results persist even if plotting fails ──
    out_dir_csv = Path(out_dir_csv)
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        'center_eV':      center,
        'stderr_eV':      stderr,
        'classification': classification,
        'redchi':         result.redchi,
    }]).to_csv(out_dir_csv / 'ta5_constraint_check.csv', index=False)

    # Unconstrained fit species fills (discretion: show fills for unconstrained fit only)
    for sp in ['metal', 'interface', 'alloy', 'Ta5', 'Ta1', 'Ta3']:
        key72 = f'{sp}_7_2'
        key52 = f'{sp}_5_2'
        if key72 not in ta4f_modelDict or key52 not in ta4f_modelDict:
            continue
        col = SPECIES_COLOR.get(sp, 'gray')
        lbl = SPECIES_LABEL.get(sp, sp)
        y72 = ta4f_modelDict[key72].eval(result.params, x=x)
        y52 = ta4f_modelDict[key52].eval(result.params, x=x)
        ax1.fill_between(x, y72, alpha=0.35, color=col, label=f'{lbl} 7/2')
        ax1.fill_between(x, y52, alpha=0.35, color=col, hatch='//', label=f'{lbl} 5/2')

    # Fit envelopes: unconstrained (solid) vs constrained (dashed)
    ax1.plot(x, tot_unc, 'k-',  lw=1.2, label=f'Unconstrained fit ({center:.2f} eV)')
    ax1.plot(x, tot_con, 'k--', lw=1.0, label='Constrained fit (25.5–28 eV)')

    ax1.set_ylabel('Intensity (a.u.)')
    ax1.legend(bbox_to_anchor=(1, 1), fontsize=6)
    ax1.set_title(f'Ta5+ constraint check — BOE WK0\n{classification}', fontsize=7)
    ax1.set_xlim(31, 20)  # descending BE axis

    # Residual panel (unconstrained fit residuals)
    ax2.plot(x, residual, lw=1)
    ax2.axhline(0, linestyle='--', lw=0.8)
    ymax = max(3, 1.2 * float(np.max(np.abs(residual))))
    ax2.set_ylim(-ymax, ymax)
    ax2.fill_between(x, -1, 1, alpha=0.15, color='gray')
    ax2.set_xlabel('Binding Energy (eV)')
    ax2.set_ylabel('Residual / σ')

    plt.tight_layout()
    fig.savefig(out_dir_figs / 'ta5_constraint_check_boe_wk0.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'center_eV':      center,
        'stderr_eV':      stderr,
        'classification': classification,
        'redchi':         result.redchi,
    }
