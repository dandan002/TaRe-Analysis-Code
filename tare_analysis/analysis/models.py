# tare_analysis/analysis/models.py
import lmfit


def build_ta4f_models():
    return {
        "metal_7_2":     lmfit.models.SkewedVoigtModel(prefix="metal_7b2_"),
        "metal_5_2":     lmfit.models.SkewedVoigtModel(prefix="metal_5b2_"),
        "Ta5_7_2":       lmfit.models.GaussianModel(prefix="Ta5_7b2_"),
        "Ta5_5_2":       lmfit.models.GaussianModel(prefix="Ta5_5b2_"),
        "Ta3_7_2":       lmfit.models.GaussianModel(prefix="Ta3_7b2_"),
        "Ta3_5_2":       lmfit.models.GaussianModel(prefix="Ta3_5b2_"),
        "Ta1_7_2":       lmfit.models.GaussianModel(prefix="Ta1_7b2_"),
        "Ta1_5_2":       lmfit.models.GaussianModel(prefix="Ta1_5b2_"),
        "interface_7_2": lmfit.models.SkewedVoigtModel(prefix="interface_7b2_"),
        "interface_5_2": lmfit.models.SkewedVoigtModel(prefix="interface_5b2_"),
        "alloy_7_2":     lmfit.models.SkewedVoigtModel(prefix="alloy_7b2_"),
        "alloy_5_2":     lmfit.models.SkewedVoigtModel(prefix="alloy_5b2_"),
    }


def build_ta4f_models_s7():
    """S7 Ta 4f model dict: Ta5 pair replaced by Ta5a + Ta5b Gaussian pairs."""
    d = build_ta4f_models()
    del d['Ta5_7_2']
    del d['Ta5_5_2']
    d['Ta5a_7_2'] = lmfit.models.GaussianModel(prefix='Ta5a_7b2_')
    d['Ta5a_5_2'] = lmfit.models.GaussianModel(prefix='Ta5a_5b2_')
    d['Ta5b_7_2'] = lmfit.models.GaussianModel(prefix='Ta5b_7b2_')
    d['Ta5b_5_2'] = lmfit.models.GaussianModel(prefix='Ta5b_5b2_')
    return d


def build_re4f_models():
    return {
        "Re_metal_7_2": lmfit.models.SkewedVoigtModel(prefix="Re_metal_7b2_"),
        "Re_metal_5_2": lmfit.models.SkewedVoigtModel(prefix="Re_metal_5b2_"),
        # Reδ+ surface O-coordination shoulders at +0.22, +0.45, +0.73 eV above Re⁰
        # (Greiner et al. 2014; three distinct sub-surface coordination environments)
        "Redp1_7_2":    lmfit.models.SkewedVoigtModel(prefix="Redp1_7b2_"),
        "Redp1_5_2":    lmfit.models.SkewedVoigtModel(prefix="Redp1_5b2_"),
        "Redp2_7_2":    lmfit.models.SkewedVoigtModel(prefix="Redp2_7b2_"),
        "Redp2_5_2":    lmfit.models.SkewedVoigtModel(prefix="Redp2_5b2_"),
        "Redp3_7_2":    lmfit.models.SkewedVoigtModel(prefix="Redp3_7b2_"),
        "Redp3_5_2":    lmfit.models.SkewedVoigtModel(prefix="Redp3_5b2_"),
        # 'ReO' (Re²⁺) surface oxide at ~41.45 eV — unique to Re metal surfaces,
        # no bulk analog; sits between Redp3 and ReO2 (Greiner et al. 2014, Table 1)
        "ReO_7_2":      lmfit.models.GaussianModel(prefix="ReO_7b2_"),
        "ReO_5_2":      lmfit.models.GaussianModel(prefix="ReO_5b2_"),
        "ReO2_7_2":     lmfit.models.GaussianModel(prefix="ReO2_7b2_"),
        "ReO2_5_2":     lmfit.models.GaussianModel(prefix="ReO2_5b2_"),
        "ReO3_7_2":     lmfit.models.GaussianModel(prefix="ReO3_7b2_"),
        "ReO3_5_2":     lmfit.models.GaussianModel(prefix="ReO3_5b2_"),
        "Re2O7_7_2":    lmfit.models.GaussianModel(prefix="Re2O7_7b2_"),
        "Re2O7_5_2":    lmfit.models.GaussianModel(prefix="Re2O7_5b2_"),
    }


def ta4f_objective(params, data, modelDict):
    import numpy as np
    x, y = data.BE, data.intensity
    s = np.where(data.intensityErr > 0, data.intensityErr, 1.0)
    tot = sum(mdl.eval(params=params, x=x) for mdl in modelDict.values())
    return (tot - y) / s


def re4f_objective(params, data, modelDict):
    import numpy as np
    x, y = data.BE, data.intensity
    s = np.where(data.intensityErr > 0, data.intensityErr, 1.0)
    tot = sum(mdl.eval(params=params, x=x) for mdl in modelDict.values())
    return (tot - y) / s


# ── Reference values ─────────────────────────────────────────────────────────
#
# Ta values below are carried over from the local Ta oxide analysis workflow and
# are useful fit/label references, but the exact numeric windows used in
# `etch_main.py` are heuristic fitting bounds rather than explicitly cited
# literature limits.
#
# Re values below are aligned to the local paper:
# Greiner et al., "The Oxidation of Rhenium..." (2014),
# which reports internally consistent Re 4f7/2 values of
# Re0 40.35 eV, Re4+ 42.20 eV, Re6+ 43.10 eV, Re7+ 45.5 eV
# with 2.42 eV spin-orbit splitting.

ta4f_expected_peaks = {
    'Ta metal':       {'7/2': 22.0, '5/2': 23.9},
    'Ta interface':   {'7/2': 22.4, '5/2': 24.3},
    'Ta alloy':       {'7/2': 24.0, '5/2': 25.9},
    'Ta+1':           {'7/2': 22.7, '5/2': 24.6},
    'Ta+3':           {'7/2': 24.2, '5/2': 26.1},
    'Ta+5 (Ta2O5)':   {'7/2': 26.8, '5/2': 28.7},
}

ta4f_species_map = {
    'Ta metal': 'metal', 'Ta interface': 'interface', 'Ta alloy': 'alloy',
    'Ta+1': 'Ta1', 'Ta+3': 'Ta3', 'Ta+5 (Ta2O5)': 'Ta5',
}

# Reδ+ energy offsets from Re⁰ for the three surface O-coordination environments.
# Centers are parameterised as Re_metal_center + offset (constrained in fitting).
re4f_redp_offsets = {'Redp1': 0.22, 'Redp2': 0.45, 'Redp3': 0.73}

re4f_expected_peaks = {
    'Re metal':       {'7/2': 40.35, '5/2': 42.77},
    'Redp1 (Reδ+1)':  {'7/2': 40.57, '5/2': 42.99},
    'Redp2 (Reδ+2)':  {'7/2': 40.80, '5/2': 43.22},
    'Redp3 (Reδ+3)':  {'7/2': 41.08, '5/2': 43.50},
    'ReO (Re2+)':     {'7/2': 41.45, '5/2': 43.87},
    'ReO2 (Re4+)':    {'7/2': 42.20, '5/2': 44.62},
    'ReO3 (Re6+)':    {'7/2': 43.10, '5/2': 45.52},
    'Re2O7 (Re7+)':   {'7/2': 45.50, '5/2': 47.92},
}

re4f_species_map = {
    'Re metal': 'Re_metal',
    'Redp1 (Reδ+1)': 'Redp1', 'Redp2 (Reδ+2)': 'Redp2', 'Redp3 (Reδ+3)': 'Redp3',
    'ReO (Re2+)': 'ReO',
    'ReO2 (Re4+)': 'ReO2', 'ReO3 (Re6+)': 'ReO3', 'Re2O7 (Re7+)': 'Re2O7',
}

re4f_expected_amplitudes = {
    'BOE samples':     {'Re metal': 0.65, 'Redp1 (Reδ+1)': 0.02, 'Redp2 (Reδ+2)': 0.01, 'Redp3 (Reδ+3)': 0.01,
                        'ReO (Re2+)': 0.03, 'ReO2 (Re4+)': 0.15, 'ReO3 (Re6+)': 0.10, 'Re2O7 (Re7+)': 0.05},
    'Control samples': {'Re metal': 0.46, 'Redp1 (Reδ+1)': 0.02, 'Redp2 (Reδ+2)': 0.01, 'Redp3 (Reδ+3)': 0.01,
                        'ReO (Re2+)': 0.03, 'ReO2 (Re4+)': 0.25, 'ReO3 (Re6+)': 0.15, 'Re2O7 (Re7+)': 0.10},
}

ta4f_expected_amplitudes = {
    'BOE samples':     {'Ta metal': 0.40, 'Ta interface': 0.10, 'Ta alloy': 0.15, 'Ta+1': 0.05, 'Ta+3': 0.05, 'Ta+5 (Ta2O5)': 0.25},
    'Control samples': {'Ta metal': 0.20, 'Ta interface': 0.05, 'Ta alloy': 0.05, 'Ta+1': 0.05, 'Ta+3': 0.10, 'Ta+5 (Ta2O5)': 0.55},
}

re4f_expected_widths = {
    'Re metal':       {'sigma': 0.12, 'FWHM': 0.28},
    'Redp1 (Reδ+1)':  {'sigma': 0.12, 'FWHM': 0.28},
    'Redp2 (Reδ+2)':  {'sigma': 0.12, 'FWHM': 0.28},
    'Redp3 (Reδ+3)':  {'sigma': 0.12, 'FWHM': 0.28},
    'ReO (Re2+)':     {'sigma': 0.20, 'FWHM': 0.47},
    'ReO2 (Re4+)':    {'sigma': 0.25, 'FWHM': 0.59},
    'ReO3 (Re6+)':    {'sigma': 0.25, 'FWHM': 0.59},
    'Re2O7 (Re7+)':   {'sigma': 0.30, 'FWHM': 0.71},
}

ta4f_expected_widths = {
    'Ta metal':      {'sigma': 0.10, 'FWHM': 0.24},
    'Ta interface':  {'sigma': 0.10, 'FWHM': 0.24},
    'Ta alloy':      {'sigma': 0.10, 'FWHM': 0.24},
    'Ta+1':          {'sigma': 0.20, 'FWHM': 0.47},
    'Ta+3':          {'sigma': 0.20, 'FWHM': 0.47},
    'Ta+5 (Ta2O5)':  {'sigma': 0.20, 'FWHM': 0.47},
}

# Literature centers for Section 15 BE-shift analysis
ta4f_literature = {
    'Ta metal':       {'7/2': 22.0, '5/2': 23.9},
    'Ta interface':   {'7/2': 22.4, '5/2': 24.3},
    'Ta alloy':       {'7/2': 24.0, '5/2': 25.9},
    'Ta+1':           {'7/2': 22.7, '5/2': 24.6},
    'Ta+3':           {'7/2': 24.2, '5/2': 26.1},
    'Ta+5 (Ta2O5)':   {'7/2': 26.8, '5/2': 28.7},
}

re4f_literature = {
    'Re metal':       {'7/2': 40.35, '5/2': 42.77},
    'Redp1 (Reδ+1)':  {'7/2': 40.57, '5/2': 42.99},
    'Redp2 (Reδ+2)':  {'7/2': 40.80, '5/2': 43.22},
    'Redp3 (Reδ+3)':  {'7/2': 41.08, '5/2': 43.50},
    'ReO (Re2+)':     {'7/2': 41.45, '5/2': 43.87},
    'ReO2 (Re4+)':    {'7/2': 42.20, '5/2': 44.62},
    'ReO3 (Re6+)':    {'7/2': 43.10, '5/2': 45.52},
    'Re2O7 (Re7+)':   {'7/2': 45.50, '5/2': 47.92},
}
