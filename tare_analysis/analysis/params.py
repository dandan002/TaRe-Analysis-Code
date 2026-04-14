# tare_analysis/analysis/params.py
"""Parameter builders for all fit variants.

Stable builders replicate main.py lines 98-175 exactly.
# Must stay in sync with main.py lines 98-175 if stable fit parameters change.

Variant builders start from the stable parameters and apply targeted changes.
"""
import lmfit
from analysis.models import re4f_redp_offsets


def build_ta4f_params_stable() -> lmfit.Parameters:
    """Stable Ta 4f parameter set — mirrors main.py lines 98-131."""
    p = lmfit.Parameters()
    p.add('Ta5_7b2_center',    value=26.8, min=25.5, max=28,   vary=True)
    p.add('Ta3_7b2_center',    value=24.2, min=23.5, max=24.3, vary=True)
    p.add('Ta1_7b2_center',    value=22.7, min=22.0, max=22.9, vary=True)
    p.add('metal_7b2_center',  value=22,   min=21.0, max=23,   vary=True)
    p.add('interface_offset',  value=0.44, min=0.2,  max=0.7)
    p.add('alloy_offset',      value=2.0,  min=1.5,  max=2.5,  vary=True)
    p.add('_7b2_5b2_offset',   value=1.91, min=1.0,  max=3.0,  vary=False)
    p.add('Ta5_7b2_sigma',     value=0.16, min=0.0,  max=0.75, vary=True)
    p.add('metal_7b2_sigma',   value=0.08, min=0.0,  max=0.2,  vary=True)
    p.add('metal_7b2_skew',    value=0.0,  min=0.0,  max=0.5,  vary=True)
    p.add('Ta5_7b2_amplitude',       value=0.16,  min=0.0, max=6.0)
    p.add('Ta3_7b2_amplitude',       value=0.015, min=0.0, max=0.3)
    p.add('Ta1_7b2_amplitude',       value=0.008, min=0.0, max=0.3)
    p.add('metal_7b2_amplitude',     value=0.3,   min=0.05, max=6.0)
    p.add('interface_7b2_amplitude', value=0.03,  min=0.01, max=0.06)
    p.add('alloy_7b2_amplitude',     value=0.10,  min=0.00, max=6.0)
    p.add('_7b2_5b2_ratio',    value=1.333, min=1.2,  max=1.5)
    p.add('interface_7b2_center', vary=False, expr='interface_offset+metal_7b2_center')
    p.add('alloy_7b2_center',     vary=False, expr='alloy_offset+metal_7b2_center')
    p.add('Ta3_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p.add('Ta1_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p.add('interface_7b2_sigma', vary=False, expr='metal_7b2_sigma')
    p.add('alloy_7b2_sigma',     vary=False, expr='metal_7b2_sigma')
    p.add('interface_7b2_skew',  vary=False, expr='metal_7b2_skew')
    p.add('alloy_7b2_skew',      vary=False, expr='metal_7b2_skew')
    for species in ['Ta5', 'Ta3', 'Ta1', 'metal', 'interface', 'alloy']:
        p.add(f'{species}_5b2_center',    vary=False, expr=f'{species}_7b2_center+_7b2_5b2_offset')
        p.add(f'{species}_5b2_sigma',     vary=False, expr=f'{species}_7b2_sigma')
        p.add(f'{species}_5b2_amplitude', vary=False, expr=f'{species}_7b2_amplitude/_7b2_5b2_ratio')
    p.add('metal_5b2_skew',     vary=False, expr='metal_7b2_skew')
    p.add('interface_5b2_skew', vary=False, expr='interface_7b2_skew')
    p.add('alloy_5b2_skew',     vary=False, expr='alloy_7b2_skew')
    return p


def build_re4f_params_stable() -> lmfit.Parameters:
    """Stable Re 4f parameter set — mirrors main.py lines 133-175."""
    p = lmfit.Parameters()
    p.add('Re_metal_7b2_center', value=40.4, min=39.8, max=41.0)
    p.add('ReO2_7b2_center',     value=42.3, min=41.8, max=42.8)
    p.add('ReO3_7b2_center',     value=43.2, min=42.6, max=44.0)
    p.add('Re2O7_7b2_center',    value=45.5, min=45.0, max=46.7)
    p.add('Re_7b2_5b2_offset',   value=2.42, min=2.35, max=2.50)
    p.add('Re_7b2_5b2_ratio',    value=1.33, min=1.20, max=1.50)
    p.add('Re_metal_7b2_sigma',  value=0.10, min=0.00, max=0.30)
    p.add('Re_metal_7b2_skew',   value=0.00, min=0.00, max=0.50, vary=True)
    p.add('ReO2_7b2_sigma',      value=0.18, min=0.05, max=0.60)
    p.add('ReO3_7b2_sigma',      value=0.22, min=0.05, max=0.75)
    p.add('Re2O7_7b2_sigma',     value=0.35, min=0.10, max=0.75)
    p.add('Re_metal_7b2_amplitude', value=0.10,  min=0.00, max=6.00)
    p.add('ReO_7b2_center',    value=41.45, min=41.20, max=41.70)
    p.add('ReO_7b2_sigma',     value=0.18,  min=0.05,  max=0.50)
    p.add('ReO_7b2_amplitude', value=0.005, min=0.00,  max=1.00)
    p.add('ReO_7b2_skew',      value=0.00,  min=0.00,  max=0.30, vary=True)
    p.add('ReO2_7b2_amplitude',     value=0.020, min=0.00, max=2.00)
    p.add('ReO3_7b2_amplitude',     value=0.015, min=0.00, max=2.00)
    p.add('Re2O7_7b2_amplitude',    value=0.010, min=0.00, max=2.00)
    for sp, offset in re4f_redp_offsets.items():
        p.add(f'{sp}_offset',        value=offset, vary=False)
        p.add(f'{sp}_7b2_amplitude', value=0.005, min=0.0, max=0.5)
        p.add(f'{sp}_7b2_center',    vary=False, expr=f'Re_metal_7b2_center+{sp}_offset')
        p.add(f'{sp}_7b2_sigma',     vary=False, expr='Re_metal_7b2_sigma')
        p.add(f'{sp}_7b2_skew',      vary=False, expr='Re_metal_7b2_skew')
        p.add(f'{sp}_5b2_center',    vary=False, expr=f'{sp}_7b2_center+Re_7b2_5b2_offset')
        p.add(f'{sp}_5b2_sigma',     vary=False, expr=f'{sp}_7b2_sigma')
        p.add(f'{sp}_5b2_amplitude', vary=False, expr=f'{sp}_7b2_amplitude/Re_7b2_5b2_ratio')
        p.add(f'{sp}_5b2_skew',      vary=False, expr=f'{sp}_7b2_skew')
    for species in ['Re_metal', 'ReO', 'ReO2', 'ReO3', 'Re2O7']:
        p.add(f'{species}_5b2_center',    expr=f'{species}_7b2_center+Re_7b2_5b2_offset')
        p.add(f'{species}_5b2_sigma',     expr=f'{species}_7b2_sigma')
        p.add(f'{species}_5b2_amplitude', expr=f'{species}_7b2_amplitude/Re_7b2_5b2_ratio')
    p.add('Re_metal_5b2_skew', expr='Re_metal_7b2_skew')
    p.add('ReO_5b2_skew',      expr='ReO_7b2_skew')
    return p


# ── S6: Re fit stabilisation ──────────────────────────────────────────────────
# Changes vs stable:
#   1. ReO2 center tied to Re metal: Re_metal + ReO2_delta, delta free in [1.6, 2.0]
#   2. ReO  upper bound tightened:  41.70 → 41.65 eV
#   3. ReO3 upper bound tightened:  44.0  → 43.6  eV

def build_re4f_params_s6() -> lmfit.Parameters:
    """S6 Re 4f parameters: ReO₂ center relative to Re metal, tighter ReO/ReO₃ bounds."""
    p = lmfit.Parameters()
    p.add('Re_metal_7b2_center', value=40.4, min=39.8, max=41.0)
    # ReO2_delta is the free offset; ReO2_7b2_center derived from it.
    # Adding ReO2_delta before ReO2_7b2_center so the expr is valid at definition time.
    p.add('ReO2_delta',          value=1.9,  min=1.6,  max=2.0,  vary=True)
    p.add('ReO2_7b2_center',     vary=False, expr='Re_metal_7b2_center+ReO2_delta')
    p.add('ReO3_7b2_center',     value=43.2, min=42.6, max=43.6)   # tighter upper bound
    p.add('Re2O7_7b2_center',    value=45.5, min=45.0, max=46.7)
    p.add('Re_7b2_5b2_offset',   value=2.42, min=2.35, max=2.50)
    p.add('Re_7b2_5b2_ratio',    value=1.33, min=1.20, max=1.50)
    p.add('Re_metal_7b2_sigma',  value=0.10, min=0.00, max=0.30)
    p.add('Re_metal_7b2_skew',   value=0.00, min=0.00, max=0.50, vary=True)
    p.add('ReO2_7b2_sigma',      value=0.18, min=0.05, max=0.60)
    p.add('ReO3_7b2_sigma',      value=0.22, min=0.05, max=0.75)
    p.add('Re2O7_7b2_sigma',     value=0.35, min=0.10, max=0.75)
    p.add('Re_metal_7b2_amplitude', value=0.10,  min=0.00, max=6.00)
    p.add('ReO_7b2_center',    value=41.45, min=41.20, max=41.65)  # tighter upper bound
    p.add('ReO_7b2_sigma',     value=0.18,  min=0.05,  max=0.50)
    p.add('ReO_7b2_amplitude', value=0.005, min=0.00,  max=1.00)
    p.add('ReO_7b2_skew',      value=0.00,  min=0.00,  max=0.30, vary=True)
    p.add('ReO2_7b2_amplitude',     value=0.020, min=0.00, max=2.00)
    p.add('ReO3_7b2_amplitude',     value=0.015, min=0.00, max=2.00)
    p.add('Re2O7_7b2_amplitude',    value=0.010, min=0.00, max=2.00)
    for sp, offset in re4f_redp_offsets.items():
        p.add(f'{sp}_offset',        value=offset, vary=False)
        p.add(f'{sp}_7b2_amplitude', value=0.005, min=0.0, max=0.5)
        p.add(f'{sp}_7b2_center',    vary=False, expr=f'Re_metal_7b2_center+{sp}_offset')
        p.add(f'{sp}_7b2_sigma',     vary=False, expr='Re_metal_7b2_sigma')
        p.add(f'{sp}_7b2_skew',      vary=False, expr='Re_metal_7b2_skew')
        p.add(f'{sp}_5b2_center',    vary=False, expr=f'{sp}_7b2_center+Re_7b2_5b2_offset')
        p.add(f'{sp}_5b2_sigma',     vary=False, expr=f'{sp}_7b2_sigma')
        p.add(f'{sp}_5b2_amplitude', vary=False, expr=f'{sp}_7b2_amplitude/Re_7b2_5b2_ratio')
        p.add(f'{sp}_5b2_skew',      vary=False, expr=f'{sp}_7b2_skew')
    for species in ['Re_metal', 'ReO', 'ReO2', 'ReO3', 'Re2O7']:
        p.add(f'{species}_5b2_center',    expr=f'{species}_7b2_center+Re_7b2_5b2_offset')
        p.add(f'{species}_5b2_sigma',     expr=f'{species}_7b2_sigma')
        p.add(f'{species}_5b2_amplitude', expr=f'{species}_7b2_amplitude/Re_7b2_5b2_ratio')
    p.add('Re_metal_5b2_skew', expr='Re_metal_7b2_skew')
    p.add('ReO_5b2_skew',      expr='ReO_7b2_skew')
    return p


# ── S7: Two-component Ta oxide model ─────────────────────────────────────────
# Changes vs stable:
#   Ta5 (single Gaussian) replaced by two components linked by a free separation:
#     Ta5a — higher-BE component: free in [25.0, 27.5] eV
#     Ta5b — lower-BE component:  expr='Ta5a_7b2_center - Ta5_separation'
#             Ta5_separation free in [0.5, 2.0] eV
#   Linking Ta5b to Ta5a via a separation parameter eliminates the identity-swap
#   degeneracy seen when both centers were independently free in adjacent windows.
#   Ta5a_7b2_sigma becomes the shared oxide width (replaces Ta5_7b2_sigma).
#   Ta3 and Ta1 sigma expressions re-pointed to Ta5a_7b2_sigma.

def build_ta4f_params_s7() -> lmfit.Parameters:
    """S7 Ta 4f parameters: two-component Ta oxide with Ta5b center linked to Ta5a."""
    p = lmfit.Parameters()
    p.add('metal_7b2_center',  value=22,   min=21.0, max=23,   vary=True)
    p.add('Ta3_7b2_center',    value=24.2, min=23.5, max=24.3, vary=True)
    p.add('Ta1_7b2_center',    value=22.7, min=22.0, max=22.9, vary=True)
    p.add('interface_offset',  value=0.44, min=0.2,  max=0.7)
    p.add('alloy_offset',      value=2.0,  min=1.5,  max=2.5,  vary=True)
    p.add('_7b2_5b2_offset',   value=1.91, min=1.0,  max=3.0,  vary=False)
    # Shared oxide sigma (plays the role of Ta5_7b2_sigma in the stable fit)
    p.add('Ta5a_7b2_sigma',    value=0.16, min=0.0,  max=0.75, vary=True)
    p.add('metal_7b2_sigma',   value=0.08, min=0.0,  max=0.2,  vary=True)
    p.add('metal_7b2_skew',    value=0.0,  min=0.0,  max=0.5,  vary=True)
    # Ta5a: higher-BE oxide component, free in [25.0, 27.5] eV
    p.add('Ta5a_7b2_center',   value=26.8, min=25.0, max=27.5, vary=True)
    p.add('Ta5a_7b2_amplitude', value=0.08, min=0.0, max=6.0)
    # Ta5b: lower-BE component, center = Ta5a - separation (separation free in [0.5, 2.0] eV)
    # This eliminates the identity-swap degeneracy from independently windowed centers.
    p.add('Ta5_separation',    value=1.3,  min=0.5,  max=2.0,  vary=True)
    p.add('Ta5b_7b2_center',   vary=False, expr='Ta5a_7b2_center - Ta5_separation')
    p.add('Ta5b_7b2_sigma',    vary=False, expr='Ta5a_7b2_sigma')
    p.add('Ta5b_7b2_amplitude', value=0.08, min=0.0, max=6.0)
    p.add('Ta3_7b2_amplitude',       value=0.015, min=0.0, max=0.3)
    p.add('Ta1_7b2_amplitude',       value=0.008, min=0.0, max=0.3)
    p.add('metal_7b2_amplitude',     value=0.3,   min=0.05, max=6.0)
    p.add('interface_7b2_amplitude', value=0.03,  min=0.01, max=0.06)
    p.add('alloy_7b2_amplitude',     value=0.10,  min=0.00, max=6.0)
    p.add('_7b2_5b2_ratio',    value=1.333, min=1.2,  max=1.5)
    p.add('interface_7b2_center', vary=False, expr='interface_offset+metal_7b2_center')
    p.add('alloy_7b2_center',     vary=False, expr='alloy_offset+metal_7b2_center')
    # Ta3 and Ta1 sigma re-pointed to Ta5a_7b2_sigma (replaces Ta5_7b2_sigma)
    p.add('Ta3_7b2_sigma',       vary=False, expr='Ta5a_7b2_sigma')
    p.add('Ta1_7b2_sigma',       vary=False, expr='Ta5a_7b2_sigma')
    p.add('interface_7b2_sigma', vary=False, expr='metal_7b2_sigma')
    p.add('alloy_7b2_sigma',     vary=False, expr='metal_7b2_sigma')
    p.add('interface_7b2_skew',  vary=False, expr='metal_7b2_skew')
    p.add('alloy_7b2_skew',      vary=False, expr='metal_7b2_skew')
    for species in ['Ta5a', 'Ta5b', 'Ta3', 'Ta1', 'metal', 'interface', 'alloy']:
        p.add(f'{species}_5b2_center',    vary=False, expr=f'{species}_7b2_center+_7b2_5b2_offset')
        p.add(f'{species}_5b2_sigma',     vary=False, expr=f'{species}_7b2_sigma')
        p.add(f'{species}_5b2_amplitude', vary=False, expr=f'{species}_7b2_amplitude/_7b2_5b2_ratio')
    p.add('metal_5b2_skew',     vary=False, expr='metal_7b2_skew')
    p.add('interface_5b2_skew', vary=False, expr='interface_7b2_skew')
    p.add('alloy_5b2_skew',     vary=False, expr='alloy_7b2_skew')
    return p


# ── S8: Relative constraints for Ta¹⁺ and Ta³⁺ ───────────────────────────────
# Changes vs stable:
#   Ta1_7b2_center: expr='metal_7b2_center + Ta1_delta', Ta1_delta free in [0.4, 1.0] eV
#   Ta3_7b2_center: expr='metal_7b2_center + Ta3_delta', Ta3_delta free in [1.7, 2.7] eV
# The previous run pinned Ta1_delta=0.7 and Ta3_delta=2.2 exactly (McLellan pure-Ta
# values), producing χ²ν of 10–14. Narrow free windows keep the relative-anchoring
# benefit (no bound-saturation) while letting the alloy-specific shifts adjust.

def build_ta4f_params_s8() -> lmfit.Parameters:
    """S8 Ta 4f parameters: Ta¹⁺ and Ta³⁺ centers anchored relative to metal, deltas free."""
    p = lmfit.Parameters()
    p.add('Ta5_7b2_center',    value=26.8, min=25.5, max=28,   vary=True)
    p.add('metal_7b2_center',  value=22,   min=21.0, max=23,   vary=True)
    p.add('interface_offset',  value=0.44, min=0.2,  max=0.7)
    p.add('alloy_offset',      value=2.0,  min=1.5,  max=2.5,  vary=True)
    p.add('_7b2_5b2_offset',   value=1.91, min=1.0,  max=3.0,  vary=False)
    p.add('Ta5_7b2_sigma',     value=0.16, min=0.0,  max=0.75, vary=True)
    p.add('metal_7b2_sigma',   value=0.08, min=0.0,  max=0.2,  vary=True)
    p.add('metal_7b2_skew',    value=0.0,  min=0.0,  max=0.5,  vary=True)
    # Ta1 and Ta3 deltas free in narrow windows around McLellan pure-Ta values.
    # Allows the alloy-specific shift to adjust without releasing the relative anchor.
    p.add('Ta1_delta',         value=0.7,  min=0.4,  max=1.0,  vary=True)
    p.add('Ta3_delta',         value=2.2,  min=1.7,  max=2.7,  vary=True)
    p.add('Ta1_7b2_center',    vary=False, expr='metal_7b2_center+Ta1_delta')
    p.add('Ta3_7b2_center',    vary=False, expr='metal_7b2_center+Ta3_delta')
    p.add('Ta5_7b2_amplitude',       value=0.16,  min=0.0, max=6.0)
    p.add('Ta3_7b2_amplitude',       value=0.015, min=0.0, max=0.3)
    p.add('Ta1_7b2_amplitude',       value=0.008, min=0.0, max=0.3)
    p.add('metal_7b2_amplitude',     value=0.3,   min=0.05, max=6.0)
    p.add('interface_7b2_amplitude', value=0.03,  min=0.01, max=0.06)
    p.add('alloy_7b2_amplitude',     value=0.10,  min=0.00, max=6.0)
    p.add('_7b2_5b2_ratio',    value=1.333, min=1.2,  max=1.5)
    p.add('interface_7b2_center', vary=False, expr='interface_offset+metal_7b2_center')
    p.add('alloy_7b2_center',     vary=False, expr='alloy_offset+metal_7b2_center')
    p.add('Ta3_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p.add('Ta1_7b2_sigma',       vary=False, expr='Ta5_7b2_sigma')
    p.add('interface_7b2_sigma', vary=False, expr='metal_7b2_sigma')
    p.add('alloy_7b2_sigma',     vary=False, expr='metal_7b2_sigma')
    p.add('interface_7b2_skew',  vary=False, expr='metal_7b2_skew')
    p.add('alloy_7b2_skew',      vary=False, expr='metal_7b2_skew')
    for species in ['Ta5', 'Ta3', 'Ta1', 'metal', 'interface', 'alloy']:
        p.add(f'{species}_5b2_center',    vary=False, expr=f'{species}_7b2_center+_7b2_5b2_offset')
        p.add(f'{species}_5b2_sigma',     vary=False, expr=f'{species}_7b2_sigma')
        p.add(f'{species}_5b2_amplitude', vary=False, expr=f'{species}_7b2_amplitude/_7b2_5b2_ratio')
    p.add('metal_5b2_skew',     vary=False, expr='metal_7b2_skew')
    p.add('interface_5b2_skew', vary=False, expr='interface_7b2_skew')
    p.add('alloy_5b2_skew',     vary=False, expr='alloy_7b2_skew')
    return p
