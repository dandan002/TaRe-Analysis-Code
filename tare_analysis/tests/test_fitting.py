import numpy as np
import pytest

def test_build_ta4f_models():
    from analysis.models import build_ta4f_models
    m = build_ta4f_models()
    assert set(m.keys()) == {
        'metal_7_2', 'metal_5_2', 'Ta5_7_2', 'Ta5_5_2',
        'Ta3_7_2', 'Ta3_5_2', 'Ta1_7_2', 'Ta1_5_2',
        'interface_7_2', 'interface_5_2', 'alloy_7_2', 'alloy_5_2'
    }

def test_build_re4f_models():
    from analysis.models import build_re4f_models
    m = build_re4f_models()
    assert set(m.keys()) == {
        'Re_metal_7_2', 'Re_metal_5_2',
        'Redp1_7_2', 'Redp1_5_2', 'Redp2_7_2', 'Redp2_5_2', 'Redp3_7_2', 'Redp3_5_2',
        'ReO_7_2', 'ReO_5_2',
        'ReO2_7_2', 'ReO2_5_2', 'ReO3_7_2', 'ReO3_5_2', 'Re2O7_7_2', 'Re2O7_5_2',
    }

def test_literature_dicts():
    from analysis.models import ta4f_expected_peaks, re4f_expected_peaks
    assert ta4f_expected_peaks['Ta metal']['7/2'] == 22.0
    assert re4f_expected_peaks['Re metal']['7/2'] == 40.35

def _make_corrected(n=50):
    from analysis.fitting import CorrectedData
    BE = np.linspace(20, 30, n)
    intensity = np.exp(-0.5 * ((BE - 25) / 0.5) ** 2)
    err = np.sqrt(intensity + 0.001)
    return CorrectedData(BE=BE, intensity=intensity, intensityErr=err,
                         sample='TestSample')

def test_corrected_data_fields():
    cd = _make_corrected()
    assert cd.sample == 'TestSample'
    assert cd.etchlevel is None
    assert len(cd.BE) == 50

def test_calculate_oxide_thickness_basic():
    from analysis.fitting import calculate_oxide_thickness
    t, te = calculate_oxide_thickness(
        I_ox=1.0, I_metal=1.0,
        lambda_ox=2.8, lambda_metal=2.5,
        N_metal=5.55e22, N_ox=8.17e22,
    )
    assert t > 0
    assert te == 0.0

def test_calculate_oxide_thickness_zero_metal():
    from analysis.fitting import calculate_oxide_thickness
    t, te = calculate_oxide_thickness(
        I_ox=1.0, I_metal=0.0,
        lambda_ox=2.8, lambda_metal=2.5,
        N_metal=5.55e22, N_ox=8.17e22,
    )
    assert np.isnan(t)


def test_cabrera_mott_fit_basic():
    """Fit should recover approximate parameters for synthetic logarithmic growth."""
    from analysis.fitting import cabrera_mott_fit
    # Synthetic data: x(t) = 0.5 + 1.2 * ln(1 + t/2)
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    x = 0.5 + 1.2 * np.log(1.0 + t / 2.0)
    result = cabrera_mott_fit(t, x)
    assert result is not None
    assert abs(result['x0'] - 0.5) < 0.05
    assert abs(result['k'] - 1.2) < 0.1
    assert abs(result['tau'] - 2.0) < 0.3


def test_cabrera_mott_fit_too_few_points():
    from analysis.fitting import cabrera_mott_fit
    result = cabrera_mott_fit([0.0, 1.0], [0.5, 0.8])
    assert result is None


def test_safe_minimize_clears_unreliable_covariance_on_linalg_warning(monkeypatch):
    from scipy.linalg import LinAlgWarning
    from analysis.fitting import safe_minimize

    class FakeParam:
        def __init__(self, stderr):
            self.stderr = stderr

    class FakeResult:
        def __init__(self):
            self.covar = np.eye(2)
            self.params = {
                "a": FakeParam(0.12),
                "b": FakeParam(0.34),
            }

    def fake_minimize(*args, **kwargs):
        result = FakeResult()
        warnings.warn(
            "An ill-conditioned matrix detected: slice 0 has rcond = 1e-20.",
            LinAlgWarning,
        )
        return result

    import warnings
    import lmfit

    monkeypatch.setattr(lmfit, "minimize", fake_minimize)

    result = safe_minimize(lambda *_args, **_kwargs: 0.0, None)

    assert result.covar is None
    assert result.params["a"].stderr is None
    assert result.params["b"].stderr is None


def test_shirley_converges_on_step_spectrum():
    """Shirley iteration should produce a monotonically decreasing background on a step."""
    from xps.funcs import iteratedShirleyCorrect
    # Synthetic step: high at left (index 0), low at right (last index)
    n = 100
    step = np.concatenate([np.ones(50) * 2.0, np.ones(50) * 0.5])
    err = np.sqrt(step + 0.01)
    corrected, _, iterations = iteratedShirleyCorrect(step, err, shirleyCutoff=0.001)
    assert iterations >= 1
    assert np.all(corrected >= 0)
    # Background-corrected data should be ≥ 0 everywhere and step should be reduced
    assert corrected[0] <= step[0]
