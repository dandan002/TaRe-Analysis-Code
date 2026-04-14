import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main import CSV_OUT, PLOT_OUT, main
from xrd.config import DATA_DIR
from xrd.io import export_csv, get_wavelength, load_raw, validate


DATA_FILE = DATA_DIR / "TaRe_Full_Oxide.raw"


def test_main_import_does_not_require_gsasii_path(monkeypatch):
    monkeypatch.delenv("GSASII_PATH", raising=False)
    sys.modules.pop("xrd.config", None)
    sys.modules.pop("main", None)

    module = importlib.import_module("main")

    assert module.RAW_FILE == DATA_FILE


def test_xrd_io_import_does_not_require_xylib_installed(monkeypatch):
    monkeypatch.delenv("GSASII_PATH", raising=False)
    original_module = sys.modules.pop("xrd.io", None)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xylib":
            raise ModuleNotFoundError("No module named 'xylib'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        module = importlib.import_module("xrd.io")
        assert module.__all__ == ["load_raw", "get_wavelength", "validate", "export_csv"]
    finally:
        sys.modules.pop("xrd.io", None)
        if original_module is not None:
            sys.modules["xrd.io"] = original_module


def test_load_raw_returns_physically_plausible_arrays():
    two_theta, intensity = load_raw(DATA_FILE)

    assert isinstance(two_theta, np.ndarray)
    assert isinstance(intensity, np.ndarray)
    assert two_theta.shape == intensity.shape
    assert len(two_theta) > 0
    assert 0.0 <= float(two_theta[0]) <= 20.0
    assert 80.0 <= float(two_theta[-1]) <= 120.0
    assert np.all(intensity > 0)


def test_get_wavelength_returns_none_when_metadata_missing(monkeypatch):
    class FakeBlock:
        def get_meta(self, key):
            raise RuntimeError("missing")

    class FakeDataset:
        def get_block(self, index):
            return FakeBlock()

    monkeypatch.setattr(
        get_wavelength.__globals__["xylib"],
        "load_file",
        lambda *args, **kwargs: FakeDataset(),
    )

    assert get_wavelength(DATA_FILE) is None


def test_validate_rejects_truncated_range():
    with pytest.raises(ValueError, match=r"2θ span is 50.0°"):
        validate(np.array([20.0, 70.0]), np.array([1.0, 2.0]))


def test_validate_rejects_negative_intensity():
    with pytest.raises(ValueError, match=r"1 intensity values ≤ 0"):
        validate(np.array([20.0, 81.0, 82.0]), np.array([1.0, 1.0, -1.0]))


def test_validate_rejects_wrong_wavelength():
    with pytest.raises(ValueError, match=r"Header wavelength 1.53000 Å deviates from Cu Kα"):
        validate(np.array([20.0, 81.0]), np.array([1.0, 2.0]), wavelength=1.53)


def test_validate_accepts_real_file():
    two_theta, intensity = load_raw(DATA_FILE)
    validate(two_theta, intensity, get_wavelength(DATA_FILE))


def test_export_csv_writes_header_and_two_columns(tmp_path):
    out_path = tmp_path / "nested" / "scan.csv"

    export_csv(np.array([10.0, 11.0]), np.array([100.0, 101.0]), out_path)

    frame = pd.read_csv(out_path)
    assert list(frame.columns) == ["two_theta", "intensity"]
    assert frame.to_dict(orient="list") == {
        "two_theta": [10.0, 11.0],
        "intensity": [100.0, 101.0],
    }


def test_quicklook_plot_exports_expected_phase_metadata_and_png(tmp_path):
    from xrd.plot import PHASE_COLORS, PHASE_PEAKS, quicklook_plot

    assert set(PHASE_PEAKS) == {"Ta₂O₅", "TaRe bcc", "ReO₂", "ReO₃", "Al₂O₃"}
    assert PHASE_COLORS == {
        "Ta₂O₅": "#D55E00",
        "TaRe bcc": "#0072B2",
        "ReO₂": "#E69F00",
        "ReO₃": "#CC79A7",
        "Al₂O₃": "#009E73",
    }

    two_theta, intensity = load_raw(DATA_FILE)
    out_path = tmp_path / "plots" / "quicklook.png"
    quicklook_plot(two_theta, intensity, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_pipeline_creates_csv_and_quicklook_outputs(capsys):
    main()

    output = capsys.readouterr().out
    assert "Phase 3 complete." in output
    assert CSV_OUT.exists()
    assert PLOT_OUT.exists()
    assert CSV_OUT.read_text().splitlines()[0] == "two_theta,intensity"
