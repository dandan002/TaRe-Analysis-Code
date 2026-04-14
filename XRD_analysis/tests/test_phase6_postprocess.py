"""Phase 6: Tests for xrd/postprocess.py — rietveld_plot(), build_xrd_xps_table(),
and GSASII_TO_PLOT_KEY.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_gsasii_env(monkeypatch, tmp_path):
    """Prevent xrd.config import error — set GSASII_PATH to a temp dir."""
    monkeypatch.setenv("GSASII_PATH", str(tmp_path))


def _fresh_postprocess():
    """Re-import postprocess after clearing cached modules."""
    for key in list(sys.modules.keys()):
        if "xrd.postprocess" in key:
            sys.modules.pop(key, None)
    return importlib.import_module("xrd.postprocess")


def _make_hist_mock(rwp: float = 95.0) -> MagicMock:
    """Return a mock G2PwdrData with the given Rwp value."""
    hist = MagicMock(name="hist")
    n = 100
    hist.getdata.side_effect = lambda key: {
        "X":        np.linspace(20.0, 80.0, n),
        "Yobs":     np.ones(n) * 1000.0,
        "Ycalc":    np.ones(n) * 10.0,
        "Residual": np.zeros(n),
    }[key]
    # hist.residuals is a @property returning a dict — NOT a method
    type(hist).residuals = PropertyMock(return_value={"wR": rwp})
    hist.reflections.return_value = {}
    return hist


def _make_fake_g2sc(hist_mock: MagicMock) -> MagicMock:
    """Return a mock GSASIIscriptable module."""
    gpx_mock = MagicMock(name="gpx")
    gpx_mock.histograms.return_value = [hist_mock]
    g2sc = MagicMock(name="G2sc_module")
    g2sc.G2Project.return_value = gpx_mock
    return g2sc


# ---------------------------------------------------------------------------
# Task 1 tests: GSASII_TO_PLOT_KEY importability
# ---------------------------------------------------------------------------


class TestGsasiiToPlotKey:
    """GSASII_TO_PLOT_KEY must be importable from xrd.plot and map correctly."""

    def test_importable_from_xrd_plot(self):
        from xrd.plot import GSASII_TO_PLOT_KEY  # noqa: PLC0415
        assert isinstance(GSASII_TO_PLOT_KEY, dict)

    def test_ta2o5_maps_to_unicode(self):
        from xrd.plot import GSASII_TO_PLOT_KEY  # noqa: PLC0415
        assert GSASII_TO_PLOT_KEY["Ta2O5"] == "Ta₂O₅"

    def test_tare_maps_to_bcc(self):
        from xrd.plot import GSASII_TO_PLOT_KEY  # noqa: PLC0415
        assert GSASII_TO_PLOT_KEY["TaRe"] == "TaRe bcc"

    def test_all_five_keys_present_and_in_phase_colors(self):
        from xrd.plot import GSASII_TO_PLOT_KEY, PHASE_COLORS  # noqa: PLC0415
        expected = {"Ta2O5", "ReO2", "ReO3", "TaRe", "Al2O3"}
        assert set(GSASII_TO_PLOT_KEY.keys()) == expected
        for ascii_key, unicode_key in GSASII_TO_PLOT_KEY.items():
            assert unicode_key in PHASE_COLORS, (
                f"{ascii_key!r} → {unicode_key!r} not found in PHASE_COLORS"
            )

    def test_importable_from_xrd_postprocess(self):
        pp = _fresh_postprocess()
        assert hasattr(pp, "GSASII_TO_PLOT_KEY")


# ---------------------------------------------------------------------------
# Task 1 tests: rietveld_plot()
# ---------------------------------------------------------------------------


class TestRietveldPlotMissing:
    """rietveld_plot() raises FileNotFoundError before touching GSAS-II."""

    def test_raises_file_not_found_for_missing_gpx(self, tmp_path):
        pp = _fresh_postprocess()
        missing = tmp_path / "does_not_exist.gpx"
        with pytest.raises(FileNotFoundError, match="GPX file not found"):
            pp.rietveld_plot(missing, tmp_path / "out.png")


class TestRietveldPlotWarning:
    """rietveld_plot() prints WARNING when Rwp > threshold and still saves figure."""

    def test_warning_printed_and_figure_saved(self, tmp_path, capsys):
        gpx_file = tmp_path / "fake.gpx"
        gpx_file.write_bytes(b"fake gpx content")
        out_png = tmp_path / "rietveld.png"

        hist_mock = _make_hist_mock(rwp=95.0)
        g2sc_mock = _make_fake_g2sc(hist_mock)

        pp = _fresh_postprocess()
        with (
            patch.object(pp, "require_gsasii_path", return_value=None),
            patch.dict(sys.modules, {"GSASIIscriptable": g2sc_mock}),
        ):
            pp.rietveld_plot(gpx_file, out_png)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out, "Expected WARNING in stdout"
        assert "Rwp" in captured.out
        assert "not converged" in captured.out
        assert out_png.exists(), "Figure PNG was not created"

    def test_residuals_accessed_as_property_not_method(self, tmp_path, capsys):
        """hist.residuals must be a property dict access — not a method call."""
        gpx_file = tmp_path / "fake2.gpx"
        gpx_file.write_bytes(b"fake")
        out_png = tmp_path / "rietveld2.png"

        hist_mock = _make_hist_mock(rwp=95.0)
        g2sc_mock = _make_fake_g2sc(hist_mock)

        pp = _fresh_postprocess()
        with (
            patch.object(pp, "require_gsasii_path", return_value=None),
            patch.dict(sys.modules, {"GSASIIscriptable": g2sc_mock}),
        ):
            # Would crash with TypeError if code called hist.residuals() as a method
            pp.rietveld_plot(gpx_file, out_png)

        assert out_png.exists()


# ---------------------------------------------------------------------------
# Task 2 tests: build_xrd_xps_table()
# ---------------------------------------------------------------------------


def _write_minimal_rietveld_csv(path: Path, weight_fraction: float = 0.25) -> Path:
    """Write a minimal 5-row rietveld_results.csv for testing."""
    rows = [
        "phase,a_Å,b_Å,c_Å,alpha_deg,beta_deg,gamma_deg,a_esd,b_esd,c_esd,weight_fraction,wf_esd,Rwp",
        f"Ta2O5,4.868,5.538,6.868,90,90,90,0.001,0.001,0.001,{weight_fraction},,12.5",
        f"ReO2,4.585,4.854,5.661,90,90,90,0.001,0.001,0.001,{weight_fraction},,12.5",
        f"ReO3,6.521,6.521,6.521,90,90,90,0.001,0.001,0.001,{weight_fraction},,12.5",
        f"TaRe,3.190,3.190,3.190,90,90,90,0.001,0.001,0.001,{weight_fraction},,12.5",
        f"Al2O3,5.178,5.178,5.178,90,90,90,0.001,0.001,0.001,{weight_fraction},,12.5",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_minimal_be_csv(path: Path) -> Path:
    """Write a minimal be_shift_summary.csv with the 5 required species."""
    rows = [
        "sample,group,element,species,spin,expected_eV,fitted_eV,stderr_eV,delta_eV",
        "S1,wk0,Ta,Ta+5 (Ta2O5),7/2,26.80,25.94,0.05,-0.86",
        "S1,wk0,Re,ReO2 (Re4+),7/2,42.20,42.39,0.05,0.19",
        "S1,wk0,Re,ReO3 (Re6+),7/2,43.10,43.49,0.05,0.39",
        "S1,wk0,Ta,Ta metal,7/2,22.00,22.40,0.05,0.40",
        "S1,wk0,Re,Re metal,7/2,40.35,40.45,0.05,0.10",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


class TestBuildXrdXpsTable:
    """build_xrd_xps_table() must return correct DataFrame structure."""

    def test_column_order_matches_d04(self, tmp_path):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv")
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        df = pp.build_xrd_xps_table(rietveld_csv, be_csv)

        expected_cols = [
            "xrd_phase", "xps_species", "xps_element",
            "weight_fraction", "wf_esd",
            "be_literature_eV", "be_measured_eV",
            "Rwp",
        ]
        assert list(df.columns) == expected_cols

    def test_six_rows_four_single_plus_two_tare(self, tmp_path):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv")
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        df = pp.build_xrd_xps_table(rietveld_csv, be_csv)

        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

    def test_tare_expands_to_ta4f_and_re4f(self, tmp_path):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv")
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        df = pp.build_xrd_xps_table(rietveld_csv, be_csv)

        tare_rows = df[df["xrd_phase"] == "TaRe"]
        assert len(tare_rows) == 2
        assert tare_rows["xps_element"].tolist() == ["Ta4f", "Re4f"]

    def test_al2o3_has_substrate_species_and_nan_bes(self, tmp_path):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv")
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        df = pp.build_xrd_xps_table(rietveld_csv, be_csv)

        al_rows = df[df["xrd_phase"] == "Al2O3"]
        assert len(al_rows) == 1
        assert al_rows["xps_species"].iloc[0] == "N/A (substrate)"
        assert math.isnan(al_rows["be_literature_eV"].iloc[0])
        assert math.isnan(al_rows["be_measured_eV"].iloc[0])

    def test_warning_printed_when_all_wf_are_one(self, tmp_path, capsys):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv", weight_fraction=1.0)
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        pp.build_xrd_xps_table(rietveld_csv, be_csv)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "weight fractions" in captured.out
        assert "not converged" in captured.out

    def test_to_csv_and_to_latex_do_not_raise(self, tmp_path):
        rietveld_csv = _write_minimal_rietveld_csv(tmp_path / "rietveld.csv")
        be_csv = _write_minimal_be_csv(tmp_path / "be.csv")

        pp = _fresh_postprocess()
        df = pp.build_xrd_xps_table(rietveld_csv, be_csv)

        # CSV export
        csv_out = tmp_path / "out.csv"
        df.to_csv(csv_out, index=False)
        assert csv_out.exists()

        # LaTeX export
        latex_str = df.to_latex(index=False, na_rep="—")
        assert isinstance(latex_str, str)
        assert len(latex_str) > 0
