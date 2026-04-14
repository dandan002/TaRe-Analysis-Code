import importlib
import inspect
from pathlib import Path

import numpy as np


def test_build_phase_peaks_dict_uses_first_downloaded_cif(monkeypatch, tmp_path):
    from xrd.phases import build_phase_peaks_dict

    ta_path = tmp_path / "Ta2O5_1_mp-1.cif"
    re_path = tmp_path / "ReO2_1_mp-2.cif"
    ta_path.write_text("ta")
    re_path.write_text("re")

    calls = []

    def fake_peaks(path):
        calls.append(Path(path).name)
        return [11.1, 22.2] if "Ta2O5" in Path(path).name else [33.3]

    monkeypatch.setattr("xrd.phases.peaks_from_cif", fake_peaks)

    phase_peaks = build_phase_peaks_dict(
        {
            "Ta2O5": [ta_path],
            "ReO2": [re_path],
            "ReO3": [],
            "TaRe": [],
            "Al2O3": [],
        }
    )

    assert calls == ["Ta2O5_1_mp-1.cif", "ReO2_1_mp-2.cif"]
    assert phase_peaks == {
        "Ta₂O₅": [11.1, 22.2],
        "ReO₂": [33.3],
    }


def test_quicklook_plot_supports_phase_peak_overrides(monkeypatch, tmp_path):
    from xrd.plot import quicklook_plot

    signature = inspect.signature(quicklook_plot)
    assert "phase_peaks" in signature.parameters
    assert signature.parameters["phase_peaks"].default is None

    drawn_positions = []

    class FakeAxis:
        def __init__(self):
            self.spines = self

        def plot(self, *args, **kwargs):
            return None

        def axvline(self, position, **kwargs):
            drawn_positions.append(position)

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def set_visible(self, *args, **kwargs):
            return None

        def __getitem__(self, _key):
            return self

    class FakeFigure:
        def tight_layout(self):
            return None

        def savefig(self, path, dpi):
            Path(path).write_text(f"dpi={dpi}")

    monkeypatch.setattr("xrd.plot.plt.subplots", lambda **kwargs: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr("xrd.plot.plt.close", lambda fig: None)

    quicklook_plot(
        np.array([10.0, 20.0]),
        np.array([1.0, 2.0]),
        tmp_path / "override.png",
        phase_peaks={"Ta₂O₅": [12.5], "ReO₂": [14.5, 15.5]},
    )

    assert drawn_positions == [12.5, 14.5, 15.5]


def test_search_phases_main_runs_full_pipeline(monkeypatch, tmp_path, capsys):
    search_phases = importlib.import_module("search_phases")

    calls = []
    expected_frame = object()
    expected_results = {"Ta2O5": ["doc"]}
    expected_downloaded = {"Ta2O5": [tmp_path / "Ta2O5_1_mp-1.cif"]}
    expected_observed_peaks = np.array([41.7])
    expected_peaks = {"Ta₂O₅": [23.4, 45.6]}
    expected_scan = (np.array([10.0, 20.0]), np.array([1.0, 2.0]))

    monkeypatch.setattr(search_phases, "RAW_FILE", tmp_path / "scan.raw")
    monkeypatch.setattr(search_phases, "PLOT_OUT", tmp_path / "quicklook.png")
    monkeypatch.setattr(search_phases, "ensure_results", lambda: calls.append("ensure_results"))
    monkeypatch.setattr(search_phases, "ensure_cif_dir", lambda: calls.append("ensure_cif_dir"))
    monkeypatch.setattr(
        search_phases,
        "load_raw",
        lambda path: (calls.append(("load_raw", Path(path).name)) or expected_scan),
    )
    monkeypatch.setattr(search_phases, "get_wavelength", lambda path: 1.5406)
    monkeypatch.setattr(
        search_phases,
        "validate",
        lambda two_theta, intensity, wavelength: calls.append(("validate", wavelength)),
    )
    monkeypatch.setattr(search_phases, "_get_api_key", lambda: "secret-key")
    monkeypatch.setattr(
        search_phases,
        "search_all_phases",
        lambda api_key: calls.append(("search_all_phases", api_key)) or expected_results,
    )
    monkeypatch.setattr(
        search_phases,
        "build_candidates_df",
        lambda results: calls.append(("build_candidates_df", results)) or expected_frame,
    )
    monkeypatch.setattr(
        search_phases,
        "print_candidates_table",
        lambda frame: calls.append(("print_candidates_table", frame)),
    )
    monkeypatch.setattr(
        search_phases,
        "save_candidates_csv",
        lambda frame: calls.append(("save_candidates_csv", frame)),
    )
    monkeypatch.setattr(
        search_phases,
        "download_top_cifs",
        lambda results, api_key: calls.append(("download_top_cifs", results, api_key)) or expected_downloaded,
    )
    monkeypatch.setattr(
        search_phases,
        "find_observed_peaks",
        lambda two_theta, intensity: calls.append(
            ("find_observed_peaks", two_theta.copy(), intensity.copy())
        )
        or expected_observed_peaks,
    )
    monkeypatch.setattr(
        search_phases,
        "build_phase_peaks_dict",
        lambda downloaded, observed_peaks=None: calls.append(
            ("build_phase_peaks_dict", downloaded, observed_peaks.copy())
        )
        or expected_peaks,
    )
    monkeypatch.setattr(
        search_phases,
        "quicklook_plot",
        lambda two_theta, intensity, out_path, phase_peaks=None: calls.append(
            ("quicklook_plot", Path(out_path).name, phase_peaks)
        ),
    )

    search_phases.main()

    assert calls[:9] == [
        "ensure_results",
        "ensure_cif_dir",
        ("load_raw", "scan.raw"),
        ("validate", 1.5406),
        ("search_all_phases", "secret-key"),
        ("build_candidates_df", expected_results),
        ("print_candidates_table", expected_frame),
        ("save_candidates_csv", expected_frame),
        ("download_top_cifs", expected_results, "secret-key"),
    ]
    assert calls[9][0] == "find_observed_peaks"
    assert np.array_equal(calls[9][1], expected_scan[0])
    assert np.array_equal(calls[9][2], expected_scan[1])
    assert calls[10][0] == "build_phase_peaks_dict"
    assert calls[10][1] == expected_downloaded
    assert np.array_equal(calls[10][2], expected_observed_peaks)
    assert calls[11] == ("quicklook_plot", "quicklook.png", expected_peaks)
    assert "Phase 4 complete." in capsys.readouterr().out
