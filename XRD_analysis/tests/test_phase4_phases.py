from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xrd.config import DATA_DIR


def test_ensure_cif_dir_creates_data_cif_directory():
    from xrd.config import CIF_DIR, ensure_cif_dir

    created = ensure_cif_dir()

    assert created == CIF_DIR
    assert created == DATA_DIR / "cif"
    assert created.exists()
    assert created.is_dir()


def test_get_api_key_raises_clear_error_when_missing(monkeypatch):
    from xrd.phases import _get_api_key

    monkeypatch.delenv("MP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MP_API_KEY environment variable not set"):
        _get_api_key()


def test_build_candidates_df_sorts_by_phase_order_then_hull_energy():
    from xrd.phases import build_candidates_df

    results = {
        "ReO3": [
            SimpleNamespace(
                material_id="mp-reo3-2",
                formula_pretty="ReO3",
                symmetry=SimpleNamespace(symbol="Pm-3m"),
                energy_above_hull=0.1,
                is_stable=False,
            ),
            SimpleNamespace(
                material_id="mp-reo3-1",
                formula_pretty="ReO3",
                symmetry=SimpleNamespace(symbol="Pm-3m"),
                energy_above_hull=0.0,
                is_stable=True,
            ),
        ],
        "Ta2O5": [
            SimpleNamespace(
                material_id="mp-ta2o5-1",
                formula_pretty="Ta2O5",
                symmetry=SimpleNamespace(symbol="P2/m"),
                energy_above_hull=0.2,
                is_stable=False,
            )
        ],
        "TaRe": [
            SimpleNamespace(
                material_id="mp-tare-1",
                formula_pretty="TaRe",
                symmetry=SimpleNamespace(symbol="Im-3m"),
                energy_above_hull=None,
                is_stable=False,
            )
        ],
    }

    frame = build_candidates_df(results)

    assert list(frame.columns) == [
        "target_phase",
        "material_id",
        "formula",
        "space_group",
        "energy_above_hull",
        "is_stable",
    ]
    assert frame["target_phase"].tolist() == ["Ta2O5", "ReO3", "ReO3", "TaRe"]
    assert frame["material_id"].tolist() == [
        "mp-ta2o5-1",
        "mp-reo3-1",
        "mp-reo3-2",
        "mp-tare-1",
    ]


def test_build_candidates_df_surfaces_fallback_sentinel_row():
    from xrd.phases import build_candidates_df

    frame = build_candidates_df(
        {
            "TaRe": [{"fallback": True, "cif_path": Path("data/cif/fallback/TaRe_bcc.cif")}],
            "ReO3": [
                SimpleNamespace(
                    material_id="mp-reo3-1",
                    formula_pretty="ReO3",
                    symmetry=SimpleNamespace(symbol="Pm-3m", number=221),
                    energy_above_hull=0.0,
                    is_stable=True,
                )
            ],
        }
    )

    assert frame["target_phase"].tolist() == ["ReO3", "TaRe"]
    fallback = frame.loc[frame["target_phase"] == "TaRe"].iloc[0]
    assert fallback["material_id"] == "TaRe_fallback"
    assert fallback["formula"] == "TaRe"
    assert fallback["space_group"] == "Im-3m (#229) fallback"
    assert pd.isna(fallback["energy_above_hull"])
    assert bool(fallback["is_stable"]) is False


def test_fallback_row_sorts_last_within_tare_group():
    from xrd.phases import build_candidates_df

    frame = build_candidates_df(
        {
            "TaRe": [
                SimpleNamespace(
                    material_id="mp-t1",
                    formula_pretty="TaRe",
                    symmetry=SimpleNamespace(symbol="Im-3m", number=229),
                    energy_above_hull=0.1,
                    is_stable=False,
                ),
                {"fallback": True, "cif_path": Path("data/cif/fallback/TaRe_bcc.cif")},
            ]
        }
    )

    assert frame["material_id"].tolist() == ["mp-t1", "TaRe_fallback"]


def test_six_column_contract_unchanged_when_fallback_present():
    from xrd.phases import build_candidates_df

    frame = build_candidates_df(
        {
            "TaRe": [{"fallback": True, "cif_path": Path("data/cif/fallback/TaRe_bcc.cif")}],
        }
    )

    assert frame.columns.tolist() == [
        "target_phase",
        "material_id",
        "formula",
        "space_group",
        "energy_above_hull",
        "is_stable",
    ]


def test_find_observed_peaks_returns_dominant_peaks():
    from xrd.phases import find_observed_peaks

    two_theta = np.linspace(35.0, 45.0, 1001)
    intensity = np.full_like(two_theta, 0.2)
    intensity[np.argmin(np.abs(two_theta - 37.65))] = 5.0
    intensity[np.argmin(np.abs(two_theta - 41.70))] = 7.0

    observed = find_observed_peaks(two_theta, intensity, prominence_frac=0.01, min_distance_deg=0.1)

    assert observed == pytest.approx(np.array([37.65, 41.70]), abs=0.01)


def test_filter_peaks_to_observed_keeps_matching_peaks():
    from xrd.phases import filter_peaks_to_observed

    filtered = filter_peaks_to_observed(
        [10.0, 20.0, 30.0],
        observed_peaks=np.array([10.1, 30.4]),
        tol_deg=0.5,
    )

    assert filtered == [10.0, 30.0]


def test_filter_peaks_to_observed_excludes_all_when_no_match():
    from xrd.phases import filter_peaks_to_observed

    filtered = filter_peaks_to_observed([50.0, 60.0], observed_peaks=np.array([10.0]), tol_deg=0.5)

    assert filtered == []


def test_filter_peaks_to_observed_empty_observed_returns_empty():
    from xrd.phases import filter_peaks_to_observed

    filtered = filter_peaks_to_observed([10.0, 20.0], observed_peaks=np.array([]), tol_deg=0.5)

    assert filtered == []


def test_build_phase_peaks_dict_filters_when_observed_peaks_provided(monkeypatch, tmp_path):
    from xrd.phases import build_phase_peaks_dict

    ta_path = tmp_path / "Ta2O5_1_mp-1.cif"
    re_path = tmp_path / "ReO2_1_mp-2.cif"
    ta_path.write_text("ta")
    re_path.write_text("re")

    def fake_peaks(path):
        if "Ta2O5" in Path(path).name:
            return [20.0, 41.4, 41.9, 70.0]
        return [41.1, 60.0]

    monkeypatch.setattr("xrd.phases.peaks_from_cif", fake_peaks)

    phase_peaks = build_phase_peaks_dict(
        {
            "Ta2O5": [ta_path],
            "ReO2": [re_path],
            "ReO3": [],
            "TaRe": [],
            "Al2O3": [],
        },
        observed_peaks=np.array([41.7]),
    )

    assert phase_peaks == {
        "Ta₂O₅": [41.4, 41.9],
        "ReO₂": [],
    }


def test_build_phase_peaks_dict_backward_compatible(monkeypatch, tmp_path):
    from xrd.phases import build_phase_peaks_dict

    ta_path = tmp_path / "Ta2O5_1_mp-1.cif"
    ta_path.write_text("ta")
    monkeypatch.setattr("xrd.phases.peaks_from_cif", lambda _path: [20.0, 41.4, 70.0])

    phase_peaks = build_phase_peaks_dict(
        {
            "Ta2O5": [ta_path],
            "ReO2": [],
            "ReO3": [],
            "TaRe": [],
            "Al2O3": [],
        }
    )

    assert phase_peaks == {"Ta₂O₅": [20.0, 41.4, 70.0]}


def test_save_candidates_csv_writes_expected_columns(tmp_path, monkeypatch):
    from xrd import phases

    frame = pd.DataFrame(
        [
            {
                "target_phase": "Ta2O5",
                "material_id": "mp-1",
                "formula": "Ta2O5",
                "space_group": "P2/m",
                "energy_above_hull": 0.0,
                "is_stable": True,
            }
        ]
    )
    monkeypatch.setattr(phases, "RESULTS_DIR", tmp_path)

    out_path = phases.save_candidates_csv(frame)

    assert out_path == tmp_path / "candidates.csv"
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert saved.to_dict(orient="records") == frame.to_dict(orient="records")


def test_download_top_cifs_uses_ranked_names_and_skips_existing(tmp_path, monkeypatch):
    from xrd import phases

    written = []

    class FakeStructure:
        def __init__(self, marker):
            self.marker = marker

        def to(self, filename):
            path = Path(filename)
            path.write_text(self.marker)
            written.append(path.name)

    doc1 = SimpleNamespace(material_id="mp-1", structure=FakeStructure("first"))
    doc2 = SimpleNamespace(material_id="mp-2", structure=FakeStructure("second"))
    doc3 = SimpleNamespace(material_id="mp-3", structure=FakeStructure("third"))

    monkeypatch.setattr(phases, "TOP_N_CIFS", 2)
    monkeypatch.setattr(phases, "ensure_cif_dir", lambda: tmp_path)

    downloaded = phases.download_top_cifs({"Ta2O5": [doc1, doc2, doc3]}, api_key="unused")

    assert [path.name for path in downloaded["Ta2O5"]] == [
        "Ta2O5_1_mp-1.cif",
        "Ta2O5_2_mp-2.cif",
    ]
    assert written == ["Ta2O5_1_mp-1.cif", "Ta2O5_2_mp-2.cif"]

    written.clear()
    downloaded_again = phases.download_top_cifs({"Ta2O5": [doc1, doc2]}, api_key="unused")
    assert [path.name for path in downloaded_again["Ta2O5"]] == [
        "Ta2O5_1_mp-1.cif",
        "Ta2O5_2_mp-2.cif",
    ]
    assert written == []


def test_search_single_phase_queries_exact_formula_for_oxides():
    from xrd.phases import _search_single_phase

    calls = []

    class FakeSummary:
        def search(self, **kwargs):
            calls.append(kwargs)
            return []

    class FakeMaterials:
        def __init__(self):
            self.summary = FakeSummary()

    class FakeMPRester:
        def __init__(self):
            self.materials = FakeMaterials()

    _search_single_phase(FakeMPRester(), "ReO2")

    assert calls == [
        {
            "formula": "ReO2",
            "fields": [
                "material_id",
                "formula_pretty",
                "symmetry",
                "energy_above_hull",
                "is_stable",
            ],
        }
    ]


def test_search_single_phase_returns_tare_fallback_when_no_bcc_docs():
    from xrd import phases

    non_bcc_doc = SimpleNamespace(
        material_id="mp-tare-nonbcc",
        formula_pretty="TaRe",
        symmetry=SimpleNamespace(number=221),
        energy_above_hull=0.0,
        is_stable=True,
    )

    class FakeSummary:
        def search(self, **kwargs):
            return [non_bcc_doc]

    class FakeMaterials:
        def __init__(self):
            self.summary = FakeSummary()

    class FakeMPRester:
        def __init__(self):
            self.materials = FakeMaterials()

    results = phases._search_single_phase(FakeMPRester(), "TaRe")

    assert results == [{"fallback": True, "cif_path": phases.TARE_FALLBACK_CIF}]


def test_download_top_cifs_copies_fallback_for_tare(tmp_path, monkeypatch):
    from xrd import phases

    fake_bundled = tmp_path / "bundled" / "TaRe_bcc.cif"
    fake_bundled.parent.mkdir(parents=True)
    fake_bundled.write_text("fake cif content")

    monkeypatch.setattr(phases, "TARE_FALLBACK_CIF", fake_bundled)
    monkeypatch.setattr(phases, "ensure_cif_dir", lambda: tmp_path)

    sentinel = {"fallback": True, "cif_path": fake_bundled}
    downloaded = phases.download_top_cifs({"TaRe": [sentinel]}, api_key="unused")

    assert downloaded["TaRe"] == [tmp_path / "TaRe_1_fallback.cif"]
    assert (tmp_path / "TaRe_1_fallback.cif").read_text() == "fake cif content"


def test_download_top_cifs_reuses_existing_fallback_copy(tmp_path, monkeypatch):
    from xrd import phases

    fake_bundled = tmp_path / "bundled" / "TaRe_bcc.cif"
    fake_bundled.parent.mkdir(parents=True)
    fake_bundled.write_text("fake cif content")

    monkeypatch.setattr(phases, "TARE_FALLBACK_CIF", fake_bundled)
    monkeypatch.setattr(phases, "ensure_cif_dir", lambda: tmp_path)

    sentinel = {"fallback": True, "cif_path": fake_bundled}
    phases.download_top_cifs({"TaRe": [sentinel]}, api_key="unused")
    phases.download_top_cifs({"TaRe": [sentinel]}, api_key="unused")

    assert (tmp_path / "TaRe_1_fallback.cif").exists()
