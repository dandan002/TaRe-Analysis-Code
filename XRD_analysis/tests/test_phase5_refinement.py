import importlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def fake_gsasii_path(monkeypatch):
    """Prevent config.py from failing during import."""
    monkeypatch.setenv("GSASII_PATH", "/tmp/fake_gsasii")


@pytest.fixture()
def refinement_module():
    sys.modules.pop("xrd.config", None)
    sys.modules.pop("xrd.refinement", None)
    return importlib.import_module("xrd.refinement")


@pytest.fixture()
def refine_phases_module():
    sys.modules.pop("xrd.config", None)
    sys.modules.pop("xrd.refinement", None)
    sys.modules.pop("refine_phases", None)
    return importlib.import_module("refine_phases")


@pytest.fixture()
def mock_g2sc():
    """Return a MagicMock bundle matching the GSASIIscriptable surface."""
    module_mock = MagicMock(name="G2sc")
    hist_mock = MagicMock(name="hist")
    hist_mock.get_wR.return_value = 12.0
    phase_mock = MagicMock(name="phase")
    phase_mock.get_cell.return_value = [3.0, 3.0, 3.0, 90.0, 90.0, 90.0]
    phase_mock.get_cell_and_esd.return_value = (
        [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
        [0.001, 0.001, 0.001, 0.0, 0.0, 0.0],
    )
    phase_mock.getHAPvalues.return_value = {"Scale": [1.0, 0.35, True]}
    gpx_mock = MagicMock(name="gpx")
    gpx_mock.add_powder_histogram.return_value = hist_mock
    gpx_mock.phase.return_value = phase_mock
    module_mock.G2Project.return_value = gpx_mock
    return module_mock, gpx_mock, hist_mock, phase_mock


def test_setup_project_creates_project_histogram_and_phases(refinement_module, mock_g2sc):
    g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    refinement_module.G2sc = g2sc

    selected_cifs = {"Ta2O5": Path("ta.cif"), "ReO2": Path("reo2.cif")}

    gpx, hist = refinement_module.setup_project(
        selected_cifs=selected_cifs,
        raw_file=Path("TaRe_Full_Oxide.raw"),
        instprm_file=Path("TaRe_Cu_Ka.instprm"),
        gpx_file=Path("TaRe_refinement.gpx"),
    )

    assert gpx is gpx_mock
    assert hist is hist_mock
    g2sc.G2Project.assert_called_once_with(newgpx="TaRe_refinement.gpx")
    gpx_mock.add_powder_histogram.assert_called_once_with(
        "TaRe_Full_Oxide.raw",
        "TaRe_Cu_Ka.instprm",
        fmthint="Bruker RAW",
    )
    assert gpx_mock.add_phase.call_args_list == [
        call("ta.cif", phasename="Ta2O5", fmthint="CIF", histograms=[hist_mock]),
        call("reo2.cif", phasename="ReO2", fmthint="CIF", histograms=[hist_mock]),
    ]
    gpx_mock.save.assert_called_once_with()


def test_run_staged_refinement_tracks_decreasing_rwp(refinement_module, mock_g2sc):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [15.0, 12.0, 11.0, 10.0]

    rwp_history = refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert rwp_history == [15.0, 12.0, 11.0, 10.0]
    assert gpx_mock.do_refinements.call_count == 4


def test_run_staged_refinement_raises_when_rwp_rises(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [15.0, 12.0, 13.5]

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert exc_info.value.stage == 3
    assert exc_info.value.rwp_history == [15.0, 12.0, 13.5]
    assert "rose" in str(exc_info.value)
    assert gpx_mock.do_refinements.call_count == 3


def test_run_staged_refinement_raises_on_stage2_refinement_error(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [18.0, 100.0]
    gpx_mock.do_refinements.side_effect = [
        None,
        RuntimeError("divide by zero encountered in scalar divide"),
    ]

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert exc_info.value.stage == 2
    assert exc_info.value.rwp_history == [18.0, 100.0]
    assert "divide by zero" in str(exc_info.value)
    assert gpx_mock.do_refinements.call_count == 2


def test_run_staged_refinement_reads_gsasii_lst_failure_signals(
    refinement_module, mock_g2sc, tmp_path
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [18.0, 18.0]
    gpx_mock.filename = str(tmp_path / "stage2-failure.gpx")
    lst_file = tmp_path / "stage2-failure.lst"

    def run_stage(_stages):
        if gpx_mock.do_refinements.call_count == 2:
            lst_file.write_text(
                "**** ERROR: Refinement failed ****\n"
                "Note refinement problem:\n"
                "divide by zero encountered in scalar divide\n",
                encoding="utf-8",
            )

    gpx_mock.do_refinements.side_effect = run_stage

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert exc_info.value.stage == 2
    assert "divide by zero" in str(exc_info.value)
    assert gpx_mock.do_refinements.call_count == 2


def test_run_staged_refinement_raises_when_stage2_rwp_stays_pinned_at_100(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [42.0, 100.0]

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert exc_info.value.stage == 2
    assert exc_info.value.rwp_history == [42.0, 100.0]
    assert "100%" in str(exc_info.value)
    assert gpx_mock.do_refinements.call_count == 2


def test_run_staged_refinement_rejects_soft_singularity_signals(
    refinement_module, mock_g2sc, tmp_path
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [18.0, 12.0, 11.0]
    gpx_mock.filename = str(tmp_path / "stage3-soft-singularity.gpx")
    lst_file = tmp_path / "stage3-soft-singularity.lst"

    def run_stage(_stages):
        if gpx_mock.do_refinements.call_count == 3:
            lst_file.write_text(
                "Reported from refinement:\n"
                "Warning: Soft (SVD) singularity in the Hessian\n",
                encoding="utf-8",
            )

    gpx_mock.do_refinements.side_effect = run_stage

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert exc_info.value.stage == 3
    assert "soft" in str(exc_info.value).lower()


def test_stage2_is_scale_only_no_cell_key(refinement_module):
    """Stage 2 must not contain Cell to avoid the 42-variable Hessian singularity."""
    stage2_set = refinement_module.REFINEMENT_STAGES[1].get("set", {})
    assert "Cell" not in stage2_set, f"Stage 2 must refine Scale only. Found: {stage2_set}"
    assert "Scale" in stage2_set, f"Stage 2 must refine Scale. Found: {stage2_set}"


def test_stage3_is_cell_only_no_scale_key(refinement_module):
    """Stage 3 adds Cell; Scale is already on from Stage 2 so it stays."""
    stage3_set = refinement_module.REFINEMENT_STAGES[2].get("set", {})
    assert "Cell" in stage3_set, f"Stage 3 must refine Cell. Found: {stage3_set}"
    assert (
        "Scale" not in stage3_set
    ), f"Stage 3 set-dict should not re-declare Scale. Found: {stage3_set}"


def test_refinement_stages_has_four_entries(refinement_module):
    """Exactly 4 stages are required to avoid singular Hessian at Stage 2."""
    assert len(refinement_module.REFINEMENT_STAGES) == 4


def test_initialize_phase_scales_sets_each_phase_to_film_prior(
    refinement_module, mock_g2sc
):
    """Non-substrate phases each receive the film prior (0.075)."""
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc

    hap_a = {"Scale": [0.0, True]}
    hap_b = {"Scale": [0.0, True]}
    phase_a = MagicMock(name="phase_a")
    phase_a.getHAPvalues.return_value = hap_a
    phase_b = MagicMock(name="phase_b")
    phase_b.getHAPvalues.return_value = hap_b
    gpx_mock.phase.side_effect = lambda name: {"PhaseA": phase_a, "PhaseB": phase_b}[name]

    selected_cifs = {"PhaseA": Path("a.cif"), "PhaseB": Path("b.cif")}
    refinement_module._initialize_phase_scales(gpx_mock, hist_mock, selected_cifs)

    assert hap_a["Scale"][0] == pytest.approx(refinement_module._FILM_PHASE_PRIOR)
    assert hap_b["Scale"][0] == pytest.approx(refinement_module._FILM_PHASE_PRIOR)
    phase_a.setHAPvalues.assert_called_once_with(hap_a, [hist_mock])
    phase_b.setHAPvalues.assert_called_once_with(hap_b, [hist_mock])


def test_initialize_phase_scales_handles_scalar_scale_entry(
    refinement_module, mock_g2sc
):
    """_initialize_phase_scales handles scalar float Scale payloads."""
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hap = {"Scale": 1.0}
    phase_mock = MagicMock(name="scalar_phase")
    phase_mock.getHAPvalues.return_value = hap
    gpx_mock.phase.return_value = phase_mock

    refinement_module._initialize_phase_scales(
        gpx_mock, hist_mock, {"OnlyPhase": Path("x.cif")}
    )

    assert hap["Scale"] == pytest.approx(refinement_module._FILM_PHASE_PRIOR)
    phase_mock.setHAPvalues.assert_called_once_with(hap, [hist_mock])


def test_initialize_phase_scales_uses_substrate_aware_prior(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc

    phases: dict[str, MagicMock] = {}
    for name in ["Al2O3", "Ta", "Ta2O5", "Re", "ReO2"]:
        phase = MagicMock(name=f"{name}_phase")
        phase.getHAPvalues.return_value = {"Scale": [0.0, 0.0, True]}
        phases[name] = phase
    gpx_mock.phase.side_effect = lambda name: phases[name]

    selected_cifs = {name: Path(f"{name}.cif") for name in phases}

    refinement_module._initialize_phase_scales(gpx_mock, hist_mock, selected_cifs)

    assert phases["Al2O3"].getHAPvalues.return_value["Scale"][0] == pytest.approx(0.70)
    for name in ["Ta", "Ta2O5", "Re", "ReO2"]:
        assert phases[name].getHAPvalues.return_value["Scale"][0] == pytest.approx(0.075)


def test_run_staged_refinement_writes_singularity_diagnostics(
    refinement_module, mock_g2sc, tmp_path
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [18.0, 12.0, 12.0]
    gpx_mock.filename = str(tmp_path / "TaRe_refinement.gpx")
    diagnostics_path = tmp_path / "phase5_stage_diagnostics.json"
    selected_cifs = {
        "Al2O3": Path("Al2O3.cif"),
        "Ta": Path("Ta.cif"),
    }

    phase_scales = {
        "Al2O3": {"value": 0.70},
        "Ta": {"value": 0.075},
    }

    def phase_lookup(name: str) -> MagicMock:
        phase = MagicMock(name=f"{name}_phase")

        def get_hapvalues(_hist):
            return {"Scale": [0.0, phase_scales[name]["value"], False]}

        phase.getHAPvalues.side_effect = get_hapvalues
        return phase

    phase_cache = {name: phase_lookup(name) for name in selected_cifs}
    gpx_mock.phase.side_effect = lambda name: phase_cache[name]

    stage_calls = {"count": 0}

    def run_stage(_stages):
        stage_calls["count"] += 1
        if stage_calls["count"] in {2, 3}:
            raise RuntimeError("Soft (SVD) singularity in the Hessian")

    gpx_mock.do_refinements.side_effect = run_stage

    with pytest.raises(refinement_module.RefinementConvergenceError) as exc_info:
        refinement_module.run_staged_refinement(
            gpx_mock,
            hist_mock,
            selected_cifs=selected_cifs,
            diagnostics_path=diagnostics_path,
            histogram_input=Path("results/TaRe_Full_Oxide.csv"),
            histogram_format_hint="comma/tab/semicolon separated",
        )

    assert exc_info.value.stage == 2
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["histogram_input"].endswith("TaRe_Full_Oxide.csv")
    assert diagnostics["histogram_format_hint"] == "comma/tab/semicolon separated"
    assert diagnostics["selected_cifs"] == {
        "Al2O3": "Al2O3.cif",
        "Ta": "Ta.cif",
    }
    assert diagnostics["failure"]["reason"] == "Soft (SVD) singularity in the Hessian"
    assert diagnostics["failure"]["stage"] == 2
    assert diagnostics["gpx_path"].endswith("TaRe_refinement.gpx")
    assert diagnostics["lst_path"].endswith("TaRe_refinement.lst")
    assert [entry["stage"] for entry in diagnostics["stage_history"]] == [1, 2, 2]
    assert diagnostics["stage_history"][0]["phase_scales"]["Al2O3"] == pytest.approx(0.70)


def test_run_staged_refinement_retries_stage2_once_after_invalid_scales(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [18.0, 12.0, 11.5, 11.0, 10.5]

    selected_cifs = {
        "Al2O3": Path("Al2O3.cif"),
        "Ta": Path("Ta.cif"),
        "Re": Path("Re.cif"),
    }
    scale_state = {
        "Al2O3": {"value": 0.70},
        "Ta": {"value": 0.0},
        "Re": {"value": float("nan")},
    }
    scale_snapshots = {
        "Al2O3": [0.70, 0.70, 0.70, 0.68, 0.66],
        "Ta": [0.075, 0.0, 0.17, 0.18, 0.19],
        "Re": [0.075, float("nan"), 0.13, 0.14, 0.15],
    }

    phases: dict[str, MagicMock] = {}
    for name in selected_cifs:
        phase = MagicMock(name=f"{name}_phase")

        def get_hapvalues(_hist, *, phase_name=name):
            return {"Scale": [0.0, scale_state[phase_name]["value"], False]}

        def set_hapvalues(hap, _hists, *, phase_name=name):
            scale_state[phase_name]["value"] = hap["Scale"][0]

        phase.getHAPvalues.side_effect = get_hapvalues
        phase.setHAPvalues.side_effect = set_hapvalues
        phases[name] = phase

    gpx_mock.phase.side_effect = lambda name: phases[name]

    stage_calls = {"count": 0}

    def run_stage(_stages):
        stage_calls["count"] += 1
        call_index = stage_calls["count"]
        for phase_name, snapshots in scale_snapshots.items():
            scale_state[phase_name]["value"] = snapshots[min(call_index - 1, len(snapshots) - 1)]

    gpx_mock.do_refinements.side_effect = run_stage

    rwp_history = refinement_module.run_staged_refinement(
        gpx_mock,
        hist_mock,
        selected_cifs=selected_cifs,
    )

    assert gpx_mock.do_refinements.call_count == 5
    assert len(refinement_module.REFINEMENT_STAGES) == 4
    assert rwp_history == [18.0, 12.0, 11.5, 11.0, 10.5]
    assert phases["Ta"].setHAPvalues.call_count >= 2
    corrected_ta_scale = phases["Ta"].setHAPvalues.call_args_list[-1].args[0]["Scale"][0]
    corrected_re_scale = phases["Re"].setHAPvalues.call_args_list[-1].args[0]["Scale"][0]
    assert corrected_ta_scale > 0
    assert corrected_re_scale > 0


def test_run_staged_refinement_succeeds_with_four_stage_mock(
    refinement_module, mock_g2sc
):
    """Full 4-stage run with mocks returns the complete Rwp history."""
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [15.0, 12.0, 11.0, 10.5]

    rwp_history = refinement_module.run_staged_refinement(gpx_mock, hist_mock)

    assert rwp_history == [15.0, 12.0, 11.0, 10.5]
    assert gpx_mock.do_refinements.call_count == 4


def test_run_staged_refinement_initializes_scales_when_selected_cifs_provided(
    refinement_module, mock_g2sc, monkeypatch
):
    """run_staged_refinement calls _initialize_phase_scales when selected_cifs is given."""
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.side_effect = [15.0, 12.0, 11.0, 10.5]

    calls = []

    def fake_init(gpx, hist, selected_cifs):
        calls.append((gpx, hist, list(selected_cifs.keys())))

    monkeypatch.setattr(refinement_module, "_initialize_phase_scales", fake_init)

    selected_cifs = {"PhaseA": Path("a.cif")}
    refinement_module.run_staged_refinement(gpx_mock, hist_mock, selected_cifs=selected_cifs)

    assert len(calls) == 1
    assert calls[0][2] == ["PhaseA"]


def test_extract_results_uses_cell_esd_and_normalized_hap_scale(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.return_value = 10.5
    other_phase = MagicMock(name="other_phase")
    other_phase.get_cell.return_value = [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
    other_phase.get_cell_and_esd.return_value = (
        [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
        [0.002, 0.002, 0.002, 0.0, 0.0, 0.0],
    )
    other_phase.getHAPvalues.return_value = {"Scale": [0.65, True]}
    gpx_mock.phase.side_effect = lambda name: {
        "Ta2O5": _phase_mock,
        "Al2O3": other_phase,
    }[name]

    results = refinement_module.extract_results(
        gpx_mock,
        hist_mock,
        {"Ta2O5": Path("ta.cif"), "Al2O3": Path("alumina.cif")},
    )

    assert len(results) == 2
    result = results[0]
    assert set(result) == {
        "phase",
        "a_Å",
        "b_Å",
        "c_Å",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "a_esd",
        "b_esd",
        "c_esd",
        "weight_fraction",
        "wf_esd",
        "Rwp",
    }
    assert result["a_esd"] == 0.001
    assert pytest.approx(sum(row["weight_fraction"] for row in results), rel=1e-6) == 1.0
    assert results[0]["weight_fraction"] == pytest.approx(0.35 / (0.35 + 0.65))
    assert results[1]["weight_fraction"] == pytest.approx(0.65 / (0.35 + 0.65))
    assert result["Rwp"] == 10.5


def test_extract_results_rejects_non_positive_normalization_denominator(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, phase_mock = mock_g2sc
    phase_mock.getHAPvalues.return_value = {"Scale": [0.0, True]}

    with pytest.raises(ValueError) as exc_info:
        refinement_module.extract_results(gpx_mock, hist_mock, {"Ta2O5": Path("ta.cif")})

    assert "weight fraction" in str(exc_info.value).lower()


def test_extract_results_falls_back_to_nan_esd_on_attribute_error(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, phase_mock = mock_g2sc
    phase_mock.get_cell_and_esd.side_effect = AttributeError("missing method")

    results = refinement_module.extract_results(gpx_mock, hist_mock, {"Ta2O5": Path("ta.cif")})

    assert math.isnan(results[0]["a_esd"])


def test_extract_results_converts_fractional_rwp_to_percent(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, _phase_mock = mock_g2sc
    hist_mock.get_wR.return_value = 0.105

    results = refinement_module.extract_results(gpx_mock, hist_mock, {"Ta2O5": Path("ta.cif")})

    assert results[0]["Rwp"] == 10.5


def test_config_runtime_guidance_documents_real_phase5_contract(monkeypatch):
    monkeypatch.setenv("GSASII_PATH", "/tmp/fake_gsasii")
    sys.modules.pop("xrd.config", None)

    config = importlib.import_module("xrd.config")

    guidance = config.phase5_runtime_guidance()

    assert "/opt/homebrew/bin/python3.14" in guidance
    assert 'GSASII_PATH="$HOME/GSAS-II-src/backcompat"' in guidance
    assert "TaRe_Cu_Ka.instprm" in guidance
    assert "GSASIIscriptable" in guidance


def test_refinement_bootstrap_imports_gsasiiscriptable_from_gsasii_path(
    monkeypatch, tmp_path
):
    fake_gsasii = tmp_path / "fake_gsasii"
    fake_gsasii.mkdir()
    module_path = fake_gsasii / "GSASIIscriptable.py"
    module_path.write_text(
        "MARKER = 'fake-gsasii-path'\n"
        "class G2Project:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GSASII_PATH", str(fake_gsasii))
    sys.modules.pop("GSASIIscriptable", None)
    sys.modules.pop("xrd.config", None)
    sys.modules.pop("xrd.refinement", None)

    refinement = importlib.import_module("xrd.refinement")

    assert refinement.G2sc.MARKER == "fake-gsasii-path"
    assert sys.path[0] == str(fake_gsasii)


def test_choose_histogram_input_prefers_raw_when_probe_succeeds(refine_phases_module):
    chosen, reason = refine_phases_module.choose_histogram_input(
        raw_file=Path("data/TaRe_Full_Oxide.raw"),
        csv_fallback=Path("results/TaRe_Full_Oxide.csv"),
        probe_reader=lambda path: object(),
    )

    assert chosen == Path("data/TaRe_Full_Oxide.raw")
    assert "RAW" in reason


def test_choose_histogram_input_falls_back_to_csv_after_raw_probe_failure(
    refine_phases_module,
):
    def fail_probe(_path: Path) -> object:
        raise RuntimeError("Bruker RAW reader failed")

    chosen, reason = refine_phases_module.choose_histogram_input(
        raw_file=Path("data/TaRe_Full_Oxide.raw"),
        csv_fallback=Path("results/TaRe_Full_Oxide.csv"),
        probe_reader=fail_probe,
    )

    assert chosen == Path("results/TaRe_Full_Oxide.csv")
    assert "Bruker RAW reader failed" in reason


def test_choose_histogram_input_falls_back_to_csv_after_unusable_raw_probe(
    refine_phases_module,
):
    chosen, reason = refine_phases_module.choose_histogram_input(
        raw_file=Path("data/TaRe_Full_Oxide.raw"),
        csv_fallback=Path("results/TaRe_Full_Oxide.csv"),
        probe_reader=lambda path: None,
    )

    assert chosen == Path("results/TaRe_Full_Oxide.csv")
    assert "unusable" in reason.lower()


def test_save_results_rejects_implausible_phase5_rows(refine_phases_module, tmp_path):
    with pytest.raises(ValueError) as exc_info:
        refine_phases_module.save_results(
            [
                {
                    "phase": "Ta2O5",
                    "a_Å": 4.8,
                    "b_Å": 5.5,
                    "c_Å": 6.8,
                    "alpha_deg": 90.0,
                    "beta_deg": 90.0,
                    "gamma_deg": 90.0,
                    "a_esd": 0.01,
                    "b_esd": 0.01,
                    "c_esd": 0.01,
                    "weight_fraction": 1.0,
                    "wf_esd": math.nan,
                    "Rwp": 100.0,
                },
                {
                    "phase": "Al2O3",
                    "a_Å": 5.1,
                    "b_Å": 5.1,
                    "c_Å": 5.1,
                    "alpha_deg": 55.0,
                    "beta_deg": 55.0,
                    "gamma_deg": 55.0,
                    "a_esd": 0.01,
                    "b_esd": 0.01,
                    "c_esd": 0.01,
                    "weight_fraction": 1.0,
                    "wf_esd": math.nan,
                    "Rwp": 100.0,
                },
            ],
            tmp_path / "rietveld_results.csv",
        )

    message = str(exc_info.value).lower()
    assert "weight fraction" in message or "rwp" in message


def test_save_results_writes_plausible_csv_in_d05_column_order(
    refine_phases_module, tmp_path
):
    output_path = refine_phases_module.save_results(
        [
            {
                "phase": "Ta2O5",
                "a_Å": 4.8,
                "b_Å": 5.5,
                "c_Å": 6.8,
                "alpha_deg": 90.0,
                "beta_deg": 90.0,
                "gamma_deg": 90.0,
                "a_esd": 0.01,
                "b_esd": 0.01,
                "c_esd": 0.01,
                "weight_fraction": 0.25,
                "wf_esd": math.nan,
                "Rwp": 12.0,
            },
            {
                "phase": "Al2O3",
                "a_Å": 5.1,
                "b_Å": 5.1,
                "c_Å": 5.1,
                "alpha_deg": 55.0,
                "beta_deg": 55.0,
                "gamma_deg": 55.0,
                "a_esd": 0.01,
                "b_esd": 0.01,
                "c_esd": 0.01,
                "weight_fraction": 0.75,
                "wf_esd": math.nan,
                "Rwp": 12.0,
            },
        ],
        tmp_path / "rietveld_results.csv",
    )

    written = output_path.read_text(encoding="utf-8").splitlines()
    assert written[0].split(",") == [
        "phase",
        "a_Å",
        "b_Å",
        "c_Å",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "a_esd",
        "b_esd",
        "c_esd",
        "weight_fraction",
        "wf_esd",
        "Rwp",
    ]
    assert "Al2O3" in written[2]


def test_extract_results_supports_dict_shaped_real_gsasii_cell_payload(
    refinement_module, mock_g2sc
):
    _g2sc, gpx_mock, hist_mock, phase_mock = mock_g2sc
    phase_mock.get_cell.return_value = {
        "length_a": 4.5851,
        "length_b": 4.8538,
        "length_c": 5.6609,
        "angle_alpha": 90.0,
        "angle_beta": 90.0,
        "angle_gamma": 90.0,
    }
    phase_mock.get_cell_and_esd.return_value = (
        phase_mock.get_cell.return_value,
        {"length_a": 0.01, "length_b": 0.02, "length_c": 0.03},
    )

    results = refinement_module.extract_results(gpx_mock, hist_mock, {"ReO2": Path("reo2.cif")})

    assert results[0]["a_Å"] == 4.5851
    assert results[0]["b_esd"] == 0.02
