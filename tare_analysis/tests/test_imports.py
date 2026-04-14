# tare_analysis/tests/test_imports.py
def test_config_import():
    from config import DATA_DIR, make_run_dirs
    assert DATA_DIR.parent.name == "tare_analysis"


def test_config_exposes_qubits_data_dir():
    from config import QUBITS_DATA_DIR

    assert QUBITS_DATA_DIR.parent.name == "qubits"


def test_make_run_dirs(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    figs, csvs = config.make_run_dirs()
    assert figs.exists()
    assert csvs.exists()


def test_make_run_dirs_supports_run_suffix(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    figs, csvs = config.make_run_dirs(run_suffix="ETCH")

    assert figs.parent.name.endswith("_ETCH")
    assert csvs.parent == figs.parent

def test_peaks_import():
    from xps.peaks import peakDict, augerDict
    assert 'Ta' in peakDict
    assert 'Re' in peakDict
    assert 'O'  in augerDict
    assert peakDict['Ta']['4f'] == 19
    assert peakDict['Re']['4f'] == 44

def test_funcs_import():
    from xps.funcs import XPSMeas, XPSMeas_ion_milled, backgroundCorrect, iteratedShirleyCorrect
    assert callable(backgroundCorrect)
    assert callable(iteratedShirleyCorrect)
    import numpy as np
    m = XPSMeas(sample='S1', xrayEnergy=1486, element='Ta4f',
                BE=np.array([22.0, 22.5, 23.0]),
                intensity=np.array([100.0, 200.0, 150.0]),
                intensityErr=np.array([10.0, 14.0, 12.0]))
    assert m.sample == 'S1'
    assert m.element == 'Ta4f'

def test_import_module_importable():
    from xps.import_ import importSST, importIOS, importIOS_v2, import_ThermoAlpha, importNeo
    assert callable(importSST)
    assert callable(importIOS)
    assert callable(importIOS_v2)
    assert callable(import_ThermoAlpha)
    assert callable(importNeo)

def test_style_import():
    from plots.style import OKABE_ITO, SPECIES_COLOR, GROUP_COLOR, FIG_SINGLE
    assert 'blue' in OKABE_ITO
    assert 'BOE' in GROUP_COLOR
    assert FIG_SINGLE == (3.375, 2.8)

def test_plots_importable():
    from plots import spectra, fit_components, summary
    assert callable(spectra.plot_element_overlays)
    assert callable(fit_components.plot_ta4f_fit_components)
    assert callable(summary.stacked_fraction_plot_ta4f)


def test_summary_plot_module_exposes_timecourse_plotters():
    from plots import summary

    assert hasattr(summary, "plot_timecourse_with_group_difference")
    assert hasattr(summary, "plot_cabrera_mott_fit")


def test_summary_plot_module_exposes_depth_profile_plotter():
    from plots import summary

    assert hasattr(summary, "plot_depth_profile_by_sample")


def test_etch_main_module_importable():
    import etch_main

    assert callable(etch_main.main)


def test_ta4f_initial_parameters_match_notebook_constraints():
    import pytest
    import etch_main

    params = etch_main._build_ta4f_initial_parameters()

    assert params["Ta5_7b2_center"].value == pytest.approx(27.6)
    assert params["Ta5_7b2_center"].min == pytest.approx(26.0)
    assert params["Ta5_7b2_center"].max == pytest.approx(28.0)
    assert params["Ta5_7b2_center"].vary is True

    assert params["metal_7b2_center"].value == pytest.approx(21.95)
    assert params["metal_7b2_center"].min == pytest.approx(21.0)
    assert params["metal_7b2_center"].max == pytest.approx(22.2)
    assert params["metal_7b2_center"].vary is True

    assert params["_7b2_5b2_offset"].value == pytest.approx(29.24 - 27.35)
    assert params["_7b2_5b2_offset"].min == pytest.approx(1.0)
    assert params["_7b2_5b2_offset"].max == pytest.approx(3.0)
    assert params["_7b2_5b2_offset"].vary is True

    assert params["metal_7b2_amplitude"].value == pytest.approx(0.25)
    assert params["interface_7b2_center"].expr == "interface_offset+metal_7b2_center"
    assert params["alloy_7b2_center"].expr == "alloy_offset+metal_7b2_center"
    assert "alloy_7b2_skew" not in params
    assert "alloy_5b2_skew" not in params


def test_fit_corrected_series_builds_fresh_parameters_for_each_scan(monkeypatch):
    from types import SimpleNamespace
    import etch_main

    corrected = [SimpleNamespace(sample="ReTa03", etchlevel=1), SimpleNamespace(sample="ReTa03", etchlevel=2)]
    built_ids = []
    used_ids = []

    def fake_builder():
        params = object()
        built_ids.append(id(params))
        return params

    def fake_minimize(objective, params, args, method, nan_policy):
        used_ids.append(id(params))
        return SimpleNamespace(params={}, redchi=1.0)

    monkeypatch.setattr(etch_main.lmfit, "minimize", fake_minimize)

    out = etch_main._fit_corrected_series(
        corrected,
        objective=lambda *_args, **_kwargs: 0.0,
        params_builder=fake_builder,
        model_dict={},
    )

    assert len(out) == 2
    assert used_ids == built_ids
    assert len(set(used_ids)) == 2


def test_etch_main_wires_per_sample_etch_plotters_and_metric_outputs():
    import ast
    import inspect
    import etch_main

    tree = ast.parse(inspect.getsource(etch_main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    calls = []
    filename_prefixes = []
    metric_values = []
    fit_builders = []
    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call):
            if isinstance(stmt.func, ast.Attribute):
                calls.append(stmt.func.attr)
                for kw in stmt.keywords:
                    if kw.arg == "filename_prefix":
                        filename_prefixes.append(ast.literal_eval(kw.value))
                    if kw.arg == "metric":
                        metric_values.append(ast.literal_eval(kw.value))
            elif isinstance(stmt.func, ast.Name) and stmt.func.id == "_fit_corrected_series":
                for kw in stmt.keywords:
                    if kw.arg == "params_builder" and isinstance(kw.value, ast.Name):
                        fit_builders.append(kw.value.id)
                        break

    assert calls.count("plot_etch_profile_per_sample") == 3
    assert calls.count("plot_etch_metric_evolution_by_sample") == 4
    assert "plot_oxide_thickness" not in calls
    assert "plot_peaks_vs_ev" not in calls
    assert "plot_peaks_vs_ev_merged" not in calls
    assert "plot_depth_profile_by_sample" not in calls
    assert "plot_etch_oxide_thickness_by_sample" in calls
    assert "ta4f_thickness_depth_profile" in filename_prefixes
    assert "re4f_thickness_depth_profile" in filename_prefixes
    assert "be_shift_depth_profile" in filename_prefixes
    assert "Ta4f_binding_energy_by_species" in filename_prefixes
    assert "Ta4f_amplitude_by_species" in filename_prefixes
    assert "Re4f_binding_energy_by_species" in filename_prefixes
    assert "Re4f_amplitude_by_species" in filename_prefixes
    assert metric_values.count("binding_energy") == 2
    assert metric_values.count("amplitude") == 2
    assert set(metric_values) == {"binding_energy", "amplitude"}
    assert fit_builders == [
        "_build_ta4f_initial_parameters",
        "_build_re4f_initial_parameters",
    ]


def test_etch_main_wires_results_directories_and_depth_profile_exports():
    import ast
    import inspect
    import etch_main

    tree = ast.parse(inspect.getsource(etch_main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    fig_subdirs = None
    csv_writes = set()

    for stmt in main_fn.body:
        if isinstance(stmt, ast.For) and fig_subdirs is None:
            if isinstance(stmt.iter, ast.List):
                fig_subdirs = [
                    elt.value for elt in stmt.iter.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "to_csv"
                and call.args
                and isinstance(call.args[0], ast.BinOp)
                and isinstance(call.args[0].right, ast.Constant)
            ):
                csv_writes.add(call.args[0].right.value)

    assert "etch_profiles" in fig_subdirs
    assert "ta4f_thickness_depth_profile.csv" in csv_writes
    assert "re4f_thickness_depth_profile.csv" in csv_writes
    assert "be_shift_depth_profile.csv" in csv_writes


def test_etch_main_uses_etch_run_suffix_and_reference_profile_exports():
    import ast
    import inspect
    import etch_main

    tree = ast.parse(inspect.getsource(etch_main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    make_run_dirs_calls = []
    dict_keys = set()

    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name):
            if stmt.func.id == "make_run_dirs":
                make_run_dirs_calls.append(
                    {kw.arg: ast.literal_eval(kw.value) for kw in stmt.keywords if kw.arg}
                )
        if isinstance(stmt, ast.Dict):
            for key in stmt.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    dict_keys.add(key.value)

    assert make_run_dirs_calls == [{"run_suffix": "ETCH"}]
    assert "reference_profile" in dict_keys
    assert "sample_type" not in dict_keys
def test_etch_main_builds_be_shift_plot_frame_per_sample_species_series():
    import pandas as pd
    import etch_main

    df = pd.DataFrame(
        [
            {"sample": "ReTa03", "species": "Ta metal", "spin": "7/2", "depth_order": 0, "delta_eV": 0.2, "stderr_eV": 0.01},
            {"sample": "ReTa03", "species": "Ta metal", "spin": "7/2", "depth_order": 1, "delta_eV": 0.1, "stderr_eV": 0.02},
            {"sample": "ReTa03", "species": "Re metal", "spin": "7/2", "depth_order": 0, "delta_eV": -0.1, "stderr_eV": 0.01},
            {"sample": "ReTa03", "species": "Re metal", "spin": "7/2", "depth_order": 1, "delta_eV": -0.2, "stderr_eV": 0.02},
            {"sample": "ReTa03", "species": "Ta metal", "spin": "5/2", "depth_order": 0, "delta_eV": 0.3, "stderr_eV": 0.03},
        ]
    )

    out = etch_main._build_be_shift_depth_plot_frame(df)

    assert set(out["sample"]) == {"ReTa03 - Ta metal", "ReTa03 - Re metal"}
    assert set(out["spin"]) == {"7/2"}
    assert list(out[out["sample"] == "ReTa03 - Ta metal"]["depth_order"]) == [0, 1]
    assert list(out[out["sample"] == "ReTa03 - Re metal"]["depth_order"]) == [0, 1]


def test_etch_main_omits_week_based_exports_for_etched_runs():
    import ast
    import inspect
    import etch_main

    tree = ast.parse(inspect.getsource(etch_main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    csv_writes = set()
    plot_calls = []

    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
            if stmt.func.attr == "to_csv" and stmt.args:
                arg = stmt.args[0]
                if (
                    isinstance(arg, ast.BinOp)
                    and isinstance(arg.op, ast.Div)
                    and isinstance(arg.right, ast.Constant)
                    and isinstance(arg.right.value, str)
                ):
                    csv_writes.add(arg.right.value)
            plot_calls.append(stmt.func.attr)

    forbidden_csvs = {
        "ta4f_thickness_timecourse.csv",
        "re4f_thickness_timecourse.csv",
        "ta4f_thickness_baseline_change.csv",
        "re4f_thickness_baseline_change.csv",
        "ta4f_thickness_group_differences.csv",
        "re4f_thickness_group_differences.csv",
        "be_shift_timecourse.csv",
        "be_shift_by_group.csv",
    }

    assert csv_writes.isdisjoint(forbidden_csvs)
    assert "plot_timecourse_with_group_difference" not in plot_calls
    assert "plot_be_shift_boe_vs_control" not in plot_calls


def test_etch_main_exposes_etch_amplitude_reference_label():
    import etch_main

    assert etch_main._etch_amplitude_reference_label() == "Control-derived reference"


def test_plot_timecourse_with_group_difference_writes_png_and_sorts_difference_weeks(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_timecourse_with_group_difference

    summary_df = pd.DataFrame(
        [
            {"group": "BOE", "week": 0, "mean": 1.0, "sem": 0.1},
            {"group": "BOE", "week": 1, "mean": 1.5, "sem": 0.2},
            {"group": "Control", "week": 0, "mean": 2.0, "sem": 0.1},
            {"group": "Control", "week": 1, "mean": 2.4, "sem": 0.2},
        ]
    )
    diff_df = pd.DataFrame(
        [
            {"week": 2, "difference": 0.6, "difference_sem": 0.05},
            {"week": 0, "difference": -1.0, "difference_sem": 0.04},
            {"week": 1, "difference": -0.8, "difference_sem": 0.03},
        ]
    )

    calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        calls.append({"x": list(x), "y": list(y)})
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    plot_timecourse_with_group_difference(
        summary_df,
        diff_df,
        title="Demo",
        ylabel="Value",
        filename="demo.png",
        out_dir=out_dir,
    )

    png = out_dir / "demo.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert calls[2]["x"] == [0, 1, 2]


def test_plot_cabrera_mott_fit_writes_png_and_sorts_week_series(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_cabrera_mott_fit

    summary_df = pd.DataFrame(
        [
            {"group": "BOE", "week": 2, "mean": 1.8, "sem": 0.15},
            {"group": "Control", "week": 1, "mean": 0.9, "sem": 0.08},
            {"group": "BOE", "week": 0, "mean": 1.2, "sem": 0.10},
            {"group": "Control", "week": 0, "mean": 0.8, "sem": 0.07},
            {"group": "BOE", "week": 1, "mean": 1.5, "sem": 0.12},
        ]
    )
    fit_result = {
        "x0": 1.1,
        "x0_err": 0.1,
        "k": 0.5,
        "k_err": 0.05,
        "tau": 0.7,
        "tau_err": 0.08,
        "t_fit": [0.0, 1.0, 2.0],
        "x_fit": [1.1, 1.45, 1.7],
        "redchi": 0.9,
    }

    errorbar_calls = []
    plot_calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_plot = matplotlib.axes.Axes.plot

    def recording_errorbar(self, x, y, *args, **kwargs):
        errorbar_calls.append({"x": list(x), "y": list(y), "label": kwargs.get("label")})
        return original_errorbar(self, x, y, *args, **kwargs)

    def recording_plot(self, x, y, *args, **kwargs):
        plot_calls.append({"x": list(x), "y": list(y), "label": kwargs.get("label")})
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    plot_cabrera_mott_fit(
        summary_df,
        fit_result,
        element="Ta",
        filename="ta_cm.png",
        out_dir=out_dir,
    )

    png = out_dir / "ta_cm.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert errorbar_calls[0]["x"] == [0, 1]
    assert errorbar_calls[0]["label"] == "Control mean"
    assert errorbar_calls[1]["x"] == [0, 1, 2]
    assert errorbar_calls[1]["label"] == "BOE mean"
    assert plot_calls[0]["x"] == [0.0, 1.0, 2.0]
    assert plot_calls[0]["label"] == "Cabrera-Mott fit"


def test_plot_depth_profile_by_sample_writes_png_and_sorts_depth_order(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_depth_profile_by_sample

    df = pd.DataFrame(
        [
            {"sample": "ReTa04", "etchlevel": 2, "etchtime": 20.0, "depth_order": 2, "oxide_thickness_nm": 2.2, "thickness_err_nm": 0.2},
            {"sample": "ReTa03", "etchlevel": 2, "etchtime": 20.0, "depth_order": 2, "oxide_thickness_nm": 1.2, "thickness_err_nm": 0.2},
            {"sample": "ReTa04", "etchlevel": None, "etchtime": None, "depth_order": 0, "oxide_thickness_nm": 3.0, "thickness_err_nm": 0.3},
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "depth_order": 0, "oxide_thickness_nm": 2.0, "thickness_err_nm": 0.3},
            {"sample": "ReTa04", "etchlevel": 1, "etchtime": 10.0, "depth_order": 1, "oxide_thickness_nm": 2.6, "thickness_err_nm": 0.2},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "depth_order": 1, "oxide_thickness_nm": 1.6, "thickness_err_nm": 0.2},
        ]
    )

    calls = []
    label_calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels

    def recording_errorbar(self, x, y, *args, **kwargs):
        calls.append({"x": list(x), "y": list(y), "kwargs": dict(kwargs)})
        return original_errorbar(self, x, y, *args, **kwargs)

    def recording_set_xticklabels(self, labels, *args, **kwargs):
        label_calls.append(list(labels))
        return original_set_xticklabels(self, labels, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", recording_set_xticklabels)

    out_dir = tmp_path / "etch_profiles"
    out_dir.mkdir()
    plot_depth_profile_by_sample(
        df,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
        ylabel="Oxide Thickness (nm)",
        filename="ta4f_thickness_depth_profile.png",
        out_dir=out_dir,
    )

    png = out_dir / "ta4f_thickness_depth_profile.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert len(calls) == 2
    assert len(label_calls) == 2
    assert calls[0]["x"] == [0, 1, 2]
    assert calls[0]["y"] == [2.0, 1.6, 1.2]
    assert calls[1]["x"] == [0, 1, 2]
    assert calls[1]["y"] == [3.0, 2.6, 2.2]
    assert label_calls[0] == ["Unetched", "Lv1\n(10s)", "Lv2\n(20s)"]
    assert label_calls[1] == ["Unetched", "Lv1\n(10s)", "Lv2\n(20s)"]


def test_plot_depth_profile_by_sample_without_error_column_omits_yerr(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_depth_profile_by_sample

    df = pd.DataFrame(
        [
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "depth_order": 0, "oxide_thickness_nm": 2.0},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "depth_order": 1, "oxide_thickness_nm": 1.6},
        ]
    )

    calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    out_dir = tmp_path / "etch_profiles_no_err"
    out_dir.mkdir()
    plot_depth_profile_by_sample(
        df,
        value_col="oxide_thickness_nm",
        ylabel="Oxide Thickness (nm)",
        filename="ta4f_thickness_depth_profile_no_err.png",
        out_dir=out_dir,
    )

    png = out_dir / "ta4f_thickness_depth_profile_no_err.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert len(calls) == 1
    assert "yerr" not in calls[0]


def test_plot_metric_evolution_by_species_splits_control_and_etched_panels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import matplotlib.axes
    import matplotlib.figure
    from plots.summary import plot_metric_evolution_by_species

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="Control_1124", etchlevel=None),
        SimpleNamespace(sample="BOE_1124", etchlevel=None),
        SimpleNamespace(sample="Control_1201", etchlevel=None),
        SimpleNamespace(sample="BOE_1201", etchlevel=None),
    ]
    fit_results = [
        SimpleNamespace(params={
            "metal_7b2_center": param(22.0, 0.1),
            "Redp1_7b2_center": param(22.4, 0.1),
            "Ta5_7b2_center": param(26.7, 0.1),
            "metal_7b2_amplitude": param(0.30, 0.02),
            "Redp1_7b2_amplitude": param(0.20, 0.01),
            "Ta5_7b2_amplitude": param(0.16, 0.01),
        }),
        SimpleNamespace(params={
            "metal_7b2_center": param(21.8, 0.1),
            "Redp1_7b2_center": param(22.2, 0.1),
            "Ta5_7b2_center": param(26.5, 0.1),
            "metal_7b2_amplitude": param(0.28, 0.02),
            "Redp1_7b2_amplitude": param(0.18, 0.01),
            "Ta5_7b2_amplitude": param(0.15, 0.01),
        }),
        SimpleNamespace(params={
            "metal_7b2_center": param(22.1, 0.1),
            "Redp1_7b2_center": param(22.5, 0.1),
            "Ta5_7b2_center": param(26.8, 0.1),
            "metal_7b2_amplitude": param(0.32, 0.02),
            "Redp1_7b2_amplitude": param(0.21, 0.01),
            "Ta5_7b2_amplitude": param(0.17, 0.01),
        }),
        SimpleNamespace(params={
            "metal_7b2_center": param(21.9, 0.1),
            "Redp1_7b2_center": param(22.3, 0.1),
            "Ta5_7b2_center": param(26.6, 0.1),
            "metal_7b2_amplitude": param(0.29, 0.02),
            "Redp1_7b2_amplitude": param(0.19, 0.01),
            "Ta5_7b2_amplitude": param(0.14, 0.01),
        }),
    ]

    errorbar_calls = []
    label_calls = []
    title_calls = []
    suptitles = []
    legend_calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels
    original_set_title = matplotlib.axes.Axes.set_title
    original_suptitle = matplotlib.figure.Figure.suptitle
    original_legend = matplotlib.axes.Axes.legend

    def recording_errorbar(self, x, y, *args, **kwargs):
        errorbar_calls.append(
            {
                "x": list(x),
                "y": list(y),
                "yerr": list(kwargs.get("yerr", [])),
                "fmt": kwargs.get("fmt"),
                "label": kwargs.get("label"),
                "color": kwargs.get("color"),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    def recording_set_xticklabels(self, labels, *args, **kwargs):
        label_calls.append(list(labels))
        return original_set_xticklabels(self, labels, *args, **kwargs)

    def recording_set_title(self, title, *args, **kwargs):
        title_calls.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def recording_suptitle(self, title, *args, **kwargs):
        suptitles.append(title)
        return original_suptitle(self, title, *args, **kwargs)

    def recording_legend(self, *args, **kwargs):
        legend_calls.append(
            {
                "loc": kwargs.get("loc"),
                "bbox_to_anchor": kwargs.get("bbox_to_anchor"),
                "borderaxespad": kwargs.get("borderaxespad"),
                "fontsize": kwargs.get("fontsize"),
            }
        )
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", recording_set_xticklabels)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", recording_suptitle)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", recording_legend)

    out_dir = tmp_path / "peaks_vs_samples"
    out_dir.mkdir()
    plot_metric_evolution_by_species(
        fit_results,
        corrected_data,
        ["metal", "Redp1", "Ta5"],
        "Ta4f",
        {"metal": "#1f77b4", "Redp1": "#ff7f0e", "Ta5": "#d62728"},
        metric="binding_energy",
        filename="ta4f_binding_energy_evolution_by_species.png",
        out_dir=out_dir,
    )

    png = out_dir / "ta4f_binding_energy_evolution_by_species.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert errorbar_calls == [
        {
            "x": [0, 1],
            "y": [21.8, 21.9],
            "yerr": [0.1, 0.1],
            "fmt": "o-",
            "label": "metal",
            "color": "#1f77b4",
        },
        {
            "x": [0, 1],
            "y": [26.5, 26.6],
            "yerr": [0.1, 0.1],
            "fmt": "o-",
            "label": "Ta5",
            "color": "#d62728",
        },
        {
            "x": [0, 1],
            "y": [22.0, 22.1],
            "yerr": [0.1, 0.1],
            "fmt": "o-",
            "label": "metal",
            "color": "#1f77b4",
        },
        {
            "x": [0, 1],
            "y": [26.7, 26.8],
            "yerr": [0.1, 0.1],
            "fmt": "o-",
            "label": "Ta5",
            "color": "#d62728",
        },
    ]
    assert label_calls[:2] == [
        ["BOE_1124", "BOE_1201"],
        ["Control_1124", "Control_1201"],
    ]
    assert title_calls[:2] == [
        "Growth After BOE Etch",
        "Control (6M Oxide Growth)",
    ]
    assert legend_calls == [
        {"loc": "upper left", "bbox_to_anchor": (1.02, 1), "borderaxespad": 0, "fontsize": 9},
    ]
    assert suptitles == ["Ta4f - Binding Energy Evolution by Species"]


def test_plot_metric_evolution_by_species_rejects_mismatched_lengths(tmp_path):
    from types import SimpleNamespace
    import pytest
    from plots.summary import plot_metric_evolution_by_species

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    fit_results = [
        SimpleNamespace(params={
            "metal_7b2_center": param(22.0, 0.1),
            "metal_7b2_amplitude": param(0.30, 0.02),
        }),
    ]
    corrected_data = [
        SimpleNamespace(sample="Control_1124", etchlevel=None),
        SimpleNamespace(sample="BOE_1124", etchlevel=None),
    ]

    with pytest.raises(ValueError, match="same length"):
        plot_metric_evolution_by_species(
            fit_results,
            corrected_data,
            ["metal"],
            "Ta4f",
            {"metal": "#1f77b4"},
            metric="binding_energy",
            filename="ta4f_binding_energy_evolution_by_species.png",
            out_dir=tmp_path,
        )


def test_plot_metric_evolution_by_species_rejects_unknown_samples(tmp_path):
    from types import SimpleNamespace
    import pytest
    from plots.summary import plot_metric_evolution_by_species

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="CTRL_WK0", etchlevel=None),
    ]
    fit_results = [
        SimpleNamespace(params={
            "metal_7b2_center": param(22.0, 0.1),
        }),
    ]

    with pytest.raises(ValueError, match="Unknown"):
        plot_metric_evolution_by_species(
            fit_results,
            corrected_data,
            ["metal"],
            "Ta4f",
            {"metal": "#1f77b4"},
            metric="binding_energy",
            filename="ta4f_binding_energy_evolution_by_species.png",
            out_dir=tmp_path,
        )


def test_plot_metric_evolution_by_species_supports_amplitude_metric(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import matplotlib.axes
    from plots.summary import plot_metric_evolution_by_species

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="Control_1124", etchlevel=None),
        SimpleNamespace(sample="BOE_1124", etchlevel=None),
    ]
    fit_results = [
        SimpleNamespace(params={
            "metal_7b2_center": param(22.0, 0.1),
            "metal_7b2_amplitude": param(0.30, 0.02),
        }),
        SimpleNamespace(params={
            "metal_7b2_center": param(21.8, 0.1),
            "metal_7b2_amplitude": param(0.28, 0.02),
        }),
    ]

    errorbar_calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        errorbar_calls.append(
            {
                "x": list(x),
                "y": list(y),
                "fmt": kwargs.get("fmt"),
                "label": kwargs.get("label"),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    plot_metric_evolution_by_species(
        fit_results,
        corrected_data,
        ["metal"],
        "Ta4f",
        {"metal": "#1f77b4"},
        metric="amplitude",
        filename="ta4f_amplitude_evolution_by_species.png",
        out_dir=tmp_path,
    )

    assert errorbar_calls == [
        {"x": [0], "y": [0.28], "fmt": "s-", "label": "metal"},
        {"x": [0], "y": [0.3], "fmt": "s-", "label": "metal"},
    ]


def test_plot_normalized_amplitude_by_species_uses_fraction_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import matplotlib.axes
    from plots.summary import plot_normalized_amplitude_by_species

    corrected_data = [
        SimpleNamespace(sample="Control_1124", etchlevel=None),
        SimpleNamespace(sample="BOE_1124", etchlevel=None),
    ]
    area_fracs = [
        {"metal": 0.6, "Redp1": 0.1},
        {"metal": 0.4, "Redp1": 0.2},
    ]
    area_errs = [
        {"metal": 0.06, "Redp1": 0.01},
        {"metal": 0.04, "Redp1": 0.02},
    ]

    errorbar_calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        errorbar_calls.append(
            {
                "x": list(x),
                "y": list(y),
                "fmt": kwargs.get("fmt"),
                "label": kwargs.get("label"),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    plot_normalized_amplitude_by_species(
        area_fracs,
        area_errs,
        corrected_data,
        ["metal", "Redp1"],
        "Ta4f",
        {"metal": "#1f77b4", "Redp1": "#aaaaaa"},
        filename="ta4f_amplitude_normalized.png",
        out_dir=tmp_path,
    )

    assert errorbar_calls == [
        {"x": [0], "y": [0.4], "fmt": "s-", "label": "metal"},
        {"x": [0], "y": [0.6], "fmt": "s-", "label": "metal"},
    ]
    assert (tmp_path / "ta4f_amplitude_normalized.png").exists()


def test_main_wires_time_aware_thickness_exports():
    import ast
    import inspect

    import main

    tree = ast.parse(inspect.getsource(main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    imported_names = set()
    fig_subdirs = None
    assignments = {}
    to_csv_writes = {}
    timecourse_plot_calls = []
    cm_plot_calls = []

    def path_filename(expr):
        if (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.op, ast.Div)
            and isinstance(expr.left, ast.Name)
            and expr.left.id == "csv_dir"
            and isinstance(expr.right, ast.Constant)
            and isinstance(expr.right.value, str)
        ):
            return expr.right.value
        return None

    def call_name(expr):
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                return expr.func.id
            if isinstance(expr.func, ast.Attribute):
                return expr.func.attr
        return None

    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "analysis.statistics":
            imported_names.update(alias.name for alias in stmt.names)

    for stmt in main_fn.body:
        if isinstance(stmt, ast.For) and fig_subdirs is None:
            if (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id == "sub"
                and isinstance(stmt.iter, ast.List)
            ):
                values = [
                    elt.value for elt in stmt.iter.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                if {
                    "element_overlays",
                    "fit_components",
                    "area_fractions",
                    "peak_comparisons",
                    "peaks_vs_samples",
                    "thickness",
                    "be_shifts",
                    "timecourse",
                }.issubset(values):
                    fig_subdirs = values

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            assignments[target] = stmt.value

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "to_csv"
                and call.args
            ):
                receiver = call.func.value
                receiver_name = receiver.id if isinstance(receiver, ast.Name) else None
                if receiver_name and receiver_name in {
                    "ta_thickness_df",
                    "re_thickness_df",
                    "ta_timecourse",
                    "re_timecourse",
                    "ta_baseline",
                    "re_baseline",
                    "ta_group_diff",
                    "re_group_diff",
                }:
                    to_csv_writes[path_filename(call.args[0])] = receiver_name
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "plot_timecourse_with_group_difference"
            ):
                kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
                timecourse_plot_calls.append(
                    (
                        ast.unparse(call.args[0]),
                        ast.unparse(call.args[1]),
                        ast.literal_eval(kwargs["filename"]),
                        ast.unparse(kwargs["out_dir"]),
                    )
                )
    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
            if stmt.func.attr == "plot_cabrera_mott_fit":
                kwargs = {kw.arg: kw.value for kw in stmt.keywords if kw.arg}
                cm_plot_calls.append(
                    (
                        ast.unparse(stmt.args[0]),
                        ast.unparse(stmt.args[1]),
                        ast.unparse(kwargs["element"]),
                        ast.unparse(kwargs["filename"]),
                        ast.unparse(kwargs["out_dir"]),
                    )
                )

    assert {
        "build_timecourse_summary",
        "build_baseline_change",
        "build_group_week_differences",
    }.issubset(imported_names)
    assert fig_subdirs is not None and "timecourse" in fig_subdirs
    assert call_name(assignments.get("ta_thickness_df")) == "DataFrame"
    assert call_name(assignments.get("re_thickness_df")) == "DataFrame"
    assert ast.unparse(assignments.get("ta_thickness_df").args[0]) == "ta_thick_rows"
    assert ast.unparse(assignments.get("re_thickness_df").args[0]) == "re_thick_rows"
    assert call_name(assignments.get("ta_timecourse")) == "build_timecourse_summary"
    assert call_name(assignments.get("ta_baseline")) == "build_baseline_change"
    assert call_name(assignments.get("ta_group_diff")) == "build_group_week_differences"
    assert call_name(assignments.get("re_timecourse")) == "build_timecourse_summary"
    assert call_name(assignments.get("re_baseline")) == "build_baseline_change"
    assert call_name(assignments.get("re_group_diff")) == "build_group_week_differences"
    assert ast.unparse(assignments.get("ta_timecourse").args[0]) == "ta_thickness_df"
    assert ast.unparse(assignments.get("ta_baseline").args[0]) == "ta_thickness_df"
    assert ast.unparse(assignments.get("ta_group_diff").args[0]) == "ta_thickness_df"
    assert ast.unparse(assignments.get("re_timecourse").args[0]) == "re_thickness_df"
    assert ast.unparse(assignments.get("re_baseline").args[0]) == "re_thickness_df"
    assert ast.unparse(assignments.get("re_group_diff").args[0]) == "re_thickness_df"
    assert to_csv_writes == {
        "ta4f_thickness_timecourse.csv": "ta_timecourse",
        "re4f_thickness_timecourse.csv": "re_timecourse",
        "ta4f_thickness_baseline_change.csv": "ta_baseline",
        "re4f_thickness_baseline_change.csv": "re_baseline",
        "ta4f_thickness_group_differences.csv": "ta_group_diff",
        "re4f_thickness_group_differences.csv": "re_group_diff",
    }
    assert timecourse_plot_calls == [
        ("ta_timecourse", "ta_group_diff", "ta4f_thickness_timecourse.png", "figs_dir / 'timecourse'"),
        ("re_timecourse", "re_group_diff", "re4f_thickness_timecourse.png", "figs_dir / 'timecourse'"),
    ]
    assert cm_plot_calls == [
        ("timecourse_df", "cm", "element", "f'{element.lower()}_cabrera_mott_fit.png'", "figs_dir / 'timecourse'"),
    ]


def test_main_wires_four_metric_specific_species_evolution_plots():
    import ast
    import inspect
    import main

    tree = ast.parse(inspect.getsource(main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    new_calls = []
    legacy_calls = []

    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
            if stmt.func.attr in {"plot_peaks_vs_ev", "plot_peaks_vs_ev_merged"}:
                legacy_calls.append(stmt.func.attr)
            if stmt.func.attr == "plot_metric_evolution_by_species":
                kwargs = {kw.arg: kw.value for kw in stmt.keywords if kw.arg}
                new_calls.append(
                    (
                        ast.unparse(stmt.args[0]),
                        ast.unparse(stmt.args[1]),
                        ast.unparse(stmt.args[2]),
                        ast.literal_eval(stmt.args[3]),
                        ast.unparse(stmt.args[4]),
                        ast.literal_eval(kwargs["metric"]),
                        ast.literal_eval(kwargs["filename"]),
                        ast.unparse(kwargs["out_dir"]),
                    )
                )

    assert legacy_calls == []
    assert new_calls == [
        (
            "ta4f_fitResults",
            "Ta4fCorrected",
            "ta4f_speciesOrder",
            "Ta4f",
            "ta4f_colors",
            "binding_energy",
            "ta4f_binding_energy_evolution_by_species.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
        (
            "ta4f_fitResults",
            "Ta4fCorrected",
            "ta4f_speciesOrder",
            "Ta4f",
            "ta4f_colors",
            "amplitude",
            "ta4f_amplitude_evolution_by_species.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
        (
            "re4f_fitResults",
            "Re4fCorrected",
            "re4f_speciesOrder",
            "Re4f",
            "re4f_colors",
            "binding_energy",
            "re4f_binding_energy_evolution_by_species.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
        (
            "re4f_fitResults",
            "Re4fCorrected",
            "re4f_speciesOrder",
            "Re4f",
            "re4f_colors",
            "amplitude",
            "re4f_amplitude_evolution_by_species.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
    ]


def test_main_wires_normalized_amplitude_csvs_and_plots():
    import ast
    import inspect
    import main

    tree = ast.parse(inspect.getsource(main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    csv_filenames = []
    normalized_plot_calls = []

    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
            if stmt.func.attr == "to_csv" and stmt.args:
                arg0 = stmt.args[0]
                if (
                    isinstance(arg0, ast.BinOp)
                    and isinstance(arg0.op, ast.Div)
                    and isinstance(arg0.left, ast.Name)
                    and arg0.left.id == "csv_dir"
                    and isinstance(arg0.right, ast.Constant)
                    and isinstance(arg0.right.value, str)
                ):
                    csv_filenames.append(arg0.right.value)

            if stmt.func.attr == "plot_normalized_amplitude_by_species":
                kwargs = {kw.arg: kw.value for kw in stmt.keywords if kw.arg}
                normalized_plot_calls.append(
                    (
                        ast.unparse(stmt.args[0]),
                        ast.unparse(stmt.args[1]),
                        ast.unparse(stmt.args[2]),
                        ast.unparse(stmt.args[3]),
                        ast.literal_eval(stmt.args[4]),
                        ast.unparse(stmt.args[5]),
                        ast.literal_eval(kwargs["filename"]),
                        ast.unparse(kwargs["out_dir"]),
                    )
                )

    assert "ta4f_normalized_amplitudes.csv" in csv_filenames
    assert "re4f_normalized_amplitudes.csv" in csv_filenames
    assert normalized_plot_calls == [
        (
            "ta4f_areaFrac",
            "ta4f_areaErr",
            "Ta4fCorrected",
            "ta4f_speciesOrder",
            "Ta4f",
            "ta4f_colors",
            "ta4f_amplitude_normalized.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
        (
            "re4f_areaFrac",
            "re4f_areaErr",
            "Re4fCorrected",
            "re4f_speciesOrder",
            "Re4f",
            "re4f_colors",
            "re4f_amplitude_normalized.png",
            "figs_dir / 'peaks_vs_samples'",
        ),
    ]


def test_main_uses_safe_minimize_for_both_fit_loops():
    import ast
    import inspect
    import main

    tree = ast.parse(inspect.getsource(main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    methods = []
    for stmt in ast.walk(main_fn):
        if not isinstance(stmt, ast.Call):
            continue
        if not isinstance(stmt.func, ast.Name):
            continue
        if stmt.func.id != "safe_minimize":
            continue

        kwargs = {kw.arg: kw.value for kw in stmt.keywords if kw.arg}
        methods.append(ast.literal_eval(kwargs["method"]))

    assert methods == ["bfgs", "bfgs"]


def test_main_wires_postetch_core_overlay_plotter():
    import ast
    import inspect
    import main

    tree = ast.parse(inspect.getsource(main))
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    spectra_calls = [
        stmt.func.attr
        for stmt in ast.walk(main_fn)
        if isinstance(stmt, ast.Call)
        and isinstance(stmt.func, ast.Attribute)
        and isinstance(stmt.func.value, ast.Name)
        and stmt.func.value.id == "plot_spectra"
    ]

    assert "plot_survey_overlays" in spectra_calls
    assert "plot_postetch_core_overlays" in spectra_calls
    assert "plot_element_overlays" not in spectra_calls
