from types import SimpleNamespace


def test_fit_component_plots_use_human_readable_titles_and_gray_data_fit(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    from plots.fit_components import plot_ta4f_fit_components, plot_re4f_fit_components

    class StubModel:
        def __init__(self, value):
            self.value = value

        def eval(self, params, x):
            import numpy as np
            return np.full_like(x, self.value, dtype=float)

    data = SimpleNamespace(
        sample="BOE_WK1",
        BE=np.array([30.0, 29.0, 28.0]),
        intensity=np.array([10.0, 12.0, 11.0]),
        intensityErr=np.array([1.0, 1.0, 1.0]),
    )
    result = SimpleNamespace(params={})
    model_dict = {
        "metal_7_2": StubModel(1.0),
        "metal_5_2": StubModel(0.8),
        "interface_7_2": StubModel(0.6),
        "interface_5_2": StubModel(0.4),
        "alloy_7_2": StubModel(0.3),
        "alloy_5_2": StubModel(0.2),
        "Ta5_7_2": StubModel(0.5),
        "Ta5_5_2": StubModel(0.4),
        "Ta1_7_2": StubModel(0.2),
        "Ta1_5_2": StubModel(0.1),
        "Ta3_7_2": StubModel(0.2),
        "Ta3_5_2": StubModel(0.1),
        "Re_metal_7_2": StubModel(1.0),
        "Re_metal_5_2": StubModel(0.8),
        "ReO2_7_2": StubModel(0.6),
        "ReO2_5_2": StubModel(0.4),
        "ReO3_7_2": StubModel(0.3),
        "ReO3_5_2": StubModel(0.2),
        "Re2O7_7_2": StubModel(0.2),
        "Re2O7_5_2": StubModel(0.1),
    }

    titles = []
    errorbar_colors = []
    line_colors = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_plot = matplotlib.axes.Axes.plot

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def recording_errorbar(self, *args, **kwargs):
        errorbar_colors.append(kwargs.get("color"))
        return original_errorbar(self, *args, **kwargs)

    def recording_plot(self, *args, **kwargs):
        color = kwargs.get("color")
        if color is None and len(args) >= 3 and isinstance(args[2], str):
            color = args[2]
        line_colors.append(color)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)

    out_dir = tmp_path / "fit_components"
    out_dir.mkdir()
    plot_ta4f_fit_components(
        SimpleNamespace(**{**data.__dict__, "sample": "BOE_WK0"}),
        result,
        model_dict,
        out_dir=out_dir,
        title="Ta4f - BOE_WK0",
    )
    plot_re4f_fit_components(
        SimpleNamespace(**{**data.__dict__, "sample": "Control_WK1"}),
        result,
        model_dict,
        out_dir=out_dir,
        title="Re4f - Control_WK1",
    )

    assert (out_dir / "Ta4f_-_BOE_WK0.png").exists()
    assert (out_dir / "Re4f_-_Control_WK1.png").exists()
    assert titles[:2] == ["Ta4f 0WK Post-Etch", "Re4f Control WK1"]
    assert errorbar_colors[:2] == ["gray", "gray"]
    assert "gray" in line_colors


def test_plot_postetch_core_overlays_writes_requested_four_plots_with_week_labels(tmp_path, monkeypatch):
    from plots import spectra

    scans = [
        SimpleNamespace(sample="BOE_WK0", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="BOE_WK1", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="BOE_WK2", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK0", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK1", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK2", element="Ta4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="BOE_WK0", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="BOE_WK1", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="BOE_WK2", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK0", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK1", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
        SimpleNamespace(sample="Control_WK2", element="Re4f", etchlevel=None, BE=[1, 2], intensity=[2, 1]),
    ]
    ind_dict = {
        "Ta4f": list(range(0, 6)),
        "Re4f": list(range(6, 12)),
    }

    calls = []

    def recording_plot_subset(plot_scans, inds, title, out_path, totalNorm=False):
        calls.append(
            {
                "samples": [plot_scans[i].sample for i in inds],
                "labels": [spectra._overlay_week_label(plot_scans[i].sample) for i in inds],
                "title": title,
                "filename": out_path.name,
                "total_norm": totalNorm,
            }
        )
        out_path.touch()

    monkeypatch.setattr(spectra, "_plot_overlay_subset", recording_plot_subset)

    out_dir = tmp_path / "element_overlays"
    out_dir.mkdir()
    spectra.plot_postetch_core_overlays(scans, ind_dict, out_dir=out_dir)

    assert (out_dir / "ta4f_postetch_overlay.png").exists()
    assert (out_dir / "re4f_postetch_overlay.png").exists()
    assert (out_dir / "ta4f_control_overlay.png").exists()
    assert (out_dir / "re4f_control_overlay.png").exists()
    assert calls == [
        {
            "samples": ["BOE_WK0", "BOE_WK1", "BOE_WK2"],
            "labels": ["WK0", "WK1", "WK2"],
            "title": "Ta4f Spectra (post-etch)",
            "filename": "ta4f_postetch_overlay.png",
            "total_norm": False,
        },
        {
            "samples": ["BOE_WK0", "BOE_WK1", "BOE_WK2"],
            "labels": ["WK0", "WK1", "WK2"],
            "title": "Re4f Spectra (post-etch)",
            "filename": "re4f_postetch_overlay.png",
            "total_norm": False,
        },
        {
            "samples": ["Control_WK0", "Control_WK1", "Control_WK2"],
            "labels": ["WK0", "WK1", "WK2"],
            "title": "Ta4f Spectra (Control)",
            "filename": "ta4f_control_overlay.png",
            "total_norm": False,
        },
        {
            "samples": ["Control_WK0", "Control_WK1", "Control_WK2"],
            "labels": ["WK0", "WK1", "WK2"],
            "title": "Re4f Spectra (Control)",
            "filename": "re4f_control_overlay.png",
            "total_norm": False,
        },
    ]


def test_stacked_fraction_plot_ta4f_splits_boe_and_control_panels(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import stacked_fraction_plot_ta4f

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "etchlevel": None, "metal_frac": 0.2, "Ta5_frac": 0.8},
            {"sample": "BOE_WK1", "etchlevel": 1, "metal_frac": 0.3, "Ta5_frac": 0.7},
            {"sample": "Control_WK0", "etchlevel": None, "metal_frac": 0.4, "Ta5_frac": 0.6},
            {"sample": "Control_WK1", "etchlevel": 1, "metal_frac": 0.5, "Ta5_frac": 0.5},
        ]
    )

    titles = []
    legends = []
    xticklabels = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_legend = matplotlib.axes.Axes.legend
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def recording_legend(self, *args, **kwargs):
        legends.append(kwargs)
        return original_legend(self, *args, **kwargs)

    def recording_set_xticklabels(self, labels, *args, **kwargs):
        xticklabels.append(list(labels))
        return original_set_xticklabels(self, labels, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", recording_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", recording_set_xticklabels)

    out_dir = tmp_path / "area_fractions"
    out_dir.mkdir()
    stacked_fraction_plot_ta4f(df, ["metal", "Ta5"], out_dir=out_dir)

    assert (out_dir / "ta4f_stacked_fractions.png").exists()
    assert titles == ["Growth After BOE Etch", "Control (6M Oxide Growth)"]
    assert xticklabels == [["WK0", "WK1\nLv1"], ["WK0", "WK1\nLv1"]]
    assert legends[0]["bbox_to_anchor"] == (1.02, 1)


def test_stacked_fraction_plot_re4f_splits_boe_and_control_panels(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import stacked_fraction_plot_re4f

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "etchlevel": None, "Re_metal_frac": 0.2, "ReO3_frac": 0.8},
            {"sample": "BOE_WK1", "etchlevel": 1, "Re_metal_frac": 0.3, "ReO3_frac": 0.7},
            {"sample": "Control_WK0", "etchlevel": None, "Re_metal_frac": 0.4, "ReO3_frac": 0.6},
            {"sample": "Control_WK1", "etchlevel": 1, "Re_metal_frac": 0.5, "ReO3_frac": 0.5},
        ]
    )

    titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)

    out_dir = tmp_path / "area_fractions"
    out_dir.mkdir()
    stacked_fraction_plot_re4f(df, ["Re_metal", "ReO3"], out_dir=out_dir)

    assert (out_dir / "re4f_stacked_fractions.png").exists()
    assert titles == ["Growth After BOE Etch", "Control (6M Oxide Growth)"]


def test_plot_etch_profile_per_sample_writes_one_file_per_sample_and_sorts_etchlevel(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_etch_profile_per_sample

    df = pd.DataFrame(
        [
            {"sample": "ReTa04", "etchlevel": 2, "etchtime": 20.0, "oxide_thickness_nm": 2.2, "thickness_err_nm": 0.2},
            {"sample": "ReTa03", "etchlevel": 2, "etchtime": 20.0, "oxide_thickness_nm": 1.2, "thickness_err_nm": 0.2},
            {"sample": "ReTa04", "etchlevel": None, "etchtime": None, "oxide_thickness_nm": 3.0, "thickness_err_nm": 0.3},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "oxide_thickness_nm": 1.6, "thickness_err_nm": 0.2},
            {"sample": "ReTa04", "etchlevel": 1, "etchtime": 10.0, "oxide_thickness_nm": 2.6, "thickness_err_nm": 0.2},
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "oxide_thickness_nm": 2.0, "thickness_err_nm": 0.3},
        ]
    )

    calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        calls.append({"x": list(x), "y": list(y), "markersize": kwargs.get("markersize")})
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    out_dir = tmp_path / "etch_profiles"
    out_dir.mkdir()
    plot_etch_profile_per_sample(
        df,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
        ylabel="Oxide Thickness (nm)",
        filename_prefix="ta4f_thickness_depth_profile",
        out_dir=out_dir,
    )

    assert (out_dir / "ta4f_thickness_depth_profile_ReTa03.png").exists()
    assert (out_dir / "ta4f_thickness_depth_profile_ReTa04.png").exists()
    assert calls[0]["x"] == [1.0, 2.0]
    assert calls[0]["y"] == [1.6, 1.2]
    assert calls[1]["x"] == [1.0, 2.0]
    assert calls[1]["y"] == [2.6, 2.2]
    assert all(call["markersize"] == 5 for call in calls)


def test_plot_etch_metric_evolution_by_sample_writes_per_sample_files(tmp_path, monkeypatch):
    import matplotlib.axes
    from plots.summary import plot_etch_metric_evolution_by_sample

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="ReTa03", etchlevel=2, etchtime=20.0),
        SimpleNamespace(sample="ReTa03", etchlevel=1, etchtime=10.0),
        SimpleNamespace(sample="ReTa04", etchlevel=1, etchtime=10.0),
        SimpleNamespace(sample="ReTa04", etchlevel=2, etchtime=20.0),
        SimpleNamespace(sample="ReTa04", etchlevel=None, etchtime=None),
    ]
    fit_results = [
        SimpleNamespace(params={"metal_7b2_center": param(22.2, 0.1), "Ta5_7b2_center": param(26.7, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(22.0, 0.1), "Ta5_7b2_center": param(26.5, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(21.8, 0.1), "Ta5_7b2_center": param(26.4, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(21.7, 0.1), "Ta5_7b2_center": param(26.3, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(21.6, 0.1), "Ta5_7b2_center": param(26.2, 0.1)}),
    ]

    calls = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, x, y, *args, **kwargs):
        calls.append({"x": list(x), "markersize": kwargs.get("markersize")})
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)

    out_dir = tmp_path / "peaks_vs_samples"
    out_dir.mkdir()
    plot_etch_metric_evolution_by_sample(
        fit_results,
        corrected_data,
        ["metal", "Ta5"],
        "Ta4f",
        {"metal": "#1f77b4", "Ta5": "#d62728"},
        metric="binding_energy",
        filename_prefix="Ta4f_binding_energy_by_species",
        out_dir=out_dir,
    )

    assert (out_dir / "Ta4f_binding_energy_by_species_ReTa03.png").exists()
    assert (out_dir / "Ta4f_binding_energy_by_species_ReTa04.png").exists()
    assert calls[0]["x"] == [1.0, 2.0]
    assert calls[1]["x"] == [1.0, 2.0]
    assert calls[2]["x"] == [1.0, 2.0]
    assert calls[3]["x"] == [1.0, 2.0]
    assert all(call["markersize"] == 5 for call in calls)


def test_plot_peaks_vs_ev_merged_uses_single_right_side_legend(tmp_path, monkeypatch):
    import matplotlib.axes
    from plots.summary import plot_peaks_vs_ev_merged

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="BOE_1124", etchlevel=None),
        SimpleNamespace(sample="Control_1124", etchlevel=1),
    ]
    fit_results = [
        SimpleNamespace(
            params={
                "metal_7b2_center": param(21.8, 0.1),
                "metal_7b2_amplitude": param(0.30, 0.02),
                "Ta5_7b2_center": param(26.5, 0.1),
                "Ta5_7b2_amplitude": param(0.14, 0.01),
            }
        ),
        SimpleNamespace(
            params={
                "metal_7b2_center": param(21.9, 0.1),
                "metal_7b2_amplitude": param(0.31, 0.02),
                "Ta5_7b2_center": param(26.6, 0.1),
                "Ta5_7b2_amplitude": param(0.15, 0.01),
            }
        ),
    ]

    legends = []
    original_legend = matplotlib.axes.Axes.legend

    def recording_legend(self, *args, **kwargs):
        legends.append(kwargs)
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", recording_legend)

    out_dir = tmp_path / "peaks_vs_samples"
    out_dir.mkdir()
    plot_peaks_vs_ev_merged(
        fit_results,
        corrected_data,
        ["metal", "Ta5"],
        "Ta4f",
        {"metal": "#1f77b4", "Ta5": "#d62728"},
        out_dir=out_dir,
    )

    assert (out_dir / "Ta4f_peaks_vs_samples_merged.png").exists()
    assert legends == [
        {"loc": "upper left", "bbox_to_anchor": (1.02, 1), "borderaxespad": 0, "fontsize": 9}
    ]


def test_plot_etch_oxide_thickness_by_sample_splits_boe_and_control_panels(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_etch_oxide_thickness_by_sample

    ta_df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "etchlevel": 1, "oxide_thickness_nm": 1.2, "thickness_err_nm": 0.2},
            {"sample": "BOE_WK1", "etchlevel": 3, "oxide_thickness_nm": 1.6, "thickness_err_nm": 0.2},
            {"sample": "Control_WK0", "etchlevel": 0, "oxide_thickness_nm": 2.6, "thickness_err_nm": 0.2},
            {"sample": "Control_WK1", "etchlevel": 1, "oxide_thickness_nm": 2.2, "thickness_err_nm": 0.2},
        ]
    )
    re_df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "etchlevel": 1, "oxide_thickness_nm": 0.8, "thickness_err_nm": 0.1},
            {"sample": "BOE_WK1", "etchlevel": 3, "oxide_thickness_nm": 0.9, "thickness_err_nm": 0.1},
            {"sample": "Control_WK0", "etchlevel": 0, "oxide_thickness_nm": 1.3, "thickness_err_nm": 0.1},
            {"sample": "Control_WK1", "etchlevel": 1, "oxide_thickness_nm": 1.1, "thickness_err_nm": 0.1},
        ]
    )

    titles = []
    ticks = []
    xticklabels = []
    legends = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_set_xticks = matplotlib.axes.Axes.set_xticks
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels
    original_legend = matplotlib.axes.Axes.legend

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def recording_set_xticks(self, xticks, *args, **kwargs):
        ticks.append(list(xticks))
        return original_set_xticks(self, xticks, *args, **kwargs)

    def recording_set_xticklabels(self, labels, *args, **kwargs):
        xticklabels.append(list(labels))
        return original_set_xticklabels(self, labels, *args, **kwargs)

    def recording_legend(self, *args, **kwargs):
        legends.append(kwargs)
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", recording_set_xticks)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", recording_set_xticklabels)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", recording_legend)

    out_dir = tmp_path / "thickness"
    out_dir.mkdir()
    plot_etch_oxide_thickness_by_sample(ta_df, re_df, out_dir=out_dir)

    assert (out_dir / "oxide_thickness_comparison.png").exists()
    assert titles == ["Growth After BOE Etch", "Control (6M Oxide Growth)"]
    assert ticks == [[0, 1], [0, 1]]
    assert xticklabels == [["WK0\nLv1", "WK1\nLv3"], ["WK0\nLv0", "WK1\nLv1"]]
    assert legends[0]["bbox_to_anchor"] == (1.02, 1)


def test_plot_oxide_thickness_writes_one_plot_per_element_with_boe_then_control_order(tmp_path, monkeypatch):
    import matplotlib.axes
    from plots.summary import plot_oxide_thickness

    ta_corrected = [
        SimpleNamespace(sample="Control_WK1"),
        SimpleNamespace(sample="BOE_WK0"),
        SimpleNamespace(sample="Control_WK0"),
        SimpleNamespace(sample="BOE_WK1"),
    ]
    re_corrected = [
        SimpleNamespace(sample="Control_WK1"),
        SimpleNamespace(sample="BOE_WK0"),
        SimpleNamespace(sample="Control_WK0"),
        SimpleNamespace(sample="BOE_WK1"),
    ]

    titles = []
    ticks = []
    xticklabels = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_set_xticks = matplotlib.axes.Axes.set_xticks
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def recording_set_xticks(self, xticks, *args, **kwargs):
        ticks.append(list(xticks))
        return original_set_xticks(self, xticks, *args, **kwargs)

    def recording_set_xticklabels(self, labels, *args, **kwargs):
        xticklabels.append(list(labels))
        return original_set_xticklabels(self, labels, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", recording_set_xticks)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", recording_set_xticklabels)

    out_dir = tmp_path / "thickness"
    out_dir.mkdir()
    plot_oxide_thickness(
        ta_corrected,
        [2.2, 1.1, 2.0, 1.4],
        [0.2, 0.1, 0.2, 0.1],
        re_corrected,
        [1.4, 0.8, 1.2, 0.9],
        [0.1, 0.1, 0.1, 0.1],
        out_dir=out_dir,
    )

    assert (out_dir / "ta_oxide_thickness_comparison.png").exists()
    assert (out_dir / "re_oxide_thickness_comparison.png").exists()
    assert not (out_dir / "oxide_thickness_comparison.png").exists()
    assert titles == ["Tantalum Oxide Layer Thickness", "Rhenium Oxide Layer Thickness"]
    assert ticks == [[0, 1, 3, 4], [0, 1, 3, 4]]
    assert xticklabels == [
        ["WK0", "WK1", "WK0", "WK1"],
        ["WK0", "WK1", "WK0", "WK1"],
    ]


def test_plot_depth_profile_by_sample_uses_week_labels_in_titles(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.axes
    from plots.summary import plot_depth_profile_by_sample

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "depth_order": 0, "etchlevel": None, "etchtime": None, "oxide_thickness_nm": 2.1},
            {"sample": "BOE_WK0", "depth_order": 1, "etchlevel": 1, "etchtime": 10.0, "oxide_thickness_nm": 1.8},
            {"sample": "Control_WK1", "depth_order": 0, "etchlevel": None, "etchtime": None, "oxide_thickness_nm": 1.5},
        ]
    )

    titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)

    out_dir = tmp_path / "depth_profiles"
    out_dir.mkdir()
    plot_depth_profile_by_sample(
        df,
        value_col="oxide_thickness_nm",
        ylabel="Oxide Thickness (nm)",
        filename="depth_profile.png",
        out_dir=out_dir,
    )

    assert (out_dir / "depth_profile.png").exists()
    assert titles == ["WK0", "WK1"]


def test_plot_etch_metric_evolution_by_sample_uses_week_labels_in_titles(tmp_path, monkeypatch):
    import matplotlib.axes
    from plots.summary import plot_etch_metric_evolution_by_sample

    def param(value, stderr):
        return SimpleNamespace(value=value, stderr=stderr)

    corrected_data = [
        SimpleNamespace(sample="BOE_WK0", etchlevel=2, etchtime=20.0),
        SimpleNamespace(sample="BOE_WK0", etchlevel=1, etchtime=10.0),
        SimpleNamespace(sample="Control_WK1", etchlevel=1, etchtime=10.0),
    ]
    fit_results = [
        SimpleNamespace(params={"metal_7b2_center": param(22.2, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(22.0, 0.1)}),
        SimpleNamespace(params={"metal_7b2_center": param(21.8, 0.1)}),
    ]

    titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def recording_set_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", recording_set_title)

    out_dir = tmp_path / "peaks_vs_samples"
    out_dir.mkdir()
    plot_etch_metric_evolution_by_sample(
        fit_results,
        corrected_data,
        ["metal"],
        "Ta4f",
        {"metal": "#1f77b4"},
        metric="binding_energy",
        filename_prefix="Ta4f_binding_energy_by_species",
        out_dir=out_dir,
    )

    assert (out_dir / "Ta4f_binding_energy_by_species_BOE_WK0.png").exists()
    assert (out_dir / "Ta4f_binding_energy_by_species_Control_WK1.png").exists()
    assert titles == [
        "Ta4f - Binding Energy Evolution\nWK0",
        "Ta4f - Binding Energy Evolution\nWK1",
    ]
