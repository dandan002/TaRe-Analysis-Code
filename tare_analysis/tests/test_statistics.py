# tare_analysis/tests/test_statistics.py
import numpy as np
import pandas as pd

def test_sample_group():
    from analysis.statistics import sample_group
    assert sample_group('BOE_1124') == 'BOE'
    assert sample_group('Control_1124') == 'Control'
    assert sample_group('Mystery') == 'Unknown'

def test_sample_week_extracts_integer_from_standard_name():
    from analysis.statistics import sample_week
    assert sample_week("BOE_WK3") == 3
    assert sample_week("Control_WK0") == 0

def test_sample_week_returns_nan_when_missing():
    from analysis.statistics import sample_week
    import numpy as np
    assert np.isnan(sample_week("CTRL"))

def test_sample_timepoint_key_maps_baseline_label_to_week_zero_pairing_key():
    from analysis.statistics import sample_timepoint_key
    assert sample_timepoint_key("BOE_WK0") == ("BOE", 0)
    assert sample_timepoint_key("Control_WK2") == ("Control", 2)


def test_sample_display_label_shortens_standard_week_names():
    from analysis.statistics import sample_display_label
    assert sample_display_label("BOE_WK0") == "WK0"
    assert sample_display_label("Control_WK2") == "WK2"
    assert sample_display_label("ReTa03") == "ReTa03"

def test_weighted_mean_shift_equal_weights():
    from analysis.statistics import weighted_mean_shift
    mean, unc = weighted_mean_shift([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    assert abs(mean - 1.0) < 1e-10
    assert unc > 0

def test_weighted_mean_shift_floor():
    from analysis.statistics import weighted_mean_shift, STDERR_FLOOR
    # stderrs of 0 should be floored, not cause divide-by-zero
    mean, unc = weighted_mean_shift([0.5, 0.5], [0.0, 0.0])
    assert abs(mean - 0.5) < 1e-10

def test_pearson_annot_small_n():
    from analysis.statistics import pearson_annot
    result = pearson_annot([1.0], [1.0])
    assert 'N/A' in result

def test_pearson_annot_significant():
    from analysis.statistics import pearson_annot
    xs = [1, 2, 3, 4, 5]
    ys = [1, 2, 3, 4, 5]
    result = pearson_annot(xs, ys)
    assert 'significant' in result
    assert 'r = 1.00' in result


def test_build_timecourse_summary_computes_group_means_by_week():
    import pandas as pd
    from analysis.statistics import build_timecourse_summary

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "metric": 1.0},
            {"sample": "BOE_WK1", "metric": 2.0},
            {"sample": "Control_WK0", "metric": 3.0},
            {"sample": "Control_WK1", "metric": 5.0},
        ]
    )

    out = build_timecourse_summary(df, value_col="metric")
    assert set(out.columns) == {"group", "week", "mean", "std", "count", "sem"}
    assert set(out["group"]) == {"BOE", "Control"}
    assert set(out["week"]) == {0, 1}
    assert out.loc[(out["group"] == "BOE") & (out["week"] == 1), "mean"].iloc[0] == 2.0


def test_build_baseline_change_uses_week_zero_within_group():
    import pandas as pd
    from analysis.statistics import build_baseline_change

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "metric": 1.0},
            {"sample": "BOE_WK1", "metric": 2.5},
            {"sample": "Control_WK0", "metric": 4.0},
            {"sample": "Control_WK1", "metric": 3.5},
        ]
    )

    out = build_baseline_change(df, value_col="metric")
    assert out.loc[out["sample"] == "BOE_WK1", "delta_from_wk0"].iloc[0] == 1.5
    assert out.loc[out["sample"] == "Control_WK1", "delta_from_wk0"].iloc[0] == -0.5


def test_build_group_week_differences_only_uses_common_weeks():
    import pandas as pd
    from analysis.statistics import build_group_week_differences

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "metric": 1.0},
            {"sample": "BOE_WK1", "metric": 2.0},
            {"sample": "BOE_WK3", "metric": 9.0},
            {"sample": "Control_WK0", "metric": 3.0},
            {"sample": "Control_WK1", "metric": 6.0},
        ]
    )

    out = build_group_week_differences(df, value_col="metric")
    assert set(out["week"]) == {0, 1}
    assert 3 not in set(out["week"])


def test_build_baseline_change_collapses_repeated_week_zero_rows():
    import pandas as pd
    from analysis.statistics import build_baseline_change

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "metric": 1.0},
            {"sample": "BOE_WK0", "metric": 3.0},
            {"sample": "BOE_WK1", "metric": 4.0},
            {"sample": "Control_WK0", "metric": 2.0},
        ]
    )

    out = build_baseline_change(df, value_col="metric")
    assert len(out) == len(df)
    assert set(out.columns) == {"sample", "metric", "group", "week", "baseline_value", "delta_from_wk0"}
    assert out.loc[out["sample"] == "BOE_WK1", "baseline_value"].iloc[0] == 2.0
    assert out.loc[out["sample"] == "BOE_WK1", "delta_from_wk0"].iloc[0] == 2.0


def test_build_baseline_change_drops_unlabeled_samples_before_baseline():
    import pandas as pd
    from analysis.statistics import build_baseline_change

    df = pd.DataFrame(
        [
            {"sample": "BOE_1124", "metric": 9.0},
            {"sample": "BOE_WK0", "metric": 1.0},
            {"sample": "BOE_WK1", "metric": 4.0},
        ]
    )

    out = build_baseline_change(df, value_col="metric")
    assert len(out) == 2
    assert "BOE_1124" not in set(out["sample"])
    assert not out["week"].isna().any()


def test_build_be_shift_timecourse_summarizes_by_element_species_group_and_week():
    import pandas as pd
    from analysis.statistics import build_be_shift_timecourse

    df = pd.DataFrame(
        [
            {"sample": "BOE_WK0", "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": 0.1},
            {"sample": "BOE_WK1", "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": 0.2},
            {"sample": "Control_WK0", "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": 0.3},
        ]
    )

    out = build_be_shift_timecourse(df)
    assert set(out["week"]) == {0, 1}
    assert set(out["group"]) == {"BOE", "Control"}


def test_etch_sort_key_orders_unetched_before_etched_then_by_level_and_time():
    from analysis.statistics import etch_sort_key

    rows = [
        {"etchlevel": 3, "etchtime": 45.0},
        {"etchlevel": None, "etchtime": None},
        {"etchlevel": 1, "etchtime": 30.0},
        {"etchlevel": 1, "etchtime": 10.0},
    ]

    ordered = sorted(rows, key=lambda row: etch_sort_key(row["etchlevel"], row["etchtime"]))
    assert ordered == [
        {"etchlevel": None, "etchtime": None},
        {"etchlevel": 1, "etchtime": 10.0},
        {"etchlevel": 1, "etchtime": 30.0},
        {"etchlevel": 3, "etchtime": 45.0},
    ]


def test_build_depth_profile_summary_sorts_within_sample_and_preserves_metrics():
    import pandas as pd
    from analysis.statistics import build_depth_profile_summary

    df = pd.DataFrame(
        [
            {"sample": "ReTa03", "etchlevel": 2, "etchtime": 20.0, "oxide_thickness_nm": 1.2, "thickness_err_nm": 0.2},
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "oxide_thickness_nm": 2.1, "thickness_err_nm": 0.3},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "oxide_thickness_nm": 1.8, "thickness_err_nm": 0.2},
            {"sample": "ReTa04", "etchlevel": 1, "etchtime": 15.0, "oxide_thickness_nm": 0.9, "thickness_err_nm": 0.1},
        ]
    )

    out = build_depth_profile_summary(
        df,
        value_col="oxide_thickness_nm",
        error_col="thickness_err_nm",
    )

    reta03 = out[out["sample"] == "ReTa03"]
    assert pd.isna(reta03.iloc[0]["etchlevel"])
    assert list(reta03["etchlevel"].iloc[1:]) == [1, 2]
    assert list(reta03["depth_order"]) == [0, 1, 2]
    assert reta03.iloc[2]["oxide_thickness_nm"] == 1.2

    reta04 = out[out["sample"] == "ReTa04"]
    assert list(reta04["depth_order"]) == [0]


def test_build_depth_profile_summary_raises_when_error_column_is_missing():
    import pandas as pd
    import pytest
    from analysis.statistics import build_depth_profile_summary

    df = pd.DataFrame(
        [
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "oxide_thickness_nm": 1.8},
        ]
    )

    with pytest.raises(KeyError):
        build_depth_profile_summary(
            df,
            value_col="oxide_thickness_nm",
            error_col="thickness_err_nm",
        )


def test_build_be_shift_depth_profile_sorts_by_sample_species_spin_and_depth():
    import pandas as pd
    from analysis.statistics import build_be_shift_depth_profile

    df = pd.DataFrame(
        [
            {"sample": "ReTa03", "etchlevel": 2, "etchtime": 20.0, "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": -0.1, "stderr_eV": 0.02},
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": 0.3, "stderr_eV": 0.03},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 10.0, "element": "Ta4f", "species": "Ta metal", "spin": "7/2", "delta_eV": 0.1, "stderr_eV": 0.02},
            {"sample": "ReTa03", "etchlevel": None, "etchtime": None, "element": "O1s", "species": "Oxide", "spin": "1/2", "delta_eV": 0.7, "stderr_eV": 0.05},
            {"sample": "ReTa03", "etchlevel": 1, "etchtime": 8.0, "element": "O1s", "species": "Oxide", "spin": "1/2", "delta_eV": 0.4, "stderr_eV": 0.04},
        ]
    )

    out = build_be_shift_depth_profile(df)
    ta4f = out[(out["element"] == "Ta4f") & (out["species"] == "Ta metal")]
    assert pd.isna(ta4f.iloc[0]["etchlevel"])
    assert list(ta4f["etchlevel"].iloc[1:]) == [1, 2]
    assert list(ta4f["depth_order"]) == [0, 1, 2]
    assert list(ta4f["delta_eV"]) == [0.3, 0.1, -0.1]

    oxide = out[(out["element"] == "O1s") & (out["species"] == "Oxide")]
    assert list(oxide["depth_order"]) == [0, 1]
