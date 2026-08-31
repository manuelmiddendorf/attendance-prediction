from __future__ import annotations

import pandas as pd

from src.feature_engineering import add_class_context_features


def _build_features() -> pd.DataFrame:
    """Create prediction instances with distinctive existing feature values.

    Returns
    -------
    pd.DataFrame
        Incremental feature table containing canonical baseline columns.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio B"],
            "course": ["Course A", "Course B"],
            "instructor": ["Instructor A", "Instructor C"],
            "class_start": pd.to_datetime(
                ["2025-05-20 11:00:00", "2025-05-21 18:00:00"]
            ),
            "prediction_time": pd.to_datetime(
                ["2025-05-19 11:00:00", "2025-05-20 18:00:00"]
            ),
            "final_attendance_count": [5, 3],
            "attendance_count": [4, 2],
            "waiting_list_length": pd.Series([7, 0], index=[8, 3], dtype="Int64"),
            "waiting_list": [[{"ID": "w1"}, {"ID": "w2"}], [{"ID": "w3"}]],
            "capacity": pd.Series([5, 10], index=[8, 3], dtype="Int64"),
            "occupancy_rate": [91.0, 92.0],
            "has_waiting_list": [False, True],
            "weekday": ["stored-weekday-a", "stored-weekday-b"],
            "class_hour": [-1, -2],
        },
        index=[8, 3],
    )


def _build_attendance_history() -> pd.DataFrame:
    """Create attendance history with eligible and leaking examples.

    Returns
    -------
    pd.DataFrame
        History containing eligible, too-old, mismatched, current, and future
        classes for temporal leakage tests.
    """

    return pd.DataFrame(
        {
            "course": [
                "Course A",
                "Course A",
                "Course A",
                "Course A",
                "Course A",
                "Course B",
            ],
            "instructor": [
                "Instructor A",
                "Instructor A",
                "Instructor B",
                "Instructor A",
                "Instructor A",
                "Instructor C",
            ],
            "class_start": pd.to_datetime(
                [
                    "2025-05-10 11:00:00",
                    "2025-02-01 11:00:00",
                    "2025-05-12 11:00:00",
                    "2025-05-19 11:00:00",
                    "2025-05-20 11:00:00",
                    "2025-05-15 18:00:00",
                ]
            ),
            "attendance_list": [
                ["m1", "m2"],
                ["m1"] * 10,
                ["m1"] * 20,
                ["m1"] * 30,
                ["m1"] * 40,
                ["m1", "m2", "m3"],
            ],
        },
        index=[20, 21, 22, 23, 24, 25],
    )


def test_add_class_context_features_preserves_existing_pipeline_state() -> None:
    features = _build_features()
    attendance_history = _build_attendance_history()
    features_before = features.copy(deep=True)
    history_before = attendance_history.copy(deep=True)

    result = add_class_context_features(features, attendance_history)

    pd.testing.assert_frame_equal(features, features_before)
    pd.testing.assert_frame_equal(attendance_history, history_before)
    pd.testing.assert_frame_equal(result.loc[:, features.columns], features_before)
    pd.testing.assert_index_equal(result.index, features.index)
    assert len(result) == len(features)
    assert result["final_attendance_count"].equals(
        features["final_attendance_count"]
    )

    canonical_baseline_columns = [
        "attendance_count",
        "waiting_list_length",
        "occupancy_rate",
        "has_waiting_list",
        "weekday",
        "class_hour",
    ]
    pd.testing.assert_frame_equal(
        result[canonical_baseline_columns],
        features[canonical_baseline_columns],
    )
    redundant_alias_columns = {
        "waiting_count",
        "waiting_list_count",
        "current_fill_ratio",
        "class_weekday",
    }
    assert not redundant_alias_columns & set(result.columns)


def test_add_class_context_features_uses_existing_baseline_counts() -> None:
    result = add_class_context_features(
        _build_features(),
        _build_attendance_history(),
    )

    assert result["total_demand_count"].tolist() == [11, 2]
    assert result["total_demand_fill_ratio"].tolist() == [2.2, 0.2]


def test_add_class_context_features_rejects_non_positive_capacity_for_ratio() -> None:
    features = pd.concat(
        [_build_features().iloc[[0]].copy() for _ in range(3)],
        ignore_index=True,
    )
    features.index = [5, 2, 9]
    features["capacity"] = pd.Series(
        [pd.NA, 0, -1],
        index=features.index,
        dtype="Int64",
    )

    result = add_class_context_features(features, _build_attendance_history())

    assert result["total_demand_fill_ratio"].isna().all()
    pd.testing.assert_index_equal(result.index, features.index)


def test_add_class_context_features_excludes_current_and_future_outcomes() -> None:
    result = add_class_context_features(
        _build_features(),
        _build_attendance_history(),
    )

    assert result["course_instructor_attendance_mean_90d"].tolist() == [2.0, 3.0]
    assert result["course_instructor_history_count_90d"].tolist() == [1, 1]
