from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    add_baseline_features,
    add_historical_features,
    add_member_history_features,
)


def _build_prediction_instances() -> pd.DataFrame:
    """Create a tiny prediction-instance table for baseline feature tests.

    Returns
    -------
    pd.DataFrame
        Small canonical prediction-instance DataFrame with two rows.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio B"],
            "course": ["Course A", "Course B"],
            "class_start": pd.to_datetime(
                ["2025-05-20 11:00:00", "2025-05-19 18:00:00"]
            ),
            "instructor": ["Instructor A", "Instructor B"],
            "final_attendance_count": [3, 9],
            "attendance_list": [["m1", "m2"], ["m3", "m4", "m5", "m6"]],
            "event_timestamp": pd.to_datetime(
                ["2025-05-19 10:00:00", "2025-05-18 18:00:00"]
            ),
            "class_date": pd.to_datetime(["2025-05-20", "2025-05-19"]),
            "is_holiday": pd.Series([False, False], dtype="boolean"),
            "is_holiday_week": pd.Series([False, False], dtype="boolean"),
            "capacity": pd.Series([12, 4], dtype="Int64"),
            "waiting_list": [[], [{"ID": "w1"}, {"ID": "w2"}]],
            "waiting_list_length": pd.Series([0, 2], dtype="Int64"),
            "event_order": [10, 20],
            "prediction_horizon": [24, 24],
            "prediction_time": pd.to_datetime(
                ["2025-05-19 11:00:00", "2025-05-18 18:00:00"]
            ),
            "snapshot_age_hours": [1.0, 0.0],
        }
    )


def test_add_baseline_features_builds_expected_columns_and_values() -> None:
    prediction_instances = _build_prediction_instances()

    baseline_features = add_baseline_features(prediction_instances)

    assert list(baseline_features.columns) == [
        *prediction_instances.columns.tolist(),
        "attendance_count",
        "available_spots",
        "occupancy_rate",
        "is_full",
        "has_waiting_list",
        "weekday",
        "class_hour",
    ]
    assert baseline_features["attendance_count"].tolist() == [2, 4]
    assert baseline_features["available_spots"].tolist() == [10, 0]
    assert baseline_features["occupancy_rate"].tolist() == [2 / 12, 1.0]
    assert baseline_features["is_full"].tolist() == [False, True]
    assert baseline_features["has_waiting_list"].tolist() == [False, True]
    assert baseline_features["weekday"].tolist() == ["Tuesday", "Monday"]
    assert baseline_features["class_hour"].tolist() == [11, 18]


def test_add_baseline_features_preserves_rows_target_and_identifiers() -> None:
    prediction_instances = _build_prediction_instances()
    prediction_identifier_columns = [
        "studio",
        "course",
        "class_start",
        "prediction_horizon",
    ]

    baseline_features = add_baseline_features(prediction_instances)

    assert len(baseline_features) == len(prediction_instances)
    assert baseline_features["final_attendance_count"].equals(
        prediction_instances["final_attendance_count"]
    )
    assert prediction_instances["attendance_list"].map(len).equals(
        baseline_features["attendance_count"]
    )
    assert prediction_instances["waiting_list_length"].equals(
        baseline_features["waiting_list_length"]
    )
    assert "waiting_count" not in baseline_features.columns
    assert prediction_instances.duplicated(prediction_identifier_columns).sum() == 0
    assert baseline_features.duplicated(prediction_identifier_columns).sum() == 0
    assert "attendance_list" in baseline_features.columns
    assert "waiting_list" in baseline_features.columns


def test_prediction_identifier_excludes_prediction_time() -> None:
    prediction_instances = pd.concat(
        [_build_prediction_instances().iloc[[0]]] * 2,
        ignore_index=True,
    )
    prediction_instances.loc[1, "prediction_time"] += pd.Timedelta(hours=1)
    prediction_identifier_columns = [
        "studio",
        "course",
        "class_start",
        "prediction_horizon",
    ]

    assert "prediction_time" not in prediction_identifier_columns
    assert (
        prediction_instances.duplicated(
            prediction_identifier_columns,
            keep=False,
        ).sum()
        == 2
    )


def _build_historical_prediction_instances() -> pd.DataFrame:
    """Create prediction instances for historical-feature tests.

    Returns
    -------
    pd.DataFrame
        Small prediction-instance DataFrame with two future classes.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio B"],
            "course": ["Course A", "Course B"],
            "class_start": pd.to_datetime(
                ["2025-05-21 11:00:00", "2025-05-21 18:00:00"]
            ),
            "instructor": ["Instructor A", "Instructor C"],
            "final_attendance_count": [3, 7],
            "attendance_list": [["m1", "m2"], ["m3"]],
            "event_timestamp": pd.to_datetime(
                ["2025-05-19 10:00:00", "2025-05-19 16:00:00"]
            ),
            "class_date": pd.to_datetime(["2025-05-21", "2025-05-21"]),
            "is_holiday": pd.Series([False, False], dtype="boolean"),
            "is_holiday_week": pd.Series([False, False], dtype="boolean"),
            "capacity": pd.Series([12, 14], dtype="Int64"),
            "waiting_list": [[], []],
            "waiting_list_length": pd.Series([0, 0], dtype="Int64"),
            "event_order": [1, 2],
            "prediction_horizon": [24, 24],
            "prediction_time": pd.to_datetime(
                ["2025-05-20 10:00:00", "2025-05-20 10:00:00"]
            ),
            "snapshot_age_hours": [24.0, 18.0],
        }
    )


def _build_attendance_history() -> pd.DataFrame:
    """Create a tiny attendance-history table for historical-feature tests.

    Returns
    -------
    pd.DataFrame
        Small attendance-history DataFrame with past, future, and equal-time
        examples for validating the temporal cutoff.
    """

    return pd.DataFrame(
        {
            "studio": [
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio B",
            ],
            "course": [
                "Course A",
                "Course A",
                "Course A",
                "Course C",
                "Course A",
                "Course A",
                "Course B",
            ],
            "instructor": [
                "Instructor A",
                "Instructor A",
                "Instructor D",
                "Instructor A",
                "Instructor A",
                "Instructor A",
                "Instructor C",
            ],
            "class_start": pd.to_datetime(
                [
                    "2025-05-10 11:00:00",
                    "2025-04-01 11:00:00",
                    "2024-12-15 11:00:00",
                    "2025-05-19 16:00:00",
                    "2025-05-20 12:00:00",
                    "2025-05-20 10:00:00",
                    "2025-05-21 18:00:00",
                ]
            ),
            "attendance_list": [
                ["a1", "a2", "a3", "a4"],
                ["b1", "b2", "b3", "b4", "b5", "b6"],
                ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
                ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"],
                ["e1"] * 50,
                ["f1"] * 40,
                ["g1"] * 7,
            ],
        }
    )


def test_add_historical_features_builds_expected_statistics() -> None:
    prediction_instances = _build_historical_prediction_instances()
    attendance_history = _build_attendance_history()

    features = add_baseline_features(prediction_instances)
    features = add_historical_features(features, attendance_history)

    assert len(features) == len(prediction_instances)
    assert features["final_attendance_count"].equals(
        prediction_instances["final_attendance_count"]
    )
    assert features.at[0, "course_attendance_count_30d"] == 1
    assert features.at[0, "course_attendance_mean_30d"] == 4.0
    assert pd.isna(features.at[0, "course_attendance_std_30d"])
    assert features.at[0, "course_attendance_count_90d"] == 2
    assert features.at[0, "course_attendance_mean_90d"] == 5.0
    assert features.at[0, "course_attendance_std_90d"] == pytest.approx(2**0.5)
    assert features.at[0, "course_attendance_count_365d"] == 3
    assert features.at[0, "course_attendance_mean_365d"] == 6.0
    assert features.at[0, "course_attendance_std_365d"] == pytest.approx(2.0)
    assert features.at[0, "instructor_attendance_count_30d"] == 2
    assert features.at[0, "studio_attendance_count_30d"] == 2
    assert features.at[1, "instructor_attendance_count_365d"] == 0
    assert features.at[1, "studio_attendance_count_365d"] == 0
    assert pd.isna(features.at[1, "instructor_attendance_mean_365d"])
    assert pd.isna(features.at[1, "studio_attendance_std_365d"])


def test_add_historical_features_excludes_equal_and_future_classes() -> None:
    prediction_instances = _build_historical_prediction_instances()
    attendance_history = _build_attendance_history()

    features = add_baseline_features(prediction_instances)
    features = add_historical_features(features, attendance_history)

    assert features.at[0, "course_attendance_count_365d"] == 3
    assert features.at[0, "course_attendance_mean_365d"] == 6.0
    assert features.at[0, "course_attendance_count_30d"] == 1


def _build_member_history_prediction_instances() -> pd.DataFrame:
    """Create prediction instances for member-history feature tests.

    Returns
    -------
    pd.DataFrame
        Small prediction-instance DataFrame with one populated and one empty
        booking list.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio A"],
            "course": ["Course A", "Course A"],
            "class_start": pd.to_datetime(
                ["2025-05-21 11:00:00", "2025-05-22 11:00:00"]
            ),
            "instructor": ["Instructor A", "Instructor A"],
            "final_attendance_count": [3, 0],
            "attendance_list": [["m1", "m2"], []],
            "event_timestamp": pd.to_datetime(
                ["2025-05-19 09:00:00", "2025-05-20 09:00:00"]
            ),
            "class_date": pd.to_datetime(["2025-05-21", "2025-05-22"]),
            "is_holiday": pd.Series([False, False], dtype="boolean"),
            "is_holiday_week": pd.Series([False, False], dtype="boolean"),
            "capacity": pd.Series([12, 12], dtype="Int64"),
            "waiting_list": [[], []],
            "waiting_list_length": pd.Series([0, 0], dtype="Int64"),
            "event_order": [1, 2],
            "prediction_horizon": [24, 24],
            "prediction_time": pd.to_datetime(
                ["2025-05-20 10:00:00", "2025-05-21 10:00:00"]
            ),
            "snapshot_age_hours": [25.0, 26.0],
        }
    )


def _build_member_history_attendance() -> pd.DataFrame:
    """Create attendance history for member-history feature tests.

    Returns
    -------
    pd.DataFrame
        Small canonical attendance history with past, equal-time, and future
        examples for leakage-safe member aggregation tests.
    """

    return pd.DataFrame(
        {
            "studio": [
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
                "Studio A",
            ],
            "course": [
                "Course A",
                "Course C",
                "Course A",
                "Course D",
                "Course A",
                "Course A",
            ],
            "instructor": [
                "Instructor A",
                "Instructor A",
                "Instructor C",
                "Instructor D",
                "Instructor A",
                "Instructor A",
            ],
            "class_start": pd.to_datetime(
                [
                    "2025-05-10 11:00:00",
                    "2025-05-15 16:00:00",
                    "2025-05-18 11:00:00",
                    "2025-02-01 17:00:00",
                    "2025-05-20 10:00:00",
                    "2025-05-21 11:00:00",
                ]
            ),
            "attendance_list": [
                ["m1"],
                ["m1"],
                ["m2"],
                ["m2"],
                ["m1"],
                ["m1", "m2"],
            ],
            "waiting_list": [[], [], [], [], [], []],
            "instructor_helper": ["x", "x", "x", "x", "x", "x"],
            "capacity": pd.Series([12, 12, 12, 12, 12, 12], dtype="Int64"),
            "class_date": pd.to_datetime(
                [
                    "2025-05-10",
                    "2025-05-15",
                    "2025-05-18",
                    "2025-02-01",
                    "2025-05-20",
                    "2025-05-21",
                ]
            ),
        }
    ).rename(columns={"instructor_helper": "unused"})


def test_add_member_history_features_builds_expected_aggregates() -> None:
    prediction_instances = _build_member_history_prediction_instances()
    attendance_history = _build_member_history_attendance()

    features = add_baseline_features(prediction_instances)
    features["course_attendance_mean_30d"] = [5.0, 6.0]
    features = add_member_history_features(features, attendance_history)

    assert len(features) == len(prediction_instances)
    assert features["final_attendance_count"].equals(
        prediction_instances["final_attendance_count"]
    )
    assert features.at[0, "member_attendance_count_mean_30d"] == 1.5
    assert features.at[0, "members_with_history_count_30d"] == 2
    assert features.at[0, "members_with_history_share_30d"] == 1.0
    assert features.at[0, "member_course_affinity_mean_30d"] == pytest.approx(0.75)
    assert features.at[0, "member_instructor_affinity_mean_30d"] == pytest.approx(0.5)
    assert features.at[0, "member_attendance_count_mean_365d"] == 2.0
    assert features.at[0, "member_course_affinity_mean_365d"] == pytest.approx(0.5)
    assert features.at[0, "member_instructor_affinity_mean_365d"] == pytest.approx(
        0.5
    )
    assert features.at[0, "course_attendance_mean_30d"] == 5.0
    assert features.at[1, "members_with_history_count_30d"] == 0
    assert pd.isna(features.at[1, "members_with_history_share_30d"])
    assert pd.isna(features.at[1, "member_course_affinity_mean_30d"])
    assert pd.isna(features.at[1, "member_instructor_affinity_mean_30d"])


def test_add_member_history_features_excludes_equal_and_future_classes() -> None:
    prediction_instances = _build_member_history_prediction_instances().iloc[[0]].copy()
    attendance_history = _build_member_history_attendance()

    features = add_baseline_features(prediction_instances)
    features = add_member_history_features(features, attendance_history)

    assert features.at[0, "member_attendance_count_mean_30d"] == 1.5
    assert features.at[0, "members_with_history_count_30d"] == 2
    assert features.at[0, "member_course_affinity_mean_30d"] == pytest.approx(0.75)
