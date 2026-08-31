from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import add_member_reliability_features


def _build_current_features(prediction_horizon: int = 24) -> pd.DataFrame:
    """Create a tiny current feature table for reliability tests.

    Parameters
    ----------
    prediction_horizon : int, default=24
        Prediction horizon of the current instance, in hours.

    Returns
    -------
    pd.DataFrame
        One-row feature table with the columns required for member reliability.
    """

    class_start = pd.Timestamp("2025-05-20 11:00:00") + pd.Timedelta(
        hours=prediction_horizon - 24
    )

    return pd.DataFrame(
        {
            "studio": ["Studio A"],
            "course": ["Course A"],
            "class_start": [class_start],
            "instructor": ["Instructor A"],
            "final_attendance_count": [1],
            "attendance_list": [["m1"]],
            "prediction_horizon": [prediction_horizon],
            "prediction_time": [class_start - pd.Timedelta(hours=prediction_horizon)],
        }
    )


def _build_reliability_attendance_history() -> pd.DataFrame:
    """Create historical attendance outcomes for reliability tests.

    Returns
    -------
    pd.DataFrame
        Historical attendance table with attended, missed, missing-snapshot,
        and future-class examples.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A"] * 6,
            "course": ["Course A"] * 6,
            "class_start": pd.to_datetime(
                [
                    "2025-05-10 11:00:00",
                    "2025-05-11 11:00:00",
                    "2025-05-12 11:00:00",
                    "2025-05-13 11:00:00",
                    "2025-05-14 11:00:00",
                    "2025-05-20 12:00:00",
                ]
            ),
            "attendance_list": [
                ["m1"],
                ["m1"],
                ["m1"],
                [],
                ["m1"],
                ["m1"],
            ],
        }
    )


def _build_reliability_booking_history() -> pd.DataFrame:
    """Create booking-event histories for reliability tests.

    Returns
    -------
    pd.DataFrame
        Historical booking-event table with one missing-horizon snapshot case
        and one future class that should be ignored.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A"] * 7,
            "course": ["Course A"] * 7,
            "class_start": pd.to_datetime(
                [
                    "2025-05-10 11:00:00",
                    "2025-05-11 11:00:00",
                    "2025-05-11 11:00:00",
                    "2025-05-12 11:00:00",
                    "2025-05-13 11:00:00",
                    "2025-05-14 11:00:00",
                    "2025-05-20 12:00:00",
                ]
            ),
            "event_timestamp": pd.to_datetime(
                [
                    "2025-05-09 09:00:00",
                    "2025-05-10 10:00:00",
                    "2025-05-10 10:00:00",
                    "2025-05-11 08:00:00",
                    "2025-05-12 08:00:00",
                    "2025-05-13 12:00:00",
                    "2025-05-19 10:00:00",
                ]
            ),
            "attendance_list": [
                ["m1"],
                [],
                ["m1"],
                ["m1"],
                ["m1"],
                ["m1"],
                ["m1"],
            ],
            "event_order": [0, 0, 1, 0, 0, 0, 0],
        }
    )


def _build_horizon_dependence_attendance_history() -> pd.DataFrame:
    """Create historical attendance outcomes for horizon-dependence tests.

    Returns
    -------
    pd.DataFrame
        Historical attendance table with two attended classes.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio A"],
            "course": ["Course A", "Course A"],
            "class_start": pd.to_datetime(
                ["2025-05-10 11:00:00", "2025-05-11 11:00:00"]
            ),
            "attendance_list": [["m1"], ["m1"]],
        }
    )


def _build_horizon_dependence_booking_history() -> pd.DataFrame:
    """Create booking-event histories for horizon-dependence tests.

    Returns
    -------
    pd.DataFrame
        Historical booking-event table where one class has an early enough
        snapshot for both horizons and one only for the 24-hour horizon.
    """

    return pd.DataFrame(
        {
            "studio": ["Studio A", "Studio A"],
            "course": ["Course A", "Course A"],
            "class_start": pd.to_datetime(
                ["2025-05-10 11:00:00", "2025-05-11 11:00:00"]
            ),
            "event_timestamp": pd.to_datetime(
                ["2025-05-08 10:00:00", "2025-05-10 10:00:00"]
            ),
            "attendance_list": [["m1"], ["m1"]],
            "event_order": [0, 0],
        }
    )


def test_add_member_reliability_features_builds_same_horizon_show_up_rate() -> None:
    features = _build_current_features(prediction_horizon=24)
    booking_events_history = _build_reliability_booking_history()
    attendance_history = _build_reliability_attendance_history()

    features = add_member_reliability_features(
        features,
        booking_events_history,
        attendance_history,
    )

    assert features.at[0, "member_booked_at_horizon_count_mean_30d"] == 4.0
    assert features.at[0, "members_with_reliability_history_count_30d"] == 1
    assert features.at[0, "members_with_reliability_history_share_30d"] == 1.0
    assert features.at[0, "member_show_up_rate_mean_30d"] == pytest.approx(0.75)


def test_add_member_reliability_features_ignores_missing_historical_snapshot() -> None:
    features = _build_current_features(prediction_horizon=24)
    booking_events_history = _build_reliability_booking_history()
    attendance_history = _build_reliability_attendance_history()

    features = add_member_reliability_features(
        features,
        booking_events_history,
        attendance_history,
    )

    assert features.at[0, "member_booked_at_horizon_count_mean_30d"] == 4.0


def test_add_member_reliability_features_ignores_future_classes() -> None:
    features = _build_current_features(prediction_horizon=24)
    booking_events_history = _build_reliability_booking_history()
    attendance_history = _build_reliability_attendance_history()

    features = add_member_reliability_features(
        features,
        booking_events_history,
        attendance_history,
    )

    assert features.at[0, "member_show_up_rate_mean_30d"] == pytest.approx(0.75)


def test_add_member_reliability_features_depends_on_current_horizon() -> None:
    features_24h = _build_current_features(prediction_horizon=24)
    features_48h = _build_current_features(prediction_horizon=48)
    features = pd.concat([features_24h, features_48h], ignore_index=True)
    booking_events_history = _build_horizon_dependence_booking_history()
    attendance_history = _build_horizon_dependence_attendance_history()

    features = add_member_reliability_features(
        features,
        booking_events_history,
        attendance_history,
    )

    assert features.at[0, "member_booked_at_horizon_count_mean_30d"] == 2.0
    assert features.at[0, "member_show_up_rate_mean_30d"] == 1.0
    assert features.at[1, "member_booked_at_horizon_count_mean_30d"] == 1.0
    assert features.at[1, "member_show_up_rate_mean_30d"] == 1.0
