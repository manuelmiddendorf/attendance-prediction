from __future__ import annotations

import pandas as pd

from src.feature_engineering import add_booking_dynamics_features


def _build_booking_dynamics_features() -> pd.DataFrame:
    """Create a one-row feature table for booking-dynamics tests.

    Returns
    -------
    pd.DataFrame
        Minimal feature table with the columns required by the booking-
        dynamics feature step.
    """

    return pd.DataFrame(
        {
            "studio": ["Cb"],
            "course": ["Di 11:00"],
            "class_start": pd.to_datetime(["2025-05-20 11:00:00"]),
            "instructor": ["Marta"],
            "final_attendance_count": [3],
            "attendance_list": [["m1", "m2"]],
            "event_timestamp": pd.to_datetime(["2025-05-20 09:30:00"]),
            "prediction_horizon": [1],
            "prediction_time": pd.to_datetime(["2025-05-20 10:00:00"]),
        }
    )


def _build_booking_dynamics_history() -> pd.DataFrame:
    """Create booking-event history for booking-dynamics tests.

    Returns
    -------
    pd.DataFrame
        Booking timeline with positive and negative observed transitions
        across several windows and one post-prediction event that must be
        ignored.
    """

    return pd.DataFrame(
        {
            "studio": ["Cb"] * 7,
            "course": ["Di 11:00"] * 7,
            "class_start": pd.to_datetime(["2025-05-20 11:00:00"] * 7),
            "event_timestamp": pd.to_datetime(
                [
                    "2025-05-17 11:00:00",
                    "2025-05-18 12:00:00",
                    "2025-05-19 09:30:00",
                    "2025-05-19 12:00:00",
                    "2025-05-20 04:30:00",
                    "2025-05-20 09:30:00",
                    "2025-05-20 10:15:00",
                ]
            ),
            "attendance_list": [
                ["m1"],
                ["m1", "m2"],
                ["m1"],
                ["m1", "m2"],
                ["m1", "m2", "m3"],
                ["m1", "m2"],
                ["m1", "m2", "m3", "m4"],
            ],
            "event_order": [0, 1, 2, 3, 4, 5, 6],
        }
    )


def _build_same_timestamp_features() -> pd.DataFrame:
    """Create a one-row feature table for same-timestamp ordering tests.

    Returns
    -------
    pd.DataFrame
        Minimal feature table whose current booking state depends on
        ``event_order`` within one timestamp.
    """

    return pd.DataFrame(
        {
            "studio": ["Cb"],
            "course": ["Di 11:00"],
            "class_start": pd.to_datetime(["2025-05-20 11:00:00"]),
            "instructor": ["Marta"],
            "final_attendance_count": [1],
            "attendance_list": [["m1"]],
            "event_timestamp": pd.to_datetime(["2025-05-20 09:00:00"]),
            "prediction_horizon": [1],
            "prediction_time": pd.to_datetime(["2025-05-20 10:00:00"]),
        }
    )


def _build_same_timestamp_history() -> pd.DataFrame:
    """Create booking-event history with multiple states at one timestamp.

    Returns
    -------
    pd.DataFrame
        Booking timeline where ``event_order`` determines the final
        pre-prediction sequence of observed transitions.
    """

    return pd.DataFrame(
        {
            "studio": ["Cb"] * 3,
            "course": ["Di 11:00"] * 3,
            "class_start": pd.to_datetime(["2025-05-20 11:00:00"] * 3),
            "event_timestamp": pd.to_datetime(
                [
                    "2025-05-20 08:00:00",
                    "2025-05-20 09:00:00",
                    "2025-05-20 09:00:00",
                ]
            ),
            "attendance_list": [
                ["m1"],
                ["m1", "m2"],
                ["m1"],
            ],
            "event_order": [0, 0, 1],
        }
    )


def test_add_booking_dynamics_features_builds_expected_window_counts() -> None:
    features = _build_booking_dynamics_features()
    booking_events_history = _build_booking_dynamics_history()

    features = add_booking_dynamics_features(features, booking_events_history)

    assert features.at[0, "observed_booking_increase_events_6h"] == 1
    assert features.at[0, "observed_booking_decrease_events_6h"] == 1
    assert features.at[0, "observed_net_booking_change_6h"] == 0
    assert features.at[0, "observed_booking_increase_events_24h"] == 2
    assert features.at[0, "observed_booking_decrease_events_24h"] == 1
    assert features.at[0, "observed_net_booking_change_24h"] == 1
    assert features.at[0, "observed_booking_increase_events_72h"] == 3
    assert features.at[0, "observed_booking_decrease_events_72h"] == 2
    assert features.at[0, "observed_net_booking_change_72h"] == 1
    assert features.at[0, "max_observed_attendance_before_prediction"] == 3
    assert features.at[0, "attendance_drop_from_observed_peak"] == 1


def test_add_booking_dynamics_features_ignores_post_prediction_events() -> None:
    features = _build_booking_dynamics_features()
    booking_events_history = _build_booking_dynamics_history()

    features = add_booking_dynamics_features(features, booking_events_history)

    assert features.at[0, "observed_booking_increase_events_6h"] == 1
    assert features.at[0, "observed_net_booking_change_6h"] == 0
    assert features.at[0, "max_observed_attendance_before_prediction"] == 3


def test_add_booking_dynamics_features_uses_event_order_with_same_timestamp() -> None:
    features = _build_same_timestamp_features()
    booking_events_history = _build_same_timestamp_history()

    features = add_booking_dynamics_features(features, booking_events_history)

    assert features.at[0, "observed_booking_increase_events_6h"] == 1
    assert features.at[0, "observed_booking_decrease_events_6h"] == 1
    assert features.at[0, "observed_net_booking_change_6h"] == 0
    assert features.at[0, "max_observed_attendance_before_prediction"] == 2
    assert features.at[0, "attendance_drop_from_observed_peak"] == 1
