from __future__ import annotations

import pandas as pd
import pytest

from src.modeling import (
    PREDICTION_IDENTIFIER_COLUMNS,
    assign_temporal_split,
    validate_prediction_identity,
    validate_prediction_time,
)

TRAINING_END = pd.Timestamp("2026-02-01")
VALIDATION_END = pd.Timestamp("2026-05-01")


def _build_prediction_instances() -> pd.DataFrame:
    """Create prediction instances around both temporal boundaries.

    Returns
    -------
    pd.DataFrame
        Small prediction-instance table with a non-default index.
    """

    class_start = pd.to_datetime(
        [
            "2026-01-31 18:00:00",
            "2026-02-01 00:00:00",
            "2026-05-01 00:00:00",
        ]
    )
    prediction_horizon = pd.Series([24, 24, 24], index=[8, 3, 12])

    return pd.DataFrame(
        {
            "studio": ["Cb", "Cb", "Nk"],
            "course": ["Sa 18:00", "So 00:00", "Fr 00:00"],
            "class_start": class_start,
            "prediction_horizon": prediction_horizon,
            "prediction_time": class_start - pd.to_timedelta(
                prediction_horizon.to_numpy(),
                unit="h",
            ),
        },
        index=[8, 3, 12],
    )


def test_assign_temporal_split_handles_boundaries_and_preserves_index() -> None:
    features = _build_prediction_instances()

    split = assign_temporal_split(features, TRAINING_END, VALIDATION_END)

    assert split.tolist() == ["train", "validation", "test"]
    pd.testing.assert_index_equal(split.index, features.index)


def test_assign_temporal_split_does_not_mutate_input() -> None:
    features = _build_prediction_instances()
    features_before = features.copy(deep=True)

    assign_temporal_split(features, TRAINING_END, VALIDATION_END)

    pd.testing.assert_frame_equal(features, features_before)


@pytest.mark.parametrize(
    ("training_end", "validation_end"),
    [
        (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-02-01")),
        (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-01")),
    ],
)
def test_assign_temporal_split_rejects_invalid_boundaries(
    training_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> None:
    features = _build_prediction_instances()

    with pytest.raises(ValueError, match="training_end must be earlier"):
        assign_temporal_split(features, training_end, validation_end)


def test_assign_temporal_split_rejects_timezone_mismatch() -> None:
    features = _build_prediction_instances()
    features["class_start"] = features["class_start"].dt.tz_localize(
        "Europe/Berlin"
    )

    with pytest.raises(ValueError, match="must use the same timezone"):
        assign_temporal_split(features, TRAINING_END, VALIDATION_END)


def test_multiple_horizons_for_one_class_remain_in_one_split() -> None:
    features = pd.DataFrame(
        {
            "studio": ["Cb", "Cb"],
            "course": ["Di 11:00", "Di 11:00"],
            "class_start": pd.to_datetime(
                ["2026-03-10 11:00:00", "2026-03-10 11:00:00"]
            ),
            "prediction_horizon": [24, 48],
            "prediction_time": pd.to_datetime(
                ["2026-03-09 11:00:00", "2026-03-08 11:00:00"]
            ),
        }
    )

    split = assign_temporal_split(features, TRAINING_END, VALIDATION_END)

    assert split.nunique() == 1
    assert split.iloc[0] == "validation"
    validate_prediction_identity(features)


def test_prediction_identifier_columns_match_contract() -> None:
    assert PREDICTION_IDENTIFIER_COLUMNS == [
        "studio",
        "course",
        "class_start",
        "prediction_horizon",
    ]


def test_prediction_identity_ignores_prediction_time() -> None:
    features = pd.DataFrame(
        {
            "studio": ["Cb", "Cb"],
            "course": ["Di 11:00", "Di 11:00"],
            "class_start": pd.to_datetime(
                ["2026-03-10 11:00:00", "2026-03-10 11:00:00"]
            ),
            "prediction_horizon": [24, 24],
            "prediction_time": pd.to_datetime(
                ["2026-03-09 11:00:00", "2026-03-09 12:00:00"]
            ),
            "event_timestamp": pd.to_datetime(
                ["2026-03-09 10:00:00", "2026-03-09 11:00:00"]
            ),
        }
    )

    with pytest.raises(ValueError, match="2 duplicate prediction-instance rows"):
        validate_prediction_identity(features)


@pytest.mark.parametrize(
    ("column", "missing_value"),
    [
        ("studio", pd.NA),
        ("course", pd.NA),
        ("class_start", pd.NaT),
        ("prediction_horizon", pd.NA),
    ],
)
def test_prediction_identity_rejects_missing_identifier_values(
    column: str,
    missing_value: object,
) -> None:
    features = _build_prediction_instances()
    if column == "prediction_horizon":
        features[column] = features[column].astype("Int64")
    features.loc[8, column] = missing_value

    with pytest.raises(ValueError) as error:
        validate_prediction_identity(features)

    message = str(error.value)
    assert "Found 1 rows with missing prediction identifier values" in message
    assert column in message
    assert "Diagnostic sample" in message


def test_inconsistent_prediction_time_is_detected_separately() -> None:
    features = _build_prediction_instances()
    features.loc[3, "prediction_time"] += pd.Timedelta(hours=1)

    validate_prediction_identity(features)

    with pytest.raises(ValueError, match="1 rows with inconsistent prediction_time"):
        validate_prediction_time(features)
