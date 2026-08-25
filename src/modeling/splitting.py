"""Assign and validate chronological modeling-data partitions.

This module provides the reusable temporal split used by Notebook 04.
It also validates prediction identity and prediction-time consistency before
the data enters model development.
The functions preserve their inputs and fail clearly when a data contract is
violated, preventing ambiguous rows or temporal inconsistencies from being
silently accepted.
"""

from __future__ import annotations

import pandas as pd

PREDICTION_IDENTIFIER_COLUMNS = [
    "studio",
    "course",
    "class_start",
    "prediction_horizon",
]


def assign_temporal_split(
    features: pd.DataFrame,
    training_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.Series:
    """Assign chronological train, validation, and test labels.

    Parameters
    ----------
    features : pd.DataFrame
        Feature table containing a datetime-like ``class_start`` column.
    training_end : pd.Timestamp
        Exclusive upper class-start boundary for the training split.
    validation_end : pd.Timestamp
        Exclusive upper class-start boundary for the validation split.

    Returns
    -------
    pd.Series
        String-valued split labels aligned to the original DataFrame index.

    Raises
    ------
    KeyError
        If ``class_start`` is absent.
    TypeError
        If ``class_start`` is not datetime-like.
    ValueError
        If boundaries are missing, reversed, timezone-incompatible, or a
        class start is missing.
    """

    if "class_start" not in features.columns:
        raise KeyError("features must contain a 'class_start' column.")
    if not pd.api.types.is_datetime64_any_dtype(features["class_start"]):
        raise TypeError("features['class_start'] must be datetime-like.")

    training_end = pd.Timestamp(training_end)
    validation_end = pd.Timestamp(validation_end)

    if pd.isna(training_end) or pd.isna(validation_end):
        raise ValueError("Split boundaries must not be missing.")
    if training_end >= validation_end:
        raise ValueError("training_end must be earlier than validation_end.")

    class_timezone = features["class_start"].dt.tz
    if training_end.tz != class_timezone or validation_end.tz != class_timezone:
        raise ValueError(
            "Split boundaries and class_start must use the same timezone."
        )
    if features["class_start"].isna().any():
        raise ValueError("class_start must not contain missing values.")

    split = pd.Series(pd.NA, index=features.index, dtype="string", name="split")
    split.loc[features["class_start"] < training_end] = "train"
    split.loc[
        features["class_start"].ge(training_end)
        & features["class_start"].lt(validation_end)
    ] = "validation"
    split.loc[features["class_start"] >= validation_end] = "test"

    return split


def validate_prediction_identity(features: pd.DataFrame) -> None:
    """Validate uniqueness of the four-column prediction identifier.

    Parameters
    ----------
    features : pd.DataFrame
        Feature table containing studio, course, class start, and prediction
        horizon columns.

    Returns
    -------
    None
        Returns only when every prediction identifier is unique.

    Raises
    ------
    KeyError
        If an identifier column is absent.
    ValueError
        If identifier values are missing or duplicate prediction-instance
        rows exist.
    """

    missing_columns = [
        column
        for column in PREDICTION_IDENTIFIER_COLUMNS
        if column not in features.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing prediction identifier columns: {missing_columns}")

    missing_value_mask = features[PREDICTION_IDENTIFIER_COLUMNS].isna()
    affected_row_mask = missing_value_mask.any(axis=1)
    affected_row_count = int(affected_row_mask.sum())

    if affected_row_count:
        affected_columns = [
            column
            for column in PREDICTION_IDENTIFIER_COLUMNS
            if missing_value_mask[column].any()
        ]
        diagnostic_sample = features.loc[
            affected_row_mask,
            PREDICTION_IDENTIFIER_COLUMNS,
        ].head(5)
        raise ValueError(
            f"Found {affected_row_count} rows with missing prediction identifier "
            f"values in columns {affected_columns}. Diagnostic sample:\n"
            f"{diagnostic_sample.to_string(index=False)}"
        )

    duplicate_mask = features.duplicated(
        PREDICTION_IDENTIFIER_COLUMNS,
        keep=False,
    )
    duplicate_row_count = int(duplicate_mask.sum())

    if duplicate_row_count:
        diagnostic_columns = PREDICTION_IDENTIFIER_COLUMNS + [
            column
            for column in ["prediction_time", "event_timestamp"]
            if column in features.columns
        ]
        diagnostic_sample = features.loc[
            duplicate_mask,
            diagnostic_columns,
        ].head(5)
        raise ValueError(
            f"Found {duplicate_row_count} duplicate prediction-instance rows "
            f"under {PREDICTION_IDENTIFIER_COLUMNS}. Diagnostic sample:\n"
            f"{diagnostic_sample.to_string(index=False)}"
        )


def validate_prediction_time(features: pd.DataFrame) -> None:
    """Validate the deterministic prediction-time relationship.

    Parameters
    ----------
    features : pd.DataFrame
        Feature table containing ``class_start``, ``prediction_horizon``, and
        ``prediction_time``. Prediction horizons are measured in hours.

    Returns
    -------
    None
        Returns only when every prediction time satisfies the contract.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If datetime or prediction-horizon dtypes are incompatible.
    ValueError
        If datetime timezones differ or inconsistent rows exist.
    """

    required_columns = [
        "studio",
        "course",
        "class_start",
        "prediction_horizon",
        "prediction_time",
    ]
    missing_columns = [
        column for column in required_columns if column not in features.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing prediction-time columns: {missing_columns}")

    for column in ["class_start", "prediction_time"]:
        if not pd.api.types.is_datetime64_any_dtype(features[column]):
            raise TypeError(f"features['{column}'] must be datetime-like.")
    if not pd.api.types.is_numeric_dtype(features["prediction_horizon"]):
        raise TypeError("features['prediction_horizon'] must be numeric hours.")
    if features["class_start"].dt.tz != features["prediction_time"].dt.tz:
        raise ValueError("class_start and prediction_time must use the same timezone.")

    expected_prediction_time = features["class_start"] - pd.to_timedelta(
        features["prediction_horizon"],
        unit="h",
    )
    inconsistent_mask = features["prediction_time"].ne(expected_prediction_time)
    inconsistent_row_count = int(inconsistent_mask.sum())

    if inconsistent_row_count:
        diagnostic_columns = [
            "studio",
            "course",
            "class_start",
            "prediction_horizon",
            "prediction_time",
        ]
        diagnostic_sample = features.loc[
            inconsistent_mask,
            diagnostic_columns,
        ].copy()
        diagnostic_sample["expected_prediction_time"] = expected_prediction_time.loc[
            inconsistent_mask
        ]
        raise ValueError(
            f"Found {inconsistent_row_count} rows with inconsistent "
            "prediction_time values. Diagnostic sample:\n"
            f"{diagnostic_sample.head(5).to_string(index=False)}"
        )
