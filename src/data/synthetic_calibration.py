"""Build aggregate calibration summaries and privacy sanity checks.

This module is the only development utility that compares private and public
data. It consumes DataFrames through the normal data layer and returns only
rounded aggregate statistics or boolean overlap checks; it never writes
private inputs, entity parameters, or mapping tables. The synthetic generator
does not import this module and can therefore run in a fresh public checkout.
These checks support practical review but do not constitute a formal privacy
or anonymity guarantee.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import RAW_ATTENDANCE_FILENAME, RAW_EVENT_FILENAME
from .representation import prepare_attendance, prepare_booking_events

_CLASS_COLUMNS = ["studio", "course", "class_start"]
_QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _member_count(value: object) -> float:
    """Return collection length while preserving a missing value.

    Parameters
    ----------
    value : object
        Canonical member-list value.

    Returns
    -------
    float
        Collection length or ``NaN`` when the source value is missing.
    """

    return float(len(value)) if isinstance(value, list) else float("nan")


def _rounded_quantiles(values: pd.Series, digits: int = 1) -> dict[str, float]:
    """Return a small rounded quantile summary.

    Parameters
    ----------
    values : pd.Series
        Numeric observations to summarize.
    digits : int, optional
        Number of decimal places retained in the safe public summary.

    Returns
    -------
    dict[str, float]
        Quantiles keyed by compact probability labels.
    """

    quantiles = values.dropna().astype("float64").quantile(_QUANTILE_LEVELS)
    return {
        f"p{int(level * 100):02d}": round(float(value), digits)
        for level, value in quantiles.items()
    }


def _prediction_instances_at_24h(
    booking_events: pd.DataFrame,
    attendance: pd.DataFrame,
) -> pd.DataFrame:
    """Construct aggregate-analysis instances at the project's 24-hour horizon.

    Parameters
    ----------
    booking_events : pd.DataFrame
        Canonical booking snapshots.
    attendance : pd.DataFrame
        Canonical final attendance records.

    Returns
    -------
    pd.DataFrame
        Latest eligible snapshot for every matched class with usable data.
    """

    clean_attendance = attendance.dropna().loc[
        lambda frame: ~frame.duplicated(_CLASS_COLUMNS, keep=False)
    ]
    events = booking_events.dropna(
        subset=[*_CLASS_COLUMNS, "event_timestamp", "attendance_list"]
    ).copy()
    events["event_order"] = range(len(events))
    outcomes = clean_attendance.loc[
        :, [*_CLASS_COLUMNS, "attendance_list"]
    ].rename(columns={"attendance_list": "final_attendance_list"})
    matched = outcomes.merge(events, on=_CLASS_COLUMNS, how="inner")
    prediction_time = matched["class_start"] - pd.Timedelta(hours=24)
    eligible = matched.loc[matched["event_timestamp"] <= prediction_time]
    return (
        eligible.sort_values(["event_timestamp", "event_order"])
        .groupby(_CLASS_COLUMNS)
        .tail(1)
        .copy()
    )


def build_calibration_summary(
    booking_events_raw: pd.DataFrame,
    attendance_raw: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute rounded, non-row-level statistics for synthetic calibration.

    Parameters
    ----------
    booking_events_raw : pd.DataFrame
        Raw private or synthetic booking-event table loaded through ``io.py``.
    attendance_raw : pd.DataFrame
        Raw private or synthetic final-attendance table loaded through
        ``io.py``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Coarse scale, distribution, trajectory, and member-recurrence
        summaries. No category labels, timestamps, member values, or rows are
        included.
    """

    booking_events = prepare_booking_events(booking_events_raw)
    attendance = prepare_attendance(attendance_raw)
    booking_events["booked_count"] = booking_events["attendance_list"].map(
        _member_count
    )
    booking_events["waiting_count"] = booking_events["waiting_list"].map(
        _member_count
    )
    attendance["final_attendance_count"] = attendance["attendance_list"].map(
        _member_count
    )
    attendance["waiting_count"] = attendance["waiting_list"].map(_member_count)

    plausible_attendance = attendance.loc[
        attendance["class_start"] >= pd.Timestamp("2000-01-01")
    ]
    active_weeks = plausible_attendance["class_start"].dt.to_period("W").nunique()
    class_groups = booking_events.groupby(_CLASS_COLUMNS, sort=False)
    lead_hours = (
        booking_events["class_start"] - booking_events["event_timestamp"]
    ).dt.total_seconds() / 3600

    instances = _prediction_instances_at_24h(booking_events, attendance)
    instances["booking_count_24h"] = instances["attendance_list"].map(len)
    instances["final_attendance_count"] = instances["final_attendance_list"].map(
        len
    )
    instances["booking_error_24h"] = (
        instances["final_attendance_count"] - instances["booking_count_24h"]
    )

    member_class_counts = Counter(
        member_id
        for member_list in plausible_attendance["attendance_list"]
        if isinstance(member_list, list)
        for member_id in set(member_list)
    )

    return {
        "scale": {
            "booking_rows_rounded": int(round(len(booking_events), -1)),
            "attendance_rows_rounded": int(round(len(attendance), -1)),
            "booking_classes_rounded": int(round(class_groups.ngroups, -1)),
            "classes_per_week": round(
                len(plausible_attendance) / max(active_weeks, 1),
                1,
            ),
            "studio_count": int(attendance["studio"].nunique()),
            "course_count_rounded": int(
                round(attendance["course"].nunique() / 5) * 5
            ),
            "instructor_count": int(attendance["instructor"].nunique()),
        },
        "attendance": {
            "final_count_quantiles": _rounded_quantiles(
                plausible_attendance["final_attendance_count"]
            ),
            "capacity_quantiles": _rounded_quantiles(
                plausible_attendance["capacity"]
            ),
            "waiting_list_share": round(
                float(plausible_attendance["waiting_count"].gt(0).mean()),
                2,
            ),
            "mean_final_attendance": round(
                float(plausible_attendance["final_attendance_count"].mean()),
                1,
            ),
            "std_final_attendance": round(
                float(plausible_attendance["final_attendance_count"].std()),
                1,
            ),
        },
        "booking_events": {
            "snapshots_per_class_quantiles": _rounded_quantiles(
                class_groups.size()
            ),
            "lead_hours_quantiles": _rounded_quantiles(lead_hours),
            "waiting_list_snapshot_share": round(
                float(booking_events["waiting_count"].gt(0).mean()),
                2,
            ),
        },
        "prediction_24h": {
            "instances_rounded": int(round(len(instances), -1)),
            "booking_count_quantiles": _rounded_quantiles(
                instances["booking_count_24h"]
            ),
            "final_minus_booking_quantiles": _rounded_quantiles(
                instances["booking_error_24h"]
            ),
            "booking_count_mae": round(
                float(instances["booking_error_24h"].abs().mean()),
                2,
            ),
            "booking_count_rmse": round(
                float(np.sqrt(instances["booking_error_24h"].pow(2).mean())),
                2,
            ),
            "booking_final_correlation": round(
                float(
                    instances[
                        ["booking_count_24h", "final_attendance_count"]
                    ].corr().iloc[0, 1]
                ),
                2,
            ),
        },
        "members": {
            "distinct_attendees_rounded": int(round(len(member_class_counts), -1)),
            "classes_per_member_quantiles": _rounded_quantiles(
                pd.Series(list(member_class_counts.values()), dtype="float64")
            ),
        },
    }


def _collect_member_ids(frame: pd.DataFrame) -> set[str]:
    """Collect member identifiers without exposing other waiting-list fields.

    Parameters
    ----------
    frame : pd.DataFrame
        Canonical event or attendance table containing member-list columns.

    Returns
    -------
    set[str]
        IDs found in attendance lists and waiting-list ``ID`` fields.
    """

    member_ids: set[str] = set()
    for column in ["attendance_list", "waiting_list"]:
        for member_list in frame[column]:
            if not isinstance(member_list, list):
                continue
            for member in member_list:
                if isinstance(member, str):
                    member_ids.add(member)
                elif isinstance(member, dict) and isinstance(member.get("ID"), str):
                    member_ids.add(member["ID"])
    return member_ids


def _raw_row_fingerprints(frame: pd.DataFrame) -> set[tuple[str, ...]]:
    """Convert raw rows to comparable string fingerprints in memory.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw source table.

    Returns
    -------
    set[tuple[str, ...]]
        Exact full-row fingerprints used only for local overlap checks.
    """

    normalized = frame.fillna("<missing>").astype(str)
    return set(normalized.itertuples(index=False, name=None))


def _class_summary_fingerprints(
    booking_events: pd.DataFrame,
    attendance: pd.DataFrame,
) -> set[tuple[int, ...]]:
    """Build coarse high-dimensional class summaries for copy detection.

    Parameters
    ----------
    booking_events : pd.DataFrame
        Canonical booking snapshots.
    attendance : pd.DataFrame
        Canonical final attendance rows.

    Returns
    -------
    set[tuple[int, ...]]
        Behavioral fingerprints that intentionally omit labels and dates.
    """

    events = booking_events.copy()
    events["booked_count"] = events["attendance_list"].map(_member_count)
    events["waiting_count"] = events["waiting_list"].map(_member_count)
    summaries = events.groupby(_CLASS_COLUMNS).agg(
        event_count=("event_timestamp", "size"),
        maximum_booking=("booked_count", "max"),
        maximum_waiting=("waiting_count", "max"),
    )
    outcomes = attendance.assign(
        final_count=attendance["attendance_list"].map(_member_count)
    ).set_index(_CLASS_COLUMNS)
    combined = summaries.join(outcomes[["capacity", "final_count"]], how="inner")
    return {
        tuple(int(value) for value in row)
        for row in combined[
            [
                "capacity",
                "event_count",
                "maximum_booking",
                "maximum_waiting",
                "final_count",
            ]
        ].dropna().itertuples(index=False, name=None)
    }


def _member_summary_fingerprints(
    attendance: pd.DataFrame,
) -> set[tuple[int, int, int, int]]:
    """Build coarse member activity summaries for copy-detection review.

    Parameters
    ----------
    attendance : pd.DataFrame
        Canonical final attendance records.

    Returns
    -------
    set[tuple[int, int, int, int]]
        Activity count, active-year count, and rounded studio/course
        concentration fingerprints. IDs and category labels are omitted.
    """

    member_rows: list[dict[str, object]] = []
    for row in attendance.itertuples(index=False):
        if not isinstance(row.attendance_list, list):
            continue
        for member_id in set(row.attendance_list):
            member_rows.append(
                {
                    "member_id": member_id,
                    "year": row.class_start.year,
                    "studio": row.studio,
                    "course": row.course,
                }
            )
    if not member_rows:
        return set()

    member_history = pd.DataFrame(member_rows)
    fingerprints: set[tuple[int, int, int, int]] = set()
    for _, history in member_history.groupby("member_id"):
        studio_concentration = history["studio"].value_counts(normalize=True).max()
        course_concentration = history["course"].value_counts(normalize=True).max()
        fingerprints.add(
            (
                len(history),
                history["year"].nunique(),
                int(round(studio_concentration * 10)),
                int(round(course_concentration * 10)),
            )
        )
    return fingerprints


def run_privacy_sanity_checks(
    original_booking_raw: pd.DataFrame,
    original_attendance_raw: pd.DataFrame,
    synthetic_booking_raw: pd.DataFrame,
    synthetic_attendance_raw: pd.DataFrame,
    generated_paths: dict[str, Path] | None = None,
) -> dict[str, bool | float]:
    """Check for direct overlap between private and synthetic datasets.

    Parameters
    ----------
    original_booking_raw : pd.DataFrame
        Private raw booking events loaded locally.
    original_attendance_raw : pd.DataFrame
        Private raw final attendance loaded locally.
    synthetic_booking_raw : pd.DataFrame
        Generated public raw booking events.
    synthetic_attendance_raw : pd.DataFrame
        Generated public raw final attendance.
    generated_paths : dict[str, Path] | None, optional
        Known files written by the current generation operation. When
        provided, their filenames are checked against the two expected CSV
        outputs.

    Returns
    -------
    dict[str, bool | float]
        Direct non-overlap checks and a diagnostic behavioral-summary overlap
        share. These are practical sanity checks, not a privacy proof.
    """

    original_booking = prepare_booking_events(original_booking_raw)
    original_attendance = prepare_attendance(original_attendance_raw)
    synthetic_booking = prepare_booking_events(synthetic_booking_raw)
    synthetic_attendance = prepare_attendance(synthetic_attendance_raw)

    original_member_ids = _collect_member_ids(original_booking) | _collect_member_ids(
        original_attendance
    )
    synthetic_member_ids = _collect_member_ids(
        synthetic_booking
    ) | _collect_member_ids(synthetic_attendance)
    original_class_keys = set(
        original_attendance[_CLASS_COLUMNS].itertuples(index=False, name=None)
    )
    synthetic_class_keys = set(
        synthetic_attendance[_CLASS_COLUMNS].itertuples(index=False, name=None)
    )
    original_class_summaries = _class_summary_fingerprints(
        original_booking,
        original_attendance,
    )
    synthetic_class_summaries = _class_summary_fingerprints(
        synthetic_booking,
        synthetic_attendance,
    )
    summary_denominator = max(len(synthetic_class_summaries), 1)
    original_member_summaries = _member_summary_fingerprints(original_attendance)
    synthetic_member_summaries = _member_summary_fingerprints(synthetic_attendance)
    member_summary_denominator = max(len(synthetic_member_summaries), 1)

    checks: dict[str, bool | float] = {
        "no_member_id_overlap": not bool(
            original_member_ids & synthetic_member_ids
        ),
        "synthetic_member_namespace_only": all(
            member_id.startswith("member_") for member_id in synthetic_member_ids
        ),
        "no_studio_name_overlap": not bool(
            set(original_attendance["studio"])
            & set(synthetic_attendance["studio"])
        ),
        "no_instructor_name_overlap": not bool(
            set(original_attendance["instructor"])
            & set(synthetic_attendance["instructor"])
        ),
        "no_exact_class_key_overlap": not bool(
            original_class_keys & synthetic_class_keys
        ),
        "no_exact_booking_row_overlap": not bool(
            _raw_row_fingerprints(original_booking_raw)
            & _raw_row_fingerprints(synthetic_booking_raw)
        ),
        "no_exact_attendance_row_overlap": not bool(
            _raw_row_fingerprints(original_attendance_raw)
            & _raw_row_fingerprints(synthetic_attendance_raw)
        ),
        "class_summary_exact_overlap_share": round(
            len(original_class_summaries & synthetic_class_summaries)
            / summary_denominator,
            3,
        ),
        "member_summary_exact_overlap_share": round(
            len(original_member_summaries & synthetic_member_summaries)
            / member_summary_denominator,
            3,
        ),
    }

    if generated_paths is not None:
        expected_filenames = {
            RAW_EVENT_FILENAME,
            RAW_ATTENDANCE_FILENAME,
        }
        written_filenames = {path.name for path in generated_paths.values()}
        checks["only_expected_generation_files_written"] = (
            written_filenames <= expected_filenames
        )

    return checks
