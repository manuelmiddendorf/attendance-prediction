"""Generate calibrated, privacy-preserving synthetic studio data.

The module creates an entirely new population of members, classes, and
booking histories for the public notebook workflow. Its small generative
model is calibrated only to coarse aggregate properties of the private data.
Persistent member traits make historical reliability informative, while
class demand, late bookings, cancellations, no-shows, and waiting lists
produce realistic event-driven full-state snapshots. Generation is fully
deterministic for a fixed random seed and never reads private source files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import RAW_ATTENDANCE_FILENAME, RAW_EVENT_FILENAME

DEFAULT_SYNTHETIC_SEED = 20260825


@dataclass(frozen=True)
class SyntheticConfig:
    """Privacy-safe, rounded parameters for public synthetic generation.

    Parameters
    ----------
    attendance_start : str
        First Monday considered for the longer attendance-history period.
    attendance_end : str
        Last date considered for generated classes.
    booking_event_start : str
        Beginning of the shorter detailed booking-event observation period.
    member_count : int
        Size of the newly sampled synthetic member population.
    recent_member_count : int
        Stable recent cohort available during detailed event observation.
    event_coverage_probability : float
        Probability that a recent class receives detailed event snapshots.
    class_skip_probability : float
        Probability that a recurring scheduled class does not take place.
    booking_demand_offset : float
        Small global adjustment to typical booking-attempt volume.
    """

    attendance_start: str = "2020-08-03"
    attendance_end: str = "2026-07-12"
    booking_event_start: str = "2025-05-05"
    member_count: int = 700
    recent_member_count: int = 205
    event_coverage_probability: float = 0.75
    class_skip_probability: float = 0.04
    booking_demand_offset: float = 0.8


@dataclass(frozen=True)
class CourseSchedule:
    """Definition of one fictional recurring class.

    Parameters
    ----------
    studio : str
        Fictional studio label.
    course : str
        Course label containing weekday and start time.
    weekday : int
        Monday-based weekday index used to construct class dates.
    capacity : int
        Maximum number of simultaneously booked members.
    instructor : str
        Fictional instructor label.
    expected_bookings : float
        Typical number of booking attempts before class start.
    preference_group : int
        Broad synthetic preference group used for recurring member behavior.
    show_up_adjustment : float
        Small class-specific shift in show-up probability.
    """

    studio: str
    course: str
    weekday: int
    capacity: int
    instructor: str
    expected_bookings: float
    preference_group: int
    show_up_adjustment: float


@dataclass(frozen=True)
class SyntheticDatasets:
    """Container for generated raw-schema datasets.

    Parameters
    ----------
    event_log : pd.DataFrame
        Full-state booking snapshots using the original raw CSV headers.
    attendance_log : pd.DataFrame
        Final class attendance rows using the original raw CSV headers.
    """

    event_log: pd.DataFrame
    attendance_log: pd.DataFrame


PUBLIC_SYNTHETIC_CONFIG = SyntheticConfig()


def _course_schedules() -> list[CourseSchedule]:
    """Return the fully fictional recurring weekly schedule.

    Returns
    -------
    list[CourseSchedule]
        Synthetic classes with broad demand and capacity variation.
    """

    return [
        CourseSchedule("Studio A", "Mo 07:30", 0, 9, "Instructor A", 5.0, 0, 0.02),
        CourseSchedule("Studio A", "Mo 17:30", 0, 12, "Instructor B", 12.0, 1, -0.02),
        CourseSchedule("Studio A", "Di 09:30", 1, 9, "Instructor C", 6.5, 2, 0.03),
        CourseSchedule("Studio A", "Di 19:30", 1, 12, "Instructor A", 13.0, 3, -0.03),
        CourseSchedule("Studio A", "Mi 12:00", 2, 12, "Instructor D", 7.0, 4, 0.01),
        CourseSchedule("Studio A", "Do 08:30", 3, 9, "Instructor B", 5.5, 5, 0.04),
        CourseSchedule("Studio A", "Do 18:30", 3, 12, "Instructor E", 13.5, 0, -0.04),
        CourseSchedule("Studio A", "Fr 16:30", 4, 12, "Instructor C", 8.0, 1, 0.00),
        CourseSchedule("Studio A", "Sa 10:30", 5, 12, "Instructor D", 9.0, 2, 0.02),
        CourseSchedule("Studio B", "Mo 10:30", 0, 9, "Instructor F", 5.5, 3, 0.04),
        CourseSchedule("Studio B", "Di 07:30", 1, 9, "Instructor E", 4.5, 4, 0.03),
        CourseSchedule("Studio B", "Di 17:30", 1, 12, "Instructor D", 9.0, 5, -0.01),
        CourseSchedule("Studio B", "Mi 09:30", 2, 9, "Instructor A", 6.0, 0, 0.05),
        CourseSchedule("Studio B", "Mi 19:00", 2, 12, "Instructor F", 13.0, 1, -0.05),
        CourseSchedule("Studio B", "Do 12:30", 3, 12, "Instructor C", 7.5, 2, 0.00),
        CourseSchedule("Studio B", "Fr 08:00", 4, 9, "Instructor B", 4.5, 3, 0.04),
        CourseSchedule("Studio B", "Sa 16:00", 5, 12, "Instructor E", 8.0, 4, -0.02),
        CourseSchedule("Studio B", "So 11:30", 6, 12, "Instructor F", 9.5, 5, 0.01),
    ]


def _create_member_population(
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> pd.DataFrame:
    """Sample new members with persistent activity and reliability traits.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator controlling all member-level sampling.
    config : SyntheticConfig
        Public generation parameters including population size and dates.

    Returns
    -------
    pd.DataFrame
        Synthetic member parameters with no relationship to real individuals.
    """

    attendance_start = pd.Timestamp(config.attendance_start)
    attendance_end = pd.Timestamp(config.attendance_end)
    event_start = pd.Timestamp(config.booking_event_start)
    historical_count = config.member_count - config.recent_member_count

    historical_join_offsets = rng.integers(
        -730,
        max((event_start - attendance_start).days - 180, 1),
        size=historical_count,
    )
    historical_start = attendance_start + pd.to_timedelta(
        historical_join_offsets,
        unit="D",
    )
    historical_duration = np.clip(
        rng.gamma(shape=2.0, scale=420, size=historical_count),
        150,
        2_200,
    )
    historical_end = np.maximum(
        historical_start + pd.Timedelta(days=120),
        np.minimum(
            historical_start + pd.to_timedelta(historical_duration, unit="D"),
            event_start
            - pd.to_timedelta(
                rng.integers(15, 540, size=historical_count),
                unit="D",
            ),
        ),
    )

    recent_start = event_start - pd.to_timedelta(
        rng.integers(180, 1_100, size=config.recent_member_count),
        unit="D",
    )
    recent_end = attendance_end + pd.to_timedelta(
        rng.integers(90, 900, size=config.recent_member_count),
        unit="D",
    )
    active_start = pd.DatetimeIndex(historical_start).append(
        pd.DatetimeIndex(recent_start)
    )
    active_end = pd.DatetimeIndex(historical_end).append(pd.DatetimeIndex(recent_end))

    reliability_group = rng.choice(
        ["lower", "typical", "higher"],
        size=config.member_count,
        p=[0.20, 0.50, 0.30],
    )
    reliability = np.empty(config.member_count)
    group_parameters = {
        "lower": (2.2, 4.8),
        "typical": (7.0, 3.0),
        "higher": (14.0, 1.8),
    }
    for group, (alpha, beta) in group_parameters.items():
        mask = reliability_group == group
        reliability[mask] = rng.beta(alpha, beta, size=int(mask.sum()))

    return pd.DataFrame(
        {
            "member_id": [
                f"member_{member_number:04d}"
                for member_number in range(1, config.member_count + 1)
            ],
            "active_start": active_start,
            "active_end": active_end,
            "activity_weight": rng.lognormal(0.0, 1.1, config.member_count),
            "reliability": reliability,
            "cancellation_tendency": rng.beta(2.0, 7.0, config.member_count),
            "booking_lead_factor": rng.lognormal(0.0, 0.35, config.member_count),
            "preferred_studio": rng.choice(
                ["Studio A", "Studio B"],
                size=config.member_count,
            ),
            "preferred_group": rng.integers(0, 6, size=config.member_count),
            "preferred_instructor": rng.choice(
                [
                    "Instructor A",
                    "Instructor B",
                    "Instructor C",
                    "Instructor D",
                    "Instructor E",
                    "Instructor F",
                ],
                size=config.member_count,
            ),
        }
    )


def _holiday_context(class_day: pd.Timestamp) -> tuple[bool, bool]:
    """Return fictional holiday and holiday-week indicators for one date.

    Parameters
    ----------
    class_day : pd.Timestamp
        Date of the generated class.

    Returns
    -------
    tuple[bool, bool]
        Individual-holiday and broader holiday-week indicators.
    """

    day_of_year = class_day.dayofyear
    holiday_week = (
        day_of_year <= 7
        or 90 <= day_of_year <= 98
        or 211 <= day_of_year <= 225
        or day_of_year >= 354
    )
    holiday = (class_day.month, class_day.day) in {
        (1, 6),
        (5, 9),
        (8, 15),
        (12, 26),
    }
    return holiday, holiday_week


def _member_sampling_weights(
    active_members: pd.DataFrame,
    schedule: CourseSchedule,
) -> np.ndarray:
    """Calculate member selection weights for one class.

    Parameters
    ----------
    active_members : pd.DataFrame
        Members whose synthetic activity period includes the class date.
    schedule : CourseSchedule
        Current fictional course definition.

    Returns
    -------
    np.ndarray
        Positive probabilities combining activity and broad preferences.
    """

    weights = active_members["activity_weight"].to_numpy(copy=True)
    weights *= np.where(active_members["preferred_studio"] == schedule.studio, 2.0, 1.0)
    weights *= np.where(
        active_members["preferred_group"] == schedule.preference_group,
        2.4,
        1.0,
    )
    weights *= np.where(
        active_members["preferred_instructor"] == schedule.instructor,
        1.5,
        1.0,
    )
    return weights / weights.sum()


def _sample_booking_lead_hours(
    members: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample booking lead times from a rounded, long-tailed distribution.

    Parameters
    ----------
    members : pd.DataFrame
        Selected booking candidates with persistent lead-time tendencies.
    rng : np.random.Generator
        Random generator controlling lead-time sampling.

    Returns
    -------
    np.ndarray
        Lead times in hours, rounded to five-minute intervals and capped at
        one week before class.
    """

    base_lead = rng.lognormal(np.log(42.0), 1.05, len(members))
    lead_hours = base_lead * members["booking_lead_factor"].to_numpy()
    mixture_draw = rng.random(len(members))
    late_mask = mixture_draw < 0.12
    early_mask = mixture_draw > 0.82
    lead_hours[late_mask] = rng.uniform(0.0, 4.0, late_mask.sum())
    lead_hours[early_mask] = rng.uniform(96.0, 168.0, early_mask.sum())
    return np.round(np.clip(lead_hours, 0.0, 168.0) * 12) / 12


def _waiting_entry(member_id: str) -> dict[str, object]:
    """Create one fictional raw waiting-list dictionary.

    Parameters
    ----------
    member_id : str
        Independently generated synthetic member identifier.

    Returns
    -------
    dict[str, object]
        Non-private dictionary resembling the raw waiting-list structure.
    """

    member_number = member_id.removeprefix("member_")
    return {
        "ID": member_id,
        "email": f"member-{member_number}@synthetic.invalid",
        "name": f"Synthetic Member {member_number}",
    }


def _serialize(value: object) -> str:
    """Serialize one raw JSON field in a deterministic compact format.

    Parameters
    ----------
    value : object
        JSON-compatible member or waiting-list collection.

    Returns
    -------
    str
        Compact JSON representation suitable for the raw CSV schema.
    """

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _append_snapshot(
    rows: list[dict[str, object]],
    schedule: CourseSchedule,
    class_start: pd.Timestamp,
    event_timestamp: pd.Timestamp,
    booked_members: list[str],
    waiting_members: list[str],
    is_holiday: bool,
    is_holiday_week: bool,
    event_sequence: int,
) -> None:
    """Append a complete booking-state snapshot after one simulated event.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Accumulator for raw booking-event records.
    schedule : CourseSchedule
        Current fictional course definition.
    class_start : pd.Timestamp
        Start timestamp of the generated class.
    event_timestamp : pd.Timestamp
        Time of the simulated state-changing event.
    booked_members : list[str]
        Members booked immediately after the event.
    waiting_members : list[str]
        Members waiting immediately after the event.
    is_holiday : bool
        Whether the class date is an individual synthetic holiday.
    is_holiday_week : bool
        Whether the class falls in a synthetic holiday period.
    event_sequence : int
        Stable order for snapshots sharing the same timestamp.

    Returns
    -------
    None
        The snapshot is appended to ``rows`` in place.
    """

    waiting_entries = [_waiting_entry(member_id) for member_id in waiting_members]
    rows.append(
        {
            "studio": schedule.studio,
            "course": schedule.course,
            "attendance list": _serialize(booked_members),
            "current_date": event_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "class_date": class_start.normalize().strftime("%Y-%m-%d"),
            "is_holiday": is_holiday,
            "is_holiday_week": is_holiday_week,
            "maxnr": schedule.capacity,
            "waiting list": _serialize(waiting_entries),
            "warteliste_length": len(waiting_entries),
            "_event_sequence": event_sequence,
        }
    )


def _simulate_class(
    schedule: CourseSchedule,
    class_start: pd.Timestamp,
    members: pd.DataFrame,
    record_events: bool,
    booking_demand_offset: float,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Simulate bookings, cancellations, waiting lists, and attendance.

    Parameters
    ----------
    schedule : CourseSchedule
        Current fictional recurring class.
    class_start : pd.Timestamp
        Start timestamp for this newly generated class.
    members : pd.DataFrame
        Complete synthetic population with persistent latent traits.
    record_events : bool
        Whether detailed full-state booking snapshots should be returned.
    booking_demand_offset : float
        Public global adjustment to expected booking-attempt volume.
    rng : np.random.Generator
        Random generator controlling this class simulation.

    Returns
    -------
    tuple[dict[str, object], list[dict[str, object]]]
        One final-attendance row and zero or more booking snapshots.
    """

    class_day = class_start.normalize()
    is_holiday, is_holiday_week = _holiday_context(class_day)
    seasonal_adjustment = {
        1: -0.8,
        2: -0.2,
        3: 0.3,
        4: 0.2,
        5: 0.0,
        6: -0.2,
        7: -0.8,
        8: -0.6,
        9: 0.4,
        10: 0.5,
        11: 0.2,
        12: -0.9,
    }[class_start.month]
    context_adjustment = seasonal_adjustment
    context_adjustment -= 1.8 if is_holiday else 0.0
    context_adjustment -= 0.9 if is_holiday_week else 0.0

    expected_bookings = max(
        0.5,
        schedule.expected_bookings
        + booking_demand_offset
        + context_adjustment
        + rng.normal(0.0, 1.25),
    )
    booking_attempt_count = int(
        np.clip(rng.poisson(expected_bookings), 0, schedule.capacity + 6)
    )

    active_members = members.loc[
        (members["active_start"] <= class_start)
        & (members["active_end"] >= class_start)
    ]
    booking_attempt_count = min(booking_attempt_count, len(active_members))
    if booking_attempt_count:
        selected_indices = rng.choice(
            active_members.index,
            size=booking_attempt_count,
            replace=False,
            p=_member_sampling_weights(active_members, schedule),
        )
        candidates = members.loc[selected_indices].copy()
        candidates["booking_lead_hours"] = _sample_booking_lead_hours(candidates, rng)
    else:
        candidates = members.iloc[0:0].copy()
        candidates["booking_lead_hours"] = pd.Series(dtype="float64")

    actions: list[tuple[float, int, str, str]] = []
    for candidate in candidates.itertuples(index=False):
        actions.append(
            (
                float(candidate.booking_lead_hours),
                0,
                "book",
                candidate.member_id,
            )
        )
        cancellation_probability = np.clip(
            0.03
            + 0.40 * candidate.cancellation_tendency
            + 0.45 * (1.0 - candidate.reliability),
            0.02,
            0.50,
        )
        if rng.random() < cancellation_probability:
            cancellation_lead = float(
                candidate.booking_lead_hours
                * rng.beta(1.1, 4.0)
            )
            actions.append((cancellation_lead, 1, "cancel", candidate.member_id))

    actions.sort(key=lambda action: (-action[0], action[1], action[3]))
    booked_members: list[str] = []
    waiting_members: list[str] = []
    snapshot_rows: list[dict[str, object]] = []
    event_sequence = 0

    for lead_hours, _, action, member_id in actions:
        event_timestamp = class_start - pd.Timedelta(hours=lead_hours)

        if action == "book":
            if len(booked_members) < schedule.capacity:
                booked_members.append(member_id)
            else:
                waiting_members.append(member_id)
        elif member_id in booked_members:
            booked_members.remove(member_id)
        elif member_id in waiting_members:
            waiting_members.remove(member_id)
        else:
            continue

        if record_events:
            _append_snapshot(
                snapshot_rows,
                schedule,
                class_start,
                event_timestamp,
                booked_members,
                waiting_members,
                is_holiday,
                is_holiday_week,
                event_sequence,
            )
            event_sequence += 1

        if (
            action == "cancel"
            and waiting_members
            and len(booked_members) < schedule.capacity
        ):
            promoted_member = waiting_members.pop(0)
            booked_members.append(promoted_member)
            if record_events:
                _append_snapshot(
                    snapshot_rows,
                    schedule,
                    class_start,
                    event_timestamp,
                    booked_members,
                    waiting_members,
                    is_holiday,
                    is_holiday_week,
                    event_sequence,
                )
                event_sequence += 1

    booked_details = members.set_index("member_id").loc[booked_members]
    class_show_adjustment = schedule.show_up_adjustment
    class_show_adjustment -= 0.08 if is_holiday_week else 0.0
    class_show_adjustment -= 0.12 if is_holiday else 0.0
    class_show_adjustment += rng.normal(0.0, 0.10)
    if rng.random() < 0.09:
        class_show_adjustment -= rng.uniform(0.18, 0.36)

    final_attendees: list[str] = []
    for member_id, member in booked_details.iterrows():
        show_up_probability = np.clip(
            0.14 + 0.93 * member["reliability"] + class_show_adjustment,
            0.05,
            0.98,
        )
        if rng.random() < show_up_probability:
            final_attendees.append(member_id)

    waiting_entries = [_waiting_entry(member_id) for member_id in waiting_members]
    attendance_row = {
        "studio": schedule.studio,
        "course": schedule.course,
        "attendance list": _serialize(final_attendees),
        "waiting list": _serialize(waiting_entries),
        "instructor": schedule.instructor,
        "date": (
            class_start.tz_localize("Europe/Berlin")
            .tz_convert("UTC")
            .strftime("%Y-%m-%dT%H:%M:%S.000Z")
        ),
        "maxnr": schedule.capacity,
    }
    return attendance_row, snapshot_rows


def create_synthetic_raw_datasets(
    seed: int = DEFAULT_SYNTHETIC_SEED,
    config: SyntheticConfig = PUBLIC_SYNTHETIC_CONFIG,
) -> SyntheticDatasets:
    """Create calibrated synthetic datasets using the original raw headers.

    Parameters
    ----------
    seed : int, optional
        Random seed that makes the complete simulation deterministic.
    config : SyntheticConfig, optional
        Privacy-safe public parameters controlling scale and time coverage.

    Returns
    -------
    SyntheticDatasets
        Newly simulated event and attendance DataFrames in raw CSV form.
    """

    rng = np.random.default_rng(seed)
    members = _create_member_population(rng, config)
    schedules = _course_schedules()
    week_starts = pd.date_range(
        config.attendance_start,
        config.attendance_end,
        freq="W-MON",
    )
    event_start = pd.Timestamp(config.booking_event_start)

    attendance_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for week_start in week_starts:
        for schedule in schedules:
            class_day = week_start + pd.Timedelta(days=schedule.weekday)
            if class_day > pd.Timestamp(config.attendance_end):
                continue
            _, holiday_week = _holiday_context(class_day)
            skip_probability = config.class_skip_probability + (
                0.08 if holiday_week else 0.0
            )
            if rng.random() < skip_probability:
                continue

            class_time = schedule.course.split()[1]
            class_start = pd.Timestamp(f"{class_day.date()} {class_time}")
            record_events = (
                class_start >= event_start
                and rng.random() < config.event_coverage_probability
            )
            attendance_row, class_event_rows = _simulate_class(
                schedule,
                class_start,
                members,
                record_events,
                config.booking_demand_offset,
                rng,
            )
            attendance_rows.append(attendance_row)
            event_rows.extend(class_event_rows)

    attendance_log = (
        pd.DataFrame(attendance_rows)
        .sort_values(["studio", "course", "date"], kind="stable")
        .reset_index(drop=True)
    )
    event_log = (
        pd.DataFrame(event_rows)
        .sort_values(
            [
                "studio",
                "course",
                "class_date",
                "current_date",
                "_event_sequence",
            ],
            kind="stable",
        )
        .drop(columns="_event_sequence")
        .reset_index(drop=True)
    )

    return SyntheticDatasets(event_log=event_log, attendance_log=attendance_log)


def ensure_synthetic_data_files(
    repo_root: Path,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    overwrite: bool = False,
    config: SyntheticConfig = PUBLIC_SYNTHETIC_CONFIG,
) -> dict[str, Path]:
    """Write deterministic synthetic CSVs to ``data/synthetic``.

    Parameters
    ----------
    repo_root : Path
        Repository root containing the public ``data`` directory.
    seed : int, optional
        Random seed used for deterministic generation.
    overwrite : bool, optional
        Whether existing public synthetic CSVs should be replaced.
    config : SyntheticConfig, optional
        Privacy-safe public parameters used by the generator.

    Returns
    -------
    dict[str, Path]
        Paths to the two generated or reused CSV files. No mapping artifacts
        are created.
    """

    data_dir = repo_root / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    event_path = data_dir / RAW_EVENT_FILENAME
    attendance_path = data_dir / RAW_ATTENDANCE_FILENAME

    if overwrite or not event_path.exists() or not attendance_path.exists():
        datasets = create_synthetic_raw_datasets(seed=seed, config=config)
        datasets.event_log.to_csv(event_path, index=False)
        datasets.attendance_log.to_csv(attendance_path, index=False)

    return {"event_log": event_path, "attendance_log": attendance_path}


def create_synthetic_data(
    seed: int = DEFAULT_SYNTHETIC_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return both raw-schema synthetic DataFrames.

    Parameters
    ----------
    seed : int, optional
        Random seed used for deterministic generation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Synthetic event and attendance tables using the original CSV schema.
    """

    datasets = create_synthetic_raw_datasets(seed=seed)
    return datasets.event_log.copy(), datasets.attendance_log.copy()
