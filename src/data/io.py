"""Input utilities for raw datasets.

This module loads the original or synthetic datasets without modifying their
contents. Parsing, canonical representation, cleaning, and validation are
handled elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_EVENT_FILENAME = "event_log.csv"
RAW_ATTENDANCE_FILENAME = "attendance_log.csv"


def load_data(use_synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw booking-events and attendance datasets from CSV files.

    Parameters
    ----------
    use_synthetic : bool
        If ``True``, load the deterministic synthetic CSV files from
        ``data/synthetic``. Otherwise load the production CSV files from
        ``data/raw``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two unmodified pandas DataFrames containing the booking-events table
        and the attendance table in their original raw schema.
    """

    data_dir = PROJECT_ROOT / "data" / ("synthetic" if use_synthetic else "raw")

    booking_events = pd.read_csv(data_dir / RAW_EVENT_FILENAME)
    attendance = pd.read_csv(data_dir / RAW_ATTENDANCE_FILENAME)

    return booking_events, attendance
