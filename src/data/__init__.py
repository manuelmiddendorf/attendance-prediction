"""Public data-layer interface for raw I/O and canonical representations.

This package groups the reusable dataset logic used across the notebooks and
later pipeline stages.
It exposes raw loading functions, canonical representation builders,
synthetic-data generators, and low-level validation helpers from one stable
import surface.
That keeps notebook code compact while ensuring that repeated data semantics
are defined only once inside ``src``.
"""

from .io import (
    PROJECT_ROOT,
    RAW_ATTENDANCE_FILENAME,
    RAW_EVENT_FILENAME,
    load_data,
)
from .representation import (
    prepare_attendance,
    prepare_booking_events,
)
from .synthetic import (
    DEFAULT_SYNTHETIC_SEED,
    SyntheticDatasets,
    create_synthetic_data,
    create_synthetic_raw_datasets,
    ensure_synthetic_data_files,
)

__all__ = [
    "DEFAULT_SYNTHETIC_SEED",
    "PROJECT_ROOT",
    "RAW_ATTENDANCE_FILENAME",
    "RAW_EVENT_FILENAME",
    "SyntheticDatasets",
    "create_synthetic_data",
    "create_synthetic_raw_datasets",
    "ensure_synthetic_data_files",
    "load_data",
    "prepare_attendance",
    "prepare_booking_events",
]
