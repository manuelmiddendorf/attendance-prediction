"""Public interface for feature-engineering steps used in the notebooks.

This package contains reusable transformations that convert modeling-ready
prediction instances into feature tables for later machine-learning stages.
It keeps notebook code concise by moving feature logic into one canonical
implementation under ``src``.
The package is intentionally lightweight and currently exposes only the
baseline, historical, member-history, member-reliability, booking-dynamics,
and class-context feature sets used in Notebook 03.
"""

from .baseline import add_baseline_features
from .booking_dynamics import add_booking_dynamics_features
from .class_context import add_class_context_features
from .historical import add_historical_features
from .member_history import add_member_history_features
from .member_reliability import add_member_reliability_features

__all__ = [
    "add_baseline_features",
    "add_booking_dynamics_features",
    "add_class_context_features",
    "add_historical_features",
    "add_member_history_features",
    "add_member_reliability_features",
]
