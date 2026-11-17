"""Multi-object tracking for the pgoggles analysis pipeline.

Kalman filter + IoU + (greedy or Hungarian) assignment, producing
persistent ``track_id``s across frames. See ``MultiObjectTracker`` for the
main entry point; ``TrackedBox``/``TrackingMode`` live in ``pgoggles.overlays``
(the SDK's canonical location for pipeline dataclasses/enums) and are
re-exported here for convenience.
"""

from ..overlays import TrackedBox, TrackingMode
from .track import Track
from .tracker import MultiObjectTracker

__all__ = [
    "MultiObjectTracker",
    "Track",
    "TrackedBox",
    "TrackingMode",
]
