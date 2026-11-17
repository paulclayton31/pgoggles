"""OC-SORT observation-centric corrections layered on top of the SORT core.

Three pieces, each usable independently:

- :func:`direction_consistency_matrix` / :func:`augmented_cost_matrix` --
  Observation-Centric Momentum (OCM). Penalizes/rewards candidate matches
  during assignment based on whether a detection continues a track's
  established real-world motion direction, computed from real observations
  rather than the (potentially drifted) Kalman-predicted state.
- :func:`interpolate_boxes` / :func:`reupdate_kalman` -- Observation-centric
  Re-Update (ORU). When a lost track is re-matched, rebuilds its Kalman
  velocity state by replaying a straight-line interpolated trajectory
  between the last real observation and the new one, instead of trusting
  whatever drift accumulated during blind coasting.

Observation-Centric Recovery (OCR, the secondary matching pass against
``last_observation`` instead of the predicted box) has no dedicated helper
here -- it's just another call to :func:`pgoggles.tracking.iou.iou_matrix`
and one of :mod:`pgoggles.tracking.assignment`'s match functions, orchestrated
in ``tracker.py``.

This module is a leaf: it depends only on numpy, ``iou.py``'s ``Box`` type,
and ``kalman.py``'s ``KalmanBoxTracker`` -- it does not import ``Track``, so
``track.py`` can import from here without a cycle.
"""

from __future__ import annotations

import numpy as np

from .iou import Box
from .kalman import KalmanBoxTracker


def _center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _normalized(vector: tuple[float, float]) -> tuple[float, float] | None:
    """Unit vector, or ``None`` if ``vector`` has ~zero magnitude."""

    norm = float(np.hypot(vector[0], vector[1]))
    if norm < 1e-6:
        return None
    return (vector[0] / norm, vector[1] / norm)


def direction_consistency_matrix(
    track_histories: list[list[Box]],
    detections: list[Box],
    delta_t: int,
) -> np.ndarray:
    """Cosine-similarity matrix (``n_tracks x n_dets``) of motion direction agreement.

    For each track, the "established direction" is the center-movement
    vector from an observation ``delta_t`` real-observation steps back
    (clamped to the oldest available) to the most recent real observation.
    Tracks with fewer than two real observations -- or with a track-side
    vector too small to define a direction -- contribute a neutral row of
    ``0.0`` (neither rewarding nor penalizing any candidate).

    For each candidate detection, the "candidate direction" is the
    center-movement vector from the track's most recent real observation to
    the detection. A near-zero candidate vector (detection sits essentially
    on the last observation) is also scored ``0.0``, not ``NaN``.

    Parameters
    ----------
    track_histories : list[list[Box]]
        Per track, real observations ordered oldest -> newest.
    detections : list[Box]
        This frame's candidate detection boxes.
    delta_t : int
        How many real-observation steps back to look for the track-side
        direction vector, smoothing over single-frame detector noise.
    """

    n_tracks = len(track_histories)
    n_dets = len(detections)
    matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
    if n_tracks == 0 or n_dets == 0:
        return matrix

    det_centers = [_center(d) for d in detections]

    for i, history in enumerate(track_histories):
        if len(history) < 2:
            continue

        newest = history[-1]
        back_idx = max(0, len(history) - 1 - delta_t)
        reference = history[back_idx]

        newest_center = _center(newest)
        reference_center = _center(reference)
        track_vector = (
            newest_center[0] - reference_center[0],
            newest_center[1] - reference_center[1],
        )
        track_unit = _normalized(track_vector)
        if track_unit is None:
            continue

        for j, det_center in enumerate(det_centers):
            candidate_vector = (
                det_center[0] - newest_center[0],
                det_center[1] - newest_center[1],
            )
            candidate_unit = _normalized(candidate_vector)
            if candidate_unit is None:
                continue
            matrix[i, j] = track_unit[0] * candidate_unit[0] + track_unit[1] * candidate_unit[1]

    return matrix


def augmented_cost_matrix(
    iou_mat: np.ndarray,
    track_histories: list[list[Box]],
    detections: list[Box],
    ocm_weight: float,
    delta_t: int,
) -> np.ndarray:
    """``iou_mat`` blended with direction-consistency (OCM), gated on raw overlap.

    ``augmented[t, d] = iou_mat[t, d] + ocm_weight * direction_consistency[t, d]``
    wherever ``iou_mat[t, d] > 0``, and exactly ``0.0`` everywhere else.

    The gate is a correctness requirement, not a style choice: without it, a
    pair of spatially-disjoint boxes (zero IoU -- almost certainly different
    objects) could still accumulate enough momentum bonus to cross
    ``iou_threshold`` purely from moving in a similar direction. Momentum may
    only re-rank or rescue a pair that already has *some* physical overlap;
    it can never fabricate a match out of zero overlap.
    """

    if iou_mat.size == 0:
        return iou_mat.copy()

    direction_mat = direction_consistency_matrix(track_histories, detections, delta_t)
    augmented = iou_mat + ocm_weight * direction_mat
    return np.where(iou_mat > 0, augmented, 0.0)


def interpolate_boxes(start: Box, end: Box, steps: int) -> list[Box]:
    """Linearly interpolate ``steps`` boxes from ``start`` to ``end``.

    Exclusive of ``start``, inclusive of ``end`` -- the last returned box is
    exactly ``end`` (not a floating-point approximation of it).

    Parameters
    ----------
    steps : int
        Must be >= 1. ``steps=1`` returns ``[end]``.
    """

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    boxes: list[Box] = []
    for i in range(1, steps + 1):
        t = i / steps
        boxes.append(tuple(s + (e - s) * t for s, e in zip(start, end)))

    boxes[-1] = tuple(float(v) for v in end)
    return boxes


def reupdate_kalman(last_observation: Box, new_observation: Box, steps: int) -> KalmanBoxTracker:
    """Reconstruct a fresh Kalman filter via straight-line replay (ORU).

    Builds a brand-new :class:`KalmanBoxTracker`, seeded at
    ``last_observation``, then replays ``steps`` virtual ``(predict,
    update)`` pairs through it along the interpolated straight line to
    ``new_observation`` -- so the returned filter's velocity state reflects
    the interpolated trajectory rather than whatever drift a filter that
    had instead taken ``steps`` consecutive blind ``predict()``-only calls
    would have accumulated.

    Deliberately returns a *new* filter rather than mutating one in place:
    the caller (:class:`~pgoggles.tracking.track.Track`) already advanced
    its live filter's state once this frame via its own :meth:`predict`
    call (``MultiObjectTracker`` calls ``predict()`` on every live track,
    every frame, before assignment) -- replaying on top of that
    already-advanced state would double-count a ``predict()`` step.
    Rebuilding fresh from ``last_observation`` avoids that entirely: the
    caller is expected to *replace* its filter with the one returned here,
    not layer this on top of the existing one.

    Purely a :class:`KalmanBoxTracker`-level operation -- does not touch any
    :class:`~pgoggles.tracking.track.Track` bookkeeping
    (``age``/``hits``/``time_since_update``); the caller owns that
    separately.
    """

    kf = KalmanBoxTracker(bbox=last_observation)
    for box in interpolate_boxes(last_observation, new_observation, steps):
        kf.predict()
        kf.update(box)
    return kf
