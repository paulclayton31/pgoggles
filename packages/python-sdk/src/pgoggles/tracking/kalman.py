"""Constant-velocity Kalman filter for single-object box tracking.

Hand-rolled, numpy-only implementation of the SORT-style box filter
(Bewley et al., "Simple Online and Realtime Tracking") — no ``filterpy``
dependency, since a 7-state linear system is small enough to own directly
with plain matrix algebra.

State vector ``x = [cx, cy, s, r, vx, vy, vs]``:
    cx, cy -- box center
    s      -- scale (area = width * height)
    r      -- aspect ratio (assumed constant; no ``vr`` term)
    vx, vy, vs -- velocities for cx, cy, s

Measurement vector ``z = [cx, cy, s, r]`` (aspect ratio observed directly,
not predicted).
"""

from __future__ import annotations

import numpy as np

from .iou import Box

_STATE_DIM = 7
_MEASUREMENT_DIM = 4

# Constant-velocity state transition: position/scale accumulate their
# paired velocity terms each step; aspect ratio and velocities carry over
# unchanged.
_F = np.eye(_STATE_DIM)
_F[0, 4] = 1.0  # cx += vx
_F[1, 5] = 1.0  # cy += vy
_F[2, 6] = 1.0  # s  += vs

# Measurement observes cx, cy, s, r directly (no velocity terms).
_H = np.zeros((_MEASUREMENT_DIM, _STATE_DIM))
_H[0, 0] = 1.0
_H[1, 1] = 1.0
_H[2, 2] = 1.0
_H[3, 3] = 1.0

_EPS = 1e-6


def _bbox_to_z(bbox: Box) -> np.ndarray:
    """``(x1, y1, x2, y2)`` -> ``[cx, cy, s, r]``."""

    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, _EPS)
    h = max(y2 - y1, _EPS)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([cx, cy, s, r], dtype=np.float64)


def _x_to_bbox(x: np.ndarray) -> Box:
    """``[cx, cy, s, r, ...]`` -> ``(x1, y1, x2, y2)``."""

    cx, cy, s, r = x[0], x[1], x[2], x[3]
    s = max(s, _EPS)
    r = max(r, _EPS)
    w = float(np.sqrt(s * r))
    h = float(s / w) if w > _EPS else _EPS
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxTracker:
    """Tracks a single box's position/scale over time with a linear KF."""

    def __init__(self, bbox: Box) -> None:
        self._x = np.zeros((_STATE_DIM,), dtype=np.float64)
        self._x[:4] = _bbox_to_z(bbox)

        # High initial uncertainty on the (unobservable-at-birth) velocity
        # terms, moderate uncertainty on the directly-observed position/scale
        # -- standard SORT initialization.
        self._P = np.eye(_STATE_DIM) * 10.0
        self._P[4:, 4:] *= 100.0

        # Process noise: small overall, larger still on the scale-velocity
        # term (area changes are the noisiest signal to predict).
        self._Q = np.eye(_STATE_DIM) * 0.01
        self._Q[4:, 4:] *= 0.01
        self._Q[-1, -1] *= 0.01

        # Measurement noise: position/aspect trusted more than scale.
        self._R = np.eye(_MEASUREMENT_DIM)
        self._R[2:, 2:] *= 10.0

    def predict(self) -> Box:
        """Advance state by one step; return the predicted box."""

        self._x = _F @ self._x
        self._P = _F @ self._P @ _F.T + self._Q
        return _x_to_bbox(self._x)

    def update(self, bbox: Box) -> None:
        """Correct state toward a matched detection."""

        z = _bbox_to_z(bbox)
        y = z - _H @ self._x
        s = _H @ self._P @ _H.T + self._R
        k = self._P @ _H.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(_STATE_DIM) - k @ _H) @ self._P

    def to_bbox(self) -> Box:
        """Current state as ``(x1, y1, x2, y2)``, without advancing time."""

        return _x_to_bbox(self._x)
