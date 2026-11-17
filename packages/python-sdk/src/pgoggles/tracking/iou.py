"""IoU (Intersection-over-Union) geometry for multi-object tracking.

Operates on plain ``(x1, y1, x2, y2)`` float tuples rather than
``BoundingBox``/``TrackedBox`` directly, so it stays trivially
unit-testable with synthetic boxes and is reusable from both raw
detections and Kalman-predicted track state.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    """Return the Intersection-over-Union of two ``(x1, y1, x2, y2)`` boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def iou_matrix(tracks: list[Box], detections: list[Box]) -> np.ndarray:
    """Return an ``(len(tracks), len(detections))`` matrix of pairwise IoU."""

    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for i, track_box in enumerate(tracks):
        for j, det_box in enumerate(detections):
            matrix[i, j] = iou(track_box, det_box)
    return matrix
