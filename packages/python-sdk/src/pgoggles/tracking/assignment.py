"""Track-to-detection assignment strategies for MultiObjectTracker.

Both functions take an IoU cost matrix (rows = live tracks, columns =
current-frame detections) and a minimum-IoU gate, and return the same
shape of result, so ``MultiObjectTracker`` can swap strategies without
branching on the return type.
"""

from __future__ import annotations

import numpy as np

# (matched (track_idx, det_idx) pairs, unmatched track indices, unmatched detection indices)
MatchResult = tuple[list[tuple[int, int]], list[int], list[int]]


def greedy_match(iou_mat: np.ndarray, iou_threshold: float) -> MatchResult:
    """Repeatedly pick the highest-IoU pair until none clear the gate.

    Correct and sufficient whenever there are 0-1 tracks and 0-1
    detections (there is at most one possible pairing to begin with),
    and a cheap, adequate approximation for the general sparse-scene case
    ``TrackingMode.KALMAN_IOU`` targets.
    """

    n_tracks, n_dets = iou_mat.shape
    unmatched_tracks = set(range(n_tracks))
    unmatched_dets = set(range(n_dets))
    matches: list[tuple[int, int]] = []

    if n_tracks == 0 or n_dets == 0:
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    # Flatten and sort all (track, det) pairs by IoU descending, then greedily
    # accept the best remaining pair whose row/column are both still free.
    pairs = [
        (iou_mat[t, d], t, d)
        for t in range(n_tracks)
        for d in range(n_dets)
        if iou_mat[t, d] >= iou_threshold
    ]
    pairs.sort(key=lambda item: item[0], reverse=True)

    for score, t, d in pairs:
        if t in unmatched_tracks and d in unmatched_dets:
            matches.append((t, d))
            unmatched_tracks.discard(t)
            unmatched_dets.discard(d)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def hungarian_match(iou_mat: np.ndarray, iou_threshold: float) -> MatchResult:
    """Globally-optimal assignment via ``scipy.optimize.linear_sum_assignment``.

    ``scipy`` is imported lazily inside this function -- importing
    ``pgoggles.tracking`` never requires scipy, only calling this function
    does. When ``iou_mat`` has shape ``(<=1, <=1)``, this short-circuits to
    :func:`greedy_match` (provably equivalent when there is at most one
    possible pairing), skipping the scipy import entirely -- this is the
    only place that optimization lives, so ``MultiObjectTracker`` itself
    never has to special-case matrix shape.

    Raises
    ------
    ImportError
        Re-raised with an actionable message if scipy is missing and the
        matrix is larger than the trivial case above.
    """

    n_tracks, n_dets = iou_mat.shape
    if n_tracks <= 1 and n_dets <= 1:
        return greedy_match(iou_mat, iou_threshold)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "TrackingMode.FULL requires scipy for Hungarian assignment. "
            "Install it with: pip install pgoggles[tracking]"
        ) from exc

    # linear_sum_assignment minimizes cost; IoU is a similarity (higher is
    # better), so negate it to turn "maximize IoU" into "minimize cost".
    row_idx, col_idx = linear_sum_assignment(-iou_mat)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for t, d in zip(row_idx, col_idx):
        if iou_mat[t, d] >= iou_threshold:
            matches.append((int(t), int(d)))
            matched_tracks.add(int(t))
            matched_dets.add(int(d))

    unmatched_tracks = [t for t in range(n_tracks) if t not in matched_tracks]
    unmatched_dets = [d for d in range(n_dets) if d not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets
