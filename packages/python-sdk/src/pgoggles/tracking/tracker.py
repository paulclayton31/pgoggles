"""Multi-object tracker: orchestrates predict -> assign -> update -> prune per frame."""

from __future__ import annotations

from ..overlays import BoundingBox, TrackedBox, TrackingMode
from .assignment import greedy_match, hungarian_match
from .iou import iou_matrix
from .oc_sort import augmented_cost_matrix
from .track import Track


class MultiObjectTracker:
    """Assign persistent track IDs to per-frame detections.

    Call :meth:`update` once per frame with that frame's raw detections
    (e.g. ``OverlayState.boxes``); it returns a list of :class:`TrackedBox`
    for every currently-confirmed track.

    Tracking is bucketed per detection ``label`` internally -- a "car" and
    a "person" box are never matched to the same track, even with high
    spatial IoU, since a :class:`Track`'s label is fixed at creation.
    ``track_id`` values are still drawn from one global monotonic counter
    across all labels, so IDs never collide across classes.

    Built on OC-SORT (Cao et al. 2022) on top of the plain SORT core
    (Kalman filter + IoU + greedy/Hungarian assignment): three
    observation-centric corrections, each independently toggleable and
    default-on, address the plain constant-velocity model's weakness
    against real (non-straight-line) motion:

    - **OCM** (``use_ocm``/``ocm_weight``/``ocm_delta_t``) -- during
      primary assignment, candidate matches are rewarded/penalized by how
      well they continue a track's established real motion direction (see
      :func:`pgoggles.tracking.oc_sort.augmented_cost_matrix`).
    - **ORU** (``use_oru``) -- when a track reconnects after a genuine gap,
      its Kalman velocity state is rebuilt via straight-line interpolation
      between its last real observation and the new one, rather than
      trusting drift accumulated during blind coasting (see
      :func:`pgoggles.tracking.oc_sort.reupdate_kalman`, wired in via
      :meth:`Track.update`).
    - **OCR** (``use_ocr``) -- a secondary matching pass compares
      still-unmatched tracks against remaining detections using each
      track's *last real observation* rather than its (possibly drifted)
      Kalman-predicted box -- recovers matches primary assignment misses
      when an object stops abruptly.

    Parameters
    ----------
    mode : TrackingMode
        Constructor default AND a freely-mutable public attribute post-
        construction (mirrors ``AnalysisPipeline.highest_confidence`` --
        the established pattern for runtime-toggleable pipeline behavior
        in this codebase). ``DISABLED`` is a cheap passthrough returning
        ``track_id=-1`` sentinels with zero bookkeeping; existing track
        state is left completely untouched while disabled (not aged, not
        deleted) -- re-enabling resumes matching against whatever state
        existed at disable time, which may now be stale. Otherwise
        controls only assignment *strategy* -- ``KALMAN_IOU`` uses cheap
        greedy assignment, ``FULL`` uses globally-optimal Hungarian
        assignment -- for both the primary and OCR recovery passes; it
        does not gate whether OCM/ORU/OCR themselves run.
    iou_threshold : float
        Minimum IoU (or OCM-augmented score, when ``use_ocm`` is set) for a
        track/detection pair to be considered a valid match. Default 0.3
        (SORT default).
    min_hits : int
        Consecutive matched frames required before a track is
        "confirmed" (and therefore included in :meth:`update`'s return
        value). Default 3.
    max_age : int
        Frames-since-last-match after which an unmatched track is
        deleted. Default 5. This counts *inference* frames -- ``update()``
        is only meant to be called on frames where detection actually ran
        (e.g. from ``AnalysisPipeline``, which itself skips frames via
        scene-change gating), not on every camera frame.
    use_ocm : bool
        Enable Observation-Centric Momentum in the primary assignment
        pass. Default ``True``.
    ocm_weight : float
        Weight of the direction-consistency bonus/malus blended into the
        primary assignment cost matrix when ``use_ocm`` is set. Default
        0.2. Structurally cannot turn a zero-IoU pair into a match
        regardless of this value (see
        :func:`pgoggles.tracking.oc_sort.augmented_cost_matrix`).
    ocm_delta_t : int
        How many real-observation steps back OCM looks to establish a
        track's motion direction, smoothing over single-frame detector
        noise. Default 3.
    use_oru : bool
        Enable Observation-centric Re-Update when a track reconnects after
        a genuine gap. Default ``True``. Has a real behavioral cost beyond
        matching -- it also changes the filter's covariance, not just its
        velocity -- so it's independently toggleable rather than implied
        by ``use_ocm``/``use_ocr``.
    use_ocr : bool
        Enable the Observation-Centric Recovery secondary matching pass.
        Default ``True``.
    observation_history_size : int
        How many real observations each track retains for OCM's direction
        estimate. Default 5.

    Notes
    -----
    A confirmed track is reported every frame it survives, even when not
    matched this exact frame -- it "coasts" on its Kalman-predicted
    position for up to ``max_age`` frames (bridging brief misses/
    occlusion). ``TrackedBox.time_since_update`` tells the caller how
    stale a given box is (0 = matched this frame); callers that only want
    exact-frame matches can filter on ``time_since_update == 0``.
    """

    def __init__(
        self,
        *,
        mode: TrackingMode = TrackingMode.FULL,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_age: int = 5,
        use_ocm: bool = True,
        ocm_weight: float = 0.2,
        ocm_delta_t: int = 3,
        use_oru: bool = True,
        use_ocr: bool = True,
        observation_history_size: int = 5,
    ) -> None:
        self.mode = mode
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.use_ocm = use_ocm
        self.ocm_weight = ocm_weight
        self.ocm_delta_t = ocm_delta_t
        self.use_oru = use_oru
        self.use_ocr = use_ocr
        self.observation_history_size = observation_history_size
        self._tracks_by_label: dict[str, list[Track]] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """Snapshot of all live tracks (tentative + confirmed), for introspection/tests."""

        return [t for label_tracks in self._tracks_by_label.values() for t in label_tracks]

    def update(self, boxes: list[BoundingBox]) -> list[TrackedBox]:
        """Advance tracking state by one frame; return confirmed TrackedBoxes."""

        if self.mode == TrackingMode.DISABLED:
            return [
                TrackedBox(
                    x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                    label=b.label, confidence=b.confidence,
                    track_id=-1,
                )
                for b in boxes
            ]

        # 1. Predict all existing tracks (advance Kalman state one frame).
        for label_tracks in self._tracks_by_label.values():
            for track in label_tracks:
                track.predict()

        # 2. Bucket this frame's detections by label.
        detections_by_label: dict[str, list[BoundingBox]] = {}
        for b in boxes:
            detections_by_label.setdefault(b.label, []).append(b)

        output: list[TrackedBox] = []
        all_labels = set(self._tracks_by_label) | set(detections_by_label)
        assign_fn = greedy_match if self.mode == TrackingMode.KALMAN_IOU else hungarian_match

        for label in all_labels:
            tracks = self._tracks_by_label.get(label, [])
            dets = detections_by_label.get(label, [])

            track_boxes = [t.current_box() for t in tracks]
            det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

            iou_mat = iou_matrix(track_boxes, det_boxes)

            # 3. Primary assignment: OCM-augmented cost (if enabled) via the
            # configured strategy.
            if self.use_ocm and tracks and dets:
                cost_mat = augmented_cost_matrix(
                    iou_mat,
                    [t.observation_history for t in tracks],
                    det_boxes,
                    self.ocm_weight,
                    self.ocm_delta_t,
                )
            else:
                cost_mat = iou_mat

            matches, unmatched_tracks, unmatched_dets = assign_fn(cost_mat, self.iou_threshold)

            # 4. Apply primary matches (Track.update() triggers ORU internally
            # when reconnecting after a genuine gap).
            for t_idx, d_idx in matches:
                tracks[t_idx].update(dets[d_idx])

            # 5. OCR: secondary recovery pass against each remaining
            # unmatched track's last REAL observation, not its (possibly
            # drifted) Kalman-predicted current_box() -- recovers matches
            # primary assignment misses when an object stops abruptly.
            # Deliberately does not re-run OCM here: OCR's whole premise is
            # that Kalman-predicted motion just diverged from reality, so
            # re-weighting by the very momentum that failed would be
            # self-defeating.
            if self.use_ocr and unmatched_tracks and unmatched_dets:
                recovery_tracks = [tracks[t_idx] for t_idx in unmatched_tracks]
                recovery_dets = [dets[d_idx] for d_idx in unmatched_dets]
                recovery_iou = iou_matrix(
                    [t.last_observation for t in recovery_tracks],
                    [(d.x1, d.y1, d.x2, d.y2) for d in recovery_dets],
                )
                r_matches, r_unmatched_tracks, r_unmatched_dets = assign_fn(recovery_iou, self.iou_threshold)

                for r_t, r_d in r_matches:
                    recovery_tracks[r_t].update(recovery_dets[r_d])

                # Map recovery-local indices back to the original per-label
                # tracks/dets indices, so pruning/new-track-spawn below sees
                # only what's still genuinely unmatched after both passes.
                unmatched_tracks = [unmatched_tracks[r_t] for r_t in r_unmatched_tracks]
                unmatched_dets = [unmatched_dets[r_d] for r_d in r_unmatched_dets]

            # 6. Create new tentative tracks for detections unmatched by
            # both the primary and recovery passes.
            new_tracks = []
            for d_idx in unmatched_dets:
                track_id = self._next_id
                self._next_id += 1
                new_tracks.append(
                    Track(
                        track_id=track_id,
                        detection=dets[d_idx],
                        min_hits=self.min_hits,
                        observation_history_size=self.observation_history_size,
                        use_oru=self.use_oru,
                    )
                )

            # 7. Prune tracks that have gone too long without a match.
            surviving = [t for t in tracks if t.time_since_update <= self.max_age]
            surviving.extend(new_tracks)
            self._tracks_by_label[label] = surviving

            # 8. Materialize output: every confirmed track survives into the
            # result, including ones coasting on a predicted (unmatched)
            # position this frame.
            output.extend(t.to_tracked_box() for t in surviving if t.confirmed)

        return output
