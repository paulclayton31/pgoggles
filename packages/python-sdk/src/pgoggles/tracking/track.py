"""Single-track bookkeeping: lifecycle state + wrapped Kalman filter."""

from __future__ import annotations

from collections import deque

from ..overlays import BoundingBox, TrackedBox
from .iou import Box
from .kalman import KalmanBoxTracker
from .oc_sort import reupdate_kalman


class Track:
    """One tracked object across frames.

    Lifecycle
    ---------
    - Created *tentative* (``confirmed`` is ``False``) on an unmatched
      detection.
    - ``hits`` increments on every matched :meth:`update`; the track
      becomes ``confirmed`` once ``hits >= min_hits`` (``min_hits`` is a
      :class:`~pgoggles.tracking.tracker.MultiObjectTracker`-level
      parameter, passed in at construction so ``Track`` itself stays a
      dumb bookkeeping object with no config surface of its own).
    - ``time_since_update`` resets to 0 on match, increments by 1 on every
      :meth:`predict` that isn't followed by a match; the owning
      ``MultiObjectTracker`` deletes the track once
      ``time_since_update > max_age``.

    ``label`` is fixed at creation and never changes -- combined with
    per-label bucketing in ``MultiObjectTracker``, this is what prevents a
    track from ever "swapping" identity across detection classes.

    OC-SORT observation history
    ----------------------------
    Alongside the live Kalman filter, a track keeps a small buffer of its
    *real* (matched) observations -- distinct from the filter's own
    (potentially drifted) predicted state. This backs two OC-SORT
    corrections: :attr:`last_observation`/:attr:`observation_history` feed
    Observation-Centric Momentum (see
    :func:`pgoggles.tracking.oc_sort.augmented_cost_matrix`) and Observation-
    Centric Recovery (matching against ``last_observation`` instead of
    :meth:`current_box`, in ``MultiObjectTracker``), and :meth:`update`
    itself performs Observation-centric Re-Update when reconnecting after a
    gap (see :func:`pgoggles.tracking.oc_sort.reupdate_kalman`).
    """

    def __init__(
        self,
        track_id: int,
        detection: BoundingBox,
        min_hits: int,
        *,
        observation_history_size: int = 5,
        use_oru: bool = True,
    ) -> None:
        self.track_id = track_id
        self.label = detection.label
        self.confidence = detection.confidence
        self._kalman = KalmanBoxTracker(
            bbox=(detection.x1, detection.y1, detection.x2, detection.y2)
        )
        self._min_hits = min_hits
        self._use_oru = use_oru
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        initial_box: Box = (detection.x1, detection.y1, detection.x2, detection.y2)
        self._observations: deque[Box] = deque([initial_box], maxlen=observation_history_size)

    @property
    def last_observation(self) -> Box:
        """Most recent *real* (matched) box.

        Distinct from :meth:`current_box`, which is the live Kalman-
        predicted state and may have drifted since the last real match --
        this is what Observation-Centric Recovery matches against instead.
        """

        return self._observations[-1]

    @property
    def observation_history(self) -> list[Box]:
        """Real observations, oldest -> newest, for Observation-Centric Momentum."""

        return list(self._observations)

    def predict(self) -> tuple[float, float, float, float]:
        """Advance the Kalman state by one frame; return the predicted box."""

        self.age += 1
        self.time_since_update += 1
        return self._kalman.predict()

    def current_box(self) -> tuple[float, float, float, float]:
        """Current Kalman-state box as ``(x1, y1, x2, y2)``, without advancing time."""

        return self._kalman.to_bbox()

    def update(self, detection: BoundingBox) -> None:
        """Correct the Kalman state with a matched detection this frame.

        ``MultiObjectTracker`` calls :meth:`predict` on every live track,
        every frame, before assignment -- so ``time_since_update`` is
        always >= 1 by the time ``update`` runs, even for a track matched
        on every consecutive frame (that mandatory per-frame ``predict()``
        accounts for the ``1``). A genuine gap -- at least one prior frame
        where this track went unmatched -- is therefore
        ``time_since_update > 1``, not ``> 0``.

        When there's a genuine gap, performs Observation-centric Re-Update:
        rebuilds the filter from scratch by replaying a straight-line
        interpolated trajectory between :attr:`last_observation` and this
        new detection (see :func:`pgoggles.tracking.oc_sort.reupdate_kalman`),
        rather than trusting whatever drift accumulated during blind
        coasting. When there's no real gap (``time_since_update == 1``),
        behavior is identical to a plain Kalman correction on the
        already-this-frame-predicted state -- no reset, no interpolation
        overhead, and critically no redundant extra ``predict()`` call
        (layering a replay on top of the already-advanced live filter would
        double-count that frame's predict step and corrupt the velocity
        estimate).
        """

        new_box: Box = (detection.x1, detection.y1, detection.x2, detection.y2)
        if self._use_oru and self.time_since_update > 1:
            self._kalman = reupdate_kalman(self.last_observation, new_box, steps=self.time_since_update)
        else:
            self._kalman.update(new_box)

        self._observations.append(new_box)
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= self._min_hits

    def to_tracked_box(self) -> TrackedBox:
        """Materialize current Kalman state as a :class:`TrackedBox`."""

        x1, y1, x2, y2 = self._kalman.to_bbox()
        return TrackedBox(
            x1=round(x1),
            y1=round(y1),
            x2=round(x2),
            y2=round(y2),
            label=self.label,
            confidence=self.confidence,
            track_id=self.track_id,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            confirmed=self.confirmed,
        )
