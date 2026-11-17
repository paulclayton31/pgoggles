"""Detection hook registry for the pgoggles pipeline.

Provides an event-driven onDetect system modelled after React's useEffect:
multiple concurrent hooks, per-label filtering, STALE/ACTIVE modes,
debounce, and cleanup/lifecycle support.

Usage::

    from pgoggles.hooks import HookRegistry, DetectionMode

    registry = HookRegistry()

    handle = registry.register(
        labels={\"person\"},
        mode=DetectionMode.STALE,
        callback=lambda boxes, context: print(f\"{len(boxes)} person(s) detected\"),
    )

    # ... later ...
    handle.unregister()

    # Cleanup all hooks
    registry.clear()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pgoggles.pipeline.draw import DrawPipeline

from .overlays import BoundingBox, DetectionMode, TrackedBox


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DetectionContext:
    """Read-only context passed to every onDetect callback.

    Parameters
    ----------
    frame_id : int
        The pipeline frame ID at the time of detection.
    timestamp : float
        Unix timestamp (from ``time.perf_counter``) of the detection.
    boxes : list[BoundingBox | TrackedBox]
        The full list of bounding boxes from the current overlay state
        (already filtered by the hook's label and confidence criteria).
        ``TrackedBox`` (a ``BoundingBox`` plus a persistent ``track_id``)
        when the owning ``AnalysisPipeline`` was configured with a
        ``MultiObjectTracker``, plain ``BoundingBox`` otherwise.
    frame : numpy.ndarray | None
        The raw captured frame (BGR ``np.ndarray``) at the time of detection.
        ``None`` when the pipeline was not configured to expose frames.
    overlay_state : dict[str, Any] | None
        Arbitrary metadata carried by the analysis pipeline's overlay state.
        Currently ``None`` — reserved for future fields such as scene-change
        score, inference latency, etc.
    """

    frame_id: int
    timestamp: float
    boxes: list[BoundingBox | TrackedBox]
    frame: np.ndarray | None = None
    overlay_state: dict[str, Any] | None = None

    draw_pipeline: DrawPipeline | None = None


# Type alias for the user-facing callback signature.
# Receives matched boxes and a rich context object, along with the draw pipeline (in case the user wants to draw from the onDetect).
OnDetectCallback = Callable[[list[BoundingBox | TrackedBox], DetectionContext, DrawPipeline], None]
# Type alias for the optional cleanup function returned by a callback.
OnDetectCleanup = Callable[[], None] | None


# ---------------------------------------------------------------------------
# Internal hook descriptor
# ---------------------------------------------------------------------------

@dataclass
class _HookDescriptor:
    """Internal representation of a single registered hook."""

    labels: set[str]
    confidence_threshold: float
    mode: DetectionMode
    debounce_ms: float
    callback: OnDetectCallback
    cleanup: OnDetectCleanup | None = None

    # Debounce / stale-mode state (managed by the registry)
    _last_fired: float = field(default=0.0, repr=False)
    _was_present: bool = field(default=False, repr=False)
    # Last frame's matching boxes, kept so GONE-mode hooks (which fire on a
    # frame with zero matching boxes, by definition) can still tell the
    # callback what was there right before it disappeared.
    _last_matched: list[BoundingBox | TrackedBox] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Public handle
# ---------------------------------------------------------------------------

class HookHandle:
    """Handle returned by :meth:`HookRegistry.register`.

    Calling :meth:`unregister` removes the hook from the registry and
    invokes its cleanup function (if any).
    """

    def __init__(self, registry: HookRegistry, descriptor: _HookDescriptor) -> None:
        self._registry = registry
        self._descriptor = descriptor
        self._unregistered = False

    @property
    def unregistered(self) -> bool:
        """Whether this hook has been unregistered."""
        return self._unregistered

    def unregister(self) -> None:
        """Remove this hook from the registry and run its cleanup."""
        if self._unregistered:
            return
        self._unregistered = True
        self._registry._unregister_descriptor(self._descriptor)
        if self._descriptor.cleanup is not None:
            self._descriptor.cleanup()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class HookRegistry:
    """Manages a collection of onDetect hooks.

    Multiple hooks can be registered concurrently, each with its own
    label filter, confidence threshold, mode, debounce, and callback.

    The registry is **not thread-safe** — hooks should be registered and
    evaluated from a single thread (typically the analysis thread).
    """

    def __init__(self) -> None:
        self._hooks: list[_HookDescriptor] = []

    def register(
        self,
        labels: set[str],
        mode: DetectionMode = DetectionMode.STALE,
        *,
        confidence_threshold: float = 0.75,
        debounce_ms: float = 500.0,
        callback: OnDetectCallback,
        cleanup: OnDetectCleanup = None,
    ) -> HookHandle:
        """Register a new onDetect hook.

        Parameters
        ----------
        labels : set[str]
            Detection labels that trigger the hook (e.g. ``{\"person\"}``).
            Empty set matches all detections.
        mode : DetectionMode
            ``STALE`` fires once when detection first appears (debounced).
            ``ACTIVE`` fires every frame while detection is present (debounced).
            ``GONE`` fires once when a previously-present detection
            disappears (debounced). Since there is nothing matching on the
            firing frame by definition, the callback receives the last
            matching boxes from the frame before the detection disappeared
            (see :meth:`evaluate`).
        confidence_threshold : float
            Minimum confidence score for a detection to trigger the hook.
            Defaults to 0.75.
        debounce_ms : float
            Minimum milliseconds between hook invocations. Defaults to 500 ms.
        callback : Callable[[list[BoundingBox | TrackedBox], DetectionContext], None]
            The function to call when the detection condition is satisfied.
            Receives the list of matching bounding boxes and a
            :class:`DetectionContext` object.
        cleanup : Callable[[], None] | None
            Optional cleanup function. Called when the hook is unregistered
            or when the registry is cleared.

        Returns
        -------
        HookHandle
            A handle that can be used to unregister the hook.
        """
        descriptor = _HookDescriptor(
            labels=labels,
            confidence_threshold=confidence_threshold,
            mode=mode,
            debounce_ms=debounce_ms,
            callback=callback,
            cleanup=cleanup,
        )
        self._hooks.append(descriptor)
        return HookHandle(self, descriptor)

    def _unregister_descriptor(self, descriptor: _HookDescriptor) -> None:
        """Remove a descriptor from the registry (internal)."""
        self._hooks.remove(descriptor)

    def evaluate(
        self,
        boxes: list[BoundingBox | TrackedBox],
        frame_id: int,
        timestamp: float,
        draw_pipeline: DrawPipeline,
        frame: np.ndarray | None = None,
        overlay_state: dict[str, Any] | None = None,
    ) -> None:
        """Evaluate all registered hooks against the current detection state.

        This method should be called once per frame from the analysis thread,
        after inference has produced the latest bounding boxes.

        Parameters
        ----------
        boxes : list[BoundingBox | TrackedBox]
            All bounding boxes from the current overlay state (``TrackedBox``
            when tracking is enabled upstream, plain ``BoundingBox`` otherwise).
        frame_id : int
            The pipeline frame ID.
        timestamp : float
            Unix timestamp (from ``time.perf_counter``) of the detection.
        frame : numpy.ndarray | None
            The raw captured frame at the time of detection.
        overlay_state : dict[str, Any] | None
            Arbitrary metadata carried by the analysis pipeline's overlay state.

        Notes
        -----
        Presence tracking (``_was_present`` / ``_last_matched``) is updated
        for every hook on every call, regardless of whether that hook ends
        up firing. Edge-triggered modes (``STALE``, ``GONE``) depend on
        seeing every presence/absence transition; skipping the update on a
        debounced frame would let a transition slip by unnoticed.
        """
        for desc in self._hooks:
            # Filter boxes by labels and confidence
            if desc.labels:
                matched = [
                    b for b in boxes
                    if b.label in desc.labels and b.confidence >= desc.confidence_threshold
                ]
            else:
                matched = [
                    b for b in boxes
                    if b.confidence >= desc.confidence_threshold
                ]

            is_present = len(matched) > 0
            was_present = desc._was_present
            now = timestamp

            should_fire = False
            if desc.mode == DetectionMode.STALE:
                if is_present and not was_present:
                    should_fire = True
            elif desc.mode == DetectionMode.ACTIVE:
                if is_present:
                    should_fire = True
            elif desc.mode == DetectionMode.GONE:
                if was_present and not is_present:
                    should_fire = True

            # Update presence/last-seen-boxes state unconditionally (see Notes).
            if is_present:
                desc._last_matched = matched
            desc._was_present = is_present

            if not should_fire:
                continue

            # Check debounce cooldown
            elapsed_ms = (now - desc._last_fired) * 1000
            if elapsed_ms < desc.debounce_ms:
                continue

            # GONE fires on a frame with zero matching boxes by definition —
            # hand the callback the last boxes seen before the disappearance
            # instead of the (empty) current match.
            fired_boxes = desc._last_matched if desc.mode == DetectionMode.GONE else matched

            context = DetectionContext(
                frame_id=frame_id,
                timestamp=now,
                draw_pipeline=draw_pipeline,
                boxes=fired_boxes,
                frame=frame,
                overlay_state=overlay_state,
            )
            desc.callback(fired_boxes, context, draw_pipeline)
            desc._last_fired = now

    def clear(self) -> None:
        """Remove all hooks and run their cleanup functions."""
        for desc in self._hooks:
            if desc.cleanup is not None:
                desc.cleanup()
        self._hooks.clear()

    @property
    def count(self) -> int:
        """Number of currently registered hooks."""
        return len(self._hooks)
