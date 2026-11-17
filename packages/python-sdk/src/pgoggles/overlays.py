"""Display primitives and overlay message constructors for the pgoggles pipeline.

This module serves two roles:

1. **Pipeline dataclasses** – frozen dataclasses that carry detection results
   and frame metadata between the analysis thread and the draw thread:
   ``FramePacket``, ``BoundingBox``, ``OverlayState``.

2. **Overlay message constructors** – thin factory functions that build
   ``Overlay`` messages (from :mod:`.messages`) for the device display layer:
   ``text()``, ``bounding_box()``, ``reticle()``, ``chevron()``,
   ``notification()``.

The pipeline dataclasses live in this file so that the draw thread can
import everything it needs from a single location.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .messages import Overlay, OverlayType

# Colours in BGR (OpenCV default) — shared with draw.py
_WHITE = (245, 245, 245)
_BLACK = (0, 0, 0)
_GREEN = (70, 230, 90)
class DrawMode(str, Enum):
    """Drawing mode for the draw thread — selects which overlay primitive to render per detection."""

    BOUNDING_BOX = "bounding_box"  # green rectangle + label badge (default)
    TEXT = "text"                  # text label centred on detection
    RETICLE = "reticle"            # crosshair at detection centre
    CHEVRON = "chevron"            # directional chevron at detection centre


class ArrowDirection(str, Enum):
    """Direction for independent HUD arrow overlays."""

    NONE = "none"   # No arrow drawn
    LEFT = "left"
    UP = "up"
    RIGHT = "right"
    DOWN = "down"


class DetectionMode(str, Enum):
    """Trigger mode for the onDetect hook."""

    STALE = "stale"    # Fire once per detection appearance (debounced)
    ACTIVE = "active"  # Fire every frame while detection is present (debounced)
    GONE = "gone"      # Fire once when a present detection disappears (debounced)


class TrackingMode(str, Enum):
    """Assignment strategy for :class:`pgoggles.tracking.MultiObjectTracker`."""

    DISABLED = "disabled"      # Passthrough; track_id=-1 sentinel, no bookkeeping
    KALMAN_IOU = "kalman_iou"  # Kalman predict/update + IoU cost, greedy assignment
    FULL = "full"              # Kalman + IoU + Hungarian global-optimal assignment


# ---------------------------------------------------------------------------
# Pipeline dataclasses (moved from pipeline/types.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FramePacket:
    """A captured video frame plus metadata needed across pipeline threads."""

    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass(frozen=True)
class BoundingBox:
    """A detected object bounding box in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float


@dataclass(frozen=True)
class TrackedBox(BoundingBox):
    """A :class:`BoundingBox` enriched with a persistent track identity.

    Produced by :class:`pgoggles.tracking.MultiObjectTracker`. Structurally
    compatible with ``BoundingBox`` wherever ``.label``/``.confidence``
    duck-typing is relied upon (``HookRegistry.evaluate``,
    ``OverlayState.has_labels``).

    ``track_id`` is a stable identifier for one tracked object across
    frames; ``-1`` is the sentinel used when tracking is toggled off
    (:attr:`pgoggles.overlays.TrackingMode.DISABLED`) but a tracker
    instance is still in use. ``time_since_update`` is 0 when this box
    came from a real match this frame, and increases each frame the
    track "coasts" on its Kalman-predicted position without a match.
    """

    track_id: int
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = True


@dataclass(frozen=True)
class OverlayState:
    """The latest renderable analysis result for a video frame.

    Carries only raw ``BoundingBox`` objects from the analysis thread.
    The draw thread selects which overlay primitive to render per
    detection via a ``DrawMode``.

    When ``highlighted_box`` is set, the draw thread renders only that
    single detection instead of the full ``boxes`` list.
    """

    frame_id: int
    timestamp: float
    boxes: list[BoundingBox] = field(default_factory=list)

    inference_ms: float = 0.0
    scene_change_score: float = 0.0
    highlighted_boxes: list[BoundingBox] | None = None

    def best_detection(self) -> BoundingBox | None:
        """Return the bounding box with the highest confidence, or ``None``."""

        if not self.boxes:
            return None
        return max(self.boxes, key=lambda b: b.confidence)

    def has_labels(self, labels: set[str]) -> list[BoundingBox]:
        """Return boxes whose label is in the given set."""

        return [b for b in self.boxes if b.label in labels]


@dataclass(frozen=True)
class ExposedFrame:
    """A frame (or frame subset) published via :meth:`.AnalysisPipeline.expose_frame`.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame data. Can be a full frame or a cropped region.
    frame_id : int | None
        The pipeline frame ID at the time of publishing.
    labels : set[str]
        Detection labels associated with this frame.
    tags : list[str]
        Arbitrary tags for filtering.
    """

    frame: np.ndarray
    frame_id: int | None = None
    labels: set[str] = field(default_factory=set)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DrawCommand pattern — command objects for the draw pipeline
# ---------------------------------------------------------------------------

class DrawCommand(ABC):
    """Base class for a draw command in the Command pattern.

    Each concrete command encapsulates the data and drawing logic for
    a single overlay primitive.  The ``draw`` method mutates an OpenCV
    frame in-place.

    Commands are identified by a unique ``command_id`` (UUID) so the
    draw pipeline can track them across frames for persistent rendering.

    Subclasses must define ``x`` and ``y`` as fields (or properties)
    and implement ``draw(frame)``.
    """

    command_id: str

    @abstractmethod
    def draw(self, frame: np.ndarray) -> None:
        """Render this command onto ``frame`` in-place."""


@dataclass(frozen=True)
class DrawBoundingBox(DrawCommand):
    """Draw a green rectangle with a label badge at pixel coordinates.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Pixel coordinates of the bounding box corners.
    label : str
        Label text shown on the badge.
    confidence : float
        Confidence score shown alongside the label.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    command_id: str

    @property
    def x(self) -> int:
        return self.x1

    @property
    def y(self) -> int:
        return self.y1

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), _GREEN, 2)

        text = f"{self.label} {self.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        label_y = max(self.y1, text_height + baseline + 4)
        cv2.rectangle(
            frame,
            (self.x1, label_y - text_height - baseline - 4),
            (self.x1 + text_width + 8, label_y + baseline),
            _GREEN,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (self.x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _BLACK,
            1,
            cv2.LINE_AA,
        )


@dataclass(frozen=True)
class DrawText(DrawCommand):
    """Draw a text label centred on a position.

    Parameters
    ----------
    x, y : float
        Centre pixel coordinates for the text.
    value : str
        Text string to render.
    command_id : str
        Unique identifier for persistent tracking. Auto-generated if omitted.
    """

    x: float
    y: float
    value: str
    command_id: str

    def draw(self, frame: np.ndarray) -> None:
        text = self.value
        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        tw, th = text_size

        # Centre the text on (x, y)
        tx = int(self.x - tw // 2)
        ty = int(self.y + th // 2)

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        tx = max(0, min(tx, w - tw - 4))
        ty = max(th + baseline + 4, min(ty, h - 4))

        # Semi-transparent background bar behind text
        bg_y1 = max(0, ty - th - baseline - 6)
        bg_y2 = min(h, ty + baseline + 6)
        bg_x1 = max(0, tx - 6)
        bg_x2 = min(w, tx + tw + 6)
        if bg_y2 > bg_y1 and bg_x2 > bg_x1:
            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
            overlay_bg = roi.copy()
            cv2.rectangle(overlay_bg, (0, 0), (roi.shape[1], roi.shape[0]), _GREEN, -1)
            cv2.addWeighted(overlay_bg, 0.65, roi, 0.35, 0, roi)

        cv2.putText(
            frame,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _WHITE,
            1,
            cv2.LINE_AA,
        )


@dataclass(frozen=True)
class DrawReticle(DrawCommand):
    """Draw a crosshair (reticle) at the centre of a detection.

    Parameters
    ----------
    x, y : float
        Centre pixel coordinates for the crosshair.
    command_id : str
        Unique identifier for persistent tracking. Auto-generated if omitted.
    """

    x: float
    y: float
    command_id: str

    def draw(self, frame: np.ndarray) -> None:
        cx = int(self.x)
        cy = int(self.y)

        # Horizontal line
        cv2.line(frame, (cx - 7, cy), (cx + 7, cy), _WHITE, 2)
        # Vertical line
        cv2.line(frame, (cx, cy - 7), (cx, cy + 7), _WHITE, 2)


@dataclass(frozen=True)
class DrawChevron(DrawCommand):
    """Draw a chevron (pointing down) at a position.

    Used as a goal indicator for the user to follow.

    Parameters
    ----------
    x, y : float
        Top-centre pixel coordinate (y is the top of the chevron).
    width : int
        Width of the associated detection box, used to scale the chevron.
    command_id : str
        Unique identifier for persistent tracking. Auto-generated if omitted.
    """

    x: float
    y: float
    command_id: str
    width: int = 32

    def draw(self, frame: np.ndarray) -> None:
        cx = int(self.x)
        y_top = int(self.y)
        size = max(8, self.width // 6)

        # Down-pointing chevron: two diagonal lines meeting at bottom
        cv2.line(frame, (cx, y_top + size), (cx + size, y_top), _WHITE, 2)
        cv2.line(frame, (cx, y_top + size), (cx - size, y_top), _WHITE, 2)


# ---------------------------------------------------------------------------
# Overlay message constructors (device display primitives)
# ---------------------------------------------------------------------------

def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def text(
    value: str,
    *,
    x: float,
    y: float,
    frame_id: str | None = None,
    style: dict[str, Any] | None = None,
    expires_at_ms: int | None = None,
) -> Overlay:
    """Create a TEXT overlay at pixel/normalised coordinates."""
    return Overlay(
        overlay_id=_id("text"),
        type=OverlayType.TEXT.value,
        frame_id=frame_id,
        expires_at_ms=expires_at_ms,
        geometry={"x": x, "y": y},
        style=style or {},
        payload={"text": value},
    )


def bounding_box(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str | None = None,
    frame_id: str | None = None,
    style: dict[str, Any] | None = None,
) -> Overlay:
    """Create a BOUNDING_BOX overlay."""
    return Overlay(
        overlay_id=_id("box"),
        type=OverlayType.BOUNDING_BOX.value,
        frame_id=frame_id,
        geometry={"x": x, "y": y, "width": width, "height": height},
        style=style or {},
        payload={"label": label} if label else {},
    )


def reticle(
    *,
    x: float = 0.5,
    y: float = 0.5,
    frame_id: str | None = None,
    style: dict[str, Any] | None = None,
) -> Overlay:
    """Create a RETICLE (crosshair) overlay at normalised coordinates."""
    return Overlay(
        overlay_id=_id("reticle"),
        type=OverlayType.RETICLE.value,
        frame_id=frame_id,
        geometry={"x": x, "y": y},
        style=style or {},
        payload={},
    )


def chevron(
    *,
    x: float = 0.5,
    y: float = 0.5,
    direction: str = "up",
    frame_id: str | None = None,
    style: dict[str, Any] | None = None,
) -> Overlay:
    """Create a chevron/symbol overlay for directional guidance.

    Parameters
    ----------
    direction : str
        One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.
    """
    return Overlay(
        overlay_id=_id("chevron"),
        type=OverlayType.ICON.value,
        frame_id=frame_id,
        geometry={"x": x, "y": y},
        style=style or {},
        payload={"shape": "chevron", "direction": direction},
    )


def notification(
    message: str,
    *,
    level: str = "info",
    expires_at_ms: int | None = None,
) -> Overlay:
    """Create a NOTIFICATION overlay with optional expiry."""
    return Overlay(
        overlay_id=_id("notification"),
        type=OverlayType.NOTIFICATION.value,
        expires_at_ms=expires_at_ms,
        payload={"message": message, "level": level},
    )
