"""Threaded draw pipeline for applying user-queued commands to frames.

The :class:`DrawPipeline` owns a command queue that accepts
:class:`.DrawCommand` instances.  Its ``_draw_loop`` consumes frames
from an external source (typically the camera or display layer), applies
all pending commands to the frame, draws the HUD, and publishes the
annotated frame via a ``latest_frame`` property.

This makes the draw layer an independent pipeline stage — it starts and
stops on its own, just like :class:`.AnalysisPipeline` and
:class:`.CameraSource`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

from ..overlays import ArrowDirection, DrawCommand, OverlayState
from uuid import uuid4

logger = logging.getLogger(__name__)

# Colours in BGR (OpenCV default)
WHITE = (245, 245, 245)
HUD_BG = (20, 20, 20)
ANALYZING = (0, 210, 255)


class DrawPipeline:
    """Consume frames and apply queued ``DrawCommand`` instances.

    The pipeline runs its own worker thread that continuously pulls frames
    (pushed via :meth:`push_frame`), applies all pending commands from the
    internal command queue, draws the HUD, and publishes the annotated
    frame.

    Parameters
    ----------
    command_queue_size : int
        Maximum size of the internal command queue.  Defaults to 64.
    """

    def __init__(self, command_queue_size: int = 64) -> None:
        self._command_queue: queue.Queue[DrawCommand] = queue.Queue(
            maxsize=command_queue_size,
        )
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._output_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_overlay: OverlayState | None = None

        # Active commands: dict mapping command_id → DrawCommand
        self._active_commands_map: dict[str, DrawCommand] = {}
        # List of active command ids (short strings, easy to debug)
        self._active_command_ids: list[str] = []
        self._active_commands_lock = threading.Lock()

        # HUD metadata (updated by the caller via properties or setters)
        self._fps: float = 0.0
        self._analysis_hz: float = 0.0
        self._analyzing: bool = False
        self._arrow_direction: ArrowDirection | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Return the most recently annotated frame without blocking."""

        with self._output_lock:
            return self._latest_frame

    @property
    def command_queue(self) -> queue.Queue[DrawCommand]:
        """Access the command queue for enqueuing :class:`DrawCommand` instances."""

        return self._command_queue

    def enqueue(self, command: DrawCommand) -> None:
        """Add a :class:`DrawCommand` to the pipeline's command queue.

        The command is queued and then added to the active commands list,
        where it will be drawn on every subsequent frame until explicitly
        removed via :meth:`remove_command`.

        Parameters
        ----------
        command : DrawCommand
            The command to enqueue.
        """

        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            logger.warning(
                "Command queue full (%d); dropping oldest command",
                self._command_queue.maxsize,
            )
            try:
                self._command_queue.get_nowait()
                self._command_queue.put_nowait(command)
            except Exception:
                logger.exception("Failed to drop and enqueue command")

    def remove_command(self, command_id: str) -> bool:
        """Remove a command from the active scene by its ``command_id``.

        Removes the command from the dict, but not the list.
        The draw thread will skip any id not found in the dict, and
        automatically cleans it from the ids list on the next frame.
        This way, deletion is handled gracefully, and O(n) removal is absorbed by the draw loop over multiple frames.

        Parameters
        ----------
        command_id : str
            The unique identifier of the command to remove.

        Returns
        -------
        bool
            ``True`` if the command was found and removed, ``False`` otherwise.
        """

        with self._active_commands_lock:
            if command_id in self._active_commands_map:
                del self._active_commands_map[command_id]
                return True
        return False

    @property
    def active_commands(self) -> list[DrawCommand]:
        """Return a snapshot of the current active commands (resolved from the dict)."""

        with self._active_commands_lock:
            return [self._active_commands_map[cid] for cid in self._active_command_ids if cid in self._active_commands_map]

    @property
    def active_command_ids(self) -> list[str]:
        """Return a snapshot of the current active command ids (for debugging)."""

        with self._active_commands_lock:
            return list(self._active_command_ids)

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a raw frame into the pipeline for annotation.

        The frame is stored in a small buffer; the draw loop pulls the
        freshest frame each iteration.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate.
        """

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)

    def set_overlay(self, overlay: OverlayState | None) -> None:
        """Set the latest analysis overlay for HUD metrics."""

        with self._output_lock:
            self._latest_overlay = overlay

    def start(self) -> None:
        """Start the draw worker thread."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._draw_loop,
            name="pgoggles-draw",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the draw pipeline and wait for the worker to exit."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # HUD metadata setters (called by the display layer)
    # ------------------------------------------------------------------

    def set_fps(self, fps: float) -> None:
        """Set the current display frame rate for HUD rendering."""

        self._fps = fps

    def set_analysis_hz(self, analysis_hz: float) -> None:
        """Set the current analysis rate for HUD rendering."""

        self._analysis_hz = analysis_hz

    def set_analyzing(self, analyzing: bool) -> None:
        """Set whether the analysis thread is currently running."""

        self._analyzing = analyzing

    def set_arrow_direction(self, direction: ArrowDirection | None) -> None:
        """Set the independent HUD arrow direction."""

        self._arrow_direction = direction

    # ------------------------------------------------------------------
    # Internal draw loop
    # ------------------------------------------------------------------

    def _draw_loop(self) -> None:
        """Main loop: pull frames, apply commands, draw HUD, publish."""

        while not self._stop_event.is_set():
            # Pull the freshest frame (non-blocking)
            frame = self._pull_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Apply all pending commands
            self._apply_commands(frame)

            # Draw HUD
            inference_ms = 0.0
            scene_change_score = 0.0
            with self._output_lock:
                overlay = self._latest_overlay
                if overlay is not None:
                    inference_ms = overlay.inference_ms
                    scene_change_score = overlay.scene_change_score

            draw_hud(
                frame,
                fps=self._fps,
                analysis_hz=self._analysis_hz,
                inference_ms=inference_ms,
                scene_change_score=scene_change_score,
                analyzing=self._analyzing,
                arrow_direction=self._arrow_direction,
            )

            # Publish annotated frame
            with self._output_lock:
                self._latest_frame = frame

    def _pull_frame(self) -> np.ndarray | None:
        """Get the freshest frame from the input queue."""

        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _apply_commands(self, frame: np.ndarray) -> None:
        """Drain the command queue, add new commands to the active dict,
        clean stale ids, then draw every active command onto ``frame``."""

        # Consume queue and add to active dict (updates replace existing)
        while not self._command_queue.empty():
            try:
                command = self._command_queue.get_nowait()
                with self._active_commands_lock:
                    # Insert/update in dict
                    self._active_commands_map[command.command_id] = command
                    # Add to ids list if not already present
                    if command.command_id not in self._active_command_ids:
                        self._active_command_ids.append(command.command_id)
            except queue.Empty:
                break

        # Clean stale ids (removed commands no longer in dict)
        with self._active_commands_lock:
            self._active_command_ids = [
                cid for cid in self._active_command_ids
                if cid in self._active_commands_map
            ]
            # Snapshot of commands to draw (avoids holding lock during draw)
            commands_to_draw = [
                self._active_commands_map[cid]
                for cid in self._active_command_ids
                if cid in self._active_commands_map
            ]

        # Draw all active commands (persistent across frames)
        for command in commands_to_draw:
            command.draw(frame)


# ---------------------------------------------------------------------------
# Independent HUD arrow overlay
# ---------------------------------------------------------------------------


def _draw_arrow(frame: np.ndarray, direction: ArrowDirection) -> None:
    """Draw an independent HUD arrow at a fixed position on screen.

    When ``direction`` is ``ArrowDirection.NONE``, nothing is drawn.
    """

    if direction == ArrowDirection.NONE:
        return  # No arrow drawn

    h, w = frame.shape[:2]

    # Position: top-right quadrant, offset from HUD
    arrow_cx = w - 100
    arrow_cy = 60
    size = 20

    # Draw a subtle background circle for visibility
    cv2.circle(frame, (arrow_cx, arrow_cy), size + 6, HUD_BG, -1)
    cv2.circle(frame, (arrow_cx, arrow_cy), size + 6, WHITE, 1)

    if direction == ArrowDirection.UP:
        # Up-pointing arrow: triangle pointing up
        cv2.line(frame, (arrow_cx, arrow_cy - size), (arrow_cx, arrow_cy + size // 2), WHITE, 3)
        cv2.line(frame, (arrow_cx - size // 2, arrow_cy + size // 4), (arrow_cx, arrow_cy - size), WHITE, 3)
        cv2.line(frame, (arrow_cx + size // 2, arrow_cy + size // 4), (arrow_cx, arrow_cy - size), WHITE, 3)

    elif direction == ArrowDirection.DOWN:
        # Down-pointing arrow: triangle pointing down
        cv2.line(frame, (arrow_cx, arrow_cy + size), (arrow_cx, arrow_cy - size // 2), WHITE, 3)
        cv2.line(frame, (arrow_cx - size // 2, arrow_cy - size // 4), (arrow_cx, arrow_cy + size), WHITE, 3)
        cv2.line(frame, (arrow_cx + size // 2, arrow_cy - size // 4), (arrow_cx, arrow_cy + size), WHITE, 3)

    elif direction == ArrowDirection.LEFT:
        # Left-pointing arrow: triangle pointing left
        cv2.line(frame, (arrow_cx + size, arrow_cy), (arrow_cx - size // 2, arrow_cy), WHITE, 3)
        cv2.line(frame, (arrow_cx + size // 2, arrow_cy - size // 2), (arrow_cx + size, arrow_cy), WHITE, 3)
        cv2.line(frame, (arrow_cx + size // 2, arrow_cy + size // 2), (arrow_cx + size, arrow_cy), WHITE, 3)

    elif direction == ArrowDirection.RIGHT:
        # Right-pointing arrow: triangle pointing right
        cv2.line(frame, (arrow_cx - size, arrow_cy), (arrow_cx + size // 2, arrow_cy), WHITE, 3)
        cv2.line(frame, (arrow_cx - size // 2, arrow_cy - size // 2), (arrow_cx - size, arrow_cy), WHITE, 3)
        cv2.line(frame, (arrow_cx - size // 2, arrow_cy + size // 2), (arrow_cx - size, arrow_cy), WHITE, 3)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    analysis_hz: float,
    inference_ms: float,
    scene_change_score: float,
    analyzing: bool,
    arrow_direction: ArrowDirection | None = None,
) -> None:
    """Draw the small runtime HUD in the top-left corner."""

    lines = [
        f"FPS {fps:4.1f}",
        f"ANALYSIS {analysis_hz:4.1f} Hz",
        f"INFERENCE {inference_ms:5.1f} ms",
        f"SCENE MAD {scene_change_score:4.1f}",
    ]
    if arrow_direction is not None and arrow_direction != ArrowDirection.NONE:
        lines.append(f"ARROW {arrow_direction.value.upper()}")
    width = 188
    height = 120 if analyzing else 104
    if arrow_direction is not None and arrow_direction != ArrowDirection.NONE:
        height += 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        y += 18

    if analyzing:
        cv2.circle(frame, (24, y - 2), 5, ANALYZING, -1)
        cv2.putText(
            frame,
            "ANALYZING",
            (38, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            ANALYZING,
            1,
            cv2.LINE_AA,
        )
