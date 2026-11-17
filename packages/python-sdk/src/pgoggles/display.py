"""OpenCV display loop for the pgoggles video pipeline."""

from __future__ import annotations

from collections import deque
import time

import cv2

from .pipeline.analysis import AnalysisPipeline
from .pipeline.camera import CameraSource
from .pipeline.draw import DrawPipeline
from .overlays import (
    ArrowDirection,
    BoundingBox,
    DrawBoundingBox,
    DrawChevron,
    DrawCommand,
    DrawMode,
    DrawReticle,
    DrawText,
    OverlayState,
)

_DRAW_MODES: list[DrawMode] = [
    DrawMode.BOUNDING_BOX,
    DrawMode.TEXT,
    DrawMode.RETICLE,
    DrawMode.CHEVRON,
]

_ARROW_DIRECTIONS: list[ArrowDirection] = [
    ArrowDirection.NONE,
    ArrowDirection.LEFT,
    ArrowDirection.UP,
    ArrowDirection.RIGHT,
    ArrowDirection.DOWN,
]


class DisplayWindow:
    """Render the latest camera frame and overlay state in an OpenCV window.

    The display window owns a ``DrawPipeline`` that handles frame annotation
    via the Command pattern.  It builds ``DrawCommand`` instances from the
    current ``OverlayState`` and ``draw_mode``, enqueues them, and displays
    the annotated result.
    """

    def __init__(
        self,
        camera_source: CameraSource,
        analysis_pipeline: AnalysisPipeline,
        draw_pipeline: DrawPipeline | None = None,
        *,
        fps_target: float = 30.0,
        window_name: str = "pgoggles webcam demo",
    ) -> None:
        self.camera_source = camera_source
        self.analysis_pipeline = analysis_pipeline
        self.draw_pipeline = draw_pipeline or DrawPipeline()
        self.fps_target = fps_target
        self.window_name = window_name
        self.highest_confidence = False
        self._frame_timestamps: deque[float] = deque(maxlen=30)

    def run(self) -> None:
        """Run the blocking display loop until the user presses ``q``.
        """

        self.draw_pipeline.start()

        delay_ms = max(1, int(1000.0 / self.fps_target)) if self.fps_target > 0 else 1
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                packet = self.camera_source.latest_frame
                if packet is None:
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                frame = packet.frame.copy()
                self._frame_timestamps.append(time.perf_counter())
                overlay = self.analysis_pipeline.get_latest_overlay()

                # Push the raw frame into the draw pipeline
                self.draw_pipeline.push_frame(frame)

                # Sync HUD metadata into the draw pipeline
                self.draw_pipeline.set_fps(self.fps)
                self.draw_pipeline.set_analysis_hz(self.analysis_pipeline.analysis_hz)
                self.draw_pipeline.set_analyzing(self.analysis_pipeline.analyzing)
                self.draw_pipeline.set_overlay(overlay)

                # Get the annotated frame from the draw pipeline
                annotated = self.draw_pipeline.latest_frame
                if annotated is not None:
                    cv2.imshow(self.window_name, annotated)

                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self.draw_pipeline.stop()
            self.analysis_pipeline.stop()
            self.camera_source.stop()
            cv2.destroyWindow(self.window_name)

    # def _build_commands(self, overlay: OverlayState | None) -> None:
    #     """Convert the current overlay into DrawCommand instances and enqueue them."""

    #     if overlay is None:
    #         return

    #     boxes = overlay.highlighted_boxes if overlay.highlighted_boxes else overlay.boxes
    #     for box in boxes:
    #         command = self._box_to_command(box)
    #         if command is not None:
    #             self.draw_pipeline.enqueue(command)

    # def _box_to_command(self, box: BoundingBox) -> DrawCommand | None:
    #     """Convert a BoundingBox to a DrawCommand based on the current draw mode."""
    #     mode = self.draw_mode
    #     if mode == DrawMode.BOUNDING_BOX:
    #         return DrawBoundingBox(
    #             x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
    #             label=box.label, confidence=box.confidence,
    #         )
    #     elif mode == DrawMode.TEXT:
    #         cx = (box.x1 + box.x2) // 2
    #         cy = (box.y1 + box.y2) // 2
    #         return DrawText(x=cx, y=cy, value=box.label)
    #     elif mode == DrawMode.RETICLE:
    #         cx = (box.x1 + box.x2) // 2
    #         cy = (box.y1 + box.y2) // 2
    #         return DrawReticle(x=cx, y=cy)
    #     elif mode == DrawMode.CHEVRON:
    #         cx = (box.x1 + box.x2) // 2
    #         cy = box.y1
    #         width = box.x2 - box.x1
    #         return DrawChevron(x=cx, y=cy, width=width)
    #     return None

    # def _cycle_mode(self) -> None:
    #     """Advance draw_mode to the next mode in the cycle."""

    #     idx = _DRAW_MODES.index(self.draw_mode)
    #     next_mode = _DRAW_MODES[(idx + 1) % len(_DRAW_MODES)]
    #     if isinstance(self._draw_mode, dict):
    #         self._draw_mode["draw_mode"] = next_mode
    #     else:
    #         self._draw_mode = next_mode  # type: ignore[assignment]

    # def _cycle_arrow(self) -> None:
    #     """Advance arrow_direction through NONE, LEFT, UP, RIGHT, DOWN."""

    #     if self.arrow_direction is None or self.arrow_direction == ArrowDirection.DOWN:
    #         self.arrow_direction = _ARROW_DIRECTIONS[0]
    #     else:
    #         idx = _ARROW_DIRECTIONS.index(self.arrow_direction)
    #         next_idx = (idx + 1) % len(_ARROW_DIRECTIONS)
    #         self.arrow_direction = _ARROW_DIRECTIONS[next_idx]

    @property
    def fps(self) -> float:
        """Actual display rate over the recent timestamp window."""

        if len(self._frame_timestamps) < 2:
            return 0.0
        elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_timestamps) - 1) / elapsed
