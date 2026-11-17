"""Run the pgoggles onDetect hook demo.

This demo showcases the **full pgoggles pipeline** from camera capture
through analysis, detection hooks, draw commands, and display.  It
demonstrates:

1. **Starting the pipeline** — camera source, analysis pipeline, draw
   pipeline, and display window.
2. **Registering an onDetect hook** — a callback that fires when the
   analysis detects a ``"person"`` label.
3. **Exposing frames** — publishing detected frames for external
   consumers (a user-defined exposed-frame display).
4. **Forming DrawCommands** — converting bounding boxes from
   ``OverlayState`` into ``DrawCommand`` instances based on scene
   logic (position-dependent directional text).
5. **Sending commands to the display** — enqueuing commands into the
   ``DrawPipeline`` for annotation.

Usage::

    python examples/on_detect_demo.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pgoggles.hooks import DetectionContext
from pgoggles.overlays import (
    ArrowDirection,
    BoundingBox,
    DetectionMode,
    DrawBoundingBox,
    DrawChevron,
    DrawReticle,
    DrawText,
    OverlayState,
)
from pgoggles.pipeline import AnalysisPipeline, CameraSource, DrawPipeline
from pgoggles.display import DisplayWindow


CONFIG = {
    "CAMERA_INDEX": 0,
    "ANALYSIS_HZ_MAX": 10.0,
    "SCENE_CHANGE_THRESHOLD": 8.0,
    # Export once in an environment with ultralytics: yolo export model=yolov8n.pt format=onnx
    "YOLO_MODEL": "yolov8n.onnx",
    "DISPLAY_FPS_TARGET": 30.0,
}


# ---------------------------------------------------------------------------
# User-level helper: convert BoundingBox → DrawCommand
# ---------------------------------------------------------------------------

def _box_to_command(box: BoundingBox) -> DrawBoundingBox | None:
    """Convert a BoundingBox to a DrawCommand based on the current draw mode.

    This is user-level code — a pgoggles user would write their own
    mapping function to convert detection results into draw commands.
    """
    if box is None:
        return None
    return DrawBoundingBox(
            x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
            label=box.label,
            confidence=box.confidence,
            command_id= "box"
    )

# ---------------------------------------------------------------------------
# User-level helper: draw detection notification on frame
# ---------------------------------------------------------------------------

def _draw_detection_notification(
    text: str,
    frame: np.ndarray,
) -> DrawText:
    """Draw a detection notification in the top-right corner of the frame.

    This is user-level code — a pgoggles user would write their own
    notification function to display detection results on the frame.
    The ``DrawText`` command draws at a specific (x, y) position
    relative to a detection box; for a fixed screen-space notification
    we draw directly on the frame.

    Parameters
    ----------
    frame : numpy.ndarray
        The BGR frame to draw on (mutated in-place).
    text : str
        The notification text to display.
    """

    return DrawText(x=frame.shape[1] - 10, y=30, value=text, command_id="notification")


# ---------------------------------------------------------------------------
# User-level helper: render exposed frames in a second window
# ---------------------------------------------------------------------------

def _render_exposed_frame(
    analysis: AnalysisPipeline,
    window_name: str,
    last_frame_id: int | None,
    labels: set[str] | None = None,
) -> int | None:
    """Render the latest exposed frame in a second window (user-level).

    This is user-level code — a pgoggles user would write their own
    exposed frame rendering logic. The ``DisplayWindow`` no longer
    handles this; it is entirely the user's responsibility.

    Parameters
    ----------
    analysis : AnalysisPipeline
        The analysis pipeline with exposed frames.
    window_name : str
        OpenCV window name for the exposed frame display.
    last_frame_id : int | None
        The last exposed frame ID that was rendered (for change detection).
    labels : set[str] | None
        Optional label filter for exposed frames.

    Returns
    -------
    int | None
        The new last exposed frame ID, or None if no new frame was rendered.
    """

    exposed = analysis.get_exposed_frames(labels=labels)
    if not exposed:
        return last_frame_id

    latest = exposed[-1]
    if latest.frame_id == last_frame_id:
        return last_frame_id  # no new frame to display

    frame = latest.frame.copy()
    cv2.putText(
        frame,
        f"exposed frame #{latest.frame_id}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.imshow(window_name, frame)
    return latest.frame_id


# ---------------------------------------------------------------------------
# User-level logic: determine directional text from bounding box position
# ---------------------------------------------------------------------------

def _direction_from_box(box: BoundingBox, frame_width: int) -> str:
    """Determine directional text based on the bounding box horizontal position.

    Returns ``"left"``, ``"center"``, or ``"right"`` based on whether the
    box centre falls in the left third, middle third, or right third of the
    frame.
    """

    cx = (box.x1 + box.x2) // 2
    third = frame_width // 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    else:
        return "right"


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. Start the camera source
    camera = CameraSource(camera_index=CONFIG["CAMERA_INDEX"])

    # 2. Create the draw pipeline (independent threaded stage)
    draw_pipeline = DrawPipeline()

    # 3. Start the analysis pipeline (YOLO inference)
    analysis = AnalysisPipeline(
        camera,
        hz_max=CONFIG["ANALYSIS_HZ_MAX"],
        scene_change_threshold=CONFIG["SCENE_CHANGE_THRESHOLD"],
        yolo_model=CONFIG["YOLO_MODEL"],
        draw_pipeline=draw_pipeline,
    )

    # 4. Configure the display window (independent threaded stage)
    display_window = DisplayWindow(
        camera_source=camera,
        analysis_pipeline=analysis,
        draw_pipeline=draw_pipeline,
        fps_target=30.0,
        window_name="pgoggles onDetect demo — person appears, draws directional notification ",
    )

    # Detection state: tracks whether a person is currently detected
    # so we can draw the notification each frame.
    person_detected = False

    # Exposed frame rendering state (user-level, not part of DisplayWindow)
    exposed_window_name = "pgoggles exposed frame"
    last_exposed_frame_id: int | None = None

    # 4. Register the onDetect hook — fires when a person is detected
    def _on_person_detected(
        boxes: list[BoundingBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:
        """Callback fired when a person is first detected (STALE mode).

        This demonstrates:
        - Accessing detection results (boxes, confidence, frame_id)
        - Exposing the detected frame for external consumers
        - Setting detection state for the main loop
        - Mutating draw mode at runtime
        """
        nonlocal person_detected

        best = max(boxes, key=lambda b: b.confidence) if boxes else None
        if best:
            logging.info(
                "onDetect: person detected — %s @ %.2f confidence (frame %s)",
                best.label,
                best.confidence,
                context.frame_id,
            )
        # Expose the detected frame for external consumers
        if context.frame is not None:
            analysis.expose_frame(
                context.frame,
                labels={"person"},
                tags=["on_detect"],
            )

            # # Render the exposed frame in the separate window for visualization.
            # # These three separate tasks can be declared with several onDetect hooks, or combined into one hook.  Here we combine them for simplicity.
            # last_exposed_frame_id = _render_exposed_frame(
            #     analysis,
            #     exposed_window_name,
            #     None,
            #     labels={"person"},
            # )

        # Set detection state so the main loop draws the notification
        person_detected = True

    def _draw_person_notification(
        boxes: list[BoundingBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:
        """Draw a notification on the frame while a person is detected."""
        nonlocal person_detected

        if context.frame is not None and boxes:
            best = max(boxes, key=lambda b: b.confidence)
            direction = _direction_from_box(best, context.frame.shape[1])
            notification_text = f"Person detected: {direction}"
            draw_command = _draw_detection_notification(notification_text, context.frame)
            draw_pipeline.enqueue(draw_command)

    def _remove_person_notification_on_cellphone(
        boxes: list[BoundingBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:
        """Remove the person notification when a cellphone is detected (ACTIVE mode)."""
        if context.frame is not None:
            if draw_pipeline.remove_command("notification"):
                logging.info("onDetect: person notification removed due to cellphone detection")
            else:
                logging.info("onDetect: no person notification to remove due to cellphone detection")

    def _on_person_cleanup() -> None:
        """Cleanup when the hook is unregistered."""
        logging.info("onDetect: person hook cleaned up")

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.STALE,
        debounce_ms=1000.0,
        callback=_on_person_detected,
        cleanup=_on_person_cleanup,
    )

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.STALE,
        debounce_ms=0.0,
        callback=_draw_person_notification,
    )

    analysis.hooks.register(
        labels={"cell phone"},
        mode=DetectionMode.ACTIVE,
        debounce_ms=0.0,
        callback=_remove_person_notification_on_cellphone,
    )

    # 5. Start all pipeline stages
    camera.start()
    draw_pipeline.start()
    analysis.start()

    # # 6. Open display window for annotated frames (user-level, not part of DisplayWindow)
    # delay_ms = max(1, int(1000.0 / CONFIG["DISPLAY_FPS_TARGET"])) if CONFIG["DISPLAY_FPS_TARGET"] > 0 else 1
    # cv2.namedWindow(exposed_window_name, cv2.WINDOW_NORMAL)

    # 7. Run the displayWindow (its main loop invokes the draw pipeline and analysis pipeline); blocking until the user presses "q" to quit.
    display_window.run()


if __name__ == "__main__":
    main()
