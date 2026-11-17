"""OpenCV compositing helpers for analysis overlays.

Draws bounding boxes and a runtime HUD onto a frame in-place.
Imports pipeline dataclasses from :mod:`.overlays` (the canonical
location for display primitives).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..overlays import BoundingBox, OverlayState

GREEN = (70, 230, 90)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
HUD_BG = (20, 20, 20)
ANALYZING = (0, 210, 255)


def draw_overlay(
    frame: np.ndarray,
    overlay: OverlayState | None,
    *,
    fps: float,
    analysis_hz: float,
    analyzing: bool,
) -> np.ndarray:
    """Draw boxes and pipeline HUD onto ``frame`` in place."""

    if overlay is not None:
        for box in overlay.boxes:
            draw_box(frame, box)

    inference_ms = overlay.inference_ms if overlay else 0.0
    scene_change_score = overlay.scene_change_score if overlay else 0.0
    draw_hud(
        frame,
        fps=fps,
        analysis_hz=analysis_hz,
        inference_ms=inference_ms,
        scene_change_score=scene_change_score,
        analyzing=analyzing,
    )
    return frame


def draw_box(frame: np.ndarray, box: BoundingBox) -> None:
    """Draw one detection box with a label badge."""

    cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), GREEN, 2)

    label = f"{box.label} {box.confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    label_y = max(box.y1, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (box.x1, label_y - text_height - baseline - 4),
        (box.x1 + text_width + 8, label_y + baseline),
        GREEN,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (box.x1 + 4, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BLACK,
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    analysis_hz: float,
    inference_ms: float,
    scene_change_score: float,
    analyzing: bool,
) -> None:
    """Draw the small runtime HUD in the top-left corner."""

    lines = [
        f"FPS {fps:4.1f}",
        f"ANALYSIS {analysis_hz:4.1f} Hz",
        f"INFERENCE {inference_ms:5.1f} ms",
        f"SCENE MAD {scene_change_score:4.1f}",
    ]
    width = 188
    height = 104 if analyzing else 88
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
