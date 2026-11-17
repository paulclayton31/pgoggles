"""Threaded video pipeline for pgoggles webcam and future camera transports."""

from .analysis import AnalysisPipeline
from .camera import CameraSource
from .draw import DrawPipeline
from .overlay import draw_overlay

# Re-export pipeline dataclasses from the canonical overlays module
# for backward-compatible imports (e.g. ``from pgoggles.pipeline import ...``).
from ..overlays import ArrowDirection, BoundingBox, DrawMode, ExposedFrame, FramePacket, OverlayState

__all__ = [
    "AnalysisPipeline",
    "ArrowDirection",
    "BoundingBox",
    "CameraSource",
    "DrawMode",
    "DrawPipeline",
    "ExposedFrame",
    "FramePacket",
    "OverlayState",
    "draw_overlay",
]
