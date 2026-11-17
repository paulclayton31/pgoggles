"""Run the pgoggles webcam video pipeline MVP."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pgoggles.display import DisplayWindow
from pgoggles.overlays import DrawMode
from pgoggles.pipeline import AnalysisPipeline, CameraSource


CONFIG = {
    "CAMERA_INDEX": 0,
    "ANALYSIS_HZ_MAX": 10.0,
    "SCENE_CHANGE_THRESHOLD": 8.0,
    # Export once in an environment with ultralytics: yolo export model=yolov8n.pt format=onnx
    "YOLO_MODEL": "yolov8n.onnx",
    "DISPLAY_FPS_TARGET": 30.0,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    camera = CameraSource(camera_index=CONFIG["CAMERA_INDEX"])
    analysis = AnalysisPipeline(
        camera,
        hz_max=CONFIG["ANALYSIS_HZ_MAX"],
        scene_change_threshold=CONFIG["SCENE_CHANGE_THRESHOLD"],
        yolo_model=CONFIG["YOLO_MODEL"],
    )

    camera.start()
    analysis.start()
    DisplayWindow(
        camera,
        analysis,
        fps_target=CONFIG["DISPLAY_FPS_TARGET"],
        draw_mode=DrawMode.BOUNDING_BOX,
    ).run()


if __name__ == "__main__":
    main()
