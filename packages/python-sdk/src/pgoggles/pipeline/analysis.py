"""Threaded YOLO analysis pipeline for camera frames."""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
from pgoggles.pipeline.draw import DrawPipeline

from .camera import CameraSource
from ..hooks import HookRegistry
from ..overlays import BoundingBox, ExposedFrame, FramePacket, OverlayState
from ..tracking.tracker import MultiObjectTracker

logger = logging.getLogger(__name__)

COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class AnalysisPipeline:
    """Consume camera frames and publish the latest object-detection overlay."""

    def __init__(
        self,
        camera_source: CameraSource,
        *,
        hz_max: float = 10.0,
        scene_change_threshold: float = 8.0,
        yolo_model: str = "yolov8n.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        highest_confidence: bool = False,
        draw_pipeline: DrawPipeline,
        tracker: MultiObjectTracker | None = None,
    ) -> None:
        self.camera_source = camera_source
        self.hz_max = hz_max
        self.scene_change_threshold = scene_change_threshold
        self.yolo_model = yolo_model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.highest_confidence = highest_confidence
        self.draw_pipeline = draw_pipeline
        # Optional multi-object tracker: when set, onDetect hooks receive
        # TrackedBox (with a persistent track_id) instead of raw BoundingBox.
        # Public and mutable post-construction, same pattern as
        # highest_confidence. None (the default) is a true no-op: hook_boxes
        # in _analysis_loop stays literally `overlay.boxes`, so behavior is
        # byte-for-byte identical to before this was added.
        self.tracker = tracker

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._overlay_lock = threading.Lock()
        self._latest_overlay: OverlayState | None = None
        self._analysis_timestamps: deque[float] = deque(maxlen=30)
        self._last_analyzed_frame: np.ndarray | None = None
        self._last_analysis_at = 0.0
        self._session: Any | None = None
        self._input_name: str | None = None
        self._input_size = (640, 640)
        self._in_flight = False
        self._in_flight_lock = threading.Lock()
        self._hooks = HookRegistry()

        # Exposed frames buffer for external consumers
        self._exposed_lock = threading.Lock()
        self._exposed_frames: deque[ExposedFrame] = deque(maxlen=100)

    @property
    def hooks(self) -> HookRegistry:
        """Access the hook registry for registering onDetect callbacks."""
        return self._hooks

    def start(self) -> None:
        """Start the analysis worker thread."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._analysis_loop,
            name="pgoggles-analysis",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop analysis and wait briefly for the worker to exit."""

        self._stop_event.set()
        self._hooks.clear()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest_overlay(self) -> OverlayState | None:
        """Return the most recent overlay state without blocking display."""

        with self._overlay_lock:
            return self._latest_overlay

    def expose_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: int | None = None,
        labels: set[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Publish a frame (or frame subset) for external consumers.

        This is a fire-and-forget publish — no callbacks or pub/sub.
        External consumers retrieve published frames via
        :meth:`get_exposed_frames`.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame data to publish. Can be a full frame or a cropped
            region of interest.
        frame_id : int | None
            Optional frame ID. Defaults to the latest overlay's frame ID.
        labels : set[str] | None
            Optional set of detection labels associated with this frame.
        tags : list[str] | None
            Optional list of arbitrary tags for filtering.
        """
        with self._overlay_lock:
            if frame_id is None and self._latest_overlay is not None:
                frame_id = self._latest_overlay.frame_id

        exposed = ExposedFrame(
            frame=frame,
            frame_id=frame_id,
            labels=labels if labels is not None else set(),
            tags=tags if tags is not None else [],
        )
        with self._exposed_lock:
            self._exposed_frames.append(exposed)

    def get_exposed_frames(
        self,
        *,
        labels: set[str] | None = None,
        tags: list[str] | None = None,
        frame_id: int | None = None,
    ) -> list[ExposedFrame]:
        """Retrieve published frames, optionally filtered.

        Parameters
        ----------
        labels : set[str] | None
            If provided, only return frames whose ``labels`` intersect
            with this set.
        tags : list[str] | None
            If provided, only return frames whose ``tags`` contain all
            of the specified tags.
        frame_id : int | None
            If provided, only return frames matching this ID.

        Returns
        -------
        list[ExposedFrame]
            A snapshot of matching exposed frames.
        """
        with self._exposed_lock:
            frames = list(self._exposed_frames)

        result: list[ExposedFrame] = []
        for ef in frames:
            if frame_id is not None and ef.frame_id != frame_id:
                continue
            if labels is not None and not ef.labels.intersection(labels):
                continue
            if tags is not None and not all(t in ef.tags for t in tags):
                continue
            result.append(ef)
        return result

    @property
    def analyzing(self) -> bool:
        """Whether the analysis worker is currently running YOLO inference."""

        with self._in_flight_lock:
            return self._in_flight

    @property
    def analysis_hz(self) -> float:
        """Actual completed analysis rate over the recent timestamp window."""

        if len(self._analysis_timestamps) < 2:
            return 0.0
        elapsed = self._analysis_timestamps[-1] - self._analysis_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._analysis_timestamps) - 1) / elapsed

    def _analysis_loop(self) -> None:
        self._load_model()

        while not self._stop_event.is_set():
            try:
                packet = self.camera_source.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()
            min_interval = 1.0 / self.hz_max if self.hz_max > 0 else 0.0
            if now - self._last_analysis_at < min_interval:
                continue

            scene_change_score = self._scene_change_score(packet)
            if (
                self._last_analyzed_frame is not None
                and scene_change_score < self.scene_change_threshold
            ):
                self._publish_scene_score(packet, scene_change_score)
                continue

            self._last_analysis_at = now
            overlay = self._analyze(packet, scene_change_score)
            if overlay is None:
                continue

            self._last_analyzed_frame = _analysis_frame(packet.frame)
            with self._overlay_lock:
                self._latest_overlay = overlay
            self._analysis_timestamps.append(time.perf_counter())

            # If a tracker is configured, hand hooks TrackedBox (with a
            # persistent track_id) instead of raw BoundingBox. overlay.boxes
            # itself is never mutated/replaced -- DrawPipeline/draw_overlay
            # keep seeing raw detector output regardless of tracking.
            hook_boxes: list[BoundingBox] = overlay.boxes
            if self.tracker is not None:
                hook_boxes = self.tracker.update(overlay.boxes)

            # Evaluate registered onDetect hooks against the published overlay
            self._hooks.evaluate(
                boxes=hook_boxes,
                frame_id=overlay.frame_id,
                timestamp=overlay.timestamp,
                frame=packet.frame,
                draw_pipeline=self.draw_pipeline,
                overlay_state={
                    "inference_ms": overlay.inference_ms,
                    "scene_change_score": overlay.scene_change_score,
                },
            )

    def _load_model(self) -> None:
        model_path = Path(self.yolo_model)
        if not model_path.exists():
            logger.warning(
                "YOLO ONNX model %s was not found. Export once with: "
                "yolo export model=yolov8n.pt format=onnx",
                self.yolo_model,
            )
            return

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            model_input = self._session.get_inputs()[0]
            self._input_name = model_input.name
            self._input_size = _input_size_from_shape(model_input.shape)
        except Exception as exc:
            logger.warning("YOLO ONNX model %s could not be loaded: %s", self.yolo_model, exc)

    def _analyze(self, packet: FramePacket, scene_change_score: float) -> OverlayState | None:
        if self._session is None or self._input_name is None:
            return OverlayState(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                boxes=[],
                inference_ms=0.0,
                scene_change_score=scene_change_score,
            )

        self._set_in_flight(True)
        started = time.perf_counter()
        try:
            blob, scale, pad = _preprocess(packet.frame, self._input_size)
            outputs = self._session.run(None, {self._input_name: blob})
            inference_ms = (time.perf_counter() - started) * 1000.0
            boxes = _boxes_from_outputs(
                outputs[0],
                packet.frame.shape,
                scale,
                pad,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
            highlighted_boxes = None
            if self.highest_confidence and boxes:
                highlighted_boxes = [_highest_confidence(boxes)]
            return OverlayState(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                boxes=boxes,
                inference_ms=inference_ms,
                scene_change_score=scene_change_score,
                highlighted_boxes=highlighted_boxes
            )
        except Exception as exc:
            logger.warning("YOLO ONNX inference failed on frame %s: %s", packet.frame_id, exc)
            return None
        finally:
            self._set_in_flight(False)

    def _scene_change_score(self, packet: FramePacket) -> float:
        if self._last_analyzed_frame is None:
            return 255.0

        current = _analysis_frame(packet.frame)
        return float(np.mean(cv2.absdiff(current, self._last_analyzed_frame)))

    def _set_in_flight(self, value: bool) -> None:
        with self._in_flight_lock:
            self._in_flight = value

    def _publish_scene_score(self, packet: FramePacket, scene_change_score: float) -> None:
        with self._overlay_lock:
            previous = self._latest_overlay
            if previous is None:
                self._latest_overlay = OverlayState(
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                    boxes=[],
                    inference_ms=0.0,
                    scene_change_score=scene_change_score,
                )
                return

            self._latest_overlay = OverlayState(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                boxes=previous.boxes,
                inference_ms=previous.inference_ms,
                scene_change_score=scene_change_score,
            )


def _highest_confidence(boxes: list[BoundingBox]) -> BoundingBox:
    """Return the bounding box with the highest confidence score."""

    if not boxes:
        raise ValueError("Cannot select highest confidence from empty box list")
    return max(boxes, key=lambda b: b.confidence)


def _analysis_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def _input_size_from_shape(shape: list[Any]) -> tuple[int, int]:
    height = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 640
    width = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 640
    return width, height


def _preprocess(
    frame: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    input_width, input_height = input_size
    height, width = frame.shape[:2]
    scale = min(input_width / width, input_height / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, scale, (pad_x, pad_y)


def _boxes_from_outputs(
    output: np.ndarray,
    frame_shape: tuple[int, ...],
    scale: float,
    pad: tuple[int, int],
    *,
    confidence_threshold: float,
    nms_threshold: float,
) -> list[BoundingBox]:
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    frame_height, frame_width = frame_shape[:2]
    pad_x, pad_y = pad
    boxes_xywh: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    for prediction in predictions:
        class_scores = prediction[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue

        cx, cy, width, height = prediction[:4]
        x1 = int(round((cx - width / 2 - pad_x) / scale))
        y1 = int(round((cy - height / 2 - pad_y) / scale))
        x2 = int(round((cx + width / 2 - pad_x) / scale))
        y2 = int(round((cy + height / 2 - pad_y) / scale))
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, confidence_threshold, nms_threshold)
    if len(keep) == 0:
        return []

    boxes: list[BoundingBox] = []
    for index in np.array(keep).flatten():
        x, y, width, height = boxes_xywh[int(index)]
        class_id = class_ids[int(index)]
        label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else str(class_id)
        boxes.append(
            BoundingBox(
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                label=label,
                confidence=scores[int(index)],
            )
        )
    return boxes
