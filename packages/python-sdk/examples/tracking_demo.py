"""Run the pgoggles multi-object tracking + identity demo.

Combines:

    YOLO person detection
        -> OC-SORT multi-object tracking
        -> persistent track_id
        -> InsightFace embedding
        -> pgvector database lookup
        -> cached identity per track

Example:

    Track 1 | Paul

If Paul leaves and returns later, OC-SORT may assign a new track:

    Track 7 | Identifying...
        -> face embedding
        -> database lookup
        -> Track 7 | Paul

Face recognition is NOT run every frame. Once a track is successfully
identified, the name is cached for that track_id.

Usage::

    python examples/tracking_demo.py
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # packages/python-sdk
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO_ROOT = Path(__file__).resolve().parents[3]  # pgoggles/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# pgoggles imports
# ---------------------------------------------------------------------------

from pgoggles.hooks import DetectionContext
from pgoggles.overlays import DetectionMode, DrawBoundingBox, TrackedBox
from pgoggles.pipeline import AnalysisPipeline, CameraSource, DrawPipeline
from pgoggles.display import DisplayWindow
from pgoggles.tracking import MultiObjectTracker, TrackingMode


CONFIG = {
    "CAMERA_INDEX": 0,
    "ANALYSIS_HZ_MAX": 10.0,
    "SCENE_CHANGE_THRESHOLD": 8.0,

    "YOLO_MODEL": str(ROOT / "yolov8n.onnx"),

    "DISPLAY_FPS_TARGET": 30.0,
    "PERSON_CONFIDENCE_THRESHOLD": 0.5,

    # Tracker
    "IOU_THRESHOLD": 0.3,
    "MIN_HITS": 3,
    "MAX_AGE": 5,

    # OC-SORT
    "USE_OCM": True,
    "OCM_WEIGHT": 0.2,
    "OCM_DELTA_T": 3,
    "USE_ORU": True,
    "USE_OCR": True,
    "OBSERVATION_HISTORY_SIZE": 5,

    # ---------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------

    # Because InsightFace returns normalized embeddings, we can compare
    # them with L2 distance.
    #
    # Smaller distance = more similar faces.
    #
    # This is an initial experimental threshold and should eventually
    # be calibrated using your own enrolled/test images.
    "MATCH_L2_THRESHOLD": 1.0,

    # If we fail to recognize a track, don't try again on every frame.
    # At 10 analysis frames/sec, 15 frames ~= 1.5 seconds.
    "IDENTITY_RETRY_FRAMES": 15,
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Load environment
    # ------------------------------------------------------------------

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    if os.getenv("PGOGGLES_IDENTITY_RESEARCH_ENABLED") != "1":
        raise SystemExit(
            "Identity research is disabled.\n"
            "Set PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 in Z:\\pgoggles\\.env"
        )

    # Import these only after identity research has been explicitly enabled.
    from research.identity.db.database import get_session
    from research.identity.db.queries import find_nearest
    from research.identity.embeddings.face_embedder import FaceEmbedder

    # ------------------------------------------------------------------
    # Camera / draw pipeline
    # ------------------------------------------------------------------

    camera = CameraSource(
        camera_index=CONFIG["CAMERA_INDEX"]
    )

    draw_pipeline = DrawPipeline()

    # ------------------------------------------------------------------
    # OC-SORT tracker
    # ------------------------------------------------------------------

    tracker = MultiObjectTracker(
        mode=TrackingMode.FULL,
        iou_threshold=CONFIG["IOU_THRESHOLD"],
        min_hits=CONFIG["MIN_HITS"],
        max_age=CONFIG["MAX_AGE"],
        use_ocm=CONFIG["USE_OCM"],
        ocm_weight=CONFIG["OCM_WEIGHT"],
        ocm_delta_t=CONFIG["OCM_DELTA_T"],
        use_oru=CONFIG["USE_ORU"],
        use_ocr=CONFIG["USE_OCR"],
        observation_history_size=CONFIG["OBSERVATION_HISTORY_SIZE"],
    )

    # ------------------------------------------------------------------
    # Analysis pipeline
    # ------------------------------------------------------------------

    analysis = AnalysisPipeline(
        camera,
        hz_max=CONFIG["ANALYSIS_HZ_MAX"],
        scene_change_threshold=CONFIG["SCENE_CHANGE_THRESHOLD"],
        yolo_model=CONFIG["YOLO_MODEL"],
        draw_pipeline=draw_pipeline,
        tracker=tracker,
    )

    display_window = DisplayWindow(
        camera_source=camera,
        analysis_pipeline=analysis,
        draw_pipeline=draw_pipeline,
        fps_target=CONFIG["DISPLAY_FPS_TARGET"],
        window_name="pgoggles — OC-SORT + Face Identity",
    )

    # ------------------------------------------------------------------
    # Track identity state
    # ------------------------------------------------------------------

    seen_track_ids: set[int] = set()

    # Example:
    #
    # track_identities[7] = {
    #     "person_id": 1,
    #     "name": "Paul",
    #     "resolved": True,
    #     "pending": False,
    #     "last_attempt_frame": 123,
    # }
    #
    track_identities: dict[int, dict] = {}

    identity_lock = threading.Lock()

    # Queue items are:
    #
    #     (track_id, person_crop)
    #
    embed_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(
        maxsize=4
    )

    stop_event = threading.Event()

    # ------------------------------------------------------------------
    # InsightFace
    # ------------------------------------------------------------------

    logging.info("Loading InsightFace model...")

    embedder = FaceEmbedder()

    logging.info("InsightFace ready.")

    # ------------------------------------------------------------------
    # Background identity worker
    # ------------------------------------------------------------------

    def _identity_worker() -> None:

        # SQLAlchemy Session lives entirely inside this thread.
        session = get_session()

        try:
            while not stop_event.is_set():

                try:
                    track_id, crop = embed_queue.get(timeout=0.5)

                except queue.Empty:
                    continue

                try:
                    # --------------------------------------------------
                    # Person crop -> 512-dimensional face embedding
                    # --------------------------------------------------

                    embedding = embedder.embed(crop)

                    if embedding is None:

                        logging.info(
                            "Track %s: no face detected",
                            track_id,
                        )

                        with identity_lock:

                            identity = track_identities.get(track_id)

                            if identity is not None:
                                identity["pending"] = False
                                identity["name"] = "No Face"

                        continue

                    # --------------------------------------------------
                    # Find closest enrolled person
                    # --------------------------------------------------

                    matches = find_nearest(
                        session,
                        embedding,
                        limit=1,
                    )

                    if not matches:

                        with identity_lock:

                            identity = track_identities.get(track_id)

                            if identity is not None:
                                identity["pending"] = False
                                identity["name"] = "Unknown"

                        continue

                    match = matches[0]

                    # --------------------------------------------------
                    # Calculate actual L2 distance
                    # --------------------------------------------------

                    query_vector = np.asarray(
                        embedding,
                        dtype=np.float32,
                    )

                    stored_vector = np.asarray(
                        match.embedding,
                        dtype=np.float32,
                    )

                    distance = float(
                        np.linalg.norm(
                            query_vector - stored_vector
                        )
                    )

                    logging.info(
                        "Track %s: nearest=%s distance=%.4f",
                        track_id,
                        match.name,
                        distance,
                    )

                    # --------------------------------------------------
                    # Reject weak matches
                    # --------------------------------------------------

                    if distance <= CONFIG["MATCH_L2_THRESHOLD"]:

                        with identity_lock:

                            identity = track_identities.get(track_id)

                            if identity is not None:

                                identity["person_id"] = match.id
                                identity["name"] = match.name
                                identity["resolved"] = True
                                identity["pending"] = False

                        logging.info(
                            "Track %s identified as %s (person_id=%s)",
                            track_id,
                            match.name,
                            match.id,
                        )

                    else:

                        with identity_lock:

                            identity = track_identities.get(track_id)

                            if identity is not None:

                                identity["person_id"] = None
                                identity["name"] = "Unknown"
                                identity["resolved"] = False
                                identity["pending"] = False

                        logging.info(
                            "Track %s rejected nearest match %s "
                            "(distance %.4f > threshold %.4f)",
                            track_id,
                            match.name,
                            distance,
                            CONFIG["MATCH_L2_THRESHOLD"],
                        )

                except Exception:

                    logging.exception(
                        "Identity lookup failed for track %s",
                        track_id,
                    )

                    with identity_lock:

                        identity = track_identities.get(track_id)

                        if identity is not None:
                            identity["pending"] = False

        finally:
            session.close()

    identity_thread = threading.Thread(
        target=_identity_worker,
        name="pgoggles-track-identity",
        daemon=True,
    )

    identity_thread.start()

    # ------------------------------------------------------------------
    # Person ACTIVE callback
    # ------------------------------------------------------------------

    def _on_person_active(
        boxes: list[TrackedBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:

        nonlocal seen_track_ids

        current_ids = {
            b.track_id
            for b in boxes
        }

        # --------------------------------------------------------------
        # Remove tracks that disappeared
        # --------------------------------------------------------------

        stale_ids = seen_track_ids - current_ids

        for stale_id in stale_ids:

            draw_pipeline.remove_command(
                f"track-{stale_id}"
            )

            with identity_lock:
                track_identities.pop(
                    stale_id,
                    None,
                )

        # --------------------------------------------------------------
        # Process active tracks
        # --------------------------------------------------------------

        for b in boxes:

            with identity_lock:

                # ------------------------------------------------------
                # New track
                # ------------------------------------------------------

                if b.track_id not in track_identities:

                    track_identities[b.track_id] = {
                        "person_id": None,
                        "name": "Identifying...",
                        "resolved": False,
                        "pending": False,
                        "last_attempt_frame": -999999,
                    }

                identity = track_identities[b.track_id]

                # ------------------------------------------------------
                # Determine whether recognition should run
                # ------------------------------------------------------

                frames_since_attempt = (
                    context.frame_id
                    - identity["last_attempt_frame"]
                )

                should_identify = (
                    not identity["resolved"]
                    and not identity["pending"]
                    and context.frame is not None
                    and frames_since_attempt
                    >= CONFIG["IDENTITY_RETRY_FRAMES"]
                )

                if should_identify:

                    identity["pending"] = True
                    identity["last_attempt_frame"] = context.frame_id

                name = identity["name"]

            # ----------------------------------------------------------
            # Submit person's bounding-box crop for recognition
            # ----------------------------------------------------------

            if should_identify and context.frame is not None:

                frame = context.frame

                x1 = max(
                    0,
                    int(b.x1),
                )

                y1 = max(
                    0,
                    int(b.y1),
                )

                x2 = min(
                    frame.shape[1],
                    int(b.x2),
                )

                y2 = min(
                    frame.shape[0],
                    int(b.y2),
                )

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size > 0:

                    try:

                        embed_queue.put_nowait(
                            (
                                b.track_id,
                                person_crop.copy(),
                            )
                        )

                    except queue.Full:

                        logging.warning(
                            "Identity queue full; skipping track %s",
                            b.track_id,
                        )

                        with identity_lock:

                            current_identity = track_identities.get(
                                b.track_id
                            )

                            if current_identity is not None:
                                current_identity["pending"] = False

                else:

                    with identity_lock:

                        current_identity = track_identities.get(
                            b.track_id
                        )

                        if current_identity is not None:
                            current_identity["pending"] = False

            # ----------------------------------------------------------
            # Get latest identity after possible worker update
            # ----------------------------------------------------------

            with identity_lock:

                current_identity = track_identities.get(
                    b.track_id
                )

                if current_identity is not None:
                    name = current_identity["name"]
                else:
                    name = "Unknown"

            # ----------------------------------------------------------
            # Draw
            # ----------------------------------------------------------

            draw_pipeline.enqueue(
                DrawBoundingBox(
                    x1=b.x1,
                    y1=b.y1,
                    x2=b.x2,
                    y2=b.y2,
                    label=(
                        f"Track {b.track_id} | "
                        f"{name}"
                    ),
                    confidence=b.confidence,
                    command_id=f"track-{b.track_id}",
                )
            )

        seen_track_ids = current_ids

    # ------------------------------------------------------------------
    # Person GONE callback
    # ------------------------------------------------------------------

    def _on_person_gone(
        boxes: list[TrackedBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:

        nonlocal seen_track_ids

        for track_id in seen_track_ids:

            draw_pipeline.remove_command(
                f"track-{track_id}"
            )

        with identity_lock:

            for track_id in list(seen_track_ids):

                track_identities.pop(
                    track_id,
                    None,
                )

        seen_track_ids = set()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.ACTIVE,
        confidence_threshold=CONFIG[
            "PERSON_CONFIDENCE_THRESHOLD"
        ],
        debounce_ms=0.0,
        callback=_on_person_active,
    )

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.GONE,
        confidence_threshold=CONFIG[
            "PERSON_CONFIDENCE_THRESHOLD"
        ],
        debounce_ms=0.0,
        callback=_on_person_gone,
    )

    # ------------------------------------------------------------------
    # Start pipeline
    # ------------------------------------------------------------------

    camera.start()
    draw_pipeline.start()
    analysis.start()

    try:

        display_window.run()

    finally:

        stop_event.set()

        identity_thread.join(
            timeout=2.0
        )


if __name__ == "__main__":
    main()