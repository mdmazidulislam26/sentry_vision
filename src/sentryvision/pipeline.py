"""
Core video-processing pipeline: person detection (YOLOv8) -> multi-object
tracking (ByteTrack) -> face recognition -> attendance/security business logic.

This is the single production pipeline. It is intentionally framework-agnostic
(no Streamlit imports) so it can be called from the dashboard, a CLI script,
or a future batch/queue worker without modification.
"""

import os
from datetime import datetime

import cv2
import face_recognition
from ultralytics import YOLO

from . import config, database, faces


def load_model():
    """Load the YOLOv8 model. Framework-agnostic — callers that want caching
    across repeated calls (e.g. a Streamlit session) should wrap this
    themselves (see app.py), rather than this module depending on any
    particular UI framework.
    """
    return YOLO(config.YOLO_MODEL_NAME)


def process_video(video_path, known_names, known_encodings, model=None, progress_callback=None):
    """Run the full tracking + recognition + business-logic pipeline on a video file.

    Args:
        video_path: path to the input video.
        known_names: list of names, parallel to known_encodings.
        known_encodings: list of face encodings for the known people above.
        model: a pre-loaded YOLO model instance. If omitted, one is loaded
            fresh via `load_model()` — pass a cached instance in UI contexts
            to avoid reloading weights on every call.
        progress_callback: optional callable(fraction: float) invoked after
            each frame, for UI progress reporting.

    Returns:
        A summary dict: total_frames, saved_frames, attendance_logged, alerts_raised.
    """
    config.ensure_directories()

    model = model or load_model()
    conn = database.get_connection()
    cursor = conn.cursor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Failed to open the uploaded video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # track_id -> resolved name, cached so recognition only runs once per track
    track_id_to_name = {}

    # Dedup guards: business-logic events must fire exactly once per person/track
    logged_names = set()
    alerted_track_ids = set()

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            classes=[config.PERSON_CLASS_ID],
            conf=config.DETECTION_CONFIDENCE,
            iou=config.DETECTION_IOU,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            verbose=False,
        )

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                if track_id not in track_id_to_name:
                    # Skip implausibly large boxes — the tracker hasn't yet
                    # separated nearby people into distinct boxes.
                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    if frame_area > 0 and (box_area / frame_area) > config.MAX_BOX_AREA_RATIO:
                        continue

                    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if person_crop.size == 0:
                        continue

                    # face_recognition expects RGB; OpenCV frames are BGR.
                    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(person_crop_rgb)

                    # Filter out tiny false-positive detections (e.g. logos, patterns).
                    crop_area = person_crop.shape[0] * person_crop.shape[1]
                    valid_locations = [
                        (top, right, bottom, left)
                        for (top, right, bottom, left) in face_locations
                        if crop_area > 0
                        and ((right - left) * (bottom - top) / crop_area) >= config.MIN_FACE_AREA_RATIO
                    ]

                    if valid_locations:
                        face_encodings = face_recognition.face_encodings(person_crop_rgb, valid_locations)
                        encoding = face_encodings[0]  # take the largest/first valid face

                        debug_path = os.path.join(
                            config.DEBUG_DIR, f"track_{track_id}_frame_{frame_count:05d}.jpg"
                        )
                        cv2.imwrite(debug_path, person_crop)

                        name = faces.match_face(encoding, known_names, known_encodings)
                        track_id_to_name[track_id] = name

                        # --- Business logic: fires exactly once per event ---
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if name != "Unknown":
                            if name not in logged_names:
                                database.log_attendance(cursor, name, track_id, timestamp)
                                conn.commit()
                                logged_names.add(name)
                        else:
                            if track_id not in alerted_track_ids:
                                snapshot_path = os.path.join(
                                    config.ALERT_SNAPSHOT_DIR,
                                    f"unknown_track{track_id}_frame{frame_count:05d}.jpg",
                                )
                                cv2.imwrite(snapshot_path, person_crop)
                                database.log_alert(cursor, track_id, timestamp, snapshot_path)
                                conn.commit()
                                alerted_track_ids.add(track_id)
                    else:
                        # No valid face found yet in this crop; retry on a later frame.
                        track_id_to_name[track_id] = "Detecting..."

                # Draw box + label (ID + resolved name) regardless of whether
                # this frame triggered a new recognition.
                display_name = track_id_to_name.get(track_id, "Detecting...")
                label = f"ID:{track_id} {display_name}"
                color = (0, 255, 0) if display_name not in ("Unknown", "Detecting...") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1

        if frame_count % config.SAVE_INTERVAL == 0:
            output_path = os.path.join(config.OUTPUT_DIR, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        if progress_callback:
            progress_callback(min(frame_count / total_frames, 1.0))

    cap.release()
    conn.close()

    return {
        "total_frames": frame_count,
        "saved_frames": saved_count,
        "attendance_logged": len(logged_names),
        "alerts_raised": len(alerted_track_ids),
    }
