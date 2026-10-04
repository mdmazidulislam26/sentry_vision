"""
Central configuration for SentryVision: file paths and tunable constants.

Keeping these in one module means every other module (database, faces,
pipeline, app) imports the same values instead of redefining them —
avoids the drift that happens when constants are copy-pasted across
multiple notebook cells.
"""

import os

# --- Base data directory (all runtime artifacts live under here) ---
DATA_DIR = os.environ.get("SENTRYVISION_DATA_DIR", "/content")

DB_PATH = os.path.join(DATA_DIR, "sentryvision.db")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
ALERT_SNAPSHOT_DIR = os.path.join(DATA_DIR, "alerts")
DEBUG_DIR = os.path.join(DATA_DIR, "debug_face_crops")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
ENCODINGS_PATH = os.path.join(DATA_DIR, "known_faces_encodings.pkl")
UPLOADED_VIDEO_PATH = os.path.join(DATA_DIR, "uploaded_video.mp4")

# --- Detection / tracking ---
YOLO_MODEL_NAME = "yolov8n.pt"
PERSON_CLASS_ID = 0          # COCO class 0 = 'person'
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU = 0.45
TRACKER_CONFIG = "bytetrack.yaml"

# A single-person box covering more than this fraction of the frame is treated
# as an unstable/merged detection — common in the first couple of frames right
# after a track is created, before the tracker has separated nearby people
# into distinct boxes.
MAX_BOX_AREA_RATIO = 0.5

# --- Face recognition ---
# Empirically tuned: filters out small false-positive detections (e.g. a logo
# on a t-shirt) while still keeping genuine faces.
MIN_FACE_AREA_RATIO = 0.002
FACE_MATCH_TOLERANCE = 0.5

# --- Video processing ---
SAVE_INTERVAL = 30  # save one annotated frame every N frames


def ensure_directories():
    """Create every runtime directory this project writes to, if missing."""
    for d in (OUTPUT_DIR, ALERT_SNAPSHOT_DIR, DEBUG_DIR, KNOWN_FACES_DIR):
        os.makedirs(d, exist_ok=True)
