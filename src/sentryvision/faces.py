"""
Known-face management: persisting reference face encodings to disk (as a
pickle keyed by name) so they survive across separate video-processing runs,
and don't need to be recomputed from the reference photos every time.
"""

import os
import pickle

import face_recognition
import numpy as np

from . import config


def load_known_encodings():
    """Load {name: encoding} from disk. Returns an empty dict if none saved yet."""
    if os.path.exists(config.ENCODINGS_PATH):
        with open(config.ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_known_encodings(encodings_dict):
    with open(config.ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings_dict, f)


def add_known_face(uploaded_image_file, person_name):
    """Save an uploaded reference photo, compute its face encoding once, and
    persist it alongside any other known faces.

    `uploaded_image_file` is any file-like object exposing `.getbuffer()`
    (e.g. a Streamlit `UploadedFile`).

    Returns (success: bool, message: str).
    """
    image_path = os.path.join(config.KNOWN_FACES_DIR, f"{person_name}.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image_file.getbuffer())

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        os.remove(image_path)
        return False, "No face detected in this image. Please upload a clearer photo."

    known = load_known_encodings()
    known[person_name] = encodings[0]
    save_known_encodings(known)
    return True, f"Saved '{person_name}'. This person will now be recognized in any processed video."


def match_face(encoding, known_names, known_encodings):
    """Compare one face encoding against the known set.

    Returns the matched name, or "Unknown" if no match clears the tolerance
    threshold (or no known faces are registered yet).
    """
    if not known_encodings:
        return "Unknown"

    matches = face_recognition.compare_faces(
        known_encodings, encoding, tolerance=config.FACE_MATCH_TOLERANCE
    )
    face_distances = face_recognition.face_distance(known_encodings, encoding)

    if True in matches and len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        return known_names[best_match_index]
    return "Unknown"
