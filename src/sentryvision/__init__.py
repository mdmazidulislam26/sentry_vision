"""
SentryVision core package: person detection, tracking, face recognition,
and the attendance/security business logic built on top of them.

Public API:
    pipeline.process_video(...)  — run the full pipeline on a video file
    faces.add_known_face(...)    — register a new known person
    database.fetch_attendance()  — read logged attendance
    database.fetch_alerts()      — read logged security alerts
"""

from . import config, database, faces, pipeline

__all__ = ["config", "database", "faces", "pipeline"]
