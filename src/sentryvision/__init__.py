"""
SentryVision core package: person detection, tracking, face recognition,
and the attendance/security business logic built on top of them.

Public API:
    pipeline.process_video(...)  — run the full pipeline on a video file
    faces.add_known_face(...)    — register a new known person
    database.fetch_attendance()  — read logged attendance
    database.fetch_alerts()      — read logged security alerts

Submodules are intentionally NOT eagerly imported here. `faces` and
`pipeline` depend on heavy CV libraries (face_recognition/dlib, ultralytics),
while `config` and `database` are pure-stdlib. Eagerly importing everything
on `import sentryvision` would force every caller — including lightweight
tests that only need `database` — to have the full CV stack installed.
Import the specific submodule you need instead, e.g.:
    from sentryvision import database
    from sentryvision import pipeline
"""

__all__ = ["config", "database", "faces", "pipeline"]
