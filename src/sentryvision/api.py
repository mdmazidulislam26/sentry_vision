"""
FastAPI service exposing the SentryVision pipeline over HTTP.

This wraps the same `sentryvision` package used by `app.py` (Streamlit) —
no pipeline logic lives here, only request/response handling. Running both
an interactive dashboard (Streamlit) and a programmatic API (FastAPI) off
the same core package is the point of having pulled the logic out of
`app.py` in the first place.
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import config, database, faces, pipeline

app = FastAPI(
    title="SentryVision API",
    description="Person detection, tracking, and face-recognition-based attendance/security monitoring.",
    version="1.0.0",
)


class AttendanceRow(BaseModel):
    name: str
    track_id: int
    entry_time: str


class AlertRow(BaseModel):
    track_id: int
    detected_time: str
    snapshot_path: str


class ProcessVideoResponse(BaseModel):
    total_frames: int
    saved_frames: int
    attendance_logged: int
    alerts_raised: int


class EnrollResponse(BaseModel):
    success: bool
    message: str


@app.on_event("startup")
def startup():
    config.ensure_directories()


@app.get("/health")
def health():
    """Liveness check — used by the CI build-check and container orchestrators."""
    return {"status": "ok"}


@app.post("/enroll", response_model=EnrollResponse)
def enroll(name: str, file: UploadFile = File(...)):
    """Register a new known person from a reference photo."""
    success, message = faces.add_known_face(file.file, name)
    if not success:
        raise HTTPException(status_code=422, detail=message)
    return EnrollResponse(success=success, message=message)


@app.post("/process-video", response_model=ProcessVideoResponse)
def process_video(file: UploadFile = File(...)):
    """Run the full detect -> track -> recognize -> log pipeline on an uploaded video.

    Synchronous and blocking by design (matches the Streamlit app's behavior) —
    a production deployment serving large videos or concurrent requests would
    want to move this behind a background task queue instead.
    """
    known_dict = faces.load_known_encodings()
    known_names = list(known_dict.keys())
    known_encodings = list(known_dict.values())

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        summary = pipeline.process_video(tmp_path, known_names, known_encodings)
    finally:
        os.remove(tmp_path)

    return ProcessVideoResponse(**summary)


@app.get("/attendance", response_model=list[AttendanceRow])
def get_attendance():
    rows = database.fetch_attendance()
    return [AttendanceRow(name=n, track_id=t, entry_time=e) for n, t, e in rows]


@app.get("/alerts", response_model=list[AlertRow])
def get_alerts():
    rows = database.fetch_alerts()
    return [AlertRow(track_id=t, detected_time=d, snapshot_path=s) for t, d, s in rows]
