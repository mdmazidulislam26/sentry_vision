# 🛡️ SentryVision — Real-Time Smart Attendance & Security Monitoring

An end-to-end computer vision system that watches a video feed, identifies known people, logs their attendance, and raises a security alert for anyone it doesn't recognize — with a self-service Streamlit dashboard for managing known faces and reviewing results.

**🔗 Live demo:** *(add your ngrok/deployment link here when running)*

---

## 📌 Overview

**Pipeline:** Video Frame → YOLOv8 (person detection) → ByteTrack (multi-object tracking) → `face_recognition` (identity match) → business logic (attendance log / security alert) → SQLite → Streamlit dashboard.

This project combines three CV building blocks that are usually demonstrated separately — detection, tracking, and recognition — into one system with real business rules on top: each known person is logged into attendance **exactly once** per video, and each unrecognized person triggers **exactly one** security alert, no matter how many frames they appear in.

### Why tracking matters here
Running face recognition on every single frame is both wasteful and inconsistent — the same person could be logged multiple times, or missed on frames where their face is turned away. ByteTrack assigns a persistent ID to each person across frames, so recognition only needs to run **once per track**, and the result is cached and reused for every subsequent frame that track appears in.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────────┐
│ Video Frame │ --> │  YOLOv8  │ --> │  ByteTrack   │ --> │ face_recognition│
│             │     │ (person) │     │ (track IDs)  │     │ (per new track) │
└─────────────┘     └──────────┘     └──────────────┘     └────────┬────────┘
                                                                     │
                                                      ┌──────────────┴──────────────┐
                                                      │                             │
                                                known person?                 unknown person?
                                                      │                             │
                                                      ▼                             ▼
                                          attendance_log (SQLite)        security_alerts (SQLite)
                                          [dedup: once per name]      [dedup: once per track ID]
                                                      │                             │
                                                      └──────────────┬──────────────┘
                                                                     ▼
                                                          Streamlit Dashboard
                                                (Known Faces · Process Video · Attendance · Alerts · Frames)
```

---

## ✨ Features

- **Three-stage CV pipeline**: detection → tracking → recognition, each stage's output feeding the next
- **Idempotent business logic**: dedup guards ensure attendance and alerts each fire exactly once per person/track, regardless of how many frames they're visible in
- **Self-service known-faces enrollment**: add a new known person directly from the dashboard (no code, no notebook re-run) — encodings persist across sessions via a pickle file
- **False-positive filtering**: small non-face detections (e.g. a logo on clothing) are filtered out by a minimum face-area-ratio threshold, tuned empirically
- **Unstable-track filtering**: implausibly large bounding boxes (common in the first frame or two after a track is created, before nearby people are separated) are skipped rather than misrecognized
- **Debuggable by design**: every face crop used for recognition is saved to a debug directory, labeled by track ID and frame number, so a wrong `track_id → name` mapping can be traced back to the exact image that caused it

---

## 🧰 Tech Stack

- **Detection**: YOLOv8-nano (`ultralytics`), pretrained on COCO
- **Tracking**: ByteTrack (via `ultralytics`' built-in `model.track()`)
- **Recognition**: `face_recognition` (dlib-based face encoding + matching)
- **Storage**: SQLite (attendance/alerts), pickle (known-face encodings)
- **Frontend**: Streamlit
- **Deployment**: ngrok (Colab-hosted public tunnel)

---

## 📂 Project Structure

```
sentry_vision/
├── src/
│   └── sentryvision/
│       ├── __init__.py       # package public API
│       ├── config.py          # paths & tunable constants, single source of truth
│       ├── database.py        # SQLite schema + data access
│       ├── faces.py           # known-face encoding storage & matching
│       └── pipeline.py        # core detect -> track -> recognize -> log pipeline
├── app.py                      # Streamlit UI — thin layer, imports all logic from src/
├── notebooks/
│   └── sentry_vision_exploration.ipynb   # dev journey: baseline -> tracking -> recognition -> integration
├── data/
│   └── known_faces/            # known-person reference photos (gitignored; populated at runtime)
├── requirements.txt
├── .gitignore
└── README.md
```

**Why split `app.py` from `src/sentryvision/`?** The original single-notebook version had all detection/tracking/recognition/database logic inline inside the Streamlit file. Pulling it into a plain Python package means:
- the same `process_video()` pipeline is callable from a future CLI script, batch job, or API — not locked to Streamlit
- `config.py` is the single source of truth for paths and constants, instead of being redefined in multiple places
- each concern (database, face matching, pipeline) can be tested or modified independently

---

## 🚀 Setup & Run (Google Colab)

```python
# 1. Clone the repo (or upload the project folder directly)
!git clone https://github.com/<your-username>/sentry_vision.git
%cd sentry_vision

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Set secrets in Colab (🔑 icon in sidebar)
#    - NGROK_AUTHTOKEN : from dashboard.ngrok.com

# 4. Deploy
import subprocess, time
from pyngrok import ngrok
from google.colab import userdata

ngrok.set_auth_token(userdata.get('NGROK_AUTHTOKEN'))

# Clean up any tunnel/process left over from a previous session
for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)
ngrok.kill()
!pkill -f streamlit

streamlit_process = subprocess.Popen(
    ['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true']
)
time.sleep(5)

public_url = ngrok.connect(8501)
print(f"Dashboard live at: {public_url}")
```

---

## 🐛 Key Engineering Challenges & Fixes

1. **numpy ABI mismatch**: pinning `numpy` to an older version to satisfy one library broke binary compatibility with Colab's pre-installed, numpy-2.x-compiled packages (`torch`, `scipy`). Fixed by uninstalling the conflicting packages first and letting `pip` resolve a mutually compatible set on its own, rather than forcing a specific version.
2. **Per-frame detection vs. per-track identity**: an early version re-ran face recognition on every frame, which was both slow and produced duplicate attendance entries for the same person. Fixed by caching the resolved name per `track_id` (via ByteTrack) and only running recognition once per new track.
3. **False-positive face detections**: small patterns (e.g. a logo) were occasionally detected as faces. Fixed with `MIN_FACE_AREA_RATIO` — a minimum face-area-to-crop-area threshold, tuned empirically.
4. **Unstable tracks on first appearance**: a newly created track sometimes produces an implausibly large bounding box before the tracker separates nearby people. Fixed with `MAX_BOX_AREA_RATIO`, skipping recognition on a track until its box size is plausible.
5. **Idempotent business logic**: without explicit dedup, a person visible across hundreds of frames would generate hundreds of attendance rows / alerts. Fixed with `logged_names` and `alerted_track_ids` sets that gate each database write to fire exactly once.
6. **Monolithic notebook → modular package**: all pipeline logic originally lived inline inside the `%%writefile app.py` cell. Refactored into `src/sentryvision/` (config/database/faces/pipeline modules) so the pipeline is reusable outside Streamlit and each concern is independently readable.

---

## 🔮 Production Considerations (not implemented here, noted for context)

- Replace SQLite with Postgres/MySQL for concurrent multi-camera writes
- Move face-encoding storage from a local pickle to a proper vector index (e.g. FAISS) as the known-faces list grows
- Add authentication to the dashboard before exposing it beyond a local demo
- Containerize (`Dockerfile`) and deploy on a persistent host instead of a Colab + ngrok tunnel
- Add a lightweight test suite around `database.py` and `faces.py` (pure-logic modules, no video I/O needed to test)

---

## 🙋 About

Built as part of a structured AI/ML engineering transition (from a MERN/Java backend background) — the Computer Vision project in a 5-week fast-track covering RAG pipelines, agentic AI, and computer vision, extended here into a full applied system (detection + tracking + recognition + business logic + dashboard).
