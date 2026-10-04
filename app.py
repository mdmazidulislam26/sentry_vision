"""
SentryVision Dashboard — Streamlit entry point.

This file is intentionally thin: it owns UI state and layout only. All
detection/tracking/recognition/business logic lives in the `sentryvision`
package under `src/`, so the same pipeline can be reused outside Streamlit
(a CLI script, a batch job, a future API) without change.
"""

import os
import sys

import streamlit as st

# Make the `src/` package importable without a separate install step —
# convenient for running directly out of a cloned repo or a Colab session.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sentryvision import config, database, faces, pipeline  # noqa: E402

st.set_page_config(page_title="SentryVision Dashboard", layout="wide")
config.ensure_directories()


@st.cache_resource
def get_model():
    """Load the YOLO model once per Streamlit session."""
    return pipeline.load_model()


st.title("🛡️ SentryVision — Attendance & Security Dashboard")


def load_dashboard_data():
    attendance_rows = database.fetch_attendance()
    alert_rows = database.fetch_alerts()
    return attendance_rows, alert_rows


attendance_rows, alert_rows = load_dashboard_data()

col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Known Persons Logged", len(attendance_rows))
col2.metric("Security Alerts", len(alert_rows))
if col3.button("🔄 Refresh Data"):
    st.rerun()

st.divider()

tab_faces, tab_process, tab_attendance, tab_alerts, tab_frames = st.tabs(
    ["👤 Known Faces", "📤 Process Video", "📋 Attendance", "🚨 Alerts", "🎥 Processed Frames"]
)

# --- Tab: Known Faces (upload once, persisted via pickle) ---
with tab_faces:
    st.subheader("Add a Known Person")
    with st.form("add_face_form", clear_on_submit=True):
        person_name = st.text_input("Name")
        face_image = st.file_uploader("Reference photo", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Save")

        if submitted:
            if not person_name.strip():
                st.error("Please enter a name.")
            elif face_image is None:
                st.error("Please upload a photo.")
            else:
                success, message = faces.add_known_face(face_image, person_name.strip())
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.divider()
    st.subheader("Currently Known People")
    known_dict = faces.load_known_encodings()
    if not known_dict:
        st.info("No known faces saved yet. Add one above.")
    else:
        cols = st.columns(4)
        for i, name in enumerate(known_dict.keys()):
            image_path = os.path.join(config.KNOWN_FACES_DIR, f"{name}.jpg")
            with cols[i % 4]:
                if os.path.exists(image_path):
                    st.image(image_path, caption=name, width=120)

# --- Tab: Process Video (upload + trigger the pipeline) ---
with tab_process:
    st.subheader("Upload and Process a Video")

    known_dict = faces.load_known_encodings()
    known_names_list = list(known_dict.keys())
    known_encodings_list = list(known_dict.values())

    if not known_names_list:
        st.warning(
            "No known faces saved yet. Add people in the 'Known Faces' tab first "
            "(optional — the video can still be processed, everyone will show as Unknown)."
        )

    uploaded_video = st.file_uploader("Video file", type=["mp4", "mov", "avi"])

    if uploaded_video is not None and st.button("▶️ Process Video"):
        with open(config.UPLOADED_VIDEO_PATH, "wb") as f:
            f.write(uploaded_video.getbuffer())

        progress_bar = st.progress(0.0, text="Starting...")

        def update_progress(fraction):
            progress_bar.progress(fraction, text=f"Processing... {int(fraction * 100)}%")

        with st.spinner("Running detection, tracking, and recognition..."):
            summary = pipeline.process_video(
                config.UPLOADED_VIDEO_PATH,
                known_names_list,
                known_encodings_list,
                model=get_model(),
                progress_callback=update_progress,
            )

        progress_bar.progress(1.0, text="Done")
        st.success(
            f"✅ Processed {summary['total_frames']} frames. "
            f"{summary['attendance_logged']} known person(s) logged, "
            f"{summary['alerts_raised']} alert(s) raised."
        )
        st.info("Check the Attendance / Alerts / Processed Frames tabs to see the results.")

# --- Tab: Attendance ---
with tab_attendance:
    if not attendance_rows:
        st.info("No attendance entries yet.")
    else:
        st.dataframe(
            [{"Name": n, "Track ID": t, "Entry Time": e} for n, t, e in attendance_rows],
            use_container_width=True,
            height=350,
        )

# --- Tab: Alerts ---
with tab_alerts:
    if not alert_rows:
        st.info("No alerts yet.")
    else:
        MAX_ALERTS_SHOWN = 5
        for track_id, detected_time, snapshot_path in alert_rows[:MAX_ALERTS_SHOWN]:
            cols = st.columns([1, 3])
            with cols[0]:
                if os.path.exists(snapshot_path):
                    st.image(snapshot_path, width=120)
                else:
                    st.warning("Snapshot not found")
            with cols[1]:
                st.write(f"**Track ID:** {track_id}")
                st.write(f"**Detected at:** {detected_time}")
        if len(alert_rows) > MAX_ALERTS_SHOWN:
            st.caption(f"Showing latest {MAX_ALERTS_SHOWN} of {len(alert_rows)} total alerts.")

# --- Tab: Processed Frames ---
with tab_frames:
    if os.path.exists(config.OUTPUT_DIR):
        frame_files = sorted(os.listdir(config.OUTPUT_DIR))
    else:
        frame_files = []

    if frame_files:
        selected = st.select_slider("Select frame", options=frame_files)
        st.markdown(
            """
            <style>
            .frame-container img {
                max-height: 420px;
                width: auto;
                display: block;
                margin: 0 auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown('<div class="frame-container">', unsafe_allow_html=True)
            st.image(os.path.join(config.OUTPUT_DIR, selected))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No processed frames yet. Process a video in the 'Process Video' tab.")
