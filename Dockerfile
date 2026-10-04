FROM python:3.11-slim

# System dependencies:
#   - build-essential, cmake: required to compile dlib (face_recognition's backend)
#   - libopenblas-dev, liblapack-dev: dlib's linear-algebra backend
#   - libgl1, libglib2.0-0: required by opencv-python-headless at import time
#   - ffmpeg: video decoding
# This build is slow (dlib compiles from source, no prebuilt wheel for slim images) —
# expect several minutes on first build. Layer caching keeps subsequent builds fast
# as long as requirements.txt doesn't change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app.py .

# yolov8n.pt (~6MB) is downloaded automatically on first inference call if not
# present; pre-downloading here avoids a cold-start delay on the first request.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ENV SENTRYVISION_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "src.sentryvision.api:app", "--host", "0.0.0.0", "--port", "8000"]
