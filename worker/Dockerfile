FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The default PyPI torch wheel for Linux x86_64 ships with CUDA support.
RUN pip install --no-cache-dir -r requirements.txt

COPY tracker ./tracker

# Pose weights are downloaded on first run; keep them in the mounted /models
ENV YOLO_CONFIG_DIR=/models/.ultralytics
WORKDIR /models
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "tracker.main"]
CMD ["watch"]
