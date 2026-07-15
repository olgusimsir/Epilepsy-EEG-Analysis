# NeuroScan — EEG seizure-analysis web tool, containerized.
# Build:  docker build -t neuroscan .
# Run:    docker run --rm -p 8000:8000 neuroscan   → open http://localhost:8000

# 1) Base image: a slim Debian with Python 3.12 (matches your venv's 3.12.7).
FROM python:3.12-slim

# 2) System library torch/mne need at runtime (OpenMP). Cleaned up after install.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3) Everything below lives in /app inside the container.
WORKDIR /app

# 4) Install Python deps FIRST and on their own layer, so Docker caches them and
#    only re-installs when requirements.txt actually changes (fast rebuilds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy the app: source, frontend, trained models, and the server.
#    (data/ and venv/ are excluded by .dockerignore.)
COPY src/ ./src/
COPY design/ ./design/
COPY models/ ./models/
COPY serve_ui.py .

# 6) The server listens on 0.0.0.0:8000 (see serve_ui.py). Document + expose it.
ENV PORT=8000
EXPOSE 8000

# 7) What runs when the container starts.
CMD ["python", "serve_ui.py"]
