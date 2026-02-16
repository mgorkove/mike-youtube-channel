#!/bin/bash
# VM startup script — runs automatically when the GCE instance starts.
# Pulls latest code, loads secrets, runs the pipeline, sends notification,
# then shuts down the VM.
set -euo pipefail

LOG_FILE="/var/log/video-pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Video Pipeline Startup: $(date) ==="

PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id)
ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)

APP_DIR="/app"

# ---------- Install Docker if needed ----------
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq docker.io git
    systemctl start docker
fi

# ---------- Clone / pull latest code ----------
if [ -d "$APP_DIR/.git" ]; then
    echo "Pulling latest code..."
    cd "$APP_DIR" && git pull --ff-only
else
    echo "Cloning repository..."
    git clone https://github.com/mgorkove/mike-youtube-channel.git "$APP_DIR"
fi
cd "$APP_DIR"

# ---------- Load secrets from GCP Secret Manager ----------
echo "Loading secrets..."
fetch_secret() {
    gcloud secrets versions access latest --secret="$1" --project="$PROJECT_ID" 2>/dev/null
}

cat > "$APP_DIR/.env" <<EOF
GEMINI_API_KEY=$(fetch_secret "GEMINI_API_KEY")
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN")
NOTIFY_EMAIL_FROM=$(fetch_secret "NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO=$(fetch_secret "NOTIFY_EMAIL_TO")
NOTIFY_EMAIL_APP_PASSWORD=$(fetch_secret "NOTIFY_EMAIL_APP_PASSWORD")
EOF
chmod 600 "$APP_DIR/.env"

# ---------- Build Docker image ----------
echo "Building Docker image..."
docker build -t video-pipeline "$APP_DIR"

# ---------- Run pipeline ----------
echo "Starting pipeline..."
PIPELINE_EXIT=0
docker run --rm \
    --env-file "$APP_DIR/.env" \
    -v "$APP_DIR/output:/app/output" \
    video-pipeline \
    --config deploy/config.cloud.yaml || PIPELINE_EXIT=$?

echo "Pipeline exited with code: $PIPELINE_EXIT"

# ---------- Send notification ----------
echo "Sending notification..."
docker run --rm \
    --env-file "$APP_DIR/.env" \
    -v "$APP_DIR/output:/app/output" \
    --entrypoint python \
    video-pipeline \
    notify.py output/results.json || echo "Notification failed (non-fatal)"

# ---------- Shut down VM ----------
echo "=== Pipeline complete. Shutting down VM... ==="
# Clean up sensitive files
rm -f "$APP_DIR/.env"

# Stop the VM (instance schedule will handle full cleanup)
gcloud compute instances stop "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --quiet &

# Give the stop command time to register before the VM shuts down
sleep 10
shutdown -h now
