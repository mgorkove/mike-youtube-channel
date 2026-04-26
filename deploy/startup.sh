#!/bin/bash
# VM startup script — runs automatically when the GCE instance starts.
# Pulls latest code, loads secrets, runs the pipeline, sends notification,
# then shuts down the VM.
#
# The CHANNEL to run is read from instance metadata (key: "channel").
# Defaults to "mike_explains_money" if not set.
# Each channel can have its own YouTube credentials stored as separate secrets.
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

# Read channel from instance metadata (set via Cloud Scheduler or manually)
CHANNEL=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/channel 2>/dev/null \
    || echo "mike_explains_money")

echo "Channel: $CHANNEL"

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

# Base secrets (shared across channels)
cat > "$APP_DIR/.env" <<EOF
GEMINI_API_KEY=$(fetch_secret "GEMINI_API_KEY")
NOTIFY_EMAIL_FROM=$(fetch_secret "NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO=$(fetch_secret "NOTIFY_EMAIL_TO")
NOTIFY_EMAIL_APP_PASSWORD=$(fetch_secret "NOTIFY_EMAIL_APP_PASSWORD")
EOF

# Per-channel YouTube credentials
# Convention: YOUTUBE_CLIENT_ID for default, YOUTUBE_CLIENT_ID_HEARTBREAK for heartbreak_chronicles, etc.
if [ "$CHANNEL" = "mike_explains_money" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN")
EOF
elif [ "$CHANNEL" = "heartbreak_chronicles" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID_HEARTBREAK")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET_HEARTBREAK")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN_HEARTBREAK")
PEXELS_API_KEY=$(fetch_secret "PEXELS_API_KEY")
EOF
elif [ "$CHANNEL" = "rank_recon" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID_RANKS")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET_RANKS")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN_RANKS")
EOF
elif [ "$CHANNEL" = "revenge_stories" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID_REVENGE")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET_REVENGE")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN_REVENGE")
EOF
elif [ "$CHANNEL" = "senior_savvy" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID_SENIOR")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET_SENIOR")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN_SENIOR")
PEXELS_API_KEY=$(fetch_secret "PEXELS_API_KEY")
EOF
elif [ "$CHANNEL" = "revenge_chronicles" ]; then
    cat >> "$APP_DIR/.env" <<EOF
YOUTUBE_CLIENT_ID=$(fetch_secret "YOUTUBE_CLIENT_ID_CHRONICLES")
YOUTUBE_CLIENT_SECRET=$(fetch_secret "YOUTUBE_CLIENT_SECRET_CHRONICLES")
YOUTUBE_REFRESH_TOKEN=$(fetch_secret "YOUTUBE_REFRESH_TOKEN_CHRONICLES")
EOF
fi

chmod 600 "$APP_DIR/.env"

# ---------- Build Docker image ----------
echo "Building Docker image..."
docker build -t video-pipeline "$APP_DIR"

# ---------- Run pipeline ----------
# Channels with schedule_same_day (e.g. revenge_stories) always run the full
# pipeline since all videos are generated and uploaded in a single run.
# Other channels: Sunday = full pipeline, Mon-Thu = upload pending.
DAY_OF_WEEK=$(date +%u)  # 1=Monday ... 7=Sunday

PIPELINE_EXIT=0
if [ "$CHANNEL" = "revenge_stories" ] || [ "$CHANNEL" = "heartbreak_chronicles" ] || [ "$CHANNEL" = "senior_savvy" ] || [ "$CHANNEL" = "revenge_chronicles" ] || [ "$DAY_OF_WEEK" = "7" ]; then
    echo "Full pipeline run (generate + upload) for $CHANNEL..."
    docker run --rm \
        --env-file "$APP_DIR/.env" \
        -v "$APP_DIR/output:/app/output" \
        video-pipeline \
        --channel "$CHANNEL" \
        --config "channels/${CHANNEL}/config.cloud.yaml" || PIPELINE_EXIT=$?
else
    echo "Weekday: Uploading pending videos for $CHANNEL..."
    docker run --rm \
        --env-file "$APP_DIR/.env" \
        -v "$APP_DIR/output:/app/output" \
        video-pipeline \
        --channel "$CHANNEL" \
        --config "channels/${CHANNEL}/config.cloud.yaml" \
        --upload-pending || PIPELINE_EXIT=$?
fi

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
