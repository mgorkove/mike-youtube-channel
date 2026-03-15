#!/bin/bash
# One-time GCP setup for heartbreak_chronicles on a SEPARATE project/VM.
#
# This runs heartbreak_chronicles on the "gemini_project" GCP project with its
# own VM, so it doesn't share Gemini API or YouTube upload quotas with the
# revenge_stories pipeline.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project "gemini_project" with billing enabled
#   - YouTube Data API v3 enabled
#   - Secret Manager API enabled
#   - Compute Engine API enabled
#   - Cloud Scheduler API enabled
#
# Usage:
#   bash deploy/setup-gcp-heartbreak.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-gen-lang-client-0529399535}"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="video-pipeline-heartbreak"
MACHINE_TYPE="n2-standard-8"
DISK_SIZE="100"
SERVICE_ACCOUNT_NAME="video-pipeline-sa"

echo "=== Setting up Heartbreak Chronicles Pipeline on GCP ==="
echo "Project: $PROJECT"
echo "Region:  $REGION"
echo "VM:      $VM_NAME ($MACHINE_TYPE)"
echo ""

# ---------- 1. Enable APIs ----------
echo "1. Enabling required APIs..."
gcloud services enable \
    compute.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    --project="$PROJECT" --quiet

# ---------- 2. Create service account ----------
echo "2. Creating service account..."
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="Video Pipeline VM" \
    --project="$PROJECT" 2>/dev/null || echo "  (already exists)"

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"

# Grant required roles
for ROLE in roles/secretmanager.secretAccessor roles/compute.instanceAdmin.v1; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" --quiet >/dev/null
done
echo "  Service account: $SA_EMAIL"

# ---------- 3. Create secrets ----------
echo "3. Setting up Secret Manager secrets..."
echo "   You will be prompted to enter each secret value."
echo "   (Press Enter to skip if already created)"
echo ""

create_secret() {
    local name="$1"
    local description="$2"

    if gcloud secrets describe "$name" --project="$PROJECT" &>/dev/null; then
        echo "  $name: already exists (skipping)"
        return
    fi

    echo "  Enter value for $name ($description):"
    read -r -s value
    if [ -n "$value" ]; then
        echo -n "$value" | gcloud secrets create "$name" \
            --data-file=- \
            --project="$PROJECT" --quiet
        echo "  $name: created"
    else
        echo "  $name: skipped"
    fi
}

# Base secrets
create_secret "GEMINI_API_KEY" "Google Gemini API key (gemini_project)"
create_secret "NOTIFY_EMAIL_FROM" "Sender Gmail address"
create_secret "NOTIFY_EMAIL_TO" "Recipient email address"
create_secret "NOTIFY_EMAIL_APP_PASSWORD" "Gmail App Password"

# heartbreak_chronicles YouTube credentials
create_secret "YOUTUBE_CLIENT_ID_HEARTBREAK" "YouTube OAuth Client ID (heartbreak_chronicles)"
create_secret "YOUTUBE_CLIENT_SECRET_HEARTBREAK" "YouTube OAuth Client Secret (heartbreak_chronicles)"
create_secret "YOUTUBE_REFRESH_TOKEN_HEARTBREAK" "YouTube OAuth Refresh Token (heartbreak_chronicles)"

echo ""
echo "   Tip: Run 'python extract_youtube_creds.py' locally to get"
echo "   YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN."
echo ""

# ---------- 4. Create VM ----------
echo "4. Creating GCE VM..."
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "  VM already exists — updating channel metadata to heartbreak_chronicles..."
    gcloud compute instances add-metadata "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --metadata="channel=heartbreak_chronicles" \
        --quiet
else
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --boot-disk-size="${DISK_SIZE}GB" \
        --boot-disk-type="pd-ssd" \
        --image-family="ubuntu-2404-lts-amd64" \
        --image-project="ubuntu-os-cloud" \
        --service-account="$SA_EMAIL" \
        --scopes="cloud-platform" \
        --metadata-from-file="startup-script=deploy/startup.sh" \
        --metadata="channel=heartbreak_chronicles" \
        --tags="video-pipeline" \
        --quiet

    echo "  VM created: $VM_NAME"

    # Stop it immediately (it will be started by the scheduler)
    echo "  Stopping VM (will be started by scheduler)..."
    gcloud compute instances stop "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet
fi

# ---------- 5. Create Instance Schedule ----------
echo "5. Creating instance schedule..."

SCHEDULE_NAME="pipeline-schedule-heartbreak"

# Remove old schedule if attached
gcloud compute instances remove-resource-policies "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --resource-policies="$SCHEDULE_NAME" 2>/dev/null || true
gcloud compute resource-policies delete "$SCHEDULE_NAME" \
    --region="$REGION" --project="$PROJECT" --quiet 2>/dev/null || true

# Start: every day at 5:00 AM ET (10:00 UTC)
# Stop:  every day at 11:00 PM ET (4:00 UTC) — safety net
gcloud compute resource-policies create instance-schedule "$SCHEDULE_NAME" \
    --project="$PROJECT" \
    --region="$REGION" \
    --vm-start-schedule="0 10 * * *" \
    --vm-stop-schedule="0 4 * * *" \
    --timezone="UTC" \
    --description="Daily: start at 5am ET, safety-stop at 11pm ET (heartbreak_chronicles)" \
    2>/dev/null || echo "  $SCHEDULE_NAME schedule already exists"

gcloud compute instances add-resource-policies "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --resource-policies="$SCHEDULE_NAME" \
    2>/dev/null || echo "  $SCHEDULE_NAME already attached"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "VM: $VM_NAME on project: $PROJECT"
echo "Channel: heartbreak_chronicles (6 videos/day, no shorts)"
echo ""
echo "The VM will automatically:"
echo "  - Start daily at 5am ET"
echo "  - Run the full pipeline for heartbreak_chronicles"
echo "  - Shut itself down after each run"
echo "  - Safety-net stop at 11pm ET if still running"
echo "  - Send email notification after each run"
echo ""
echo "To test manually:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "To watch logs:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- tail -f /var/log/video-pipeline.log"
echo ""
echo "Estimated cost: ~\$1-3/day (on-demand, auto-shutdown) + ~\$10/month disk"
