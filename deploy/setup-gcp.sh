#!/bin/bash
# One-time GCP setup for the video pipeline.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project with billing enabled
#   - YouTube Data API v3 enabled
#   - Secret Manager API enabled
#   - Compute Engine API enabled
#   - Cloud Scheduler API enabled
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   bash deploy/setup-gcp.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="video-pipeline"
MACHINE_TYPE="n2-standard-8"
DISK_SIZE="100"
SERVICE_ACCOUNT_NAME="video-pipeline-sa"

echo "=== Setting up Video Pipeline on GCP ==="
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

create_secret "GEMINI_API_KEY" "Google Gemini API key"
create_secret "NOTIFY_EMAIL_FROM" "Sender Gmail address"
create_secret "NOTIFY_EMAIL_TO" "Recipient email address"
create_secret "NOTIFY_EMAIL_APP_PASSWORD" "Gmail App Password"

# mike_explains_money YouTube credentials (default)
create_secret "YOUTUBE_CLIENT_ID" "YouTube OAuth Client ID (mike_explains_money)"
create_secret "YOUTUBE_CLIENT_SECRET" "YouTube OAuth Client Secret (mike_explains_money)"
create_secret "YOUTUBE_REFRESH_TOKEN" "YouTube OAuth Refresh Token (mike_explains_money)"

# heartbreak_chronicles YouTube credentials + Pexels
create_secret "YOUTUBE_CLIENT_ID_HEARTBREAK" "YouTube OAuth Client ID (heartbreak_chronicles)"
create_secret "YOUTUBE_CLIENT_SECRET_HEARTBREAK" "YouTube OAuth Client Secret (heartbreak_chronicles)"
create_secret "YOUTUBE_REFRESH_TOKEN_HEARTBREAK" "YouTube OAuth Refresh Token (heartbreak_chronicles)"
create_secret "PEXELS_API_KEY" "Pexels API key (stock footage)"

# rank_recon YouTube credentials
create_secret "YOUTUBE_CLIENT_ID_RANKS" "YouTube OAuth Client ID (rank_recon)"
create_secret "YOUTUBE_CLIENT_SECRET_RANKS" "YouTube OAuth Client Secret (rank_recon)"
create_secret "YOUTUBE_REFRESH_TOKEN_RANKS" "YouTube OAuth Refresh Token (rank_recon)"

# revenge_stories YouTube credentials
create_secret "YOUTUBE_CLIENT_ID_REVENGE" "YouTube OAuth Client ID (revenge_stories)"
create_secret "YOUTUBE_CLIENT_SECRET_REVENGE" "YouTube OAuth Client Secret (revenge_stories)"
create_secret "YOUTUBE_REFRESH_TOKEN_REVENGE" "YouTube OAuth Refresh Token (revenge_stories)"

echo ""
echo "   Tip: Run 'python extract_youtube_creds.py' locally to get"
echo "   YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN."
echo ""

# ---------- 4. Create VM ----------
echo "4. Creating GCE VM..."
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    # Update channel metadata on existing VM
    echo "  VM already exists — updating channel metadata to revenge_stories..."
    gcloud compute instances add-metadata "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --metadata="channel=revenge_stories" \
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
        --metadata="channel=revenge_stories" \
        --no-address \
        --tags="video-pipeline" \
        --quiet

    echo "  VM created: $VM_NAME"

    # Stop it immediately (it will be started by the scheduler)
    echo "  Stopping VM (will be started by scheduler)..."
    gcloud compute instances stop "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet
fi

# ---------- 5. Create Instance Schedule ----------
echo "5. Creating instance schedules..."

# Remove old schedules if attached (safe to fail if they don't exist)
for POLICY in pipeline-start pipeline-stop; do
    gcloud compute instances remove-resource-policies "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --resource-policies="$POLICY" 2>/dev/null || true
    gcloud compute resource-policies delete "$POLICY" \
        --region="$REGION" --project="$PROJECT" --quiet 2>/dev/null || true
done

# Start schedule: every day at 5:00 AM ET (10:00 UTC)
# The startup script determines what to run based on channel + day of week.
# The VM shuts itself down after each run (see startup.sh).
gcloud compute resource-policies create instance-schedule "pipeline-start" \
    --project="$PROJECT" \
    --region="$REGION" \
    --vm-start-schedule="0 10 * * *" \
    --timezone="UTC" \
    --description="Start video pipeline VM daily at 5am ET" \
    2>/dev/null || echo "  pipeline-start schedule already exists"

# Safety-net stop: every day at 11:00 PM ET (4:00 UTC next day)
# Kills the VM if it's still running after 18 hours (should never happen).
gcloud compute resource-policies create instance-schedule "pipeline-stop" \
    --project="$PROJECT" \
    --region="$REGION" \
    --vm-stop-schedule="0 4 * * *" \
    --timezone="UTC" \
    --description="Safety net: stop VM daily at 11pm ET if still running" \
    2>/dev/null || echo "  pipeline-stop schedule already exists"

# Attach schedules to VM
for POLICY in pipeline-start pipeline-stop; do
    gcloud compute instances add-resource-policies "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --resource-policies="$POLICY" \
        2>/dev/null || echo "  $POLICY already attached"
done

echo ""
echo "=== Setup complete! ==="
echo ""
echo "The VM will automatically:"
echo "  - Start daily at 5am ET"
echo "  - Run the full pipeline for the configured channel"
echo "  - Shut itself down after each run (~30-90 min)"
echo "  - Safety-net stop at 11pm ET if still running"
echo "  - Send email notification after each run"
echo ""
echo "To switch channels, update the VM metadata:"
echo "  gcloud compute instances add-metadata $VM_NAME --zone=$ZONE --project=$PROJECT --metadata=channel=<CHANNEL>"
echo ""
echo "To test manually:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "To watch logs:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- tail -f /var/log/video-pipeline.log"
echo ""
echo "Estimated cost: ~\$1-3/day (on-demand, auto-shutdown) + ~\$10/month disk"
