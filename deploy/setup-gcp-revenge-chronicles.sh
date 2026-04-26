#!/bin/bash
# One-time GCP setup for revenge_chronicles on its own VM.
#
# Mirrors setup-gcp-heartbreak.sh — runs on phonic-arcana-445603-q5.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Billing enabled on the project
#   - Compute Engine, Secret Manager, Cloud Scheduler APIs enabled
#   - YouTube creds already pushed to Secret Manager:
#       YOUTUBE_CLIENT_ID_CHRONICLES
#       YOUTUBE_CLIENT_SECRET_CHRONICLES
#       YOUTUBE_REFRESH_TOKEN_CHRONICLES
#
# Usage:
#   bash deploy/setup-gcp-revenge-chronicles.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-phonic-arcana-445603-q5}"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="video-pipeline-revenge-chronicles"
MACHINE_TYPE="n2-standard-8"
DISK_SIZE="100"
SERVICE_ACCOUNT_NAME="video-pipeline-sa"

echo "=== Setting up Revenge Chronicles Pipeline on GCP ==="
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

# ---------- 2. Service account ----------
echo "2. Ensuring service account..."
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="Video Pipeline VM" \
    --project="$PROJECT" 2>/dev/null || echo "  (already exists)"

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"

for ROLE in roles/secretmanager.secretAccessor roles/compute.instanceAdmin.v1; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" --quiet >/dev/null
done
echo "  Service account: $SA_EMAIL"

# ---------- 3. Create VM ----------
echo "3. Creating GCE VM..."
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "  VM already exists — updating channel metadata..."
    gcloud compute instances add-metadata "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --metadata="channel=revenge_chronicles" \
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
        --metadata="channel=revenge_chronicles" \
        --tags="video-pipeline" \
        --quiet

    echo "  VM created: $VM_NAME"
fi

# ---------- 4. Instance schedule ----------
echo "4. Creating instance schedule (daily 8am ET start, midnight ET safety-stop)..."
SCHEDULE_NAME="pipeline-schedule-revenge-chronicles"

gcloud compute resource-policies create instance-schedule "$SCHEDULE_NAME" \
    --project="$PROJECT" \
    --region="$REGION" \
    --vm-start-schedule="0 8 * * *" \
    --vm-stop-schedule="0 0 * * *" \
    --timezone="America/New_York" \
    --description="Daily 8am ET start, midnight ET safety-stop (revenge_chronicles)" \
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
echo "Channel: revenge_chronicles (4 videos this run)"
echo ""
echo "To start the run NOW:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "To watch logs:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- tail -f /var/log/video-pipeline.log"
