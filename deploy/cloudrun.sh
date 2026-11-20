#!/usr/bin/env bash
# Build the inference image with Cloud Build and deploy it to Cloud Run.
#
#   PROJECT=my-gcp-project REGION=europe-west4 ./deploy/cloudrun.sh
#   GPU=1 PROJECT=... ./deploy/cloudrun.sh      # L4 GPU (regions with Cloud Run GPU only)
#
# Deploys with authentication REQUIRED. Uploading patient MRI to a public
# endpoint is never acceptable, even for de-identified research data; call it
# with an identity token:
#   curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ...
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PROJECT:?set PROJECT to your GCP project id}"
REGION="${REGION:-europe-west4}"
SERVICE="${SERVICE:-brats-seg}"
REPO="${REPO:-ml}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:$(git rev-parse --short HEAD)"

test -f checkpoints/nnunet_style_best.pt || { echo "checkpoints/nnunet_style_best.pt missing"; exit 1; }

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format docker \
       --location "$REGION" --project "$PROJECT"

TORCH_INDEX=https://download.pytorch.org/whl/cpu
GPU_FLAGS=()
if [[ "${GPU:-0}" == "1" ]]; then
  TORCH_INDEX=https://download.pytorch.org/whl/cu121
  GPU_FLAGS=(--gpu 1 --gpu-type nvidia-l4 --no-cpu-throttling)
fi

gcloud builds submit --project "$PROJECT" --config deploy/cloudbuild.yaml \
  --substitutions "_IMAGE=$IMAGE,_TORCH_INDEX=$TORCH_INDEX" .

gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --no-allow-unauthenticated --memory 16Gi --cpu 4 --timeout 600 --concurrency 1 \
  --max-instances 2 --min-instances 0 "${GPU_FLAGS[@]}"

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
