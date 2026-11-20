#!/usr/bin/env bash
# Create a spot GPU VM on GCP that runs a slice of the full CV study.
#
#   PROJECT=my-proj CONFIGS="nnunet_style" FOLDS="0 1" ./deploy/gcp_train_vm.sh
#
# Spot VMs cost ~60-70% less but can be pre-empted; run_cv_study.sh skips
# finished runs, so re-creating the VM resumes where it stopped. Results are
# synced to gs://$BUCKET after every run so nothing is lost on pre-emption.
#
# Requires: a GPU quota in $ZONE, a bucket, and a Kaggle token stored in
# Secret Manager as `kaggle-token` (the dataset is downloaded on the VM).
set -euo pipefail

: "${PROJECT:?}"; : "${BUCKET:?set BUCKET=gs://your-bucket}"
ZONE="${ZONE:-europe-west4-a}"
NAME="${NAME:-brats-train-$(date +%s)}"
GPU="${GPU:-nvidia-l4}"          # nvidia-l4 (g2-standard-8) or nvidia-tesla-a100 (a2-highgpu-1g)
MACHINE="${MACHINE:-g2-standard-8}"
REPO_URL="${REPO_URL:-$(git config --get remote.origin.url)}"
CONFIGS="${CONFIGS:-nnunet_style}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEEDS="${SEEDS:-42 43 44}"
ITERS="${ITERS:-20000}"

cat > /tmp/brats_startup.sh <<EOF
#!/bin/bash
set -e
mkdir -p ~/.kaggle && gcloud secrets versions access latest --secret kaggle-token > ~/.kaggle/access_token
cd /opt && (test -d brats || git clone $REPO_URL brats) && cd brats
pip install -r requirements.txt monai mlflow
gsutil -m rsync -r $BUCKET/checkpoints checkpoints || true
test -d ~/datasets/brats2021 || jupyter nbconvert --to notebook --execute brats2021_download_and_check.ipynb
python scripts/build_cache.py --n 1251
for c in $CONFIGS; do for f in $FOLDS; do for s in $SEEDS; do
  CONFIGS=\$c FOLDS=\$f SEEDS=\$s ITERS=$ITERS ./scripts/run_cv_study.sh
  gsutil -m rsync -r checkpoints $BUCKET/checkpoints
  gsutil -m rsync -r logs $BUCKET/logs
  gsutil -m rsync -r mlruns $BUCKET/mlruns || true
done; done; done
sudo shutdown -h now
EOF

gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" --accelerator "type=$GPU,count=1" \
  --provisioning-model SPOT --instance-termination-action STOP \
  --image-family pytorch-2-4-cu124-ubuntu-2204 --image-project deeplearning-platform-release \
  --boot-disk-size 200GB --maintenance-policy TERMINATE \
  --metadata install-nvidia-driver=True --metadata-from-file startup-script=/tmp/brats_startup.sh \
  --scopes cloud-platform

echo "created $NAME; follow with: gcloud compute ssh $NAME --zone $ZONE -- sudo journalctl -fu google-startup-scripts"
