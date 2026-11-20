# Model card: BraTS 2021 glioma segmentation (`nnunet_style`)

## Model details

| | |
|---|---|
| Task | Voxel-wise segmentation of adult glioma sub-regions in pre-operative multi-parametric MRI |
| Architecture | 3D U-Net, 4 levels, InstanceNorm, deep supervision; 5.6M parameters |
| Training recipe | Per-modality z-score normalisation, Dice + cross-entropy, flips + intensity jitter, 128³ patches, AdamW, cosine LR |
| Input | Four co-registered, skull-stripped, 1 mm isotropic volumes: FLAIR, T1, T1ce, T2 (BraTS preprocessing) |
| Output | Label map in the input space: 1 necrotic core, 2 oedema, 4 enhancing tumour (BraTS 2021 convention) |
| Served by | `serve/app.py` (FastAPI), `Dockerfile`, `deploy/cloudrun.sh` |
| Licence | See `LICENSE` |
| Status | Research prototype |

## Intended use

- Research and teaching: benchmarking segmentation methods on BraTS-format data.
- A starting point for annotation: pre-segmenting cases that a trained reader then corrects (`monailabel_app/`).

**Out of scope.** Diagnosis, treatment planning, surgical guidance, response assessment or any other clinical
decision. The model has no regulatory clearance (CE/MDR, FDA or Swissmedic) and has not been validated
prospectively or by clinicians.

## Training data

BraTS 2021 Task 1 training set (RSNA-ASNR-MICCAI). The reported model used a 300-patient subset
(210 train / 45 validation / 45 test, split by patient with seed 42). The data is multi-institutional,
pre-operative adult glioma only, already skull-stripped, co-registered and resampled to 1 mm³.

## Evaluation

Held-out test set: 45 patients. Mean ± standard deviation **across patients**, single training seed.

| Region | Dice | HD95 (voxels) |
|---|---|---|
| Whole tumour | 0.893 ± 0.105 | 10.7 |
| Tumour core | 0.821 ± 0.251 | 13.5 |
| Enhancing tumour | 0.793 ± 0.211 | 10.3 |

These numbers are from about 8 minutes of training. They are below published BraTS results and are
**not** official challenge scores. The full-scale protocol (all 1,251 patients, 5-fold CV × 3 seeds) and
the validation-server submission are implemented (`scripts/run_cv_study.sh`, `scripts/predict_folder.py`)
but have not been run yet. This card is updated when they are.

## Known limitations and failure modes

- **Contralateral false positives.** Non-tumour FLAIR hyperintensity (small-vessel disease, gliosis) can be
  segmented as a second lesion. Seen in the worst test case (WT Dice 0.45).
- **Enhancing tumour is bimodal.** Per patient, the model either finds the enhancing region or largely
  misses it. Optional ET suppression (`src/postprocess.py`) targets the false-positive side.
- **Domain shift is unmeasured.** Only BraTS 2021 data has been used. Other scanners, protocols, paediatric
  tumours, metastases and post-operative scans are out of distribution. `scripts/evaluate_robustness.py`
  evaluates BraTS 2023+ data and missing sequences once that data is available.
- **Missing sequences.** The model expects all four sequences. A blank channel degrades it in ways
  measured by the robustness script, not designed for.
- **Requires BraTS preprocessing.** Raw DICOM, non-skull-stripped or non-registered scans give meaningless
  output, without warning.

## Ethical and privacy considerations

- The API stores nothing: uploads are written to a temporary directory and deleted after each request.
- `deploy/cloudrun.sh` deploys with authentication required. Do not expose the endpoint publicly or send
  identifiable patient data to it without a data-processing agreement and an ethics/legal review
  (GDPR / Swiss FADP for health data).
- Every response carries a research-use-only disclaimer.

## Maintenance

Each training run logs hyperparameters, curves, GPU and wall-clock time to `logs/*_history.json`, and to
MLflow with `--mlflow`. Checkpoints fine-tuned on corrected labels are versioned
(`*_corrected_vN.pt`) and must be re-evaluated on the untouched test split before they are served.
