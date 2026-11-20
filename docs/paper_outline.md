# Paper outline (BrainLes workshop @ MICCAI / arXiv)

Working title: *How much of nnU-Net can you get on a laptop? A controlled, multi-seed comparison of
CNN and transformer segmentation for BraTS under a fixed compute budget.*

Authors: D. Frasheri, S. [surname] (to confirm), [supervisor]

Every number in the paper must come from `results/*.json` produced by the scripts named below.
Placeholders `[R:...]` mark where a result goes. None may be filled by hand.

## 1. Introduction (0.75 page)
- BraTS results are reported at very different compute budgets, which makes method comparisons hard to read.
- Contribution 1: a matched-budget comparison of five architectures (U-Net, nnU-Net-style U-Net, V-Net,
  SegResNet, Swin-UNETR), with the official nnU-Net v2 as an upper reference.
- Contribution 2: 5-fold CV × 3 seeds on all 1,251 patients, with paired tests, separating seed variance from
  patient variance.
- Contribution 3: robustness to missing sequences and to BraTS 2023 data, and whether TTA-based uncertainty
  flags failures.

## 2. Methods (2 pages)
- Data and splits: `src/data.py::kfold_split`; nnU-Net forced onto the same folds (`nnunet/convert.py splits`).
- Models and budget: `src/config.py::CONFIGS`, iterations [R:ITERS], hardware [R:GPU].
- Post-processing: mirror TTA, soft-vote ensemble, ET suppression with the threshold tuned on validation only.
- Metrics: Dice and HD95 per region (`src/metrics.py`), official lesion-wise scores from the validation server.
- Statistics: paired Wilcoxon over patients and mean ± std over seeds.

## 3. Results (2 pages)
- Table 1: CV Dice/HD95 per model, ± seed std. Source: `results/cv_summary.json` [R]
- Table 2: official validation-server scores for the best model and the ensemble [R: Synapse leaderboard]
- Table 3: TTA / ensemble / ET suppression ablation. Source: `results/postproc_summary.json` [R]
- Figure 1: Dice distributions per model (violin), CNN vs transformer.
- Figure 2: missing-sequence Dice drop per sequence. Source: `results/robustness_*_summary.json` [R]
- Figure 3: uncertainty maps, and uncertainty vs Dice (Spearman ρ) [R]

## 4. Discussion (0.75 page)
- Where the matched-budget gap to nnU-Net v2 comes from (schedule length vs architecture).
- Whether the transformer advantage reported in the literature survives a matched budget [R].
- Limitations: one dataset family, no clinical reader study.

## Checklist before submission
- [ ] All `[R:...]` filled from result files; the script and commit hash recorded for each
- [ ] Co-author approval of the final text
- [ ] Code and model card public; no patient data in the repo
- [ ] BraTS data-use citation requirements (Baid et al. 2021, Menze et al. 2015, Bakas et al. 2017)
