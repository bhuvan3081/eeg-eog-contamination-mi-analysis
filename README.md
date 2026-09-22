# EEG/EOG Contamination and Motor-Imagery EEG Analysis

This repository contains the complete executed analysis record, result tables, and portfolio figures for a study of electrooculography (EOG) contamination in motor-imagery EEG. The project examines how EOG contamination relates to classification error, how it changes during motor imagery, how suppression affects EEGNet classification and sensorimotor signal structure, and why cleaning response varies across subjects.

## Project status

This is a research/pilot analysis repository. The original executed notebook is preserved as the primary research record. The code is intentionally not fully refactored: repeated helper/model definitions remain where they existed in the analysis history so individual sections can be inspected independently. A cleaned EEGNet training module is provided separately for portfolio review.

## Main findings

- The trial-level analysis contains **2,368 matched E-session trials**: 1,247 correct and 1,121 classification errors.
- EOG contamination is analysed as a measurable predictor of classification error rather than being treated only as nuisance variance.
- Across 9 subjects, EOG contamination changes between baseline and motor imagery in a band-dependent way: Theta mean change -0.0515, Alpha/Mu +0.0205, Beta +0.0413, Gamma +0.0393 in the final corrected R² analysis.
- The final fixed λ = 0.75 validation gives mean raw accuracy **0.514845** versus cleaned accuracy **0.511558**, and mean raw Macro-F1 **0.500805** versus cleaned Macro-F1 **0.504067**. The final validation does **not** establish a significant population-level accuracy improvement from fixed λ = 0.75.
- Cleaning response is heterogeneous across subjects, motivating responder and adaptive-policy analyses. These analyses are exploratory because the subject count is 9.
- Sensorimotor ERD/ERS, beta-band dynamics, motor lateralization, EEGNet feature space, class compactness, and decision margins are used as mechanistic/associational analyses.

Taken together, the study found substantial and structured EOG contamination, but fixed regression-based suppression at λ = 0.75 did not produce a statistically significant population-level accuracy gain. That null classification result is informative: it suggests that measurable EOG contamination does not necessarily translate into a corresponding accuracy loss after EEGNet decoding. One possible explanation is that the network's temporal convolutions may already tolerate some residual EOG-related variance at this suppression strength, although the present analysis does not establish that mechanism causally. The subject-level heterogeneity therefore matters as much as the population-average effect and motivates the exploratory responder and adaptive-policy analyses.

## Why this project matters

The central research question is not simply whether EOG can be removed. It is whether EOG contamination is structured enough to affect motor-imagery EEG decoding, whether suppression changes physiologically meaningful EEG structure, and whether subjects respond differently to the same suppression policy.

## Dataset

The analysis uses BCI Competition IV 2a-style EEG/EOG data with 22 EEG channels and 3 EOG channels, sampled at 250 Hz before the analysis pipeline's downsampling step. The four motor-imagery classes are Left Hand, Right Hand, Feet, and Tongue.

Raw `.mat` files are intentionally excluded from this repository unless redistribution is explicitly permitted by the dataset license.

## Analysis pipeline

### Stage 1 — Data and EOG contamination

1. Load EEG/EOG trials.
2. Filter EEG/EOG.
3. Estimate EOG-to-EEG coupling using regression/R².
4. Evaluate trial-level EOG contamination versus classification error.
5. Analyse within-trial temporal changes in contamination.
6. Compare baseline and motor-imagery contamination.

### Stage 2 — EEGNet classification and suppression

1. Train EEGNet on training data.
2. Keep validation and held-out E-session evaluation separate.
3. Fit EOG regression using training data only.
4. Apply EOG suppression using `EEG_clean = EEG_raw - λ × EOG_component`.
5. Compare raw and cleaned classification.
6. Evaluate multiple λ values.
7. Analyse subject-level responders and adaptive policies.

### Stage 3 — Signal and representation mechanisms

1. Sensorimotor ERD/ERS.
2. Temporal beta dynamics.
3. Normalized beta class contrast.
4. Spatial motor lateralization.
5. Class separability.
6. EEGNet feature-space geometry.
7. Class-specific compactness and confusion.
8. EEGNet decision margins.

### Stage 4 — Final validation

1. Subject-level final comparison.
2. Paired statistical tests.
3. Exact responder permutation testing.
4. Leave-one-subject-out robustness analysis.

## EEGNet implementation

EEGNet is the primary classifier. The training implementation used in the notebook is also exposed in `src/eegnet_training.py` so that reviewers can inspect the actual model and training loop directly.

Main configuration: F1=16, depth multiplier D=2, F2=32, dropout=0.5, temporal kernel length 64, batch size 16, Adam learning rate 1e-3, weight decay 1e-4, cosine-annealing learning-rate schedule, maximum 200 epochs, early stopping patience 50, seed 42.

## Repository structure

```text
eeg-eog-contamination-mi-analysis/
├── notebooks/
│   └── BCC_Clean_Master_Analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── eegnet_training.py
├── results/
│   └── analysis CSV outputs should be copied here
├── figures/
│   ├── final_classification_performance.png
│   └── baseline_vs_mi_eog_r2.png
├── docs/
│   └── methodology.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

The notebook is the primary analysis record. The CSV files preserve downstream analysis results so that figures and summary tables can be regenerated without retraining models. The raw EEG/EOG dataset is kept outside the repository.

For a new environment, set `BCC_BASE_DIR` to the directory containing `Dataset/`, `results/`, and `images/`.

## Limitations

- The analysis contains 9 subjects, so subject-level responder/adaptive results are exploratory.
- Associations between EOG contamination, physiology, learned representations, and classification performance do not establish causality.
- Model-derived quantities such as decision margins are not independent physiological measurements.
- Different exploratory λ searches should not be conflated with the final fixed-condition validation.

## Citation

Publication/thesis citation will be added when the corresponding manuscript is finalized.
