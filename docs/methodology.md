# Methodology

## Stage 1 — Data and contamination

EEG and EOG are loaded from BCI Competition IV 2a-style MATLAB files. The analysis uses the 22 EEG channels and three EOG channels, filters the signals, and estimates EOG-to-EEG coupling through regression/R² measures. Trial-level contamination measures are linked to held-out EEGNet predictions to quantify the relationship between contamination and classification error.

## Stage 2 — Classification and suppression

EEGNet is trained on the training split and evaluated on held-out data. EOG regression is fit only using training data. Cleaning follows `EEG_clean = EEG_raw - λ × EOG_component`. Multiple suppression strengths are evaluated before selecting λ = 0.75 as the representative mechanistic condition used by the final corrected physiology analysis.

## Stage 3 — Mechanistic analyses

The project examines sensorimotor ERD/ERS, beta temporal dynamics, lateralization, class separability, feature-space geometry, compactness/confusion, and decision margins. These analyses are complementary and should not be interpreted as independent causal evidence.

## Stage 4 — Validation

The final validation compares raw and fixed λ = 0.75 conditions at the subject level. Paired tests and leave-one-subject-out robustness analyses are retained. Responder/adaptive analyses are treated as exploratory because the available sample is nine subjects.
