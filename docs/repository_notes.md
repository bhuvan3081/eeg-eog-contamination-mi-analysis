# Repository notes

- `notebooks/eeg_eog_contamination_analysis.ipynb` is the complete executed analysis record.
- `src/eegnet_training.py` exposes the EEGNet architecture and training/evaluation routines for direct inspection.
- Raw dataset files are excluded by `.gitignore`.
- Existing result CSVs should be copied into `results/` without altering numerical values.
## Complete-archive merge

The repository preserves the complete figure and result archive from the previous project version. The README additionally exposes a small `figures/featured/` set selected from those existing scientific outputs (plus the corrected final classification summary figure) for portfolio readability. The detailed archive remains under `figures/`, while numerical outputs remain under `results/`. No raw BCI `.mat` dataset files are included.
