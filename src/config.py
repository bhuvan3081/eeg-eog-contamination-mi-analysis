"""Central configuration for the EEG/EOG project."""

import os

BASE_DIR = os.environ.get(
    "BCC_BASE_DIR",
    "/content/drive/MyDrive/BCC"
)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "images")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
FS_RAW = 250
FS_MODEL = 125
N_EEG = 22
N_EOG = 3
SEED = 42
