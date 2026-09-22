"""EEGNet training utilities extracted from the research notebook.

This module preserves the model/training configuration used in the main analysis.
"""

import random
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 200
LEARNING_RATE = 0.001


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

class EEGNet(nn.Module):

    def __init__(
        self,
        num_channels=22,
        num_classes=4,
        samples=500,
        F1=16,
        D=2,
        F2=32,
        dropout=0.5,
        kern_length=64
    ):

        super().__init__()

        self.block1 = nn.Sequential(

            nn.Conv2d(
                1,
                F1,
                (1, kern_length),
                padding=(
                    0,
                    kern_length // 2
                ),
                bias=False
            ),

            nn.BatchNorm2d(
                F1
            ),

            nn.Conv2d(
                F1,
                F1 * D,
                (num_channels, 1),
                groups=F1,
                bias=False
            ),

            nn.BatchNorm2d(
                F1 * D
            ),

            nn.ELU(),

            nn.AvgPool2d(
                (1, 8)
            ),

            nn.Dropout(
                dropout
            )
        )

        self.block2 = nn.Sequential(

            nn.Conv2d(
                F1 * D,
                F2,
                (1, 16),
                padding=(
                    0,
                    8
                ),
                bias=False
            ),

            nn.BatchNorm2d(
                F2
            ),

            nn.ELU(),

            nn.AvgPool2d(
                (1, 8)
            ),

            nn.Dropout(
                dropout
            )
        )

        self._flatten_size = (
            self._get_flatten_size(
                num_channels,
                samples,
                F1,
                D,
                F2,
                kern_length
            )
        )

        self.fc = nn.Linear(
            self._flatten_size,
            num_classes
        )

    def _get_flatten_size(
        self,
        C,
        T,
        F1,
        D,
        F2,
        kern_length
    ):

        x = torch.zeros(
            1,
            1,
            C,
            T
        )

        x = self.block1(
            x
        )

        x = self.block2(
            x
        )

        return x.flatten(
            1
        ).shape[1]

    def forward(
        self,
        x
    ):

        x = x.permute(
            0,
            2,
            1
        ).unsqueeze(
            1
        )

        x = self.block1(
            x
        )

        x = self.block2(
            x
        )

        x = x.flatten(
            1
        )

        return self.fc(
            x
        )

def load_bci2a_with_eog(
    mat_path
):

    mat = loadmat(
        mat_path,
        squeeze_me=False
    )

    data = mat[
        "data"
    ]

    eeg_trials = []
    eog_trials = []
    labels = []
    metadata = []

    for run_idx in range(
        data.shape[1]
    ):

        run = data[
            0,
            run_idx
        ][
            0,
            0
        ]

        X_raw = run[
            "X"
        ]

        trials = run[
            "trial"
        ]

        y = run[
            "y"
        ]

        artifacts = run[
            "artifacts"
        ]

        if trials.shape[0] == 0:

            continue

        for t_idx in range(
            trials.shape[0]
        ):

            if artifacts[
                t_idx,
                0
            ] == 1:

                continue

            onset = int(
                trials[
                    t_idx,
                    0
                ]
            )

            start = onset

            end = (
                onset
                +
                4 * FS_RAW
            )

            if end > X_raw.shape[0]:

                continue

            eeg = X_raw[
                start:end,
                :N_EEG
            ]

            eog = X_raw[
                start:end,
                N_EEG:N_EEG + N_EOG
            ]

            eeg = eeg[
                ::2,
                :
            ]

            eog = eog[
                ::2,
                :
            ]

            eeg_trials.append(
                eeg
            )

            eog_trials.append(
                eog
            )

            label = int(
                y[
                    t_idx,
                    0
                ]
            ) - 1

            labels.append(
                label
            )

            metadata.append({

                "Run":
                    run_idx,

                "Trial":
                    t_idx + 1,

                "Onset":
                    onset,

                "True_Class":
                    label + 1

            })

    return (

        np.asarray(
            eeg_trials
        ),

        np.asarray(
            eog_trials
        ),

        np.asarray(
            labels
        ),

        pd.DataFrame(
            metadata
        )

    )

def filter_band(
    X,
    low,
    high
):

    """
    X:
        trials × samples × channels

    Returns:
        trials × samples × channels
    """

    X_t = X.transpose(
        0,
        2,
        1
    )

    filtered = mne.filter.filter_data(

        X_t.reshape(
            -1,
            X_t.shape[-1]
        ),

        sfreq=FS_MODEL,

        l_freq=low,

        h_freq=high,

        method="fir",

        verbose=False

    )

    filtered = filtered.reshape(
        X_t.shape
    )

    return filtered.transpose(
        0,
        2,
        1
    )

def filter_classifier_band(
    X
):

    return filter_band(
        X,
        4.0,
        38.0
    )

def fit_eog_regression(
    eeg_band,
    eog_band
):

    """
    EEG_channel =
        beta1 * EOG_left
      + beta2 * EOG_central
      + beta3 * EOG_right
      + intercept

    ONLY training data should be supplied.
    """

    eeg_flat = eeg_band.reshape(
        -1,
        N_EEG
    )

    eog_flat = eog_band.reshape(
        -1,
        N_EOG
    )

    X = np.column_stack([
        eog_flat,

        np.ones(
            len(eog_flat)
        )
    ])

    beta = np.zeros(
        (
            N_EEG,
            N_EOG
        ),
        dtype=np.float64
    )

    intercept = np.zeros(
        N_EEG,
        dtype=np.float64
    )

    for ch in range(
        N_EEG
    ):

        coeffs, _, _, _ = lstsq(
            X,
            eeg_flat[
                :,
                ch
            ],
            rcond=None
        )

        beta[
            ch,
            :
        ] = coeffs[
            :N_EOG
        ]

        intercept[
            ch
        ] = coeffs[
            N_EOG
        ]

    return (
        beta,
        intercept
    )

def estimate_eog_component(
    eog,
    beta,
    intercept
):

    component = np.einsum(
        "tsc,nc->tsn",
        eog,
        beta
    )

    component += (
        intercept.reshape(
            1,
            1,
            -1
        )
    )

    return component

def apply_lambda(
    eeg,
    eog,
    beta,
    intercept,
    lambda_value
):

    """
    EEG_lambda =
        EEG - lambda * estimated_EOG_component
    """

    component = (
        estimate_eog_component(
            eog,
            beta,
            intercept
        )
    )

    cleaned = (
        eeg
        -
        lambda_value
        *
        component
    )

    return cleaned

def make_loader(
    X,
    y,
    shuffle,
    drop_last
):

    return DataLoader(

        TensorDataset(

            torch.tensor(
                X,
                dtype=torch.float32
            ),

            torch.tensor(
                y,
                dtype=torch.long
            )

        ),

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        drop_last=drop_last
    )

def train_eegnet(
    X_train,
    y_train,
    X_val,
    y_val,
    seed
):

    set_seed(
        seed
    )

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
        drop_last=True
    )

    val_loader = make_loader(
        X_val,
        y_val,
        shuffle=False,
        drop_last=False
    )

    model = EEGNet(
        num_channels=22,
        num_classes=4,
        samples=X_train.shape[1],
        F1=16,
        D=2,
        F2=32,
        dropout=0.5,
        kern_length=64
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=1e-5
        )
    )

    best_val_acc = 0.0

    best_state = None

    patience = 50

    patience_counter = 0

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        train_correct = 0

        train_total = 0

        for (
            X_batch,
            y_batch
        ) in train_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            preds = outputs.argmax(
                dim=1
            )

            train_correct += (
                preds == y_batch
            ).sum().item()

            train_total += (
                y_batch.size(0)
            )

        train_acc = (
            train_correct
            /
            train_total
        )

        model.eval()

        val_correct = 0

        val_total = 0

        with torch.no_grad():

            for (
                X_batch,
                y_batch
            ) in val_loader:

                X_batch = X_batch.to(
                    device
                )

                y_batch = y_batch.to(
                    device
                )

                outputs = model(
                    X_batch
                )

                preds = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    preds == y_batch
                ).sum().item()

                val_total += (
                    y_batch.size(0)
                )

        val_acc = (
            val_correct
            /
            val_total
        )

        scheduler.step()

        if (
            val_acc
            >
            best_val_acc
            +
            0.0001
        ):

            best_val_acc = (
                val_acc
            )

            best_state = {

                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()

            }

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >=
                patience
            ):

                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_val_acc
    )

def evaluate_eegnet(
    model,
    X_test,
    y_test,
    metadata,
    subject,
    lambda_value
):

    model.eval()

    test_loader = make_loader(
        X_test,
        y_test,
        shuffle=False,
        drop_last=False
    )

    preds = []

    labels = []

    start = time.time()

    with torch.no_grad():

        for (
            X_batch,
            y_batch
        ) in test_loader:

            X_batch = X_batch.to(
                device
            )

            outputs = model(
                X_batch
            )

            batch_preds = (
                outputs
                .argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            batch_labels = (
                y_batch
                .cpu()
                .numpy()
            )

            preds.extend(
                batch_preds
            )

            labels.extend(
                batch_labels
            )

    inference_ms = (
        time.time() - start
    ) * 1000

    preds = np.asarray(
        preds
    )

    labels = np.asarray(
        labels
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    pred_df = metadata.copy()

    pred_df["Subject"] = (
        subject
    )

    pred_df["Model"] = (
        "EEGNet"
    )

    pred_df["Lambda"] = (
        lambda_value
    )

    pred_df[
        "Predicted_Class"
    ] = (
        preds + 1
    )

    pred_df["Correct"] = (
        pred_df[
            "True_Class"
        ]
        ==
        pred_df[
            "Predicted_Class"
        ]
    ).astype(int)

    return (
        pred_df,
        accuracy,
        macro_f1,
        inference_ms
    )

