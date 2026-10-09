# ============================================================
# 04_train_gnn.py
#
# Shared GCN + condition-specific heads
# Subject-wise 5-fold CV
# ============================================================

from pathlib import Path
from functools import lru_cache
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

from gnn_model import (
    SharedConditionGCN,
    build_spatial_graph
)

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"F:\ASD projoect"
)

CACHE_ROOT = (
    ROOT
    / "GNN_cache_v1"
)

MANIFEST_PATH = (
    CACHE_ROOT
    / "cache_manifest.csv"
)

RESULT_ROOT = (
    ROOT
    / "GNN_results_v1"
)


MIN_TRIALS = 5

MAX_TRAIN_TRIALS = 20

N_FOLDS = 5

VALIDATION_SIZE = 0.15

BATCH_SIZE = 8

MAX_EPOCHS = 120

PATIENCE = 15

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 32

EMBEDDING_DIM = 32

DROPOUT = 0.30

GRAPH_K = 4

SEED = 42


# ============================================================
# CONDITIONS
# ============================================================

EF_CONDITIONS = [
    f"{go}_{context}_{emotion}"
    for go in [
        "go",
        "nogo"
    ]
    for context in [
        "rf",
        "cf"
    ]
    for emotion in [
        "happy",
        "angry",
        "surprise",
        "neutral"
    ]
]


TOM_CONDITIONS = [
    f"{belief}_{order}_{tom}"
    for belief in [
        "true",
        "false"
    ]
    for order in [
        "first",
        "second"
    ]
    for tom in [
        "cognitive",
        "affective"
    ]
]


def get_condition_list(
    model_name
):

    if model_name == "EF":
        return EF_CONDITIONS

    elif model_name in [
        "ToM_H",
        "ToM_T"
    ]:
        return TOM_CONDITIONS

    else:
        raise ValueError(
            model_name
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CACHED NPZ LOADER
# ============================================================

@lru_cache(
    maxsize=None
)
def load_npz_X(
    path_str
):

    with np.load(
        path_str,
        allow_pickle=True
    ) as d:

        X = np.asarray(
            d["X"],
            dtype=np.float32
        )

    return X


# ============================================================
# NORMALIZATION
# ============================================================

def fit_normalizer(
    df
):
    """
    Fit channel x band mean/std using training data only.

    Output shapes:
        [62, 4]
    """

    sum_x = None
    sum_x2 = None
    count = 0

    for path in df[
        "cache_path"
    ]:

        X = load_npz_X(
            str(path)
        )

        # X:
        # trials x channels x bands

        if sum_x is None:

            sum_x = np.zeros(
                X.shape[1:],
                dtype=np.float64
            )

            sum_x2 = np.zeros(
                X.shape[1:],
                dtype=np.float64
            )

        sum_x += (
            X.sum(
                axis=0
            )
        )

        sum_x2 += (
            (X ** 2)
            .sum(
                axis=0
            )
        )

        count += (
            X.shape[0]
        )

    mean = (
        sum_x
        / count
    )

    variance = (
        sum_x2
        / count
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        1e-8
    )

    std = np.sqrt(
        variance
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        )
    )


# ============================================================
# DATASET
# ============================================================

class BagDataset(Dataset):

    def __init__(
        self,
        df,
        condition_to_idx,
        mean,
        std,
        training=False,
        max_train_trials=20
    ):

        self.df = (
            df
            .reset_index(
                drop=True
            )
        )

        self.condition_to_idx = (
            condition_to_idx
        )

        self.mean = mean
        self.std = std

        self.training = training

        self.max_train_trials = (
            max_train_trials
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[
            idx
        ]

        X = (
            load_npz_X(
                str(
                    row["cache_path"]
                )
            )
            .copy()
        )

        # ----------------------------------------
        # random trial subsampling during training
        # ----------------------------------------

        if (
            self.training
            and X.shape[0]
            > self.max_train_trials
        ):

            selected = np.random.choice(
                X.shape[0],
                size=self.max_train_trials,
                replace=False
            )

            X = X[
                selected
            ]

        # ----------------------------------------
        # training-fold normalization
        # ----------------------------------------

        X = (
            X - self.mean
        ) / self.std

        condition = str(
            row["condition"]
        )

        condition_idx = (
            self.condition_to_idx[
                condition
            ]
        )

        return {
            "X": torch.tensor(
                X,
                dtype=torch.float32
            ),

            "y": int(
                row["y"]
            ),

            "condition_idx": int(
                condition_idx
            ),

            "condition": condition,

            "subject_id": int(
                row["subject_id"]
            ),

            "group": str(
                row["group"]
            ),
        }


# ============================================================
# COLLATE VARIABLE TRIAL COUNTS
# ============================================================

def collate_bags(
    batch
):

    B = len(
        batch
    )

    max_trials = max(
        item["X"].shape[0]
        for item in batch
    )

    n_nodes = (
        batch[0]["X"].shape[1]
    )

    n_features = (
        batch[0]["X"].shape[2]
    )

    X_padded = torch.zeros(
        (
            B,
            max_trials,
            n_nodes,
            n_features
        ),
        dtype=torch.float32
    )

    trial_mask = torch.zeros(
        (
            B,
            max_trials
        ),
        dtype=torch.bool
    )

    y = torch.zeros(
        B,
        dtype=torch.long
    )

    condition_idx = torch.zeros(
        B,
        dtype=torch.long
    )

    subject_ids = []

    groups = []

    conditions = []

    for i, item in enumerate(
        batch
    ):

        T = item["X"].shape[0]

        X_padded[
            i,
            :T
        ] = item["X"]

        trial_mask[
            i,
            :T
        ] = True

        y[i] = item[
            "y"
        ]

        condition_idx[i] = item[
            "condition_idx"
        ]

        subject_ids.append(
            item["subject_id"]
        )

        groups.append(
            item["group"]
        )

        conditions.append(
            item["condition"]
        )

    return {
        "X": X_padded,
        "trial_mask": trial_mask,
        "y": y,
        "condition_idx": condition_idx,
        "subject_id": subject_ids,
        "group": groups,
        "condition": conditions,
    }


# ============================================================
# CONDITION-SPECIFIC CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    df,
    condition_to_idx
):
    """
    Balanced weights separately within each head.

    Returns
    -------
    weights:
        [n_conditions, 2]

        [:, 0] = TD weight
        [:, 1] = ASD weight
    """

    n_conditions = len(
        condition_to_idx
    )

    weights = np.ones(
        (
            n_conditions,
            2
        ),
        dtype=np.float32
    )

    for condition, idx in (
        condition_to_idx.items()
    ):

        subset = df[
            df["condition"]
            == condition
        ]

        n0 = (
            subset["y"]
            == 0
        ).sum()

        n1 = (
            subset["y"]
            == 1
        ).sum()

        total = (
            n0 + n1
        )

        if (
            n0 > 0
            and n1 > 0
        ):

            weights[
                idx,
                0
            ] = (
                total
                / (2 * n0)
            )

            weights[
                idx,
                1
            ] = (
                total
                / (2 * n1)
            )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# LOSS
# ============================================================

def weighted_bce_loss(
    logits,
    y,
    condition_idx,
    class_weights
):

    raw_loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            y.float(),
            reduction="none"
        )
    )

    sample_weights = (
        class_weights[
            condition_idx,
            y
        ]
    )

    return (
        raw_loss
        * sample_weights
    ).mean()


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    device
):

    model.train()

    losses = []

    for batch in loader:

        X = batch[
            "X"
        ].to(
            device
        )

        mask = batch[
            "trial_mask"
        ].to(
            device
        )

        y = batch[
            "y"
        ].to(
            device
        )

        condition_idx = batch[
            "condition_idx"
        ].to(
            device
        )

        optimizer.zero_grad()

        logits, _ = model(
            X,
            mask,
            condition_idx
        )

        loss = weighted_bce_loss(
            logits,
            y,
            condition_idx,
            class_weights
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

    return float(
        np.mean(
            losses
        )
    )


# ============================================================
# VALIDATION LOSS
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    class_weights,
    device
):

    model.eval()

    losses = []

    for batch in loader:

        X = batch[
            "X"
        ].to(
            device
        )

        mask = batch[
            "trial_mask"
        ].to(
            device
        )

        y = batch[
            "y"
        ].to(
            device
        )

        condition_idx = batch[
            "condition_idx"
        ].to(
            device
        )

        logits, _ = model(
            X,
            mask,
            condition_idx
        )

        loss = weighted_bce_loss(
            logits,
            y,
            condition_idx,
            class_weights
        )

        losses.append(
            loss.item()
        )

    return float(
        np.mean(
            losses
        )
    )


# ============================================================
# PREDICTIONS
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
    fold
):

    model.eval()

    rows = []

    for batch in loader:

        X = batch[
            "X"
        ].to(
            device
        )

        mask = batch[
            "trial_mask"
        ].to(
            device
        )

        condition_idx = batch[
            "condition_idx"
        ].to(
            device
        )

        logits, embeddings = model(
            X,
            mask,
            condition_idx
        )

        probs = torch.sigmoid(
            logits
        )

        logits = (
            logits
            .cpu()
            .numpy()
        )

        probs = (
            probs
            .cpu()
            .numpy()
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )

        for i in range(
            len(probs)
        ):

            row = {
                "fold": fold,

                "subject_id":
                    batch[
                        "subject_id"
                    ][i],

                "group":
                    batch[
                        "group"
                    ][i],

                "y":
                    int(
                        batch[
                            "y"
                        ][i]
                    ),

                "condition":
                    batch[
                        "condition"
                    ][i],

                "logit":
                    float(
                        logits[i]
                    ),

                "prob_ASD":
                    float(
                        probs[i]
                    ),
            }

            # save latent embedding
            for j, value in enumerate(
                embeddings[i]
            ):

                row[
                    f"embedding_{j}"
                ] = float(
                    value
                )

            rows.append(
                row
            )

    return rows


# ============================================================
# METRICS
# ============================================================

def calculate_condition_metrics(
    prediction_df
):

    results = []

    for condition, df in (
        prediction_df.groupby(
            "condition"
        )
    ):

        y = df[
            "y"
        ].to_numpy()

        prob = df[
            "prob_ASD"
        ].to_numpy()

        pred = (
            prob >= 0.5
        ).astype(int)

        if len(
            np.unique(y)
        ) == 2:

            auc = roc_auc_score(
                y,
                prob
            )

        else:

            auc = np.nan

        result = {
            "condition":
                condition,

            "n":
                len(df),

            "n_TD":
                int(
                    (y == 0)
                    .sum()
                ),

            "n_ASD":
                int(
                    (y == 1)
                    .sum()
                ),

            "AUC":
                auc,

            "accuracy":
                accuracy_score(
                    y,
                    pred
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y,
                    pred
                ),
        }

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                pred,
                labels=[
                    0,
                    1
                ]
            )
            .ravel()
        )

        result[
            "sensitivity_ASD"
        ] = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        result[
            "specificity_TD"
        ] = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def run_model(
    model_name
):

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    result_dir = (
        RESULT_ROOT
        / model_name
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest = manifest[
        manifest["model"]
        == model_name
    ].copy()

    # minimum trial threshold
    before = len(
        manifest
    )

    manifest = manifest[
        manifest["n_trials"]
        >= MIN_TRIALS
    ].copy()

    after = len(
        manifest
    )

    print(
        f"{model_name}: "
        f"removed {before-after} "
        f"subject-condition cells "
        f"with <{MIN_TRIALS} trials"
    )

    manifest[
        "y"
    ] = manifest[
        "group"
    ].map({
        "TD": 0,
        "ASD": 1
    })

    # ========================================================
    # CONDITIONS
    # ========================================================

    condition_list = (
        get_condition_list(
            model_name
        )
    )

    condition_to_idx = {
        condition: i
        for i, condition
        in enumerate(
            condition_list
        )
    }

    unknown = set(
        manifest[
            "condition"
        ]
    ) - set(
        condition_list
    )

    if unknown:

        raise RuntimeError(
            f"Unknown conditions: {unknown}"
        )

    print(
        "\nConditions:"
    )

    for i, c in enumerate(
        condition_list
    ):

        print(
            i,
            c
        )

    # ========================================================
    # CHANNELS
    # ========================================================

    first_path = str(
        manifest.iloc[
            0
        ][
            "cache_path"
        ]
    )

    with np.load(
        first_path,
        allow_pickle=True
    ) as d:

        channels = list(
            d[
                "channels"
            ].astype(str)
        )

    print(
        "\nNumber of channels:",
        len(channels)
    )

    # ========================================================
    # GRAPH
    # ========================================================

    adjacency, node_coords = (
        build_spatial_graph(
            channels,
            k=GRAPH_K
        )
    )

    np.save(
        result_dir
        / "adjacency.npy",
        adjacency.numpy()
    )

    with open(
        result_dir
        / "channels.json",
        "w"
    ) as f:

        json.dump(
            channels,
            f,
            indent=2
        )

    with open(
        result_dir
        / "conditions.json",
        "w"
    ) as f:

        json.dump(
            condition_list,
            f,
            indent=2
        )

    # ========================================================
    # UNIQUE SUBJECTS
    # ========================================================

    subjects = (
        manifest[
            [
                "subject_id",
                "y",
                "group"
            ]
        ]
        .drop_duplicates(
            subset=[
                "subject_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nSubjects:"
    )

    print(
        subjects[
            "group"
        ].value_counts()
    )

    # ========================================================
    # OUTER CV
    # ========================================================

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    all_test_predictions = []

    split_iterator = skf.split(
        subjects[
            "subject_id"
        ],
        subjects[
            "y"
        ]
    )

    for fold, (
        outer_train_idx,
        test_idx
    ) in enumerate(
        split_iterator,
        start=1
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"FOLD {fold}"
        )

        print(
            "=" * 60
        )

        outer_train_subjects = (
            subjects.iloc[
                outer_train_idx
            ]
        )

        test_subjects = (
            subjects.iloc[
                test_idx
            ]
        )

        # ----------------------------------------------------
        # internal subject-wise validation split
        # ----------------------------------------------------

        train_subject_ids, val_subject_ids = (
            train_test_split(
                outer_train_subjects[
                    "subject_id"
                ].to_numpy(),

                test_size=
                    VALIDATION_SIZE,

                random_state=
                    SEED + fold,

                stratify=
                    outer_train_subjects[
                        "y"
                    ].to_numpy()
            )
        )

        test_subject_ids = (
            test_subjects[
                "subject_id"
            ].to_numpy()
        )

        train_df = manifest[
            manifest[
                "subject_id"
            ].isin(
                train_subject_ids
            )
        ].copy()

        val_df = manifest[
            manifest[
                "subject_id"
            ].isin(
                val_subject_ids
            )
        ].copy()

        test_df = manifest[
            manifest[
                "subject_id"
            ].isin(
                test_subject_ids
            )
        ].copy()

        print(
            "Train subjects:",
            len(
                np.unique(
                    train_subject_ids
                )
            )
        )

        print(
            "Val subjects:",
            len(
                np.unique(
                    val_subject_ids
                )
            )
        )

        print(
            "Test subjects:",
            len(
                np.unique(
                    test_subject_ids
                )
            )
        )

        # ====================================================
        # NORMALIZATION - TRAIN ONLY
        # ====================================================

        mean, std = (
            fit_normalizer(
                train_df
            )
        )

        np.savez(
            result_dir
            / f"normalizer_fold_{fold}.npz",

            mean=mean,
            std=std
        )

        # ====================================================
        # DATASETS
        # ====================================================

        train_dataset = BagDataset(
            train_df,
            condition_to_idx,
            mean,
            std,
            training=True,
            max_train_trials=
                MAX_TRAIN_TRIALS
        )

        val_dataset = BagDataset(
            val_df,
            condition_to_idx,
            mean,
            std,
            training=False
        )

        test_dataset = BagDataset(
            test_df,
            condition_to_idx,
            mean,
            std,
            training=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_bags
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_bags
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_bags
        )

        # ====================================================
        # MODEL
        # ====================================================

        model = SharedConditionGCN(
            adjacency=adjacency,
            node_coords=node_coords,
            n_conditions=len(
                condition_list
            ),
            psd_features=4,
            hidden_dim=
                HIDDEN_DIM,
            embedding_dim=
                EMBEDDING_DIM,
            dropout=
                DROPOUT
        ).to(
            device
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY
        )

        class_weights = (
            calculate_class_weights(
                train_df,
                condition_to_idx
            )
            .to(
                device
            )
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        best_val_loss = np.inf

        best_state = None

        best_epoch = 0

        patience_counter = 0

        history = []

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            train_loss = (
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    class_weights,
                    device
                )
            )

            val_loss = (
                evaluate_loss(
                    model,
                    val_loader,
                    class_weights,
                    device
                )
            )

            history.append({
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss
            })

            if (
                epoch == 1
                or epoch % 5 == 0
            ):

                print(
                    f"Epoch {epoch:03d} | "
                    f"train={train_loss:.4f} | "
                    f"val={val_loss:.4f}"
                )

            if (
                val_loss
                < best_val_loss
                - 1e-4
            ):

                best_val_loss = (
                    val_loss
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                patience_counter = 0

            else:

                patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    f"Early stopping at "
                    f"epoch {epoch}"
                )

                break

        print(
            f"Best epoch: "
            f"{best_epoch}, "
            f"val loss="
            f"{best_val_loss:.4f}"
        )

        # restore best model
        model.load_state_dict(
            best_state
        )

        torch.save(
            best_state,
            result_dir
            / f"model_fold_{fold}.pt"
        )

        pd.DataFrame(
            history
        ).to_csv(
            result_dir
            / f"history_fold_{fold}.csv",
            index=False
        )

        # ====================================================
        # TEST SET
        # ====================================================

        predictions = predict(
            model,
            test_loader,
            device,
            fold
        )

        all_test_predictions.extend(
            predictions
        )

    # ========================================================
    # AGGREGATE OUTER-FOLD PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame(
        all_test_predictions
    )

    prediction_df.to_csv(
        result_dir
        / "cv_test_predictions.csv",
        index=False
    )

    metrics = (
        calculate_condition_metrics(
            prediction_df
        )
    )

    metrics.to_csv(
        result_dir
        / "condition_metrics.csv",
        index=False
    )

    # macro results
    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL HELD-OUT CONDITION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        metrics[
            [
                "condition",
                "n",
                "n_TD",
                "n_ASD",
                "AUC",
                "balanced_accuracy",
                "sensitivity_ASD",
                "specificity_TD",
            ]
        ]
        .sort_values(
            "AUC",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nMacro mean AUC:",
        metrics[
            "AUC"
        ].mean()
    )

    print(
        "Macro mean balanced accuracy:",
        metrics[
            "balanced_accuracy"
        ].mean()
    )

    print(
        "\nSaved to:"
    )

    print(
        result_dir
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "EF",
            "ToM_H",
            "ToM_T"
        ]
    )

    args = parser.parse_args()

    run_model(
        args.model
    )