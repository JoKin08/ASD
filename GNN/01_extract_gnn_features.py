# ============================================================
# 01_extract_gnn_features.py
#
# Input:
#   Existing preprocessed EF / ToM epoch .npy files
#
# Output:
#   One .npz per subject x condition
#
# Each .npz contains:
#   X            shape = [n_trials, 62, 4]
#   y            0 = TD, 1 = ASD
#   subject_id
#   condition
#   channels
#   bands
#
# IMPORTANT:
#   No global normalization is performed here.
#   Standardization will be fitted on TRAINING subjects only.
# ============================================================

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"F:\ASD projoect")

EF_ROOT = ROOT / "EF"
TOM_ROOT = ROOT / "ToM"

CACHE_ROOT = ROOT / "GNN_cache_v1"

OVERWRITE = False

GROUP_LABEL = {
    "TD": 0,
    "ASD": 1,
}

# ------------------------------------------------------------
# Frequency bands
#
# We use half-open intervals to avoid counting boundary
# frequencies twice:
#
# theta    4 <= f < 8       -> approximately 4-7 Hz
# alpha    8 <= f < 12
# lowbeta 12 <= f < 16
# beta    16 <= f <= 25
# ------------------------------------------------------------

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_beta": (12.0, 16.0),
    "beta": (16.0, 25.0),
}

BAND_NAMES = list(BANDS.keys())

# Welch segment length.
# 1 sec gives ~1-Hz frequency resolution at 1000 Hz.
WELCH_SECONDS = 1.0

# Process trials in chunks so that one large subject file
# does not consume excessive RAM.
PSD_CHUNK_SIZE = 64


# ============================================================
# BASIC HELPERS
# ============================================================

def load_epoch_file(path):
    """
    Load saved preprocessing dictionary.
    """
    obj = np.load(path, allow_pickle=True)

    if obj.shape != ():
        raise ValueError(
            f"{path} does not contain a saved dictionary."
        )

    d = obj.item()

    required = [
        "eeg",
        "channels",
        "sfreq",
        "metadata",
    ]

    missing = [
        k for k in required
        if k not in d
    ]

    if missing:
        raise KeyError(
            f"{path}: missing keys {missing}"
        )

    eeg = np.asarray(
        d["eeg"],
        dtype=np.float32
    )

    channels = [
        str(x)
        for x in d["channels"]
    ]

    sfreq = float(
        d["sfreq"]
    )

    metadata = pd.DataFrame(
        d["metadata"]
    ).reset_index(drop=True)

    if len(metadata) != eeg.shape[0]:
        raise ValueError(
            f"Metadata/epoch mismatch in {path}: "
            f"{len(metadata)} metadata rows vs "
            f"{eeg.shape[0]} epochs."
        )

    if eeg.ndim != 3:
        raise ValueError(
            f"Expected EEG shape "
            f"[trials, channels, time], got {eeg.shape}"
        )

    return eeg, channels, sfreq, metadata


def correct_mask(series):
    """
    Robust conversion of metadata 'correct' column to bool.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy()

    s = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return s.isin([
        "true",
        "1",
        "yes",
        "correct"
    ]).to_numpy()


def canonical_text(x):
    """
    Lowercase clean condition labels.
    """
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# PSD
# ============================================================

def compute_trial_bandpower_db(
    eeg,
    sfreq,
    chunk_size=PSD_CHUNK_SIZE
):
    """
    Parameters
    ----------
    eeg : ndarray
        [n_trials, n_channels, n_times]

    Returns
    -------
    features : ndarray
        [n_trials, n_channels, 5]

        Feature order:
            delta
            theta
            alpha
            low_beta
            beta

        Values are mean PSD density within each band,
        transformed to dB.
    """

    n_trials, n_channels, n_times = eeg.shape

    nperseg = min(
        n_times,
        int(round(WELCH_SECONDS * sfreq))
    )

    if nperseg < 2:
        raise ValueError(
            f"Epoch too short: n_times={n_times}"
        )

    noverlap = nperseg // 2

    features = np.empty(
        (
            n_trials,
            n_channels,
            len(BANDS)
        ),
        dtype=np.float32
    )

    for start in range(
        0,
        n_trials,
        chunk_size
    ):
        stop = min(
            start + chunk_size,
            n_trials
        )

        chunk = eeg[start:stop]

        freqs, psd = welch(
            chunk,
            fs=sfreq,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            axis=-1,
            scaling="density"
        )

        # psd:
        # [chunk_trials, channels, frequencies]

        for band_idx, (
            band_name,
            (fmin, fmax)
        ) in enumerate(BANDS.items()):

            if band_name == "beta":
                freq_mask = (
                    (freqs >= fmin)
                    & (freqs <= fmax)
                )
            else:
                freq_mask = (
                    (freqs >= fmin)
                    & (freqs < fmax)
                )

            if freq_mask.sum() == 0:
                raise RuntimeError(
                    f"No frequency bins found for "
                    f"{band_name}: {fmin}-{fmax} Hz"
                )

            # Mean spectral density within band
            band_power = np.mean(
                psd[..., freq_mask],
                axis=-1
            )

            # dB
            band_db = (
                10.0
                * np.log10(
                    band_power + 1e-20
                )
            )

            features[
                start:stop,
                :,
                band_idx
            ] = band_db.astype(
                np.float32
            )

    return features


# ============================================================
# SAVE ONE SUBJECT-CONDITION BAG
# ============================================================

def save_condition_bag(
    X,
    group,
    subject_id,
    condition,
    model_name,
    channels,
    source_file
):
    """
    X:
        [n_trials, n_channels, n_bands]
    """

    y = GROUP_LABEL[group]

    output_dir = (
        CACHE_ROOT
        / model_name
        / group
        / str(subject_id)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{condition}.npz"
    )

    if (
        output_file.exists()
        and not OVERWRITE
    ):
        return output_file

    np.savez_compressed(
        output_file,

        X=X.astype(np.float32),

        y=np.int64(y),

        subject_id=np.int64(
            subject_id
        ),

        condition=np.array(
            condition
        ),

        group=np.array(
            group
        ),

        channels=np.array(
            channels
        ),

        bands=np.array(
            BAND_NAMES
        ),

        source_file=np.array(
            str(source_file)
        ),
    )

    return output_file


# ============================================================
# EF
# ============================================================

EF_FOLDER_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<context>RF|CF)_CV_processed_numpy$",
    flags=re.IGNORECASE
)


def parse_ef_folder(folder):
    match = EF_FOLDER_RE.match(
        folder.name
    )

    if match is None:
        raise ValueError(
            f"Cannot parse EF folder name: "
            f"{folder.name}"
        )

    subject_id = int(
        match.group("subject")
    )

    context = (
        match.group("context")
        .lower()
    )

    return subject_id, context


def process_one_ef_file(
    group,
    file_path
):
    """
    One file corresponds to:
        one subject x RF/CF

    Metadata then separates:
        Go/Nogo x emotion
    """

    subject_id, context = (
        parse_ef_folder(
            file_path.parent
        )
    )

    eeg, channels, sfreq, metadata = (
        load_epoch_file(
            file_path
        )
    )

    required_columns = [
        "go_nogo",
        "emotion",
        "correct",
    ]

    for col in required_columns:
        if col not in metadata.columns:
            raise KeyError(
                f"{file_path}: "
                f"metadata lacks '{col}'"
            )

    # --------------------------------------------------------
    # Only correct trials
    # --------------------------------------------------------

    keep = correct_mask(
        metadata["correct"]
    )

    eeg = eeg[keep]
    metadata = (
        metadata.loc[keep]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove trials containing NaN / inf
    # --------------------------------------------------------

    finite_trials = np.isfinite(
        eeg
    ).all(axis=(1, 2))

    if not finite_trials.all():
        n_bad = (
            (~finite_trials).sum()
        )

        print(
            f"  {subject_id} {context}: "
            f"dropping {n_bad} non-finite trials"
        )

        eeg = eeg[finite_trials]

        metadata = (
            metadata.loc[finite_trials]
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Compute PSD ONCE for all retained trials
    # --------------------------------------------------------

    features = (
        compute_trial_bandpower_db(
            eeg,
            sfreq
        )
    )

    # --------------------------------------------------------
    # Condition names
    # --------------------------------------------------------

    metadata["go_clean"] = (
        metadata["go_nogo"]
        .map(canonical_text)
    )

    metadata["emotion_clean"] = (
        metadata["emotion"]
        .map(canonical_text)
    )

    rows = []

    combinations = (
        metadata[
            [
                "go_clean",
                "emotion_clean"
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None
        )
    )

    for go_nogo, emotion in combinations:

        condition = (
            f"{go_nogo}_"
            f"{context}_"
            f"{emotion}"
        )

        mask = (
            (metadata["go_clean"] == go_nogo)
            &
            (
                metadata["emotion_clean"]
                == emotion
            )
        ).to_numpy()

        X_condition = features[mask]

        if len(X_condition) == 0:
            continue

        cache_path = save_condition_bag(
            X=X_condition,
            group=group,
            subject_id=subject_id,
            condition=condition,
            model_name="EF",
            channels=channels,
            source_file=file_path,
        )

        rows.append({
            "model": "EF",
            "group": group,
            "subject_id": subject_id,
            "context": context,
            "condition": condition,
            "n_trials": len(
                X_condition
            ),
            "n_channels": (
                X_condition.shape[1]
            ),
            "n_features": (
                X_condition.shape[2]
            ),
            "cache_path": str(
                cache_path
            ),
            "source_file": str(
                file_path
            ),
        })

    return rows


# ============================================================
# ToM
# ============================================================

def parse_tom_subject(folder):
    """
    Works with folder names like:
        3571a_CV_processed_numpy
        3571_CV_processed_numpy
        3571_something
    """

    match = re.match(
        r"^(\d+)",
        folder.name
    )

    if match is None:
        raise ValueError(
            f"Cannot parse ToM subject from "
            f"{folder.name}"
        )

    return int(
        match.group(1)
    )


def process_one_tom_file(
    group,
    file_path,
    segment
):
    """
    segment:
        H = hearing
        T = thinking
    """

    subject_id = parse_tom_subject(
        file_path.parent
    )

    eeg, channels, sfreq, metadata = (
        load_epoch_file(
            file_path
        )
    )

    required_columns = [
        "tom",
        "order",
        "belief",
        "correct",
    ]

    for col in required_columns:
        if col not in metadata.columns:
            raise KeyError(
                f"{file_path}: "
                f"metadata lacks '{col}'"
            )

    # --------------------------------------------------------
    # Correct trials only
    # --------------------------------------------------------

    keep = correct_mask(
        metadata["correct"]
    )

    eeg = eeg[keep]

    metadata = (
        metadata.loc[keep]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove nonfinite trials
    # --------------------------------------------------------

    finite_trials = np.isfinite(
        eeg
    ).all(axis=(1, 2))

    if not finite_trials.all():

        n_bad = (
            (~finite_trials).sum()
        )

        print(
            f"  {subject_id} {segment}: "
            f"dropping {n_bad} non-finite trials"
        )

        eeg = eeg[finite_trials]

        metadata = (
            metadata.loc[finite_trials]
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Compute channel-band features
    # --------------------------------------------------------

    features = (
        compute_trial_bandpower_db(
            eeg,
            sfreq
        )
    )

    metadata["belief_clean"] = (
        metadata["belief"]
        .map(canonical_text)
    )

    metadata["order_clean"] = (
        metadata["order"]
        .map(canonical_text)
    )

    metadata["tom_clean"] = (
        metadata["tom"]
        .map(canonical_text)
    )

    rows = []

    combinations = (
        metadata[
            [
                "belief_clean",
                "order_clean",
                "tom_clean",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None
        )
    )

    model_name = (
        "ToM_H"
        if segment == "H"
        else "ToM_T"
    )

    for (
        belief,
        order,
        tom
    ) in combinations:

        condition = (
            f"{belief}_"
            f"{order}_"
            f"{tom}"
        )

        mask = (
            (
                metadata["belief_clean"]
                == belief
            )
            &
            (
                metadata["order_clean"]
                == order
            )
            &
            (
                metadata["tom_clean"]
                == tom
            )
        ).to_numpy()

        X_condition = features[mask]

        if len(X_condition) == 0:
            continue

        cache_path = save_condition_bag(
            X=X_condition,
            group=group,
            subject_id=subject_id,
            condition=condition,
            model_name=model_name,
            channels=channels,
            source_file=file_path,
        )

        rows.append({
            "model": model_name,
            "group": group,
            "subject_id": subject_id,
            "condition": condition,
            "n_trials": len(
                X_condition
            ),
            "n_channels": (
                X_condition.shape[1]
            ),
            "n_features": (
                X_condition.shape[2]
            ),
            "cache_path": str(
                cache_path
            ),
            "source_file": str(
                file_path
            ),
        })

    return rows


# ============================================================
# DISCOVERY
# ============================================================

def discover_ef_files():
    records = []

    for group in [
        "ASD",
        "TD"
    ]:

        root = (
            EF_ROOT
            / group
        )

        if not root.exists():
            print(
                "Missing:",
                root
            )
            continue

        files = sorted(
            root.rglob(
                "interpolated_epoch.npy"
            )
        )

        print(
            f"EF {group}: "
            f"found {len(files)} files"
        )

        for path in files:
            records.append(
                (group, path)
            )

    return records


def discover_tom_files(
    segment
):
    if segment == "H":
        filename = (
            "interpolated_hearing_epoch.npy"
        )
    elif segment == "T":
        filename = (
            "interpolated_thinking_epoch.npy"
        )
    else:
        raise ValueError(
            "segment must be H or T"
        )

    records = []

    for group in [
        "ASD",
        "TD"
    ]:

        root = (
            TOM_ROOT
            / group
        )

        if not root.exists():
            print(
                "Missing:",
                root
            )
            continue

        files = sorted(
            root.rglob(
                filename
            )
        )

        print(
            f"ToM {segment} {group}: "
            f"found {len(files)} files"
        )

        for path in files:
            records.append(
                (group, path)
            )

    return records


# ============================================================
# CONSISTENCY CHECK
# ============================================================

def check_channel_consistency(
    manifest
):
    """
    Verify every cache has same channel order
    within each model.
    """

    for model_name in (
        manifest["model"]
        .unique()
    ):

        subset = manifest[
            manifest["model"]
            == model_name
        ]

        reference = None

        for path in subset[
            "cache_path"
        ]:

            d = np.load(
                path,
                allow_pickle=True
            )

            channels = list(
                d["channels"]
                .astype(str)
            )

            if reference is None:
                reference = channels

            elif channels != reference:
                raise RuntimeError(
                    f"Channel order mismatch "
                    f"in model {model_name}: "
                    f"{path}"
                )

        print(
            f"{model_name}: "
            f"channel order consistent "
            f"({len(reference)} channels)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    # ========================================================
    # EF
    # ========================================================

    print(
        "\n======================"
    )
    print(
        "Processing EF"
    )
    print(
        "======================"
    )

    ef_files = (
        discover_ef_files()
    )

    for idx, (
        group,
        file_path
    ) in enumerate(
        ef_files,
        start=1
    ):

        print(
            f"[EF {idx}/{len(ef_files)}] "
            f"{group} | {file_path.parent.name}"
        )

        try:
            rows = process_one_ef_file(
                group,
                file_path
            )

            all_rows.extend(
                rows
            )

        except Exception as e:
            print(
                "  ERROR:",
                repr(e)
            )

    # ========================================================
    # ToM Hearing
    # ========================================================

    print(
        "\n======================"
    )
    print(
        "Processing ToM Hearing"
    )
    print(
        "======================"
    )

    h_files = (
        discover_tom_files(
            "H"
        )
    )

    for idx, (
        group,
        file_path
    ) in enumerate(
        h_files,
        start=1
    ):

        print(
            f"[H {idx}/{len(h_files)}] "
            f"{group} | {file_path.parent.name}"
        )

        try:

            rows = process_one_tom_file(
                group,
                file_path,
                segment="H"
            )

            all_rows.extend(
                rows
            )

        except Exception as e:
            print(
                "  ERROR:",
                repr(e)
            )

    # ========================================================
    # ToM Thinking
    # ========================================================

    print(
        "\n======================"
    )
    print(
        "Processing ToM Thinking"
    )
    print(
        "======================"
    )

    t_files = (
        discover_tom_files(
            "T"
        )
    )

    for idx, (
        group,
        file_path
    ) in enumerate(
        t_files,
        start=1
    ):

        print(
            f"[T {idx}/{len(t_files)}] "
            f"{group} | {file_path.parent.name}"
        )

        try:

            rows = process_one_tom_file(
                group,
                file_path,
                segment="T"
            )

            all_rows.extend(
                rows
            )

        except Exception as e:
            print(
                "  ERROR:",
                repr(e)
            )

    # ========================================================
    # Manifest
    # ========================================================

    if not all_rows:
        raise RuntimeError(
            "No data were successfully processed."
        )

    manifest = pd.DataFrame(
        all_rows
    )

    manifest_path = (
        CACHE_ROOT
        / "cache_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False
    )
ss
        "\nManifest saved:"
    )
    print(
        manifest_path
    )

    # ========================================================
    # Channel check
    # ========================================================

    check_channel_consistency(
        manifest
    )

    # ========================================================
    # Simple summary
    # ========================================================

    print(
        "\n======================"
    )
    print(
        "SUBJECT COUNTS"
    )
    print(
        "======================"
    )

    print(
        manifest
        .groupby(
            [
                "model",
                "group"
            ]
        )[
            "subject_id"
        ]
        .nunique()
    )

    print(
        "\n======================"
    )
    print(
        "TRIAL COUNT SUMMARY"
    )
    print(
        "======================"
    )

    print(
        manifest
        .groupby(
            [
                "model",
                "condition"
            ]
        )[
            "n_trials"
        ]
        .agg(
            [
                "count",
                "min",
                "median",
                "mean",
                "max"
            ]
        )
        .round(2)
    )


if __name__ == "__main__":
    main()