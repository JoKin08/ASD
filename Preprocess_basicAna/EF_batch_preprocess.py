from pathlib import Path
import glob
import traceback

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

# ============================================================
# 1. SUBJECTS / PATHS
# ============================================================

asd_sub_num_ef = [1261, 1371, 1441, 1271, 1241, 1301, 1541, 1461, 1501,
                  1391, 1701, 1411, 1561, 1611, 1631, 1591, 1731, 1601,
                  1661, 1780, 1331, 1401, 1291, 1220, 1251]

td_sub_num_ef = [2180, 2500, 2231, 2510, 2480, 2441, 2371, 2401, 2470,
                 2420, 2171, 2331, 2380, 2390, 2310, 2240, 2521, 2531,
                 2321, 2430, 2291, 2350, 2301]

BASE = Path(r"H:\ASD projoect\EF")

SUBJECTS = {
    "ASD": asd_sub_num_ef,
    "TD": td_sub_num_ef,
}

DATA_PATHS = {
    "ASD": BASE / "ASD",
    "TD": BASE / "TD",
}

# Choose what to run. Examples: ["ASD"], ["TD"], or ["ASD", "TD"]
RUN_GROUPS = ["ASD", "TD"]

# Matches e.g. 2371_RF_CV.cnt or 1011_CF_CV.cnt
FILE_PATTERN = "{sid}_*_CV.cnt"

# Set True only if you want to skip subjects that already have final output.
SKIP_IF_DONE = False

mne.set_log_level("WARNING")


# ============================================================
# 2. CONSTANTS
# ============================================================

trigger_codes = [
    11, 12, 13, 14,
    51, 52, 53, 54,
    101, 69, 44, 77,
    255, 3,
]

event_id_true = {str(code): code for code in trigger_codes}

stimulus_map = {
    11: {"go_nogo": "Go",   "emotion": "Happy"},
    12: {"go_nogo": "Go",   "emotion": "Angry"},
    13: {"go_nogo": "Go",   "emotion": "Surprise"},
    14: {"go_nogo": "Go",   "emotion": "Neutral"},
    51: {"go_nogo": "Nogo", "emotion": "Happy"},
    52: {"go_nogo": "Nogo", "emotion": "Angry"},
    53: {"go_nogo": "Nogo", "emotion": "Surprise"},
    54: {"go_nogo": "Nogo", "emotion": "Neutral"},
}

outcome_map = {
    101: {"outcome": "correct_press",    "correct": True},
    69:  {"outcome": "missed_press",     "correct": False},
    44:  {"outcome": "wrong_press",      "correct": False},
    77:  {"outcome": "correct_no_press", "correct": True},
}

stim_codes = set(stimulus_map)
outcome_codes = set(outcome_map)

condition_code = {
    ("Go",   "Happy"):    1,
    ("Go",   "Angry"):    2,
    ("Go",   "Surprise"): 3,
    ("Go",   "Neutral"):  4,
    ("Nogo", "Happy"):    5,
    ("Nogo", "Angry"):    6,
    ("Nogo", "Surprise"): 7,
    ("Nogo", "Neutral"):  8,
}

epoch_event_id = {
    "Go/Happy": 1,
    "Go/Angry": 2,
    "Go/Surprise": 3,
    "Go/Neutral": 4,
    "Nogo/Happy": 5,
    "Nogo/Angry": 6,
    "Nogo/Surprise": 7,
    "Nogo/Neutral": 8,
}

metadata_cols = [
    "stim_sample", "stim_time_s", "stim_code", "go_nogo", "emotion",
    "outcome_sample", "outcome_time_s", "outcome_code", "outcome",
    "correct", "outcome_latency_s", "expected_pair",
]


# ============================================================
# 3. HELPERS
# ============================================================

def expected_pair(row):
    if row["go_nogo"] == "Go":
        return row["outcome_code"] in [101, 69]
    if row["go_nogo"] == "Nogo":
        return row["outcome_code"] in [77, 44]
    return False


def reconstruct_trials(events, sfreq):
    trials = []

    for i, event in enumerate(events):
        stim_sample = int(event[0])
        stim_code = int(event[2])

        if stim_code not in stim_codes:
            continue

        outcome_code = None
        outcome_sample = None

        for j in range(i + 1, len(events)):
            next_sample = int(events[j, 0])
            next_code = int(events[j, 2])

            if next_code in outcome_codes:
                outcome_code = next_code
                outcome_sample = next_sample
                break

            if next_code in stim_codes:
                break

        stim_info = stimulus_map[stim_code]

        row = {
            "stim_sample": stim_sample,
            "stim_time_s": stim_sample / sfreq,
            "stim_code": stim_code,
            "go_nogo": stim_info["go_nogo"],
            "emotion": stim_info["emotion"],
        }

        if outcome_code is not None:
            outcome_info = outcome_map[outcome_code]
            row.update({
                "outcome_sample": outcome_sample,
                "outcome_time_s": outcome_sample / sfreq,
                "outcome_code": outcome_code,
                "outcome": outcome_info["outcome"],
                "correct": outcome_info["correct"],
                "outcome_latency_s": (outcome_sample - stim_sample) / sfreq,
            })
        else:
            row.update({
                "outcome_sample": np.nan,
                "outcome_time_s": np.nan,
                "outcome_code": np.nan,
                "outcome": "missing",
                "correct": False,
                "outcome_latency_s": np.nan,
            })

        trials.append(row)

    trial_df = pd.DataFrame(trials)

    if len(trial_df) == 0:
        raise RuntimeError("No EF trials could be reconstructed.")

    trial_df["expected_pair"] = trial_df.apply(expected_pair, axis=1)
    return trial_df


def save_epoch_dict(epochs, path):
    data = {
        "eeg": epochs.get_data(),
        "times": epochs.times,
        "channels": np.array(epochs.ch_names),
        "events": epochs.events,
        "event_id": epochs.event_id,
        "metadata": epochs.metadata.to_dict("list"),
        "sfreq": epochs.info["sfreq"],
    }
    np.save(path, data, allow_pickle=True)


# ============================================================
# 4. PROCESS ONE SUBJECT
# ============================================================

def process_subject(group, subject_id):
    data_path = DATA_PATHS[group]

    pattern = str(data_path / FILE_PATTERN.format(sid=subject_id))
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No CNT file matched: {pattern}")

    if len(files) > 1:
        raise RuntimeError(f"Multiple CNT files matched ({len(files)}): {files}")

    cnt_file = Path(files[0])
    output_dir = data_path / f"{cnt_file.stem}_processed_numpy"
    output_dir.mkdir(exist_ok=True)

    final_file = output_dir / "interpolated_epoch.npy"
    if SKIP_IF_DONE and final_file.exists():
        return {
            "group": group,
            "subject": str(subject_id),
            "file": cnt_file.name,
            "status": "already_done",
            "original_epochs": np.nan,
            "correct_epochs": np.nan,
            "after_ica": np.nan,
            "after_interpolation": np.nan,
            "avg_bad_channels": np.nan,
            "ica_removed": np.nan,
            "error": "",
        }

    # -------------------- Read raw --------------------
    raw = mne.io.read_raw_cnt(
        cnt_file,
        preload=True,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id_true,
        verbose=False,
    )

    trial_df = reconstruct_trials(events, sfreq)

    # -------------------- Filter --------------------
    raw_proc = raw.copy()
    raw_proc.filter(
        l_freq=0.05,
        h_freq=30.0,
        picks="eeg",
        verbose=False,
    )

    # -------------------- Build epochs --------------------
    epoch_events = []
    for _, row in trial_df.iterrows():
        code = condition_code[(row["go_nogo"], row["emotion"])]
        epoch_events.append([
            int(row["stim_sample"]),
            0,
            int(code),
        ])
    epoch_events = np.asarray(epoch_events, dtype=int)

    metadata = trial_df[metadata_cols].copy()

    epochs_all = mne.Epochs(
        raw_proc,
        events=epoch_events,
        event_id=epoch_event_id,
        tmin=-0.2,
        tmax=1.0,
        baseline=(-0.2, 0.0),
        metadata=metadata,
        preload=True,
        reject=None,
        detrend=None,
        verbose=False,
    )

    if len(epochs_all) == 0:
        raise RuntimeError("No epochs remained after epoch creation.")

    # Correct behavioral trials only
    correct_mask = epochs_all.metadata["correct"].to_numpy(dtype=bool)
    epochs_correct = epochs_all[correct_mask].copy()

    if len(epochs_correct) == 0:
        raise RuntimeError("No correct behavioral epochs.")

    save_epoch_dict(
        epochs_correct,
        output_dir / "filtered_epoch.npy",
    )

    # -------------------- ICA --------------------
    # Reconstruct exactly as in the single-subject workflow.
    data = np.load(
        output_dir / "filtered_epoch.npy",
        allow_pickle=True,
    ).item()

    channels = list(data["channels"])

    info = mne.create_info(
        ch_names=channels,
        sfreq=data["sfreq"],
        ch_types="eeg",
    )

    epochs = mne.EpochsArray(
        data["eeg"],
        info,
        events=data["events"],
        event_id=data["event_id"],
        tmin=data["times"][0],
        metadata=pd.DataFrame(data["metadata"]),
        verbose=False,
    )

    needed = ["HEO", "VEO", "Trigger"]
    missing = [ch for ch in needed if ch not in epochs.ch_names]
    if missing:
        raise RuntimeError(f"Missing required channels: {missing}")

    epochs.set_channel_types({
        "HEO": "eog",
        "VEO": "eog",
        "Trigger": "misc",
    }, verbose=False)

    epochs.drop_channels(["Trigger"])

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(
        montage,
        match_case=False,
        on_missing="ignore",
        verbose=False,
    )

    epochs_ica_fit = epochs.copy().filter(
        1.0,
        30.0,
        verbose=False,
    )

    ica = ICA(
        n_components=0.99,
        random_state=42,
        max_iter="auto",
    )

    ica.fit(
        epochs_ica_fit,
        picks="eeg",
        verbose=False,
    )

    heog_inds, _ = ica.find_bads_eog(
        epochs_ica_fit,
        ch_name="HEO",
        verbose=False,
    )

    veog_inds, _ = ica.find_bads_eog(
        epochs_ica_fit,
        ch_name="VEO",
        verbose=False,
    )

    eog_inds = sorted(set(heog_inds + veog_inds))
    ica.exclude = eog_inds

    epochs_after_ica = epochs.copy()
    ica.apply(epochs_after_ica, verbose=False)
    epochs_after_ica.drop_channels(["HEO", "VEO"])

    save_epoch_dict(
        epochs_after_ica,
        output_dir / "ica_epoch.npy",
    )

    # -------------------- Bad-channel detection --------------------
    data = np.load(
        output_dir / "ica_epoch.npy",
        allow_pickle=True,
    ).item()

    eeg = data["eeg"]
    channels = list(data["channels"])

    info = mne.create_info(
        ch_names=channels,
        sfreq=data["sfreq"],
        ch_types="eeg",
    )

    epochs = mne.EpochsArray(
        eeg,
        info,
        events=data["events"],
        event_id=data["event_id"],
        tmin=data["times"][0],
        metadata=pd.DataFrame(data["metadata"]),
        verbose=False,
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(
        montage,
        match_case=False,
        on_missing="ignore",
        verbose=False,
    )

    x = epochs.get_data() * 1e6  # µV
    bad_ratio = 0.01
    max_bad_channels = 20

    bad_channels_each_epoch = []

    for ep in x:
        abs_ep = np.abs(ep)

        threshold = (
            abs_ep.mean(axis=1)
            + 3 * abs_ep.std(axis=1)
        )

        exceed = abs_ep > threshold[:, None]
        exceed_ratio = exceed.mean(axis=1)
        bad = exceed_ratio > bad_ratio

        bad_channels_each_epoch.append([
            ch for ch, flag
            in zip(epochs.ch_names, bad)
            if flag
        ])

    good_epoch_idx = []
    bad_epoch_idx = []
    repaired_data = []

    for i, bad_chs in enumerate(bad_channels_each_epoch):
        if len(bad_chs) > max_bad_channels:
            bad_epoch_idx.append(i)
            continue

        ep = epochs[i].copy()

        if bad_chs:
            ep.info["bads"] = bad_chs
            ep.interpolate_bads(
                reset_bads=True,
                verbose=False,
            )

        repaired_data.append(ep.get_data()[0])
        good_epoch_idx.append(i)

    if len(repaired_data) == 0:
        raise RuntimeError("All epochs were rejected by bad-channel rule.")

    repaired_data = np.stack(repaired_data)

    remaining_bad_counts = [
        len(bad_channels_each_epoch[i])
        for i in good_epoch_idx
    ]

    avg_bad_channels = float(np.mean(remaining_bad_counts))

    epochs_interpolated = mne.EpochsArray(
        repaired_data,
        epochs.info.copy(),
        events=epochs.events[good_epoch_idx],
        event_id=epochs.event_id,
        tmin=epochs.tmin,
        metadata=epochs.metadata.iloc[
            good_epoch_idx
        ].reset_index(drop=True),
        verbose=False,
    )

    save_epoch_dict(
        epochs_interpolated,
        output_dir / "interpolated_epoch.npy",
    )

    # -------------------- ERP --------------------
    data = np.load(
        output_dir / "interpolated_epoch.npy",
        allow_pickle=True,
    ).item()

    info = mne.create_info(
        ch_names=list(data["channels"]),
        sfreq=data["sfreq"],
        ch_types="eeg",
    )

    erp_epochs = mne.EpochsArray(
        data["eeg"],
        info,
        events=data["events"],
        event_id=data["event_id"],
        tmin=data["times"][0],
        metadata=pd.DataFrame(data["metadata"]),
        verbose=False,
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    erp_epochs.set_montage(
        montage,
        match_case=False,
        on_missing="ignore",
        verbose=False,
    )

    erp_epochs.set_eeg_reference(
        "average",
        verbose=False,
    )

    if len(erp_epochs["Go"]) == 0 or len(erp_epochs["Nogo"]) == 0:
        raise RuntimeError("Go or Nogo condition has no remaining epochs.")

    evoked_go = erp_epochs["Go"].average()
    evoked_nogo = erp_epochs["Nogo"].average()

    # N2
    n2_channels = ["FCZ"]
    n2_tmin, n2_tmax = 0.300, 0.400

    n2_go_crop = evoked_go.copy().crop(n2_tmin, n2_tmax)
    n2_nogo_crop = evoked_nogo.copy().crop(n2_tmin, n2_tmax)

    n2_go_wave = n2_go_crop.get_data(
        picks=n2_channels
    ).mean(axis=0)
    n2_nogo_wave = n2_nogo_crop.get_data(
        picks=n2_channels
    ).mean(axis=0)

    n2_go = n2_go_wave.mean() * 1e6
    n2_nogo = n2_nogo_wave.mean() * 1e6
    n2_effect = n2_nogo - n2_go

    n2_go_latency = (
        n2_go_crop.times[np.argmin(n2_go_wave)] * 1000
    )
    n2_nogo_latency = (
        n2_nogo_crop.times[np.argmin(n2_nogo_wave)] * 1000
    )

    # P3/LPP
    p3_channels = ["CZ", "CPZ"]
    p3_tmin, p3_tmax = 0.500, 0.700

    p3_go_crop = evoked_go.copy().crop(p3_tmin, p3_tmax)
    p3_nogo_crop = evoked_nogo.copy().crop(p3_tmin, p3_tmax)

    p3_go_wave = p3_go_crop.get_data(
        picks=p3_channels
    ).mean(axis=0)
    p3_nogo_wave = p3_nogo_crop.get_data(
        picks=p3_channels
    ).mean(axis=0)

    p3_go = p3_go_wave.mean() * 1e6
    p3_nogo = p3_nogo_wave.mean() * 1e6
    p3_effect = p3_nogo - p3_go

    p3_go_latency = (
        p3_go_crop.times[np.argmax(p3_go_wave)] * 1000
    )
    p3_nogo_latency = (
        p3_nogo_crop.times[np.argmax(p3_nogo_wave)] * 1000
    )

    erp_results = {
        "N2": {
            "go_amplitude_uv": n2_go,
            "nogo_amplitude_uv": n2_nogo,
            "nogo_go_effect_uv": n2_effect,
            "go_latency_ms": n2_go_latency,
            "nogo_latency_ms": n2_nogo_latency,
        },
        "P3_LPP": {
            "go_amplitude_uv": p3_go,
            "nogo_amplitude_uv": p3_nogo,
            "nogo_go_effect_uv": p3_effect,
            "go_latency_ms": p3_go_latency,
            "nogo_latency_ms": p3_nogo_latency,
        },
        "n_go_trials": len(erp_epochs["Go"]),
        "n_nogo_trials": len(erp_epochs["Nogo"]),
    }

    np.save(
        output_dir / "EF_ERP_results.npy",
        erp_results,
        allow_pickle=True,
    )

    # -------------------- Subject report --------------------
    report_path = output_dir / "preprocessing_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {subject_id}\n")
        f.write(f"Group: {group}\n")
        f.write(f"CNT file: {cnt_file.name}\n")
        f.write(f"Number of original epochs: {len(epochs_all)}\n")
        f.write(f"Number of correct epochs: {len(epochs_correct)}\n")
        f.write(f"Number of epochs after ICA: {len(epochs_after_ica)}\n")
        f.write(f"Number of epochs after interpolation: {len(epochs_interpolated)}\n")
        f.write(f"Average number of bad channels for interpolation: {avg_bad_channels:.3f}\n")
        f.write(f"ICA components removed: {eog_inds}\n")

    return {
        "group": group,
        "subject": str(subject_id),
        "file": cnt_file.name,
        "status": "done",
        "original_epochs": len(epochs_all),
        "correct_epochs": len(epochs_correct),
        "after_ica": len(epochs_after_ica),
        "after_interpolation": len(epochs_interpolated),
        "avg_bad_channels": avg_bad_channels,
        "ica_removed": str(eog_inds),
        "error": "",
    }


# ============================================================
# 5. BATCH LOOP
# ============================================================

summary = []

total = sum(len(SUBJECTS[g]) for g in RUN_GROUPS)
current = 0

for group in RUN_GROUPS:
    for subject_id in SUBJECTS[group]:
        current += 1
        print(f"[{current}/{total}] {group} {subject_id}")

        try:
            result = process_subject(group, subject_id)
            summary.append(result)

            if result["status"] == "already_done":
                print("  already done")
            else:
                print("  done")

        except Exception as e:
            print(f"  skipped: {type(e).__name__}: {e}")

            summary.append({
                "group": group,
                "subject": str(subject_id),
                "file": "",
                "status": "skipped",
                "original_epochs": np.nan,
                "correct_epochs": np.nan,
                "after_ica": np.nan,
                "after_interpolation": np.nan,
                "avg_bad_channels": np.nan,
                "ica_removed": "",
                "error": f"{type(e).__name__}: {e}",
            })


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)
summary_path = BASE / "EF_batch_preprocessing_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n========== FINAL SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\nStatus counts:")
print(summary_df["status"].value_counts(dropna=False).to_string())

print("\nSaved summary to:")
print(summary_path)
