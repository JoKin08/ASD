from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

mne.set_log_level("WARNING")

# ============================================================
# 1. SUBJECTS / PATHS
# ============================================================

asd_sub_num_tom = [
    3431, 3301, 3390, 3441, 3361, 3381, 3400, 3261, 3520, 
    3711, 3690, 3291, 3371, 3351, 3331, 3311, 3271, 3191, 
    3321, 3101, 3171, 3480, 3551, 3471, 3591, 3561, 3531, 
    3671, 3511, 3741, 3631, 3751, 3640, 3730, 3781, 3771, 
    3701, 3721, 3711,
    3301, 3390, 3381, 3400, 3261, 3291, 3371, 3351, 3331, 
    3311, 3271, 3191, 3321, 3101, 3171, 3471, 3591, 3561, 
    3531, 3671, 3631,
    3200, 3081, 3041, 3061, 3161, 3071, 3111, 3121,
    3021, 3011, 3051,
    3571, 3231, 3421, 3091, 3211, 3181, 3141, 3241, 3361,
    3131, 3601, 3281, 3031, 3501, 3541, 3491, 3611, 3581,
    3461, 3450, 3411, 3221, 3151, 3250, 3341
    ]

td_sub_num_tom = [
    4260, 4430, 4281, 4120, 4390, 4011, 4380, 4581, 4441, 
    4031, 4361, 4421, 4290, 4240, 4271, 4221, 4331, 4480, 
    4560, 4320, 4231, 4541, 4340, 4500, 4451, 4400, 4410, 
    4160, 4110, 4471, 4511, 4351, 4370, 4301, 4531, 4461, 
    4520, 4130, 4491, 4250, 4601, 4610, 4640, 4631, 4550, 
    4570, 4591, 4701, 4660, 4651, 4620, 4671, 4680, 4691, 
    4710, 4721, 4731, 4751,
    4361, 4290, 4240, 4271, 4331, 4340, 4500, 4520, 4130, 
    4491, 4250, 4550, 4570,
    4150, 4181, 4171, 4210, 4091, 4080, 4071, 4141,
    4260, 4430, 4281, 4390, 4380, 4441, 4421, 4331, 4480,
    4320, 4231, 4451, 4400, 4410, 4160, 4110, 4471, 4511,
    4351, 4370, 4301, 4531, 4461
    ]

ROOTS = {
    "ASD": Path(r"F:\ASD projoect\ToM\ASD"),
    "TD": Path(r"F:\ASD projoect\ToM\TD"),
}

SUBJECTS = {
    "ASD": asd_sub_num_tom,
    "TD": td_sub_num_tom,
}

RUN_GROUPS = ["ASD", "TD"]
SKIP_IF_DONE = False
BAD_RATIO = 0.01
MAX_BAD_CHANNELS = 20

# ============================================================
# 2. MARKERS
# ============================================================

tom_markers = {
    31:  {"tom": "cognitive", "order": "first",  "belief": "false"},
    71:  {"tom": "affective", "order": "first",  "belief": "false"},
    131: {"tom": "cognitive", "order": "first",  "belief": "true"},
    171: {"tom": "affective", "order": "first",  "belief": "true"},
    32:  {"tom": "cognitive", "order": "second", "belief": "false"},
    72:  {"tom": "affective", "order": "second", "belief": "false"},
    132: {"tom": "cognitive", "order": "second", "belief": "true"},
    172: {"tom": "affective", "order": "second", "belief": "true"},
}

outcome_map = {
    101: {"outcome": "correct", "correct": True},
    69:  {"outcome": "wrong",   "correct": False},
}

all_codes = list(tom_markers.keys()) + [4] + list(outcome_map.keys())
event_id_true = {str(code): code for code in all_codes}

# ============================================================
# 3. HELPERS
# ============================================================

def load_subject_raw(subject_id, data_path):
    """Load <ID>a_CV.cnt and concatenate optional <ID>b_CV.cnt."""
    file_a = data_path / f"{subject_id}a_CV.cnt"
    file_b = data_path / f"{subject_id}b_CV.cnt"

    if not file_a.exists():
        raise FileNotFoundError(f"a file not found: {file_a}")

    raw_a = mne.io.read_raw_cnt(file_a, preload=True, verbose=False)
    parts = ["a"]

    if file_b.exists():
        raw_b = mne.io.read_raw_cnt(file_b, preload=True, verbose=False)

        if raw_a.ch_names != raw_b.ch_names:
            raise RuntimeError("a/b channel order does not match")
        if raw_a.info["sfreq"] != raw_b.info["sfreq"]:
            raise RuntimeError("a/b sampling rates do not match")

        raw = mne.concatenate_raws([raw_a, raw_b], verbose=False)
        parts.append("b")
    else:
        raw = raw_a

    return raw, "+".join(parts)


def reconstruct_trials(raw):
    events, _ = mne.events_from_annotations(raw, event_id=event_id_true, verbose=False)
    sfreq = raw.info["sfreq"]
    trials = []

    for i, event in enumerate(events):
        start_sample = int(event[0])
        code = int(event[2])

        if code not in tom_markers:
            continue

        audio_end_sample = None
        outcome_sample = None
        outcome_code = None

        for j in range(i + 1, len(events)):
            next_sample = int(events[j, 0])
            next_code = int(events[j, 2])

            if next_code == 4 and audio_end_sample is None:
                audio_end_sample = next_sample
            elif next_code in outcome_map and audio_end_sample is not None:
                outcome_sample = next_sample
                outcome_code = next_code
                break
            elif next_code in tom_markers:
                break

        info = tom_markers[code]

        trials.append({
            "start_sample": start_sample,
            "audio_end_sample": audio_end_sample,
            "outcome_sample": outcome_sample,
            "trigger": code,
            "tom": info["tom"],
            "order": info["order"],
            "belief": info["belief"],
            "outcome_code": outcome_code,
            "correct": outcome_code == 101 if outcome_code is not None else np.nan,
            "audio_duration_s": ((audio_end_sample - start_sample) / sfreq) if audio_end_sample is not None else np.nan,
            "response_time_s": ((outcome_sample - audio_end_sample) / sfreq) if outcome_sample is not None and audio_end_sample is not None else np.nan,
        })

    return pd.DataFrame(trials)


def cut_epochs(raw_proc, trial_df):
    sfreq = raw_proc.info["sfreq"]
    n_pre = int(round(0.2 * sfreq))
    n_post = int(round(2.5 * sfreq))
    expected_n = n_pre + n_post + 1

    hearing = []
    thinking = []
    kept = []

    for idx, row in trial_df.iterrows():
        q_start = int(row["start_sample"])
        audio_end = int(row["audio_end_sample"])

        h_start, h_stop = q_start - n_pre, q_start + n_post + 1
        t_start, t_stop = audio_end - n_pre, audio_end + n_post + 1

        if h_start < 0 or t_start < 0 or h_stop > raw_proc.n_times or t_stop > raw_proc.n_times:
            continue

        h = raw_proc.get_data(start=h_start, stop=h_stop)
        t = raw_proc.get_data(start=t_start, stop=t_stop)

        if h.shape[1] != expected_n or t.shape[1] != expected_n:
            continue

        hearing.append(h)
        thinking.append(t)
        kept.append(idx)

    if len(hearing) == 0:
        raise RuntimeError("No valid ToM epochs after cutting")

    hearing = np.stack(hearing)
    thinking = np.stack(thinking)
    metadata = trial_df.loc[kept].reset_index(drop=True)
    times = np.arange(hearing.shape[2]) / sfreq - 0.2

    return hearing, thinking, times, metadata


def make_epochs(eeg, times, channels, sfreq, metadata):
    ch_types = [
        "eog" if ch in ["HEO", "VEO"]
        else "misc" if ch == "Trigger"
        else "eeg"
        for ch in channels
    ]

    info = mne.create_info(channels, sfreq, ch_types=ch_types)
    epochs = mne.EpochsArray(
        eeg,
        info,
        tmin=float(times[0]),
        metadata=metadata.copy(),
        verbose=False,
    )

    montage = mne.channels.make_standard_montage("colin27_1020")
    epochs.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    return epochs


def run_ica(epochs):
    """ICA in memory only; no intermediate files are saved."""

    for ch in ["HEO", "VEO"]:
        if ch not in epochs.ch_names:
            raise RuntimeError(
                f"Missing required channel: {ch}"
            )

    # --------------------------------------------------------
    # Temporary 1–30 Hz copy ONLY for ICA fitting
    # Use IIR because epochs are relatively short.
    # --------------------------------------------------------

    epochs_ica_fit = epochs.copy().filter(
        l_freq=0.05,
        h_freq=30.0,
        picks=["eeg", "eog"],
        method="iir",
        iir_params=dict(
            order=4,
            ftype="butter"
        ),
        verbose=False,
    )

    # --------------------------------------------------------
    # Fit ICA
    # --------------------------------------------------------

    ica = ICA(
        n_components=0.99,
        random_state=42,
        max_iter="auto"
    )

    ica.fit(
        epochs_ica_fit,
        picks="eeg",
        verbose=False
    )

    # --------------------------------------------------------
    # Detect ocular components
    # --------------------------------------------------------

    heog_idx, _ = ica.find_bads_eog(
        epochs_ica_fit,
        ch_name="HEO",
        verbose=False
    )

    veog_idx, _ = ica.find_bads_eog(
        epochs_ica_fit,
        ch_name="VEO",
        verbose=False
    )

    ica.exclude = sorted(
        set(heog_idx + veog_idx)
    )

    # --------------------------------------------------------
    # Apply ICA to ORIGINAL 0.1–60 Hz epochs
    # --------------------------------------------------------

    cleaned = epochs.copy()

    ica.apply(
        cleaned,
        verbose=False
    )

    # --------------------------------------------------------
    # Baseline correction AFTER ICA
    # --------------------------------------------------------

    cleaned.apply_baseline(
        baseline=(-0.2, 0.0),
        verbose=False
    )

    # --------------------------------------------------------
    # Remove non-EEG channels
    # --------------------------------------------------------

    drop_chs = [
        ch
        for ch in [
            "HEO",
            "VEO",
            "Trigger"
        ]
        if ch in cleaned.ch_names
    ]

    if drop_chs:
        cleaned.drop_channels(
            drop_chs
        )

    return cleaned, ica.exclude


def interpolate_bad_channels(epochs, max_bad=MAX_BAD_CHANNELS, bad_ratio=BAD_RATIO):
    eeg = epochs.get_data()
    channels = epochs.ch_names

    cleaned = []
    good_idx = []
    bad_channels_record = []
    rejected_idx = []

    for i, x in enumerate(eeg):
        abs_x = np.abs(x)
        threshold = abs_x.mean(axis=1) + 3 * abs_x.std(axis=1)
        exceed_ratio = (abs_x > threshold[:, None]).mean(axis=1)
        bad_mask = exceed_ratio > bad_ratio

        bad_chs = [ch for ch, flag in zip(channels, bad_mask) if flag]

        if len(bad_chs) > max_bad:
            rejected_idx.append(i)
            continue

        ep = epochs[i].copy()

        if bad_chs:
            ep.info["bads"] = bad_chs
            ep.interpolate_bads(reset_bads=True, verbose=False)

        cleaned.append(ep.get_data()[0])
        good_idx.append(i)
        bad_channels_record.append(bad_chs)

    if len(cleaned) == 0:
        raise RuntimeError("All epochs rejected during bad-channel cleaning")

    cleaned = np.stack(cleaned)

    final_epochs = mne.EpochsArray(
        cleaned,
        epochs.info.copy(),
        tmin=epochs.tmin,
        metadata=epochs.metadata.iloc[good_idx].reset_index(drop=True),
        verbose=False,
    )

    return final_epochs, good_idx, bad_channels_record, rejected_idx


def save_final_epochs(epochs, path, bad_channels, original_epoch_idx):
    data = {
        "eeg": epochs.get_data(),
        "times": epochs.times,
        "channels": np.array(epochs.ch_names),
        "sfreq": epochs.info["sfreq"],
        "metadata": epochs.metadata.to_dict("list"),
        "bad_channels": np.array(bad_channels, dtype=object),
        "original_epoch_idx": np.array(original_epoch_idx),
    }
    np.save(path, data, allow_pickle=True)


def extract_erp(epochs):
    epochs = epochs.copy()
    epochs.set_eeg_reference("average", projection=False, verbose=False)
    epochs.apply_baseline((-0.2, 0), verbose=False)

    metadata = epochs.metadata
    conditions = {
        "false_cog": (metadata["belief"] == "false") & (metadata["tom"] == "cognitive"),
        "true_cog":  (metadata["belief"] == "true")  & (metadata["tom"] == "cognitive"),
        "false_aff": (metadata["belief"] == "false") & (metadata["tom"] == "affective"),
        "true_aff":  (metadata["belief"] == "true")  & (metadata["tom"] == "affective"),
    }

    posterior = [
        "CPZ", "CP1", "CP2", "CP3", "CP4",
        "PZ", "P1", "P2", "P3", "P4",
        "POZ", "PO3", "PO4",
    ]

    erp_defs = {
        "LPC":      {"channels": ["FZ", "F1", "F2", "FCZ", "FC1", "FC2"], "tmin": 0.300, "tmax": 0.600},
        "early_LSW": {"channels": posterior, "tmin": 0.800, "tmax": 1.400},
        "mid_LSW":   {"channels": posterior, "tmin": 1.400, "tmax": 2.100},
        "late_LSW":  {"channels": posterior, "tmin": 2.000, "tmax": 2.500},
    }

    results = {}

    for component, cfg in erp_defs.items():
        use_chs = [ch for ch in cfg["channels"] if ch in epochs.ch_names]
        if not use_chs:
            raise RuntimeError(f"No channels available for {component}")

        for condition, mask in conditions.items():
            idx = np.where(mask.to_numpy())[0]

            if len(idx) == 0:
                results[f"{component}_{condition}"] = np.nan
                continue

            x = (
                epochs[idx]
                .copy()
                .pick(use_chs)
                .crop(cfg["tmin"], cfg["tmax"])
                .get_data()
                * 1e6
            )

            results[f"{component}_{condition}"] = float(x.mean())

    for component in erp_defs:
        results[f"delta_{component}_cog"] = results[f"{component}_false_cog"] - results[f"{component}_true_cog"]
        results[f"delta_{component}_aff"] = results[f"{component}_false_aff"] - results[f"{component}_true_aff"]

    for condition, mask in conditions.items():
        results[f"n_{condition}"] = int(mask.sum())

    return results


# ============================================================
# 4. PROCESS ONE SUBJECT
# ============================================================

def process_subject(group, subject_id):
    data_path = ROOTS[group]
    output_dir = data_path / f"{subject_id}a_CV_processed_numpy"
    output_dir.mkdir(exist_ok=True)

    final_files = [
        output_dir / "interpolated_hearing_epoch.npy",
        output_dir / "interpolated_thinking_epoch.npy",
        output_dir / "ToM_ERP_hearing_results.npy",
        output_dir / "ToM_ERP_thinking_results.npy",
    ]

    if SKIP_IF_DONE and all(f.exists() for f in final_files):
        return {"group": group, "subject": subject_id, "status": "already_done"}

    raw, parts = load_subject_raw(subject_id, data_path)

    trial_df = reconstruct_trials(raw)
    n_original = len(trial_df)

    trial_df = trial_df.dropna(
        subset=["audio_end_sample", "outcome_sample", "outcome_code"]
    ).reset_index(drop=True)
    n_complete = len(trial_df)

    trial_df = trial_df[trial_df["correct"] == True].reset_index(drop=True)
    n_correct = len(trial_df)

    if n_correct == 0:
        raise RuntimeError("No correct complete trials")

    # Filter
    raw_proc = raw.copy()

    missing = [ch for ch in ["HEO", "VEO"] if ch not in raw_proc.ch_names]
    if missing:
        raise RuntimeError(f"Missing required EOG channels: {missing}")

    ch_types = {"HEO": "eog", "VEO": "eog"}
    if "Trigger" in raw_proc.ch_names:
        ch_types["Trigger"] = "misc"

    raw_proc.set_channel_types(ch_types, verbose=False)
    raw_proc.filter(
        0.1,
        60,
        picks=["eeg", "eog"],
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )

    # Cut hearing / thinking
    hearing_eeg, thinking_eeg, times, metadata = cut_epochs(raw_proc, trial_df)
    channels = list(raw_proc.ch_names)
    sfreq = raw_proc.info["sfreq"]

    hearing_epochs = make_epochs(hearing_eeg, times, channels, sfreq, metadata)
    thinking_epochs = make_epochs(thinking_eeg, times, channels, sfreq, metadata)
    n_cut = len(hearing_epochs)

    # ICA in memory only
    hearing_ica, hearing_removed = run_ica(hearing_epochs)
    thinking_ica, thinking_removed = run_ica(thinking_epochs)

    # Interpolation
    hearing_final, hearing_good_idx, hearing_bad_chs, hearing_rejected = interpolate_bad_channels(hearing_ica)
    thinking_final, thinking_good_idx, thinking_bad_chs, thinking_rejected = interpolate_bad_channels(thinking_ica)

    # Save only final EEG
    save_final_epochs(
        hearing_final,
        output_dir / "interpolated_hearing_epoch.npy",
        hearing_bad_chs,
        hearing_good_idx,
    )

    save_final_epochs(
        thinking_final,
        output_dir / "interpolated_thinking_epoch.npy",
        thinking_bad_chs,
        thinking_good_idx,
    )

    # ERP features
    hearing_erp = extract_erp(hearing_final)
    thinking_erp = extract_erp(thinking_final)

    np.save(output_dir / "ToM_ERP_hearing_results.npy", hearing_erp, allow_pickle=True)
    np.save(output_dir / "ToM_ERP_thinking_results.npy", thinking_erp, allow_pickle=True)

    hearing_avg_bad = float(np.mean([len(x) for x in hearing_bad_chs])) if hearing_bad_chs else 0.0
    thinking_avg_bad = float(np.mean([len(x) for x in thinking_bad_chs])) if thinking_bad_chs else 0.0

    report = {
        "group": group,
        "subject": subject_id,
        "status": "done",
        "raw_parts": parts,
        "original_trials": n_original,
        "complete_trials": n_complete,
        "correct_trials": n_correct,
        "cut_trials": n_cut,
        "hearing_after_ica": len(hearing_ica),
        "hearing_after_interpolation": len(hearing_final),
        "hearing_rejected": len(hearing_rejected),
        "hearing_avg_bad_channels": hearing_avg_bad,
        "hearing_ica_removed": len(hearing_removed),
        "thinking_after_ica": len(thinking_ica),
        "thinking_after_interpolation": len(thinking_final),
        "thinking_rejected": len(thinking_rejected),
        "thinking_avg_bad_channels": thinking_avg_bad,
        "thinking_ica_removed": len(thinking_removed),
    }

    with open(output_dir / "preprocessing_report.txt", "w", encoding="utf-8") as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")

    return report


# ============================================================
# 5. BATCH LOOP
# ============================================================

jobs = [(group, sid) for group in RUN_GROUPS for sid in SUBJECTS[group]]
summary = []

for i, (group, subject_id) in enumerate(jobs, start=1):
    print(f"[{i}/{len(jobs)}] {group} {subject_id}")

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
            "subject": subject_id,
            "status": "skipped",
            "error_type": type(e).__name__,
            "error": str(e),
        })


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

print("\n========== FINAL SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\nStatus counts:")
print(summary_df["status"].value_counts())

summary_path = Path(r"F:\ASD projoect\ToM") / "ToM_batch_preprocessing_summary_3.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSaved batch summary:", summary_path)
