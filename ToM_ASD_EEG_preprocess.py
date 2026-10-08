# %%
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

print("MNE version:", mne.__version__)

asd_sub_num_tom = [3571, 3231, 3421, 3091, 3211, 3181, 3141, 3241, 3361, 3131, 3601, 3281, 3031, 3501, 3541, 3491, 3611, 3581, 3461, 3450, 3411, 3221, 3151, 3250, 3341]
td_sub_num_tom = [4260, 4430, 4281, 4390, 4380, 4441, 4421, 4331, 4480, 4320, 4231, 4451, 4400, 4410, 4160, 4110, 4471, 4511, 4351, 4370, 4301, 4531, 4461]

# %%
# # for no b files
# data_path = r'H:\ASD projoect\ToM\ASD'

# data_name_a = r'3131a_CV.cnt'
# output_dir = os.path.join(data_path, data_name_a.replace('.cnt', '_processed_numpy'))
# os.makedirs(output_dir, exist_ok=True)

# files_a = sorted(glob.glob(os.path.join(data_path, data_name_a)))

# print("A files:")
# for f in files_a:
#     print(f)

# if len(files_a) == 0:
#     raise FileNotFoundError("3071a file not found")

# file_a = files_a[0]
# print(f"\nUsing file A: {file_a}")

# raw = mne.io.read_raw_cnt(
#     files_a[0],
#     preload=True
# )


# %%
data_path = r'H:\ASD projoect\ToM\TD'

data_name_a = r'4210a_CV.cnt'
data_name_b = r'4210b_CV.cnt'

output_dir = os.path.join(data_path, data_name_a.replace('.cnt', '_processed_numpy'))
os.makedirs(output_dir, exist_ok=True)

files_a = sorted(glob.glob(os.path.join(data_path, data_name_a)))
files_b = sorted(glob.glob(os.path.join(data_path, data_name_b)))

print("A files:")
for f in files_a:
    print(f)

print("\nB files:")
for f in files_b:
    print(f)
    
if len(files_a) == 0:
    raise FileNotFoundError("a file not found")

if len(files_b) == 0:
    raise FileNotFoundError("b file not found")

file_a = files_a[0]
file_b = files_b[0]
print(f"\nUsing file A: {file_a}")
print(f"Using file B: {file_b}")

# %%
raw_a = mne.io.read_raw_cnt(
    files_a[0],
    preload=True
)

raw_b = mne.io.read_raw_cnt(
    files_b[0],
    preload=True
)

print(raw_a)
print(raw_b)

# %%
print("A sfreq:", raw_a.info["sfreq"])
print("B sfreq:", raw_b.info["sfreq"])

print("A channels:", len(raw_a.ch_names))
print("B channels:", len(raw_b.ch_names))

print("Same channel order:",
      raw_a.ch_names == raw_b.ch_names)

# %%
raw = mne.concatenate_raws([
    raw_a,
    raw_b
])

print(raw)
print("Combined duration:", raw.times[-1], "seconds")

# %%
print(raw.ch_names)
print("Sampling rate:", raw.info["sfreq"])

descriptions = np.array(raw.annotations.description)

unique_desc, counts = np.unique(
    descriptions,
    return_counts=True
)

annotation_table = pd.DataFrame({
    "annotation": unique_desc,
    "count": counts
})

display(annotation_table)

# %%
for i in range(min(100, len(raw.annotations))):

    onset = raw.annotations.onset[i]
    desc = raw.annotations.description[i]

    print(
        f"{i:03d} | "
        f"{onset:10.3f} s | "
        f"{desc}"
    )

# %% [markdown]
# 31 - cognitive first-order false
# 71 - affective first-order false
# 131 - cognitive first-order true
# 171 - affective first-order true
# 
# 32 - cognitive second-order false
# 72 - affective second-order false
# 132 - cognitive second-order true
# 172 - affective second-order true

# %%
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

audio_end = 4

outcome_map = {
    101: {"outcome": "correct", "correct": True},
    69:  {"outcome": "wrong",   "correct": False},
}


# %%
all_codes = list(tom_markers.keys()) + [4] + list(outcome_map.keys())

event_id_true = {
    str(code): code
    for code in all_codes
}

events, event_id = mne.events_from_annotations(
    raw,
    event_id=event_id_true
)

print(event_id)

# %%
trials = []

for i, event in enumerate(events):

    start_sample = event[0]
    code = event[2]

    if code not in tom_markers:
        continue

    audio_end_sample = None
    outcome_sample = None
    outcome_code = None

    # search forward
    for j in range(i + 1, len(events)):

        next_sample = events[j, 0]
        next_code = events[j, 2]

        # audio ends
        if next_code == 4 and audio_end_sample is None:
            audio_end_sample = next_sample

        # response outcome
        elif next_code in [101, 69] and audio_end_sample is not None:
            outcome_sample = next_sample
            outcome_code = next_code
            break

        # next question starts before outcome -> something is wrong
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
        "correct": (
            outcome_code == 101
            if outcome_code is not None
            else np.nan
        ),

        "audio_duration_s": (
            (audio_end_sample - start_sample) / raw.info["sfreq"]
            if audio_end_sample is not None
            else np.nan
        ),

        "response_time_s": (
            (outcome_sample - audio_end_sample) / raw.info["sfreq"]
            if outcome_sample is not None
            and audio_end_sample is not None
            else np.nan
        )
    })

trial_df = pd.DataFrame(trials)

# %%
display(trial_df.head(20))
print("Total trials:", len(trial_df))

# %%
trial_df = trial_df.dropna(
    subset=["audio_end_sample", "outcome_sample", "outcome_code"]
).reset_index(drop=True)

print("Remaining complete trials:", len(trial_df))

trial_df = trial_df[
    trial_df["correct"] == True
].reset_index(drop=True)

print("Correct trials:", len(trial_df))

# %%
raw_proc = raw.copy()

raw_proc.set_channel_types({
    "HEO": "eog",
    "VEO": "eog",
    "Trigger": "misc"
})

raw_proc.filter(
    0.1,
    60,
    picks=["eeg", "eog"],
    method="iir",
    iir_params=dict(
        order=4,
        ftype="butter"
    )
)

print(raw_proc)

# %%
sfreq = raw_proc.info["sfreq"]

n_pre = int(0.2 * sfreq)
n_post = int(2.5 * sfreq)

hearing_eeg = []
thinking_eeg = []

for _, row in trial_df.iterrows():

    q_start = int(row.start_sample)
    audio_end = int(row.audio_end_sample)

    # ------------------
    # Hearing
    # -0.2 -> +2.5 s
    # ------------------
    start = q_start - n_pre
    stop = q_start + n_post + 1

    x = raw_proc.get_data(
        start=start,
        stop=stop
    )

    x = x - x[:, :n_pre].mean(
        axis=1,
        keepdims=True
    )

    hearing_eeg.append(x)

    # ------------------
    # Thinking
    # -0.2 -> +2.5 s
    # ------------------
    start = audio_end - n_pre
    stop = audio_end + n_post + 1

    x = raw_proc.get_data(
        start=start,
        stop=stop
    )

    x = x - x[:, :n_pre].mean(
        axis=1,
        keepdims=True
    )

    thinking_eeg.append(x)

hearing_eeg = np.stack(hearing_eeg)
thinking_eeg = np.stack(thinking_eeg)

times = (
    np.arange(hearing_eeg.shape[2]) / sfreq
    - 0.2
)

print("Hearing:", hearing_eeg.shape)
print("Thinking:", thinking_eeg.shape)

# %%
common_info = {
    "times": times,
    "channels": np.array(raw_proc.ch_names),
    "sfreq": sfreq,
    "metadata": trial_df.to_dict("list")
}

hearing_data = {
    "eeg": hearing_eeg,
    **common_info
}

thinking_data = {
    "eeg": thinking_eeg,
    **common_info
}

np.save(
    os.path.join(output_dir, "filtered_hearing_epoch.npy"),
    hearing_data,
    allow_pickle=True
)

np.save(
    os.path.join(output_dir, "filtered_thinking_epoch.npy"),
    thinking_data,
    allow_pickle=True
)

print("Saved filtered hearing and thinking epochs.")

# %%
hearing = np.load(
    os.path.join(output_dir, "filtered_hearing_epoch.npy"),
    allow_pickle=True
).item()

thinking = np.load(
    os.path.join(output_dir, "filtered_thinking_epoch.npy"),
    allow_pickle=True
).item()

print(hearing["eeg"].shape)
print(thinking["eeg"].shape)

# %%

def make_epochs(data):

    channels = list(data["channels"])

    ch_types = [
        "eog" if ch in ["HEO", "VEO"]
        else "misc" if ch == "Trigger"
        else "eeg"
        for ch in channels
    ]

    info = mne.create_info(
        channels,
        data["sfreq"],
        ch_types=ch_types
    )

    return mne.EpochsArray(
        data["eeg"],
        info,
        tmin=data["times"][0],
        metadata=pd.DataFrame(data["metadata"])
    )

hearing_epochs = make_epochs(hearing)
thinking_epochs = make_epochs(thinking)

# %%
from mne.preprocessing import ICA

def run_ica(input_file, output_file):

    # --------------------------------------------------
    # 1. Load filtered epochs
    # --------------------------------------------------
    data = np.load(
        os.path.join(output_dir, input_file),
        allow_pickle=True
    ).item()

    print("Loaded:", input_file)
    print("EEG shape:", data["eeg"].shape)

    channels = list(data["channels"])

    # --------------------------------------------------
    # 2. Define channel types
    # --------------------------------------------------
    ch_types = []

    for ch in channels:

        if ch in ["HEO", "VEO"]:
            ch_types.append("eog")

        elif ch == "Trigger":
            ch_types.append("misc")

        else:
            ch_types.append("eeg")

    info = mne.create_info(
        ch_names=channels,
        sfreq=data["sfreq"],
        ch_types=ch_types
    )

    # --------------------------------------------------
    # 3. Reconstruct epochs
    # --------------------------------------------------
    epochs = mne.EpochsArray(
        data["eeg"],
        info,
        tmin=data["times"][0],
        metadata=pd.DataFrame(data["metadata"])
    )

    # --------------------------------------------------
    # 4. Add electrode locations
    #    needed for ICA topography + later interpolation
    # --------------------------------------------------
    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )

    epochs.set_montage(
        montage,
        match_case=False,
        on_missing="ignore"
    )

    # --------------------------------------------------
    # 5. Fit ICA
    # --------------------------------------------------
    ica = ICA(
        n_components=0.99,
        random_state=42,
        max_iter="auto"
    )

    ica.fit(
        epochs,
        picks="eeg"
    )

    print("\nICA fitted:")
    print(ica)

    # --------------------------------------------------
    # 6. Automatically detect HEO/VEO artifacts
    # --------------------------------------------------
    heog_idx = []
    veog_idx = []

    if "HEO" in epochs.ch_names:
        heog_idx, _ = ica.find_bads_eog(
            epochs,
            ch_name="HEO"
        )

    if "VEO" in epochs.ch_names:
        veog_idx, _ = ica.find_bads_eog(
            epochs,
            ch_name="VEO"
        )

    ica.exclude = sorted(
        set(heog_idx + veog_idx)
    )

    print("\nHEO components:", heog_idx)
    print("VEO components:", veog_idx)
    print("Components removed:", ica.exclude)

    # --------------------------------------------------
    # 7. Plot components being removed
    # --------------------------------------------------
    if len(ica.exclude) > 0:

        ica.plot_components(
            picks=ica.exclude
        )

        ica.plot_properties(
            epochs,
            picks=ica.exclude
        )

    else:
        print("No EOG-related ICA components detected.")

    # --------------------------------------------------
    # 8. Apply ICA
    # --------------------------------------------------
    epochs_ica = epochs.copy()

    ica.apply(
        epochs_ica
    )

    # --------------------------------------------------
    # 9. Drop HEO, VEO, Trigger
    # --------------------------------------------------
    drop_channels = [
        ch for ch in ["HEO", "VEO", "Trigger"]
        if ch in epochs_ica.ch_names
    ]

    epochs_ica.drop_channels(
        drop_channels
    )

    # --------------------------------------------------
    # 10. Save ICA-cleaned data
    # --------------------------------------------------
    result = {
        "eeg": epochs_ica.get_data(),
        "times": epochs_ica.times,
        "channels": np.array(
            epochs_ica.ch_names
        ),
        "sfreq": epochs_ica.info["sfreq"],
        "metadata": epochs_ica.metadata.to_dict("list"),

        # keep ICA information
        "ica_exclude": np.array(
            ica.exclude
        ),
        "heog_components": np.array(
            heog_idx
        ),
        "veog_components": np.array(
            veog_idx
        ),
    }

    np.save(
        os.path.join(output_dir, output_file),
        result,
        allow_pickle=True
    )

    print("\nSaved:", os.path.join(output_dir, output_file))
    print("Final EEG shape:", result["eeg"].shape)

    return epochs_ica, ica

# %%
hearing_ica, hearing_ica_model = run_ica(
    "filtered_hearing_epoch.npy",
    "ica_hearing_epoch.npy"
)

# %%
thinking_ica, thinking_ica_model = run_ica(
    "filtered_thinking_epoch.npy",
    "ica_thinking_epoch.npy"
)

# %%

hearing = np.load(
    os.path.join(output_dir, "ica_hearing_epoch.npy"),
    allow_pickle=True
).item()

thinking = np.load(
    os.path.join(output_dir, "ica_thinking_epoch.npy"),
    allow_pickle=True
).item()

print("Hearing:", hearing["eeg"].shape)
print("Thinking:", thinking["eeg"].shape)
print(hearing.keys())

# %%
def interpolate_bad_channels(data, max_bad=20, bad_ratio=0.01):

    eeg = data["eeg"]
    channels = list(data["channels"])
    sfreq = data["sfreq"]

    info = mne.create_info(
        channels,
        sfreq,
        ch_types="eeg"
    )

    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )

    info.set_montage(
        montage,
        match_case=False,
        on_missing="ignore"
    )

    cleaned = []
    good_idx = []
    bad_channels_record = []

    for i, x in enumerate(eeg):

        abs_x = np.abs(x)

        threshold = (
            abs_x.mean(axis=1)
            + 3 * abs_x.std(axis=1)
        )

        # number of samples exceeding threshold
        exceed = (
            abs_x
            > threshold[:, None]
        )

        # proportion of abnormal samples per channel
        exceed_ratio = exceed.mean(axis=1)

        # bad channel only if >1% samples are abnormal
        bad_mask = exceed_ratio > bad_ratio

        bad_chs = [
            ch
            for ch, bad in zip(channels, bad_mask)
            if bad
        ]

        # >20 bad channels -> reject epoch
        if len(bad_chs) > max_bad:
            continue

        epoch = mne.EpochsArray(
            x[np.newaxis, :, :],
            info.copy(),
            tmin=data["times"][0],
            verbose=False
        )

        if bad_chs:
            epoch.info["bads"] = bad_chs

            epoch.interpolate_bads(
                reset_bads=True,
                verbose=False
            )

        cleaned.append(
            epoch.get_data()[0]
        )

        good_idx.append(i)
        bad_channels_record.append(bad_chs)

    cleaned = np.stack(cleaned)

    return cleaned, good_idx, bad_channels_record

# %%
hearing_cleaned, hearing_good_idx, hearing_bad_chs = interpolate_bad_channels(
    hearing,
    max_bad=20,
    bad_ratio=0.01
)

print("Original hearing epochs:", len(hearing["eeg"]))
print("Remaining:", len(hearing_cleaned))
print("Rejected:",
      len(hearing["eeg"]) - len(hearing_cleaned))

# %%
thinking_cleaned, thinking_good_idx, thinking_bad_chs = (
    interpolate_bad_channels(
        thinking,
        max_bad=20,
        bad_ratio=0.01
    )
)

print("Original:", len(thinking["eeg"]))
print("Remaining:", len(thinking_cleaned))
print("Rejected:",
      len(thinking["eeg"]) - len(thinking_cleaned))

# %%
print("Thinking bad channels per epoch:")
print([len(x) for x in thinking_bad_chs[:20]])

print("Hearing bad channels per epoch:")
print([len(x) for x in hearing_bad_chs[:20]])

# %%
# Hearing
hearing_metadata = pd.DataFrame(hearing["metadata"])

hearing_final = {
    "eeg": hearing_cleaned,
    "times": hearing["times"],
    "channels": hearing["channels"],
    "sfreq": hearing["sfreq"],

    "metadata": hearing_metadata.iloc[
        hearing_good_idx
    ].reset_index(drop=True).to_dict("list"),

    "bad_channels": np.array(
        hearing_bad_chs,
        dtype=object
    ),

    "original_epoch_idx": np.array(
        hearing_good_idx
    )
}

np.save(
    os.path.join(output_dir, "interpolated_hearing_epoch.npy"),
    hearing_final,
    allow_pickle=True
)

print("Saved hearing:", hearing_cleaned.shape)

# %%
# Thinking
thinking_metadata = pd.DataFrame(thinking["metadata"])

thinking_final = {
    "eeg": thinking_cleaned,
    "times": thinking["times"],
    "channels": thinking["channels"],
    "sfreq": thinking["sfreq"],

    "metadata": thinking_metadata.iloc[
        thinking_good_idx
    ].reset_index(drop=True).to_dict("list"),

    "bad_channels": np.array(
        thinking_bad_chs,
        dtype=object
    ),

    "original_epoch_idx": np.array(
        thinking_good_idx
    )
}

np.save(
    os.path.join(output_dir, "interpolated_thinking_epoch.npy"),
    thinking_final,
    allow_pickle=True
)

print("Saved thinking:", thinking_cleaned.shape)

# %%
import numpy as np
import pandas as pd
import mne

data = np.load(
    os.path.join(output_dir, "interpolated_hearing_epoch.npy"),
    allow_pickle=True
).item()

metadata = pd.DataFrame(data["metadata"])

print("EEG:", data["eeg"].shape)
print("Time:", data["times"][0], "to", data["times"][-1])
print(metadata.head())

# %%
channels = list(data["channels"])

info = mne.create_info(
    ch_names=channels,
    sfreq=data["sfreq"],
    ch_types="eeg"
)

montage = mne.channels.make_standard_montage(
    "standard_1020"
)

info.set_montage(
    montage,
    match_case=False,
    on_missing="ignore"
)

epochs = mne.EpochsArray(
    data["eeg"],
    info,
    tmin=data["times"][0],
    metadata=metadata,
    verbose=False
)

print(epochs)

# %%
epochs.set_eeg_reference(
    "average",
    projection=False
)

epochs.apply_baseline(
    baseline=(-0.2, 0)
)

# %%
conditions = {
    "false_cognitive":
        (metadata["belief"] == "false") &
        (metadata["tom"] == "cognitive"),

    "false_affective":
        (metadata["belief"] == "false") &
        (metadata["tom"] == "affective"),

    "true_cognitive":
        (metadata["belief"] == "true") &
        (metadata["tom"] == "cognitive"),

    "true_affective":
        (metadata["belief"] == "true") &
        (metadata["tom"] == "affective")
}

for name, mask in conditions.items():
    print(name, mask.sum())

# %%
erp_defs = {
    "LPC": {
        "tmin": 0.300,
        "tmax": 0.600,
        "channels": [
            "FZ", "F1", "F2",
            "FCZ", "FC1", "FC2"
        ]
    },

    "early_LSW": {
        "tmin": 0.800,
        "tmax": 1.400,
        "channels": [
            "CPZ", "CP1", "CP2", "CP3", "CP4",
            "PZ", "P1", "P2", "P3", "P4",
            "POZ", "PO3", "PO4"
        ]
    },

    "mid_LSW": {
        "tmin": 1.400,
        "tmax": 2.100,
        "channels": [
            "CPZ", "CP1", "CP2", "CP3", "CP4",
            "PZ", "P1", "P2", "P3", "P4",
            "POZ", "PO3", "PO4"
        ]
    },

    "late_LSW": {
        "tmin": 2.000,
        "tmax": 2.500,
        "channels": [
            "CPZ", "CP1", "CP2", "CP3", "CP4",
            "PZ", "P1", "P2", "P3", "P4",
            "POZ", "PO3", "PO4"
        ]
    }
}

# %%
results = []

for condition, mask in conditions.items():

    idx = np.where(mask.to_numpy())[0]

    for component, cfg in erp_defs.items():

        use_chs = [
            ch for ch in cfg["channels"]
            if ch in epochs.ch_names
        ]

        ep = (
            epochs[idx]
            .copy()
            .pick(use_chs)
            .crop(
                tmin=cfg["tmin"],
                tmax=cfg["tmax"]
            )
        )

        # trial × channel × time
        x = ep.get_data() * 1e6

        # average over channels and time
        trial_amp = x.mean(axis=(1, 2))

        for j, original_idx in enumerate(idx):

            results.append({
                "epoch_idx": original_idx,
                "condition": condition,
                "component": component,
                "amplitude_uV": trial_amp[j],

                "tom": metadata.iloc[original_idx]["tom"],
                "order": metadata.iloc[original_idx]["order"],
                "belief": metadata.iloc[original_idx]["belief"]
            })

erp_trial_results = pd.DataFrame(results)

erp_trial_results.head()

# %%
erp_summary = (
    erp_trial_results
    .groupby(
        ["condition", "component"]
    )["amplitude_uV"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

erp_summary

# %%
erp_trial_results.to_csv(
    os.path.join(output_dir, "ToM_hearing_ERP_trial_results.csv"),
    index=False
)

erp_summary.to_csv(
    os.path.join(output_dir, "ToM_hearing_ERP_summary.csv"),
    index=False
)

print("ERP results saved.")


