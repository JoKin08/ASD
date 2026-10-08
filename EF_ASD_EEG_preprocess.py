# %% [markdown]
# # EF ASD EEG preprocessing and trial cutting
# 
# This notebook reconstructs and preprocesses the **Emotional Face Go/Nogo (EF)** EEG trials described in the thesis.
# 
# ### Thesis-based settings used here
# - Raw format: Neuroscan `.cnt`
# - Sampling rate in the thesis: 1000 Hz
# - Band-pass filter: **0.05–30 Hz**
# - Epoch: **−200 ms to +1000 ms** relative to emotional-face onset
# - Baseline: **−200 to 0 ms**
# - Incorrect behavioral trials are excluded from the ERP analysis
# - EEG epochs exceeding **±100 µV** are rejected
# - Main conditions: **Go/Nogo × Emotion**
# - ERP windows used later in the thesis:
#   - N170: 160–250 ms
#   - N2: 300–400 ms
#   - P3/LPP: 500–700 ms
#   - Late LPP: 700–1000 ms
# 
# ### Trigger definitions supplied for this dataset
# Stimulus triggers:
# - `11` = Go Happy
# - `12` = Go Angry
# - `13` = Go Surprise
# - `14` = Go Neutral
# - `51` = Nogo Happy
# - `52` = Nogo Angry
# - `53` = Nogo Surprise
# - `54` = Nogo Neutral
# 
# Outcome triggers:
# - `101` = correct press
# - `69` = missed press when response was required
# - `44` = wrong press
# - `77` = correct no-response
# 
# Extra triggers `255` and `3` are preserved in the raw event stream but ignored during basic trial reconstruction until their role is confirmed.
# 
# > Important: the thesis used a Neuroscan eye-correction procedure ("mathematically constructed eye model"). This notebook does **not** silently replace that with ICA. The trial cutting and artifact-rejection steps below can be used as-is; ocular correction can be inserted later if needed.
# 

# %% [markdown]
# ## 1. Imports

# %%
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

print("MNE version:", mne.__version__)

asd_sub_num_ef = [1261, 1371, 1441, 1271, 1241, 1301, 1541, 1461, 1501, 1391, 1701, 1411, 1561, 1611, 1631, 1591, 1731, 1601, 1661, 1780, 1331, 1401, 1291, 1220, 1251]
td_sub_num_ef = [2180, 2500, 2231, 2510, 2480, 2441, 2371, 2401, 2470, 2420, 2171, 2331, 2380, 2390, 2310, 2240, 2521, 2531, 2321, 2430, 2291, 2350, 2301]

# %% [markdown]
# ## 2. Set file paths

# %%
# EDIT THESE PATHS

data_path = r'H:\ASD projoect\EF\TD'
data_name = r'2371_RF_CV.cnt'

# Folder used for processed output
output_dir = os.path.join(data_path, data_name.replace('.cnt', '_processed_numpy'))
os.makedirs(output_dir, exist_ok=True)

pattern = os.path.join(data_path, data_name)
files = sorted(glob.glob(pattern))

print("Search pattern:", pattern)
print(f"Found {len(files)} matching CNT file(s):")
for f in files:
    print("  ", f)

if len(files) == 0:
    raise FileNotFoundError(f"No CNT file matched: {pattern}")

# For now use the first matching file.
cnt_file = files[0]
print("\nUsing:", cnt_file)


# %% [markdown]
# ## 3. Read the CNT file

# %%
raw = mne.io.read_raw_cnt(
    cnt_file,
    preload=True
)

print(raw)
print("\nSampling rate:", raw.info["sfreq"])
print("Number of channels:", len(raw.ch_names))
print("Duration (s):", raw.times[-1])


# %% [markdown]
# ## 4. Inspect channel names and annotations

# %%
print("Channels:")
print(raw.ch_names)

print("\nAnnotations:")
print(raw.annotations)


# %% [markdown]
# ## 5. Convert annotations to events using the original trigger numbers
# 
# By default, `mne.events_from_annotations()` may map annotation strings to arbitrary integers.  
# Here we explicitly preserve the experimental trigger values so that `events[:, 2]` contains values such as `11`, `51`, `101`, etc.
# 

# %%
trigger_codes = [
    11, 12, 13, 14,
    51, 52, 53, 54,
    101, 69, 44, 77,
    255, 3
]

event_id_true = {str(code): code for code in trigger_codes}

events, event_id = mne.events_from_annotations(
    raw,
    event_id=event_id_true
)

print("event_id:")
print(event_id)

print("\nFirst 30 events:")
print(events[:30])


# %% [markdown]
# ## 6. Inspect the event sequence

# %%
sfreq = raw.info["sfreq"]

n_show = min(80, len(events))

for k, event in enumerate(events[:n_show]):
    sample = int(event[0])
    code = int(event[2])
    time_s = sample / sfreq
    print(f"{k:03d} | {time_s:10.3f} s | trigger = {code}")


# %% [markdown]
# Before continuing, visually check whether trials look approximately like:
# 
# - `11–14` followed later by `101` or `69`
# - `51–54` followed later by `77` or `44`
# 
# The code below does **not** require the outcome trigger to be immediately adjacent to the stimulus trigger. It searches forward until it finds an outcome trigger or reaches the next stimulus.
# 

# %% [markdown]
# ## 7. Define stimulus and outcome trigger meanings

# %%
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

stim_codes = set(stimulus_map.keys())
outcome_codes = set(outcome_map.keys())


# %% [markdown]
# ## 8. Reconstruct a trial table

# %%
trials = []

for i, event in enumerate(events):
    stim_sample = int(event[0])
    stim_code = int(event[2])

    if stim_code not in stim_codes:
        continue

    outcome_code = None
    outcome_sample = None

    # Search forward until outcome or next stimulus
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

print("Number of reconstructed trials:", len(trial_df))
display(trial_df.head(20))


# %% [markdown]
# ## 9. Behavioral trigger sanity checks

# %%
print("Go/Nogo × outcome trigger:")
display(
    pd.crosstab(
        trial_df["go_nogo"],
        trial_df["outcome_code"],
        margins=True
    )
)

print("\nTrial counts by condition and correctness:")
display(
    trial_df.groupby(
        ["go_nogo", "emotion", "correct"],
        dropna=False
    ).size().rename("n").reset_index()
)

print("\nMissing outcome triggers:")
display(trial_df[trial_df["outcome"] == "missing"])


# %% [markdown]
# ### Optional additional logical check
# 
# Given the supplied trigger meanings, expected combinations are:
# 
# - Go → `101` or `69`
# - Nogo → `77` or `44`
# 
# The following cell flags unexpected pairings.
# 

# %%
def expected_pair(row):
    if row["go_nogo"] == "Go":
        return row["outcome_code"] in [101, 69]
    if row["go_nogo"] == "Nogo":
        return row["outcome_code"] in [77, 44]
    return False

trial_df["expected_pair"] = trial_df.apply(expected_pair, axis=1)

unexpected = trial_df[~trial_df["expected_pair"]]

print("Unexpected stimulus-outcome pairings:", len(unexpected))
display(unexpected)


# %% [markdown]
# ## 10. Keep a copy of the original raw data
# 
# All following preprocessing is performed on a copy so that `raw` remains untouched.
# 

# %%
raw_proc = raw.copy()


# %% [markdown]
# ## 11. Band-pass filter: 0.05–30 Hz
# 
# This matches the frequency range reported for Study 1 in the thesis.
# 
# The exact filter implementation in the thesis was Neuroscan Scan 4.5 with zero-phase filtering and a reported 12 dB slope. MNE's filter implementation is not numerically identical, so this is a **Python/MNE approximation of the reported passband**, not a bit-for-bit reproduction of Scan 4.5.
# 

# %%
raw_proc.filter(
    l_freq=0.05,
    h_freq=30.0,
    picks="eeg"
)

print(raw_proc)


# %% [markdown]
# ## 12. Ocular correction placeholder
# 
# The thesis states that ocular artifacts were minimized using a *mathematically constructed eye model* for each participant in Neuroscan.
# 
# That exact method is not reproduced here because the thesis excerpt does not provide enough implementation detail.
# 
# For now, leave the data unchanged at this step. If you later obtain the original Scan workflow or want to use an alternative such as ICA, insert it here and document that it is a methodological deviation from the thesis.
# 

# %%
# No ocular correction is applied in this notebook yet.
# raw_proc remains unchanged in this cell.

print("Ocular correction: not applied yet.")


# %% [markdown]
# ## 13. Build MNE events for the 8 stimulus conditions
# 
# Each epoch is locked to **face onset**, not to the response trigger.
# 

# %%
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

epoch_events = []

for _, row in trial_df.iterrows():
    code = condition_code[(row["go_nogo"], row["emotion"])]
    epoch_events.append([
        int(row["stim_sample"]),
        0,
        int(code)
    ])

epoch_events = np.asarray(epoch_events, dtype=int)

print("epoch_events shape:", epoch_events.shape)
print(epoch_events[:20])


# %% [markdown]
# ## 14. Create metadata aligned one-to-one with epochs

# %%
metadata_cols = [
    "stim_sample",
    "stim_time_s",
    "stim_code",
    "go_nogo",
    "emotion",
    "outcome_sample",
    "outcome_time_s",
    "outcome_code",
    "outcome",
    "correct",
    "outcome_latency_s",
    "expected_pair",
]

metadata = trial_df[metadata_cols].copy()

display(metadata.head())


# %% [markdown]
# ## 15. Epoch −200 to +1000 ms and baseline-correct
# 
# The thesis reports epochs from **200 ms pre-stimulus to 1000 ms post-stimulus**.  
# Here baseline correction uses **−200 to 0 ms**.
# 

# %%
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
)

print(epochs_all)
print("Number of epochs:", len(epochs_all))


# %% [markdown]
# ## 16. Inspect trial counts before rejection

# %%
print("Counts by MNE condition:")
for cond in epoch_event_id:
    print(f"{cond:15s}: {len(epochs_all[cond])}")

print("\nCorrect / incorrect:")
print(epochs_all.metadata["correct"].value_counts(dropna=False))


# %% [markdown]
# ## 17. Exclude behaviorally incorrect trials
# 
# For the thesis ERP analysis:
# - correct Go (`101`) is retained
# - correct Nogo (`77`) is retained
# - missed Go (`69`) is excluded
# - wrong-press Nogo (`44`) is excluded
# 
# We keep `epochs_all` in memory so error trials remain available for later exploratory work.
# 

# %%
correct_mask = epochs_all.metadata["correct"].to_numpy(dtype=bool)
epochs_correct = epochs_all[correct_mask].copy()

print(epochs_correct)
print("Correct epochs retained:", len(epochs_correct))

display(
    epochs_correct.metadata[
        ["go_nogo", "emotion", "outcome_code", "correct"]
    ].value_counts().rename("n").reset_index()
)


# %%
filtered_epoch_data = {
    "eeg": epochs_correct.get_data(),
    "times": epochs_correct.times,
    "channels": np.array(epochs_correct.ch_names),
    "events": epochs_correct.events,
    "event_id": epochs_correct.event_id,
    "metadata": epochs_correct.metadata.to_dict("list"),
    "sfreq": epochs_correct.info["sfreq"],
}

save_path = os.path.join(output_dir, "filtered_epoch.npy")

np.save(
    save_path,
    filtered_epoch_data,
    allow_pickle=True
)

print("Saved to:", save_path)

# %%
import numpy as np
import mne
from pathlib import Path

load_path = os.path.join(output_dir, "filtered_epoch.npy")

data = np.load(
    load_path,
    allow_pickle=True
).item()

eeg = data["eeg"]
times = data["times"]
channels = list(data["channels"])
events = data["events"]
event_id = data["event_id"]
metadata = data["metadata"]
sfreq = data["sfreq"]

print(eeg.shape)
print(sfreq)

# %%
import pandas as pd

info = mne.create_info(
    ch_names=channels,
    sfreq=sfreq,
    ch_types="eeg"
)

metadata = pd.DataFrame(metadata)

epochs = mne.EpochsArray(
    eeg,
    info,
    events=events,
    event_id=event_id,
    tmin=times[0],
    metadata=metadata
)

print(epochs)

# %%
montage = mne.channels.make_standard_montage(
    "standard_1020"
)

epochs.set_montage(
    montage,
    on_missing="ignore"
)

# %%
epochs_ica_fit = epochs.copy().filter(
    l_freq=1.0,
    h_freq=30.0
)

# %% [markdown]
# ## 18.ICA

# %%
from mne.preprocessing import ICA

# --------------------------------------------------
# 1. Channel types
# --------------------------------------------------

epochs.set_channel_types({
    "HEO": "eog",
    "VEO": "eog",
    "Trigger": "misc"
})

epochs.drop_channels(["Trigger"])

# electrode positions, useful later for interpolation
montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(
    montage,
    match_case=False,
    on_missing="ignore"
)

# --------------------------------------------------
# 2. Fit ICA
# --------------------------------------------------

# Use a 1-Hz high-pass copy only for fitting ICA
epochs_ica_fit = epochs.copy().filter(1., 30.)

ica = ICA(
    n_components=0.99,
    random_state=42,
    max_iter="auto"
)

ica.fit(
    epochs_ica_fit,
    picks="eeg"
)

# --------------------------------------------------
# 3. Automatically detect eye-artifact components
# --------------------------------------------------

heog_inds, _ = ica.find_bads_eog(
    epochs_ica_fit,
    ch_name="HEO"
)

veog_inds, _ = ica.find_bads_eog(
    epochs_ica_fit,
    ch_name="VEO"
)

eog_inds = sorted(
    set(heog_inds + veog_inds)
)

print("ICA components to remove:", eog_inds)

ica.exclude = eog_inds

# --------------------------------------------------
# 4. Apply ICA to original filtered epochs
# --------------------------------------------------

epochs_after_ica = epochs.copy()
ica.apply(epochs_after_ica)

# HEO/VEO no longer needed
epochs_after_ica.drop_channels(["HEO", "VEO"])

print(epochs_after_ica)

# %%
# ica.plot_components(eog_inds)

# %%
ica_epoch_data = {
    "eeg": epochs_after_ica.get_data(),
    "times": epochs_after_ica.times,
    "channels": np.array(epochs_after_ica.ch_names),
    "events": epochs_after_ica.events,
    "event_id": epochs_after_ica.event_id,
    "metadata": epochs_after_ica.metadata.to_dict("list"),
    "sfreq": epochs_after_ica.info["sfreq"],
}

save_path = os.path.join(output_dir, "ica_epoch.npy")

np.save(
    save_path,
    ica_epoch_data,
    allow_pickle=True
)

print("Saved to:", save_path)
print("EEG shape:", ica_epoch_data["eeg"].shape)

# %%
data = np.load(
    os.path.join(output_dir, "ica_epoch.npy"),
    allow_pickle=True
).item()

print(data.keys())
print(data["eeg"].shape)

# %% [markdown]
# ## 19. Reject bad channels and do interpolation
# 
# MNE stores EEG in volts, therefore:
# 
# `100 µV = 100e-6 V`
# 

# %%
import mne
import pandas as pd
import numpy as np

eeg = data["eeg"]
channels = list(data["channels"])
sfreq = data["sfreq"]
events = data["events"]
event_id = data["event_id"]
metadata = pd.DataFrame(data["metadata"])
times = data["times"]

info = mne.create_info(
    ch_names=channels,
    sfreq=sfreq,
    ch_types="eeg"
)

epochs = mne.EpochsArray(
    eeg,
    info,
    events=events,
    event_id=event_id,
    tmin=times[0],
    metadata=metadata
)

# Needed for spatial interpolation
montage = mne.channels.make_standard_montage("standard_1020")

epochs.set_montage(
    montage,
    match_case=False,
    on_missing="ignore"
)

# %%
x = epochs.get_data() * 1e6   # µV

bad_ratio = 0.01

bad_channels_each_epoch = []

for ep in x:

    abs_ep = np.abs(ep)

    threshold = (
        abs_ep.mean(axis=1)
        + 3 * abs_ep.std(axis=1)
    )

    # channel × time
    exceed = (
        abs_ep
        > threshold[:, None]
    )

    # proportion of abnormal samples in each channel
    exceed_ratio = exceed.mean(axis=1)

    # bad only if >1% samples exceed threshold
    bad = exceed_ratio > bad_ratio

    bad_channels_each_epoch.append(
        [
            ch for ch, flag
            in zip(epochs.ch_names, bad)
            if flag
        ]
    )

# %%
n_bad = np.array([
    len(chs)
    for chs in bad_channels_each_epoch
])

print("Total epochs:", len(n_bad))
print("Mean bad channels:", n_bad.mean())
print("Median bad channels:", np.median(n_bad))
print("Maximum bad channels:", n_bad.max())

print("\nFirst 10 epochs:")
for i in range(10):
    print(i, bad_channels_each_epoch[i])

# %%
max_bad_channels = 20

good_epoch_idx = []
bad_epoch_idx = []
repaired_data = []

for i, bad_chs in enumerate(bad_channels_each_epoch):

    # too many bad channels -> drop whole epoch
    if len(bad_chs) > max_bad_channels:
        bad_epoch_idx.append(i)
        continue

    ep = epochs[i].copy()

    if bad_chs:
        ep.info["bads"] = bad_chs
        ep.interpolate_bads(reset_bads=True)

    repaired_data.append(ep.get_data()[0])
    good_epoch_idx.append(i)

repaired_data = np.stack(repaired_data)

print("Original epochs:", len(epochs))
print("Dropped bad epochs:", len(bad_epoch_idx))
print("Remaining epochs:", len(good_epoch_idx))
print("Dropped epoch indices:", bad_epoch_idx)



# %%
# average bad channel number for remaining epochs
remaining_bad_counts = [
    len(bad_channels_each_epoch[i])
    for i in good_epoch_idx
]
print("Average bad channel number for remaining epochs:", np.mean(remaining_bad_counts))

# %%
epochs_interpolated = mne.EpochsArray(
    repaired_data,
    epochs.info.copy(),
    events=epochs.events[good_epoch_idx],
    event_id=epochs.event_id,
    tmin=epochs.tmin,
    metadata=epochs.metadata.iloc[
        good_epoch_idx
    ].reset_index(drop=True)
)

print(epochs_interpolated)

# %%
interpolated_data = {
    "eeg": epochs_interpolated.get_data(),
    "times": epochs_interpolated.times,
    "channels": np.array(epochs_interpolated.ch_names),
    "events": epochs_interpolated.events,
    "event_id": epochs_interpolated.event_id,
    "metadata": epochs_interpolated.metadata.to_dict("list"),
    "sfreq": epochs_interpolated.info["sfreq"],
}

np.save(
    os.path.join(output_dir, "interpolated_epoch.npy"),
    interpolated_data,
    allow_pickle=True
)

# %%
data = np.load(
    os.path.join(output_dir, "interpolated_epoch.npy"),
    allow_pickle=True
).item()

print(data["eeg"].shape)
print(data["metadata"].keys())

# %%
import pandas as pd
import mne

info = mne.create_info(
    ch_names=list(data["channels"]),
    sfreq=data["sfreq"],
    ch_types="eeg"
)

epochs = mne.EpochsArray(
    data["eeg"],
    info,
    events=data["events"],
    event_id=data["event_id"],
    tmin=data["times"][0],
    metadata=pd.DataFrame(data["metadata"])
)

montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, match_case=False, on_missing="ignore")

# %%
epochs.set_eeg_reference("average")

# %%
print(
    epochs.metadata.groupby(
        ["go_nogo", "emotion"]
    ).size()
)

# %%
evoked_go = epochs["Go"].average()
evoked_nogo = epochs["Nogo"].average()

# %%
mne.viz.plot_compare_evokeds(
    {
        "Go": evoked_go,
        "Nogo": evoked_nogo
    },
    picks=["FCZ"]
)

# %%
import numpy as np

# ---------- N2 ----------
n2_channels = ["FCZ"]
n2_tmin, n2_tmax = 0.300, 0.400

# ---------- P3/LPP ----------
p3_channels = ["CZ", "CPZ"]
p3_tmin, p3_tmax = 0.500, 0.700

# N2 mean amplitude
n2_go = evoked_go.copy().crop(
    n2_tmin, n2_tmax
).get_data(picks=n2_channels).mean() * 1e6

n2_nogo = evoked_nogo.copy().crop(
    n2_tmin, n2_tmax
).get_data(picks=n2_channels).mean() * 1e6

# P3/LPP mean amplitude
p3_go = evoked_go.copy().crop(
    p3_tmin, p3_tmax
).get_data(picks=p3_channels).mean() * 1e6

p3_nogo = evoked_nogo.copy().crop(
    p3_tmin, p3_tmax
).get_data(picks=p3_channels).mean() * 1e6

n2_effect = n2_nogo - n2_go
p3_effect = p3_nogo - p3_go

print("N2")
print("Go:", n2_go, "µV")
print("Nogo:", n2_nogo, "µV")
print("Nogo-Go:", n2_effect, "µV")

print("\nP3/LPP")
print("Go:", p3_go, "µV")
print("Nogo:", p3_nogo, "µV")
print("Nogo-Go:", p3_effect, "µV")

# %%
n2_go_crop = evoked_go.copy().crop(n2_tmin, n2_tmax)
n2_nogo_crop = evoked_nogo.copy().crop(n2_tmin, n2_tmax)

n2_go_wave = n2_go_crop.get_data(
    picks=n2_channels
).mean(axis=0)

n2_nogo_wave = n2_nogo_crop.get_data(
    picks=n2_channels
).mean(axis=0)

n2_go_latency = (
    n2_go_crop.times[np.argmin(n2_go_wave)] * 1000
)

n2_nogo_latency = (
    n2_nogo_crop.times[np.argmin(n2_nogo_wave)] * 1000
)

# %%
p3_go_crop = evoked_go.copy().crop(p3_tmin, p3_tmax)
p3_nogo_crop = evoked_nogo.copy().crop(p3_tmin, p3_tmax)

p3_go_wave = p3_go_crop.get_data(
    picks=p3_channels
).mean(axis=0)

p3_nogo_wave = p3_nogo_crop.get_data(
    picks=p3_channels
).mean(axis=0)

p3_go_latency = (
    p3_go_crop.times[np.argmax(p3_go_wave)] * 1000
)

p3_nogo_latency = (
    p3_nogo_crop.times[np.argmax(p3_nogo_wave)] * 1000
)

print("N2 latency:")
print("Go:", n2_go_latency, "ms")
print("Nogo:", n2_nogo_latency, "ms")

print("\nP3 latency:")
print("Go:", p3_go_latency, "ms")
print("Nogo:", p3_nogo_latency, "ms")

# %%
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

    "n_go_trials": len(epochs["Go"]),
    "n_nogo_trials": len(epochs["Nogo"]),
}

save_path = os.path.join(output_dir, "EF_ERP_results.npy")

np.save(
    save_path,
    erp_results,
    allow_pickle=True
)

print("Saved to:", save_path)

# %%
# brief report of numbers of oringinal epoch, average numbers of bad channels for interpolation, and numbers of epochs after ICA and interpolation
print("Number of original epochs:", len(epochs_all))
print("Number of epochs after ICA:", len(epochs_after_ica))
print("Number of epochs after interpolation:", len(epochs_interpolated))
print("Average number of bad channels for interpolation:", np.mean(remaining_bad_counts))

# save the report to a text file
report_path = os.path.join(output_dir, "preprocessing_report.txt")

with open(report_path, "w") as f:
    f.write("Number of original epochs: {}\n".format(len(epochs_all)))
    f.write("Number of epochs after ICA: {}\n".format(len(epochs_after_ica)))
    f.write("Number of epochs after interpolation: {}\n".format(len(epochs_interpolated)))
    f.write("Average number of bad channels for interpolation: {}\n".format(np.mean(remaining_bad_counts)))


