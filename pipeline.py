"""Preprocessing + feature-extraction pipeline for the EEG project.

Two tasks, both from the EEG Motor Movement/Imagery dataset (eegmmidb):

  * task="state":   eyes open  vs  eyes closed   (runs 1, 2; 4 s windows)
  * task="imagery": left fist  vs  right fist    (runs 4, 8, 12; cue-locked
                                                  imagery epochs)

Steps (same for both):
  1. Load the raw EDF recordings, fix channel names, attach a 10-20 montage.
  2. Band-pass filter 0.5-40 Hz.
  3. Segment into 4 s epochs (cue-locked for imagery, fixed windows for state).
  4. Welch PSD per epoch.
  5. Extract relative band powers (delta/theta/alpha/beta/gamma) as:
       - <band>_global: average over a task-specific ROI
       - <band>_asym:   left-minus-right hemispheric asymmetry over mirrored
                        channel pairs inside the ROI
  6. Write data/features_<task>.csv, one row per epoch.

The task-specific ROIs concentrate the informative channels:
  * state uses the posterior ROI (occipital/parietal) where the alpha rhythm
    is strongest;
  * imagery uses the sensorimotor ROI (central/centro-frontal/centro-parietal)
    where mu/beta lateralization is expected.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.integrate import trapezoid

from data_utils import (
    BANDS,
    LOW_FREQ,
    HIGH_FREQ,
    load_subject,
    make_events_and_labels,
    mirror_channel_pairs,
)

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

TASK_CONFIG = {
    "state": {
        "runs": [1, 2],
        "roi_prefixes": ("O", "P", "PO"),
        "class_names": {0: "eyes open (active)", 1: "eyes closed (relaxed)"},
    },
    "imagery": {
        "runs": [4, 8, 12],
        "roi_prefixes": ("C", "CP", "FC"),
        "class_names": {1: "left fist", 2: "right fist"},
    },
}


def fixed_length_epochs(raw, reject_uv):
    """Cut 4 s non-overlapping windows across a continuous recording."""
    events = mne.make_fixed_length_events(raw, start=0, duration=4)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=4, baseline=None, picks="eeg", preload=True
    )
    epochs.drop_bad(reject=dict(eeg=reject_uv * 1e-6), verbose=False)
    return epochs


def cue_epochs(raw, reject_uv):
    """Cut cue-locked epochs around the left/right imagery events."""
    events, event_id = make_events_and_labels(raw)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-0.5, tmax=4.0, baseline=(-0.5, 0.0), picks="eeg", preload=True,
    )
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=reject_uv * 1e-6), verbose=False)
    if len(epochs) < n_before:
        print(f"  Dropped {n_before - len(epochs)} epochs for amplitude artifacts.")
    counts = {int(k): int(v) for k, v in
              zip(*np.unique(epochs.events[:, 2], return_counts=True))}
    print(f"  Kept {len(epochs)} epochs | classes: {counts}")
    return epochs


def extract_features(epochs, pairs, roi_channels):
    """Relative band powers (ROI global + ROI hemispheric asymmetry) per epoch."""
    spectrum = epochs.compute_psd(
        method="welch", fmin=LOW_FREQ, fmax=HIGH_FREQ, n_fft=512, n_overlap=256
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # (epochs, ch, freqs)

    n_epochs, _, _ = psds.shape
    ch_index = {name: i for i, name in enumerate(epochs.ch_names)}

    roi_index = [ch_index[c] for c in roi_channels]
    roi_pairs = [(l, r) for l, r in pairs if l in roi_channels and r in roi_channels]
    left_idx = [ch_index[l] for l, _ in roi_pairs]
    right_idx = [ch_index[r] for _, r in roi_pairs]

    band_freq = {band: (freqs >= lo) & (freqs <= hi) for band, (lo, hi) in BANDS.items()}
    total_power = trapezoid(psds, freqs, axis=2)

    columns = [f"{band}_global" for band in BANDS] + [f"{band}_asym" for band in BANDS]
    features = np.zeros((n_epochs, len(columns)))

    for k, band in enumerate(BANDS):
        band_idx = band_freq[band]
        band_power = trapezoid(psds[:, :, band_idx], freqs[band_idx], axis=2)
        rel = band_power / total_power  # relative power, per epoch & channel

        features[:, k] = rel[:, roi_index].mean(axis=1)  # ROI average

        asym_array = np.divide(
            rel[:, left_idx] - rel[:, right_idx],
            rel[:, left_idx] + rel[:, right_idx],
            out=np.zeros_like(rel[:, left_idx]),
            where=(rel[:, left_idx] + rel[:, right_idx]) != 0,
        )
        features[:, k + len(BANDS)] = asym_array.mean(axis=1)  # hemispheric asymmetry

    return pd.DataFrame(features, columns=columns)


def build_task_dataframe(task, subjects, reject_uv):
    """Return the pooled feature matrix (with subject / label columns) for a task."""
    cfg = TASK_CONFIG[task]
    frames = []
    for subject in subjects:
        raw = load_subject(subject=[subject], runs=cfg["runs"])
        print("  Channels:", len(raw.ch_names), "| sampling rate:", raw.info["sfreq"], "Hz")
        raw.filter(LOW_FREQ, HIGH_FREQ, picks="eeg")

        pairs = mirror_channel_pairs(raw)
        roi_channels = sorted(c for c in raw.ch_names if c.startswith(cfg["roi_prefixes"]))
        print(f"  Mirror pairs: {len(pairs)} | ROI channels: {len(roi_channels)}")

        if task == "state":
            run_epochs = []
            for run_index in range(len(cfg["runs"])):
                seg = raw.copy().crop(tmin=run_index * 60, tmax=(run_index + 1) * 60)
                run_epochs.append(fixed_length_epochs(seg, reject_uv))
            epochs = mne.concatenate_epochs(run_epochs)
            labels = np.concatenate([
                np.zeros(len(run_epochs[0]), dtype=int),
                np.ones(len(run_epochs[1]), dtype=int),
            ])
        else:
            epochs = cue_epochs(raw, reject_uv)
            labels = epochs.events[:, 2]

        X = extract_features(epochs, pairs, roi_channels)
        X["subject"] = subject
        X["label"] = labels
        print("  ->", X.shape, "rows for subject", subject)
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["state", "imagery"], default="state")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 12, 57, 107])
    parser.add_argument("--reject-uv", type=float, default=800.0)
    args = parser.parse_args()

    for directory in (DATA_DIR, FIGURE_DIR, MODEL_DIR):
        directory.mkdir(exist_ok=True)

    print(f"Building feature matrix for task '{args.task}' (subjects {args.subjects}) ...")
    X = build_task_dataframe(args.task, args.subjects, args.reject_uv)

    out_path = DATA_DIR / f"features_{args.task}.csv"
    X.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("  Shape:", X.shape)
    print("  Class counts:", X["label"].value_counts().to_dict())
    print("  Per-subject epochs:", X.groupby("subject").size().to_dict())


if __name__ == "__main__":
    main()