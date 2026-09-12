"""Scientific visualizations for each EEG task.

Produces three figures per task (state / imagery):
  1. <task>_raw_eeg.png          -- a short raw EEG segment of a few channels
  2. <task>_power_spectrum.png   -- average PSD per class with band shading
  3. <task>_feature_distributions.png -- relative band powers by class

All figures are written to figures/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

CLASS_NAMES = {
    "state": {0: "eyes open (active)", 1: "eyes closed (relaxed)"},
    "imagery": {1: "left fist", 2: "right fist"},
}
DISPLAY_CHANNELS = {
    "state": ["O1", "O2", "Oz"],
    "imagery": ["C3", "Cz", "C4"],
}
ROI_PREFIXES = {"state": ("O", "P", "PO"), "imagery": ("C", "CP", "FC")}
FIRST_SUBJECT = 1

BAND_COLORS = {
    "delta": "#4C9BE8",
    "theta": "#4CAF50",
    "alpha": "#F44336",
    "beta": "#9C27B0",
    "gamma": "#FF9800",
}


def build_first_subject_epochs(task):
    """Epochs and per-epoch labels for the first subject (pipeline settings)."""
    if task == "state":
        raw = load_subject(subject=[FIRST_SUBJECT], runs=[1, 2])
        run_epochs = []
        for run_index in range(2):
            seg = raw.copy().crop(tmin=run_index * 60, tmax=(run_index + 1) * 60)
            events = mne.make_fixed_length_events(seg, start=0, duration=4)
            run_epochs.append(mne.Epochs(seg, events, tmin=0, tmax=4, baseline=None,
                                         picks="eeg", preload=True))
        epochs = mne.concatenate_epochs(run_epochs)
        labels = np.concatenate([np.zeros(len(run_epochs[0]), dtype=int),
                                 np.ones(len(run_epochs[1]), dtype=int)])
    else:
        raw = load_subject(subject=[FIRST_SUBJECT], runs=[4, 8, 12])
        events, event_id = make_events_and_labels(raw)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4.0,
                            baseline=(-0.5, 0.0), picks="eeg", preload=True)
        labels = epochs.events[:, 2]
    return raw, epochs, labels


def plot_raw_eeg(raw, task):
    channels = [c for c in DISPLAY_CHANNELS[task] if c in raw.ch_names]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    tmax = 6.0
    data = raw.copy().crop(0, tmax).get_data(picks=channels)
    times = np.arange(data.shape[1]) / raw.info["sfreq"]
    offset = 0.0
    for row, ch in zip(data, channels):
        ax.plot(times, row * 1e6 + offset, lw=0.7, label=ch)
        offset -= np.ptp(row) * 1e6 * 1.3
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (\u00b5V, offset)")
    ax.set_title(f"Raw EEG segment ({', '.join(channels)})")
    ax.legend(loc="upper right", ncol=len(channels), fontsize=8)
    fig.tight_layout()
    out = FIGURE_DIR / f"{task}_raw_eeg.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def plot_power_spectrum(raw, epochs, labels, task):
    cfg_roi = ROI_PREFIXES[task]
    roi = sorted(c for c in epochs.ch_names if c.startswith(cfg_roi))
    roi = [c for c in roi if c in epochs.ch_names]

    spectrum = epochs.compute_psd(method="welch", fmin=LOW_FREQ, fmax=HIGH_FREQ,
                                  n_fft=512, n_overlap=256)
    psds, freqs = spectrum.get_data(return_freqs=True)
    ix = [epochs.ch_names.index(c) for c in roi]

    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in sorted(CLASS_NAMES[task]):
        mask = labels == cls
        mean_psd = psds[mask][:, ix].mean(axis=1).mean(axis=0)
        ax.semilogy(freqs, mean_psd, lw=1.6, label=CLASS_NAMES[task][cls])

    for band, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.10, color=BAND_COLORS[band])
        ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim() else 1.0,
                band.capitalize(), ha="center", va="top", fontsize=8,
                color=BAND_COLORS[band])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (\u00b5V\u00b2/Hz, log)")
    ax.set_title(f"Average power spectrum over {roi[0]}..{roi[-1]} "
                 f"(subject {FIRST_SUBJECT})")
    ax.legend()
    ax.set_ylim(bottom=1e-3)
    fig.tight_layout()
    out = FIGURE_DIR / f"{task}_power_spectrum.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def plot_feature_distributions(task):
    df = pd.read_csv(DATA_DIR / f"features_{task}.csv")
    feature_cols = [c for c in df.columns if c not in ("subject", "label")]

    n_classes = len(CLASS_NAMES[task])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, feature in zip(axes.ravel(), feature_cols):
        for cls in sorted(CLASS_NAMES[task]):
            values = df[df["label"] == cls][feature]
            parts = ax.violinplot([values], positions=[cls], widths=0.7,
                                  showmeans=True)
            parts["bodies"][0].set_facecolor(BAND_COLORS[feature.split("_")[0]])
            parts["bodies"][0].set_alpha(0.5)
        ax.set_title(feature, fontsize=9)
        ax.set_xticks(sorted(CLASS_NAMES[task]))
        ax.set_xticklabels([CLASS_NAMES[task][c] for c in sorted(CLASS_NAMES[task])],
                           fontsize=7, rotation=20)
    fig.suptitle(f"Band-power feature distributions by class ({task})", y=1.0)
    fig.tight_layout()
    out = FIGURE_DIR / f"{task}_feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["state", "imagery"], required=True)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(exist_ok=True)

    print(f"Loading subject {FIRST_SUBJECT} for raw + spectrum figures ...")
    raw, epochs, labels = build_first_subject_epochs(args.task)
    raw.filter(LOW_FREQ, HIGH_FREQ, picks="eeg")

    plot_raw_eeg(raw, args.task)
    plot_power_spectrum(raw, epochs, labels, args.task)
    plot_feature_distributions(args.task)


if __name__ == "__main__":
    main()