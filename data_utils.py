"""Shared data-loading helpers for the EEG brain-state project.

Uses the EEG Motor Movement/Imagery dataset (PhysioNet eegmmidb), which is
downloaded automatically by MNE and cached in ~/mne_data.
"""

import numpy as np
import mne
from mne.datasets import eegbci
from scipy.spatial import cKDTree

LOW_FREQ = 0.5   # Hz, lower band-pass edge
HIGH_FREQ = 40.0 # Hz, upper band-pass edge (dataset is 160 Hz, Nyquist 80 Hz)

# Standard EEG frequency bands used for feature extraction.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def load_subject(subject=107, runs=(4, 8, 12)):
    """Download (if needed) and concatenate the chosen runs for one subject.

    Runs 4, 8 and 12 are the three Task-2 recordings ("imagine opening and
    closing the left or right fist") for a given subject.

    Returns a preloaded Raw object with a standard 10-20 montage attached.
    """
    fnames = eegbci.load_data(subjects=subject, runs=list(runs))
    raws = [mne.io.read_raw_edf(fname, preload=True) for fname in fnames]
    raw = mne.concatenate_raws(raws)

    # eegmmidb channel names contain stray trailing dots (e.g. "C3..") and MNE
    # stores the EDF labels with a lowercase second letter (e.g. "Fc5").
    raw.rename_channels(lambda name: name.rstrip("."))

    # Normalise channel names to the exact standard_1005 spelling so that the
    # montage (and therefore electrode positions) matches every channel.
    standard_names = set(
        mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    )
    by_lower = {std.lower(): std for std in standard_names}
    mapping = {}
    for ch in raw.ch_names:
        if ch in standard_names:
            continue
        std = by_lower.get(ch.lower())
        if std is not None:
            mapping[ch] = std
    if mapping:
        raw.rename_channels(mapping)

    n_match = sum(1 for ch in raw.ch_names if ch in standard_names)
    print(f"  {n_match}/{len(raw.ch_names)} channels matched to standard_1005")
    raw.set_montage("standard_1005", on_missing="warn")
    return raw


def mirror_channel_pairs(raw):
    """Return a list of (left_ch, right_ch) channel pairs mirrored across the
    midline, derived from the montage electrode positions.

    Left/right hemispheric power asymmetry is what discriminates left- vs
    right-fist motor imagery (contralateral mu/beta suppression), so computing
    band-power asymmetry over symmetric channel pairs is a compact way to
    capture that signal.
    """
    positions = raw.get_montage().get_positions()["ch_pos"]
    names = [c for c in raw.ch_names
             if c in positions and np.isfinite(positions[c]).all()]

    pos = np.array([positions[c] for c in names])
    mirrored = pos * np.array([-1.0, 1.0, 1.0])  # flip the left-right axis

    dist, idx = cKDTree(mirrored).query(pos, k=1)

    pairs = []
    matched = set()
    for i in range(len(names)):
        if i in matched:
            continue
        j = int(idx[i])
        if idx[j] == i:  # mutual nearest neighbours -> symmetric pair
            a, b = names[i], names[j]
            left, right = (a, b) if positions[a][0] < positions[b][0] else (b, a)
            pairs.append((left, right))
            matched.update({i, j})
    return pairs


def make_events_and_labels(raw):
    """Extract left/right imagery events from the EDF annotations.

    Returns an (n_events, 3) events array containing only T1/T2 cues plus the
    {label name: event code} mapping (1 = left fist, 2 = right fist).
    """
    all_events, _ = mne.events_from_annotations(
        raw, event_id={"T0": 0, "T1": 1, "T2": 2}
    )
    events = all_events[np.isin(all_events[:, 2], [1, 2])]
    event_id = {"left": 1, "right": 2}
    return events, event_id