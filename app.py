"""Interactive Streamlit dashboard for EEG brain-state analysis.

Run with:  streamlit run app.py

Flow per selection:
  1. Load the subject's EEG, filter and cut the chosen epoch.
  2. Show the raw signal, its power spectrum and the extracted band features.
  3. Run the saved Random Forest and show the predicted brain state.
  4. Explain the prediction with SHAP.
"""

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import shap
import streamlit as st
from scipy.integrate import trapezoid

from data_utils import (
    BANDS,
    LOW_FREQ,
    HIGH_FREQ,
    load_subject,
    make_events_and_labels,
    mirror_channel_pairs,
)
from pipeline import TASK_CONFIG, extract_features

MODEL_DIR = "models"

st.set_page_config(page_title="EEG Brain-State Analyzer", layout="wide")

CLASS_NAMES = {"state": {0: "eyes open (active)", 1: "eyes closed (relaxed)"},
               "imagery": {1: "left fist", 2: "right fist"}}
DISPLAY_CHANNELS = {"state": ["O1", "O2", "Oz"],
                    "imagery": ["C3", "Cz", "C4"]}
BAND_COLORS = {"delta": "#4C9BE8", "theta": "#4CAF50", "alpha": "#F44336",
               "beta": "#9C27B0", "gamma": "#FF9800"}
SUBJECTS = [1, 12, 57, 107]

_EXPLAINER_CACHE = {}


@st.cache_resource
def load_model(task):
    bundle = joblib.load(f"{MODEL_DIR}/{task}_random_forest.joblib")
    return bundle


def prepare_raw(task, subject):
    raw = load_subject(subject=[subject], runs=TASK_CONFIG[task]["runs"])
    raw.filter(LOW_FREQ, HIGH_FREQ, picks="eeg")
    return raw


def build_epoch_picker(task, raw):
    """Return (epochs, labels) covering the whole task recording for a subject."""
    if task == "state":
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
        events, event_id = make_events_and_labels(raw)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4.0,
                            baseline=(-0.5, 0.0), picks="eeg", preload=True)
        labels = epochs.events[:, 2]
    fill = dict(eeg=800e-6)
    epochs.drop_bad(reject=fill, verbose=False)
    return epochs, labels


def epoch_features(epochs, pairs, task, i):
    cfg = TASK_CONFIG[task]
    roi = sorted(c for c in epochs.ch_names if c.startswith(cfg["roi_prefixes"]))
    single = epochs[[i]]
    X = extract_features(single, pairs, roi)
    return X, roi


def fig_psd(epochs, i, roi, task):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    single = epochs[[i]]
    sp = single.compute_psd(method="welch", fmin=LOW_FREQ, fmax=HIGH_FREQ,
                            n_fft=512, n_overlap=256)
    psds, freqs = sp.get_data(return_freqs=True)
    ix = [single.ch_names.index(c) for c in roi if c in single.ch_names]
    ax.semilogy(freqs, psds[0, ix].mean(axis=0), lw=1.4, color="#1E88E5")
    for band, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.12, color=BAND_COLORS[band])
        ax.text((lo + hi) / 2, ax.get_ylim()[1], band.capitalize(),
                ha="center", va="top", fontsize=8, color=BAND_COLORS[band])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (\u00b5V\u00b2/Hz, log)")
    ax.set_title("Power spectrum of this epoch")
    ax.set_ylim(bottom=1e-3)
    fig.tight_layout()
    return fig


def fig_raw(raw, task, start_s, stop_s):
    channels = [c for c in DISPLAY_CHANNELS[task] if c in raw.ch_names]
    data = raw.copy().crop(start_s, stop_s).get_data(picks=channels)
    times = np.arange(data.shape[1]) / raw.info["sfreq"] + start_s
    fig, ax = plt.subplots(figsize=(7, 3.4))
    offset = 0.0
    for row, ch in zip(data, channels):
        ax.plot(times, row * 1e6 - offset, lw=0.7, label=ch)
        offset += np.ptp(row) * 1e6 * 1.4
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (\u00b5V)")
    ax.set_title(f"Raw EEG ({', '.join(channels)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_band_features(row, feature_names, shap_abs):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sorted_idx = np.argsort(shap_abs)[::-1]
    colors = ["#F44336" if i == sorted_idx[0] else "#B0BEC5" for i in range(len(feature_names))]
    ax.bar(range(len(feature_names)), row, color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative band power")
    ax.set_title("Extracted frequency-band features")
    fig.tight_layout()
    return fig


def fig_shap(clf, feature_names, epoch_X_row):
    key = id(clf)
    explainer = _EXPLAINER_CACHE.get(key)
    if explainer is None:
        explainer = shap.TreeExplainer(clf)
        _EXPLAINER_CACHE[key] = explainer
    sv = explainer.shap_values(epoch_X_row)[..., -1][0]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = np.argsort(np.abs(sv))[::-1]
    colors = ["#E53935" if v < 0 else "#43A047" for v in sv[order]]
    ax.barh(range(len(order)), sv[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("SHAP value (pushes toward predicted class)")
    ax.set_title("Prediction explanation (SHAP)")
    fig.tight_layout()
    return fig, sv


def main():
    st.sidebar.title("EEG Brain-State Analyzer")
    task = st.sidebar.selectbox("Task", ["state", "imagery"],
                                format_func=lambda t: {
                                    "state": "State (eyes open vs closed)",
                                    "imagery": "Motor imagery (left vs right fist)"}[t])
    subject = st.sidebar.selectbox("Subject", SUBJECTS)

    try:
        bundle = load_model(task)
    except FileNotFoundError:
        st.error(f"Model missing: run `python train.py --task {task}` first.")
        return

    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = {int(k): v for k, v in bundle["class_names"].items()}

    with st.spinner("Loading and processing EEG ..."):
        raw = prepare_raw(task, subject)
        pairs = mirror_channel_pairs(raw)
        epochs, labels = build_epoch_picker(task, raw)

    st.sidebar.write(f"**{len(epochs)}** epochs available for this subject.")
    epoch_index = st.sidebar.slider(
        "Epoch", 0, len(epochs) - 1, 0,
        help="Which 4 s window / trial to analyze.")

    epoch = epochs[epoch_index]
    actual_class = int(labels[epoch_index])
    actual_name = class_names.get(actual_class, str(actual_class))
    start_s = max(0.0, float(epoch.times[0]))
    stop_s = float(epoch.times[-1])

    with st.spinner("Extracting features ..."):
        X, roi = epoch_features(epochs, pairs, task, epoch_index)
        row = X.iloc[0][feature_names].values
        probs = clf.predict_proba(X)[0]
        pred_class = int(clf.classes_[int(np.argmax(probs))])
        pred_name = class_names[pred_class]
        confidence = float(np.max(probs))

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_raw(raw, task, start_s, stop_s))
    with c2:
        st.pyplot(fig_psd(epochs, epoch_index, roi, task))

    icons = {"state": {0: "\u25cb", 1: "\u25cf"},
             "imagery": {1: "\u2190", 2: "\u2192"}}
    st.subheader(f"Prediction: {icons[task].get(pred_class, '')} {pred_name} "
                 f"(confidence {confidence:.0%})")
    st.caption(f"Actual label for this epoch: **{actual_name}**"
               f" — {'correct' if pred_class == actual_class else 'incorrect'}")

    with st.spinner("Explaining ..."):
        exp_row = X.copy()
        explains, sv_row = fig_shap(clf, feature_names, exp_row)
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(fig_band_features(row, feature_names, np.abs(sv_row)))
        with c4:
            st.pyplot(explains)

    st.caption("SHAP explanation quantifies how each frequency-band feature "
               "contributes to *this* prediction for *this* subject.")


if __name__ == "__main__":
    main()