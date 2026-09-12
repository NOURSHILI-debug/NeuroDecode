# NeuroDecode — Explainable AI for EEG Brain-State Classification

A small educational pipeline that turns raw EEG into frequency-band features,
trains a Random Forest, and explains the predictions with SHAP.

**Dataset**: PhysioNet **EEG Motor Movement/Imagery Dataset** (`eegmmidb`, 64
channels, 160 Hz), downloaded automatically by MNE into `~/mne_data`.

## Two tasks

| Task | Classes | Runs used | Epochs | Expected result |
|------|---------|-----------|--------|-----------------|
| `state`   | eyes open ("active") vs eyes closed ("relaxed") | 1, 2 (60 s each) | 4 s fixed windows | ~0.8 accuracy — **alpha blocking** dominates |
| `imagery` | left fist vs right fist (imagined) | 4, 8, 12 | cue-locked imagery | ~chance — an honest negative result |

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
python pipeline.py --task state     # or --task imagery; --subjects 1 12 57 107
python train.py     --task state
python explain.py   --task state
python visualizations.py --task state
```

Default subjects are `1, 12, 57, 107`. Add `--subjects 107` to run on subject
S107 alone.

## Preprocessing (per task)

1. **Load** the EDF recordings and concatenate the three runs.
2. **Fix channel names** — eegmmidb labels have stray dots and lowercase second
   letters (`Fc5.`); names are mapped onto the exact `standard_1005` spelling so
   the 10-20 montage gives every channel a position.
3. **Band-pass filter 0.5–40 Hz** — removes slow drift and high-frequency noise.
4. **Segment into 4 s epochs** — fixed windows for `state`, cue-locked
   (-0.5 to +4 s) for `imagery`.
5. **Artifact rejection** — drop epochs with > 800 µV peak-to-peak (this noisy
   consumer dataset otherwise loses far too much data at stricter cuts).
6. **Welch PSD** per epoch, then **relative band power** per band:

    | Band   | Hz   |
    |--------|-----:|
    | Delta  | 0.5–4 |
    | Theta  | 4–8 |
    | Alpha  | 8–13 |
    | Beta   | 13–30 |
    | Gamma  | 30–40 |

    yielding 10 features per epoch:
    - `<band>_global` — mean relative power over a task-specific ROI
      (posterior O/P/PO for `state`, sensorimotor C/CP/FC for `imagery`);
    - `<band>_asym` — left-minus-right hemispheric asymmetry over mirrored
      channel pairs inside the ROI (captures contralateral mu/beta ERD).

The ROI restriction matters: global averaging over all 64 channels dilutes the
informative signal.

## Machine learning

Random Forest (primary) and Logistic Regression (comparison):

* stratified 70/30 train/test split,
* stratified 4-fold CV,
* **leave-one-subject-out CV** (the honest estimate of cross-subject
  generalization),
* accuracy / precision / recall / F1 + confusion matrix.

Saved artifacts: `models/<task>_random_forest.joblib` and
`figures/confusion_matrix_<task>.png`.

## Explainability (SHAP)

`explain.py` computes TreeSHAP on the trained forest and saves a beeswarm
summary and a global mean-|SHAP| bar per task.

Example claim from the current run (task `state`):

> The model assigned by far the greatest importance to `alpha_global` (mean
> |SHAP| 0.147 vs ~0.05 for the next feature). Eyes-closed relative alpha power
> is ~2.7× the eyes-open value (0.33 vs 0.12). This is the classic posterior
> "alpha blocking" effect and matches the neuroscience. It does **not** establish
> that alpha universally determines alertness — only that this model relies on it.

For `imagery`, SHAP ranks `beta_asym`/`gamma_asym` first but with much smaller
magnitudes, consistent with the classifier being near chance: the simple
frequency features we use capture too little single-trial lateralization to
decode left vs right fist imagery reliably in this dataset. Reporting this
negative result is intentional — it shows the pipeline does **not** invent
signals that are absent.

## Visualizations

`visualizations.py --task <task>` saves:

* `<task>_raw_eeg.png` — a raw trace segment of three channels,
* `<task>_power_spectrum.png` — average PSD per class with the five bands shaded,
* `<task>_feature_distributions.png` — band-power distributions per class.

## Interactive dashboard

```powershell
streamlit run app.py
```

Pick a task and subject in the sidebar, step through epochs, and the app shows
the raw signal, power spectrum, extracted band features, the Random Forest's
predicted brain state, and a per-epoch SHAP explanation. Needs the trained
models (`python train.py --task state && python train.py --task imagery`).

## Reproducing current numbers

```
python pipeline.py --task state
python pipeline.py --task imagery
python train.py --task state && python train.py --task imagery
python explain.py --task state && python explain.py --task imagery
python visualizations.py --task state && python visualizations.py --task imagery
```

## Disclaimer

Educational prototype, not a medical device. Results are illustrative of the
pipeline, not population-level claims. No fabricated EEG data — everything
comes from the real PhysioNet recordings.