"""Train and evaluate classifiers for one of the EEG tasks.

Task "state":
    eyes open (0, "active") vs eyes closed (1, "relaxed")
Task "imagery":
    left fist (1) vs right fist (2)

Approach:
  1. Load the matching feature matrix from data/features_<task>.csv.
  2. Stratified train/test split (70/30).
  3. Fit a Random Forest and a Logistic Regression.
  4. Report accuracy, precision, recall, F1 and a confusion matrix per model,
     plus a stratified 4-fold CV and a leave-one-subject-out (LOSO) CV for a
     robust estimate of cross-subject generalization.
  5. Save the Random Forest for SHAP analysis (models/<task>_random_forest.joblib).

All metrics are reported honestly: with a handful of subjects, treat numbers
as illustrative of the pipeline rather than definitive claims about the
population.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

CLASS_NAMES = {
    "state": {0: "eyes open", 1: "eyes closed"},
    "imagery": {1: "left fist", 2: "right fist"},
}


def build_models():
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    lr = LogisticRegression(max_iter=2000, random_state=42)
    return {"Random Forest": rf, "Logistic Regression": lr}


def report_metrics(name, y_true, y_pred, class_names, logger):
    logger(f"{name}: accuracy {accuracy_score(y_true, y_pred):.3f}")
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    for i, cls in enumerate(np.unique(y_true)):
        logger(f"   class '{class_names[cls]}'  precision {prec[i]:.3f}  "
               f"recall {rec[i]:.3f}  F1 {f1[i]:.3f}")
    logger(classification_report(y_true, y_pred,
                                 target_names=list(class_names.values()),
                                 zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["state", "imagery"], required=True)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    class_names = CLASS_NAMES[args.task]

    df = pd.read_csv(DATA_DIR / f"features_{args.task}.csv")
    X = df.drop(columns=["subject", "label"])
    y = df["label"].values
    groups = df["subject"].values
    print(f"Task '{args.task}': {len(df)} epochs, {X.shape[1]} features, "
          f"classes {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = build_models()
    rf = models["Random Forest"]
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n=== Held-out test (30%) | {name} ===")
        report_metrics(name, y_test, y_pred, class_names, print)

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        logo = LeaveOneGroupOut()
        loso = cross_val_score(clf, X, y, cv=logo, groups=groups, scoring="accuracy")
        print(f"  {name}: stratified CV acc {scores.mean():.3f} +/- {scores.std():.3f} | "
              f"LOSO (by subject) acc {loso.mean():.3f} "
              f"(per subject: {np.round(loso, 2).tolist()})")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, rf.predict(X_test),
        display_labels=list(class_names.values()),
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"Random Forest confusion matrix (held-out, n={len(y_test)})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"confusion_matrix_{args.task}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figures/confusion_matrix_{args.task}.png")

    joblib.dump(
        {"model": rf, "feature_names": list(X.columns), "task": args.task,
         "class_names": class_names},
        MODEL_DIR / f"{args.task}_random_forest.joblib",
    )
    print(f"Saved models/{args.task}_random_forest.joblib")


if __name__ == "__main__":
    main()