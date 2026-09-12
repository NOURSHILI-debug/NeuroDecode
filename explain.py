"""SHAP-based explanation of the trained Random Forest.

For each task this:
  1. Loads the saved model and feature matrix.
  2. Computes TreeSHAP values on all epochs.
  3. Saves a summary (beeswarm) plot and a global mean-|SHAP| bar plot,
     and prints the ranked feature importance.

Scientific note: SHAP importance describes *this* model trained on *this*
dataset. It does not prove that a frequency band universally drives a brain
state; it tells us which features the model leaned on.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["state", "imagery"], required=True)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(exist_ok=True)

    bundle = joblib.load(MODEL_DIR / f"{args.task}_random_forest.joblib")
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]

    df = __import__("pandas").read_csv(DATA_DIR / f"features_{args.task}.csv")
    X = df[feature_names]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # shap returns (n_epochs, n_features, n_classes); pick the last class.
    class_keys = sorted(class_names)
    if isinstance(shap_values, list):
        values = shap_values[-1]
        class_key = class_keys[-1]
    else:
        values = shap_values[..., -1]
        class_key = class_keys[-1]
    class_label = class_names[class_key]

    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"shap_summary_{args.task}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/shap_summary_{args.task}.png")

    plt.figure()
    shap.summary_plot(values, X, plot_type="bar", show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"shap_importance_{args.task}.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved figures/shap_importance_{args.task}.png")

    order = np.argsort(np.abs(values).mean(axis=0))[::-1]
    print(f"\nGlobal feature importance (mean |SHAP| for class '{class_label}'):")
    for rank, idx in enumerate(order, start=1):
        name = feature_names[int(idx)]
        print(f"  {rank:2d}. {name:<14} {np.abs(values[:, int(idx)]).mean():.4f}")


if __name__ == "__main__":
    main()