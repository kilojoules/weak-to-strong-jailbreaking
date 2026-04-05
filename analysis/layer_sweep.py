"""
Layer Sweep: Train probes at every layer and plot AUC vs layer.

This runs LOCALLY on CPU after hidden states have been extracted on GPU.

Usage:
    python layer_sweep.py \
        --hidden_states_dir ../data/stealth_hard_s42/hidden_states \
        --output_dir ../figures

Outputs:
    - layer_sweep_auc.png: AUC vs layer number
    - layer_sweep_results.json: per-layer metrics
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def load_hidden_states(hidden_states_dir):
    """Load all hidden states from .pt files."""
    hs_dir = Path(hidden_states_dir)
    all_samples = []

    for pt_file in sorted(hs_dir.glob("hidden_states_round_*.pt")):
        samples = torch.load(pt_file, weights_only=False)
        all_samples.extend(samples)
        print(f"  Loaded {len(samples)} samples from {pt_file.name}")

    print(f"Total: {len(all_samples)} samples")
    return all_samples


def build_dataset_for_layer(samples, layer_idx):
    """Build X, y arrays for a specific layer."""
    X = []
    y = []

    for s in samples:
        if layer_idx in s["hidden_states"]:
            X.append(s["hidden_states"][layer_idx].numpy())
            y.append(s["label"])

    X = np.array(X)
    y = np.array(y)

    return X, y


def evaluate_probe(X, y, n_splits=5):
    """Train and evaluate a logistic regression probe with stratified CV."""
    if len(np.unique(y)) < 2:
        return {"auc": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n_pos": 0, "n_neg": len(y)}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, f1s, precs, recs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        if len(np.unique(y_test)) >= 2:
            aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))

    return {
        "auc": float(np.mean(aucs)) if aucs else 0.5,
        "auc_std": float(np.std(aucs)) if aucs else 0.0,
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }


def plot_layer_sweep(results, output_dir):
    """Plot AUC vs layer number."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot. Results saved to JSON.")
        return

    layers = sorted(results.keys())
    aucs = [results[l]["auc"] for l in layers]
    stds = [results[l].get("auc_std", 0) for l in layers]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(layers, aucs, yerr=stds, marker="o", capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=14)
    ax.set_title("Jailbreak Detection AUC by Layer", fontsize=16)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate peak
    peak_layer = layers[np.argmax(aucs)]
    peak_auc = max(aucs)
    ax.annotate(
        f"Peak: Layer {peak_layer}\nAUC = {peak_auc:.3f}",
        xy=(peak_layer, peak_auc),
        xytext=(peak_layer + 3, peak_auc - 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=12,
        color="red",
    )

    plt.tight_layout()
    out_path = Path(output_dir) / "layer_sweep_auc.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer sweep: probe AUC vs layer")
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../figures")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading hidden states...")
    samples = load_hidden_states(args.hidden_states_dir)

    # Get available layers
    layers = sorted(samples[0]["hidden_states"].keys())
    print(f"Available layers: {layers}")
    print(f"Label distribution: {sum(s['label'] for s in samples)} positive, "
          f"{sum(1 - s['label'] for s in samples)} negative")

    # Run probe at each layer
    results = {}
    for layer in layers:
        print(f"\nLayer {layer}:")
        X, y = build_dataset_for_layer(samples, layer)
        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        metrics = evaluate_probe(X, y, n_splits=args.n_splits)
        results[layer] = metrics
        print(f"  AUC: {metrics['auc']:.3f} +/- {metrics.get('auc_std', 0):.3f}")
        print(f"  F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")

    # Save results
    json_path = Path(args.output_dir) / "layer_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot
    plot_layer_sweep(results, args.output_dir)

    # Summary
    peak_layer = max(results, key=lambda l: results[l]["auc"])
    print(f"\n{'='*50}")
    print(f"PEAK DETECTION: Layer {peak_layer}, AUC = {results[peak_layer]['auc']:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
