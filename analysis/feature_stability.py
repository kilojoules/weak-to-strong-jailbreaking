"""
Analysis 7: Cross-Round Feature Stability

Track which SAE features are associated with jailbreaks across self-play
rounds. Low turnover = architectural vulnerabilities. High turnover = cat-and-mouse.

Uses hidden states already extracted (from Analysis 1).

Usage:
    python feature_stability.py \
        --hidden_states_dir ../data/stealth_hard_s42/hidden_states \
        --sae_path sae.pt \
        --output_dir ../figures
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, l1_coeff=5.0):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.l1_coeff = l1_coeff

    def encode(self, x):
        return torch.relu(self.encoder(x))


def load_sae(sae_path):
    data = torch.load(sae_path, weights_only=False, map_location="cpu")
    sae = SparseAutoencoder(data["d_model"], data["n_features"], data["l1_coeff"])
    sae.load_state_dict(data["sae_state_dict"])
    sae.eval()
    return sae, data["normalize_scale"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../figures")
    parser.add_argument("--layer_idx", type=int, default=16,
                        help="Which layer's hidden states to use (must exist in extracted data)")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sae, scale = load_sae(args.sae_path)
    print(f"SAE: {sae.encoder.out_features} features, scale={scale:.3f}")

    # Load hidden states and group by round
    hs_dir = Path(args.hidden_states_dir)
    rounds_data = {}

    for pt_file in sorted(hs_dir.glob("hidden_states_round_*.pt")):
        round_num = int(pt_file.stem.split("_")[-1])
        samples = torch.load(pt_file, weights_only=False)

        # Filter to wins only, and pick the closest available layer
        wins = [s for s in samples if s["label"] == 1]
        if not wins:
            print(f"Round {round_num}: no wins, skipping")
            continue

        # Find best available layer
        available_layers = sorted(wins[0]["hidden_states"].keys())
        layer = min(available_layers, key=lambda l: abs(l - args.layer_idx))

        # Encode through SAE
        features_list = []
        for s in wins:
            hs = s["hidden_states"][layer]
            hs_norm = hs / scale
            with torch.no_grad():
                feat = sae.encode(hs_norm.unsqueeze(0)).squeeze(0).numpy()
            features_list.append(feat)

        features = np.array(features_list)
        # Mean activation per feature across wins
        mean_act = features.mean(axis=0)
        top_k_idx = set(np.argsort(mean_act)[-args.top_k:])

        rounds_data[round_num] = {
            "n_wins": len(wins),
            "top_k": top_k_idx,
            "mean_activations": mean_act,
        }
        print(f"Round {round_num}: {len(wins)} wins, top-{args.top_k} features extracted")

    if len(rounds_data) < 2:
        print("Not enough rounds with wins. Exiting.")
        return

    # Compute Jaccard similarity between all round pairs
    round_nums = sorted(rounds_data.keys())
    n_rounds = len(round_nums)
    jaccard_matrix = np.zeros((n_rounds, n_rounds))

    for i, r1 in enumerate(round_nums):
        for j, r2 in enumerate(round_nums):
            s1 = rounds_data[r1]["top_k"]
            s2 = rounds_data[r2]["top_k"]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    # Feature persistence: how many rounds does each feature appear in top-k?
    feature_counts = {}
    for rd in rounds_data.values():
        for f in rd["top_k"]:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    persistent = {f: c for f, c in feature_counts.items() if c >= 0.8 * n_rounds}
    transient = {f: c for f, c in feature_counts.items() if c <= 0.2 * n_rounds}

    # Turnover rate: fraction of top-k that changes between consecutive rounds
    turnover_rates = []
    for i in range(len(round_nums) - 1):
        r1 = round_nums[i]
        r2 = round_nums[i + 1]
        s1 = rounds_data[r1]["top_k"]
        s2 = rounds_data[r2]["top_k"]
        turnover = 1.0 - len(s1 & s2) / len(s1 | s2)
        turnover_rates.append(turnover)

    print(f"\n{'='*60}")
    print("FEATURE STABILITY RESULTS")
    print(f"{'='*60}")
    print(f"Rounds analyzed: {n_rounds}")
    print(f"Top-k: {args.top_k}")
    print(f"Mean consecutive Jaccard: {np.mean([jaccard_matrix[i,i+1] for i in range(n_rounds-1)]):.3f}")
    print(f"Mean consecutive turnover: {np.mean(turnover_rates):.3f}")
    print(f"Persistent features (>80% rounds): {len(persistent)}")
    print(f"Transient features (<20% rounds): {len(transient)}")

    if persistent:
        print(f"\nPersistent feature IDs: {sorted(persistent.keys())}")

    # Save results
    results = {
        "n_rounds": n_rounds,
        "top_k": args.top_k,
        "layer_idx": args.layer_idx,
        "mean_jaccard_consecutive": float(np.mean([jaccard_matrix[i,i+1] for i in range(n_rounds-1)])),
        "mean_turnover": float(np.mean(turnover_rates)),
        "n_persistent": len(persistent),
        "n_transient": len(transient),
        "persistent_features": {str(k): v for k, v in sorted(persistent.items())},
        "turnover_by_round": [float(t) for t in turnover_rates],
        "jaccard_matrix": jaccard_matrix.tolist(),
        "round_nums": round_nums,
    }
    json_path = Path(args.output_dir) / "feature_stability_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Jaccard heatmap
    ax = axes[0]
    im = ax.imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_rounds))
    ax.set_xticklabels(round_nums, fontsize=7)
    ax.set_yticks(range(n_rounds))
    ax.set_yticklabels(round_nums, fontsize=7)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Round", fontsize=12)
    ax.set_title(f"Jaccard Similarity of Top-{args.top_k} Features", fontsize=13)
    plt.colorbar(im, ax=ax)

    # Right: turnover rate over rounds
    ax = axes[1]
    ax.plot(range(len(turnover_rates)), turnover_rates, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel("Round Transition", fontsize=12)
    ax.set_ylabel("Feature Turnover Rate", fontsize=12)
    ax.set_title("Feature Turnover Between Consecutive Rounds", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(turnover_rates), color='red', linestyle='--', alpha=0.5,
               label=f"Mean: {np.mean(turnover_rates):.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(args.output_dir) / "feature_stability_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved to {plot_path}")
    print("DONE")


if __name__ == "__main__":
    main()
