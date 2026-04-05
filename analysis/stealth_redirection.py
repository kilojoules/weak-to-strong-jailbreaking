"""
Analysis 3: Stealth Feature Redirection Map

Compare SAE feature activations between stealth wins and control wins (losses
used as baseline). Identify which features stealth suppresses (probe-visible)
vs amplifies (alternative compliance routes).

Usage:
    python stealth_redirection.py \
        --data_dir ../data/stealth_hard_s42/rounds \
        --sae_path sae.pt \
        --output_dir ../figures \
        --n_wins 50 --n_losses 50

Requires: victim model (Llama-3.1-8B-Instruct) on GPU for hidden state extraction.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class SparseAutoencoder(nn.Module):
    """Same architecture as turnstile/probe.py."""
    def __init__(self, d_model, n_features, l1_coeff=5.0):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.l1_coeff = l1_coeff

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)


def load_sae(sae_path):
    """Load trained SAE."""
    data = torch.load(sae_path, weights_only=False, map_location="cpu")
    sae = SparseAutoencoder(data["d_model"], data["n_features"], data["l1_coeff"])
    sae.load_state_dict(data["sae_state_dict"])
    sae.eval()
    scale = data["normalize_scale"]
    print(f"SAE loaded: {data['n_features']} features, scale={scale:.3f}")
    return sae, scale


def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def extract_hidden_at_breach(model, tokenizer, messages, layer_idx, device="cuda"):
    """Extract hidden state at the last token before victim responds at the breach turn.
    Uses the last turn's user message as the breach point."""
    # Find the last user message
    user_msgs = [(i, m) for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_msgs:
        return None

    # Build context up to last user message
    last_user_idx = user_msgs[-1][0]
    context = messages[:last_user_idx + 1]

    text = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get hidden state at specified layer, last token
    hs = outputs.hidden_states[layer_idx + 1]  # +1 for embedding offset
    return hs[0, -1, :].cpu().float()


def load_conversations(data_dir, n_wins=50, n_losses=50):
    """Load win and loss conversations."""
    wins, losses = [], []
    for rf in sorted(Path(data_dir).glob("round_*.jsonl")):
        with open(rf) as f:
            for line in f:
                conv = json.loads(line)
                msgs = conv.get("conversation", conv if isinstance(conv, list) else [])
                is_win = conv.get("unsafe", False) or conv.get("win", False)
                if is_win and len(wins) < n_wins:
                    wins.append(msgs)
                elif not is_win and len(losses) < n_losses:
                    losses.append(msgs)
                if len(wins) >= n_wins and len(losses) >= n_losses:
                    return wins, losses
    print(f"Loaded {len(wins)} wins, {len(losses)} losses")
    return wins, losses


def get_sae_features(model, tokenizer, sae, scale, conversations, layer_idx, device):
    """Extract SAE feature activations for a set of conversations."""
    all_features = []
    for i, msgs in enumerate(conversations):
        hs = extract_hidden_at_breach(model, tokenizer, msgs, layer_idx, device)
        if hs is not None:
            hs_norm = hs / scale
            with torch.no_grad():
                features = sae.encode(hs_norm).numpy()
            all_features.append(features)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(conversations)} done")
    return np.array(all_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../figures")
    parser.add_argument("--layer_idx", type=int, default=15,
                        help="Layer to extract hidden states from (default: middle)")
    parser.add_argument("--n_wins", type=int, default=50)
    parser.add_argument("--n_losses", type=int, default=50)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SAE
    sae, scale = load_sae(args.sae_path)

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Load conversations
    wins, losses = load_conversations(args.data_dir, args.n_wins, args.n_losses)

    # Extract SAE features
    print(f"\nExtracting SAE features at layer {args.layer_idx}...")
    print("Processing wins...")
    win_features = get_sae_features(model, tokenizer, sae, scale, wins, args.layer_idx, args.device)
    print("Processing losses...")
    loss_features = get_sae_features(model, tokenizer, sae, scale, losses, args.layer_idx, args.device)

    print(f"\nWin features shape: {win_features.shape}")
    print(f"Loss features shape: {loss_features.shape}")

    # Compute difference: wins - losses
    win_mean = win_features.mean(axis=0)
    loss_mean = loss_features.mean(axis=0)
    delta = win_mean - loss_mean

    # Compute per-feature Cohen's d
    win_std = win_features.std(axis=0) + 1e-8
    loss_std = loss_features.std(axis=0) + 1e-8
    pooled_std = np.sqrt((win_std**2 + loss_std**2) / 2)
    cohens_d = delta / pooled_std

    # Top amplified and suppressed features
    n_top = 50
    top_amplified_idx = np.argsort(cohens_d)[-n_top:][::-1]
    top_suppressed_idx = np.argsort(cohens_d)[:n_top]

    print(f"\n{'='*60}")
    print("TOP AMPLIFIED FEATURES (wins > losses)")
    print(f"{'='*60}")
    for i, idx in enumerate(top_amplified_idx[:20]):
        print(f"  Feature {idx:5d}: d={cohens_d[idx]:+.3f}  win_mean={win_mean[idx]:.4f}  loss_mean={loss_mean[idx]:.4f}")

    print(f"\n{'='*60}")
    print("TOP SUPPRESSED FEATURES (wins < losses)")
    print(f"{'='*60}")
    for i, idx in enumerate(top_suppressed_idx[:20]):
        print(f"  Feature {idx:5d}: d={cohens_d[idx]:+.3f}  win_mean={win_mean[idx]:.4f}  loss_mean={loss_mean[idx]:.4f}")

    # Feature activity statistics
    n_active_wins = (win_features > 0).mean(axis=0)  # fraction of wins where feature fires
    n_active_losses = (loss_features > 0).mean(axis=0)

    # Save results
    results = {
        "layer_idx": args.layer_idx,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_features": int(win_features.shape[1]),
        "top_amplified": [
            {"feature_idx": int(idx), "cohens_d": float(cohens_d[idx]),
             "win_mean": float(win_mean[idx]), "loss_mean": float(loss_mean[idx]),
             "win_active_frac": float(n_active_wins[idx]),
             "loss_active_frac": float(n_active_losses[idx])}
            for idx in top_amplified_idx
        ],
        "top_suppressed": [
            {"feature_idx": int(idx), "cohens_d": float(cohens_d[idx]),
             "win_mean": float(win_mean[idx]), "loss_mean": float(loss_mean[idx]),
             "win_active_frac": float(n_active_wins[idx]),
             "loss_active_frac": float(n_active_losses[idx])}
            for idx in top_suppressed_idx
        ],
    }

    json_path = Path(args.output_dir) / "stealth_redirection_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Scatter: win_mean vs loss_mean colored by Cohen's d
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter of all features
    ax = axes[0]
    sc = ax.scatter(loss_mean, win_mean, c=cohens_d, cmap="RdBu_r", s=1, alpha=0.5,
                    vmin=-1, vmax=1)
    ax.plot([0, max(win_mean.max(), loss_mean.max())],
            [0, max(win_mean.max(), loss_mean.max())], 'k--', alpha=0.3)
    ax.set_xlabel("Mean activation (losses)", fontsize=12)
    ax.set_ylabel("Mean activation (wins)", fontsize=12)
    ax.set_title("SAE Feature Activations: Wins vs Losses", fontsize=13)
    plt.colorbar(sc, ax=ax, label="Cohen's d")

    # Right: top features ranked by |d|
    ax = axes[1]
    top_n = 30
    top_idx = np.argsort(np.abs(cohens_d))[-top_n:]
    top_d = cohens_d[top_idx]
    colors = ['#c0392b' if d > 0 else '#2980b9' for d in top_d]
    y_pos = range(len(top_idx))
    ax.barh(y_pos, top_d, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"F{idx}" for idx in top_idx], fontsize=7)
    ax.set_xlabel("Cohen's d (wins - losses)", fontsize=12)
    ax.set_title(f"Top {top_n} Differentially Active Features", fontsize=13)
    ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plot_path = Path(args.output_dir) / "stealth_redirection_map.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # 2. Distribution of Cohen's d values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cohens_d, bins=100, alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='k', linewidth=1)
    ax.axvline(x=np.percentile(cohens_d, 1), color='blue', linestyle='--', alpha=0.5, label=f'1st pct: {np.percentile(cohens_d, 1):.2f}')
    ax.axvline(x=np.percentile(cohens_d, 99), color='red', linestyle='--', alpha=0.5, label=f'99th pct: {np.percentile(cohens_d, 99):.2f}')
    ax.set_xlabel("Cohen's d (wins - losses)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Feature Effect Sizes", fontsize=13)
    ax.legend()
    plt.tight_layout()
    dist_path = Path(args.output_dir) / "stealth_redirection_distribution.png"
    plt.savefig(dist_path, dpi=150)
    print(f"Saved distribution to {dist_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()
