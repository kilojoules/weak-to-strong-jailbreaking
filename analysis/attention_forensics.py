"""
Analysis 5: Attention Forensics at Breach Turn

Extract attention weights from all heads at the first compliance token.
Map attention back to prior-turn adversary tokens to see if the 3B adversary
has learned to place "attention hooks" that exploit the 8B's architecture.

Usage:
    python attention_forensics.py \
        --data_dir ../data/stealth_hard_s42/rounds \
        --output_dir ../figures \
        --n_convos 30
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def get_attention_at_generation(model, tokenizer, messages, target_layers=None, device="cuda"):
    """
    Run a forward pass using hooks to capture attention at specific layers only.
    This avoids OOM from storing all 32 layers of attention matrices.
    """
    if target_layers is None:
        target_layers = [28, 31]  # Focus on decision layers (from logit lens)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]

    # Use hooks to capture attention at target layers only
    captured_attns = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # For LlamaAttention, output is (attn_output, attn_weights, past_kv)
            # We need to enable attention output for this to work
            # Instead, compute attention manually from Q, K
            pass
        return hook_fn

    # Simpler approach: run with output_attentions but truncate context
    model.config.output_attentions = True
    with torch.no_grad():
        outputs = model(**inputs)
    model.config.output_attentions = False

    attn_to_last = torch.zeros(seq_len)
    for li in target_layers:
        if li < len(outputs.attentions):
            layer_attn = outputs.attentions[li]
            attn_to_last += layer_attn[0, :, -1, :].mean(dim=0).cpu().float()
    attn_to_last /= len(target_layers)

    # Decode tokens and annotate with speaker
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    # Find turn boundaries by looking for role markers in the tokenized text
    # This is approximate — we look for the chat template markers
    token_roles = []
    current_role = "system"
    for i, tok in enumerate(tokens):
        # Heuristic: detect role switches from chat template
        text_so_far = tokenizer.decode(input_ids[:i+1])
        if "user\n" in text_so_far[-20:]:
            current_role = "user"
        elif "assistant\n" in text_so_far[-25:]:
            current_role = "assistant"
        token_roles.append(current_role)

    return attn_to_last.numpy(), tokens, token_roles


def load_conversations(data_dir, n_wins=30, n_losses=30):
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
    return wins, losses


def analyze_attention(model, tokenizer, conversations, label, device):
    """Analyze attention patterns for a set of conversations."""
    role_attention = {"user": [], "assistant": [], "system": []}

    for i, msgs in enumerate(conversations):
        attn, tokens, roles = get_attention_at_generation(model, tokenizer, msgs, device=device)

        # Aggregate attention by role
        for role in ["user", "assistant", "system"]:
            mask = np.array([1 if r == role else 0 for r in roles])
            role_attn = (attn * mask).sum()
            role_attention[role].append(float(role_attn))

        if (i + 1) % 10 == 0:
            print(f"  {label}: {i+1}/{len(conversations)} done")

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for k, v in role_attention.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../figures")
    parser.add_argument("--n_convos", type=int, default=30)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model, args.device)
    wins, losses = load_conversations(args.data_dir, args.n_convos, args.n_convos)
    print(f"Loaded {len(wins)} wins, {len(losses)} losses")

    print("\nAnalyzing wins...")
    win_attn = analyze_attention(model, tokenizer, wins, "wins", args.device)
    print("Analyzing losses...")
    loss_attn = analyze_attention(model, tokenizer, losses, "losses", args.device)

    # Print results
    print(f"\n{'='*60}")
    print("ATTENTION BY SPEAKER ROLE")
    print(f"{'='*60}")
    for role in ["user", "assistant", "system"]:
        w = win_attn[role]
        l = loss_attn[role]
        print(f"\n{role.upper()} tokens:")
        print(f"  Wins:   {w['mean']:.4f} ± {w['std']:.4f}")
        print(f"  Losses: {l['mean']:.4f} ± {l['std']:.4f}")
        print(f"  Gap:    {w['mean'] - l['mean']:+.4f}")

    # Save
    results = {
        "n_wins": len(wins), "n_losses": len(losses),
        "wins": {k: {"mean": v["mean"], "std": v["std"]} for k, v in win_attn.items()},
        "losses": {k: {"mean": v["mean"], "std": v["std"]} for k, v in loss_attn.items()},
    }
    json_path = Path(args.output_dir) / "attention_forensics_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    roles = ["user", "assistant"]
    x = np.arange(len(roles))
    width = 0.35

    win_vals = [win_attn[r]["mean"] for r in roles]
    loss_vals = [loss_attn[r]["mean"] for r in roles]
    win_errs = [win_attn[r]["std"] / np.sqrt(len(wins)) for r in roles]
    loss_errs = [loss_attn[r]["std"] / np.sqrt(len(losses)) for r in roles]

    ax.bar(x - width/2, win_vals, width, yerr=win_errs, label=f"Wins (n={len(wins)})",
           color="#c0392b", alpha=0.8, capsize=4)
    ax.bar(x + width/2, loss_vals, width, yerr=loss_errs, label=f"Losses (n={len(losses)})",
           color="#2980b9", alpha=0.8, capsize=4)

    ax.set_ylabel("Mean Attention Weight", fontsize=12)
    ax.set_title("Attention to Adversary vs Victim Tokens\nat Generation Point", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Adversary (user)", "Victim (assistant)"], fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = Path(args.output_dir) / "attention_forensics_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved to {plot_path}")
    print("DONE")


if __name__ == "__main__":
    main()
