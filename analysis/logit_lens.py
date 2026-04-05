"""
Logit Lens Turn-by-Turn: Track compliance vs refusal probabilities
across layers and turns for multi-turn jailbreak conversations.

Produces a 2D heatmap (layer × turn) showing where/when the safety decision flips.

Usage:
    python logit_lens.py \
        --data_dir ../data/stealth_hard_s42/rounds \
        --output_dir ../figures \
        --n_convos 40 \
        --model meta-llama/Llama-3.1-8B-Instruct

Outputs:
    - logit_lens_heatmap_wins.png
    - logit_lens_heatmap_losses.png
    - logit_lens_results.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Compliance and refusal token sets
COMPLIANCE_STARTERS = ["Sure", "Here", "Okay", "Certainly", "Of", "Step", "To", "The", "In", "Yes"]
REFUSAL_STARTERS = ["Sorry", "I cannot", "I can't", "I'm unable", "I apologize", "Unfortunately", "No"]


def load_model(model_name, device="cuda"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    # Get token IDs for compliance/refusal starters
    compliance_ids = []
    for word in COMPLIANCE_STARTERS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        compliance_ids.extend(ids[:1])  # first token only
    refusal_ids = []
    for word in REFUSAL_STARTERS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        refusal_ids.extend(ids[:1])

    compliance_ids = list(set(compliance_ids))
    refusal_ids = list(set(refusal_ids))
    # Remove overlap
    overlap = set(compliance_ids) & set(refusal_ids)
    compliance_ids = [i for i in compliance_ids if i not in overlap]
    refusal_ids = [i for i in refusal_ids if i not in overlap]

    print(f"Compliance tokens ({len(compliance_ids)}): {[tokenizer.decode([i]) for i in compliance_ids]}")
    print(f"Refusal tokens ({len(refusal_ids)}): {[tokenizer.decode([i]) for i in refusal_ids]}")

    return model, tokenizer, compliance_ids, refusal_ids


def get_logit_lens_at_turn(model, tokenizer, messages_up_to_turn, compliance_ids, refusal_ids, device="cuda"):
    """
    Apply logit lens: at each layer, project hidden state through the unembedding
    matrix and compute P(compliance) - P(refusal) at the last token position.
    """
    text = tokenizer.apply_chat_template(messages_up_to_turn, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 is embedding, 1..n_layers are layer outputs
    n_layers = len(outputs.hidden_states) - 1
    lm_head = model.lm_head  # unembedding matrix

    # Also get the model's norm layer if it exists (for proper logit lens)
    norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm

    results = {}
    for layer_idx in range(0, n_layers, 4):  # every 4th layer
        hs = outputs.hidden_states[layer_idx + 1]  # +1 for embedding offset
        last_token = hs[0, -1, :].unsqueeze(0)  # [1, hidden_dim]

        # Apply RMSNorm if available (important for logit lens accuracy)
        if norm is not None:
            last_token = norm(last_token)

        # Project through unembedding (match dtype to lm_head weights)
        logits = lm_head(last_token.to(lm_head.weight.dtype))[0]  # [vocab_size]
        probs = torch.softmax(logits, dim=0)

        p_comply = sum(probs[tid].item() for tid in compliance_ids)
        p_refuse = sum(probs[tid].item() for tid in refusal_ids)

        results[layer_idx] = {
            "p_comply": p_comply,
            "p_refuse": p_refuse,
            "delta": p_comply - p_refuse,
        }

    # Also do the final layer
    if (n_layers - 1) not in results:
        hs = outputs.hidden_states[n_layers]
        last_token = hs[0, -1, :].unsqueeze(0)
        if norm is not None:
            last_token = norm(last_token)
        logits = lm_head(last_token.to(lm_head.weight.dtype))[0]
        probs = torch.softmax(logits, dim=0)
        p_comply = sum(probs[tid].item() for tid in compliance_ids)
        p_refuse = sum(probs[tid].item() for tid in refusal_ids)
        results[n_layers - 1] = {"p_comply": p_comply, "p_refuse": p_refuse, "delta": p_comply - p_refuse}

    return results


def process_conversation(model, tokenizer, messages, compliance_ids, refusal_ids, device="cuda"):
    """Process all turns of a conversation, returning logit lens at each (layer, turn)."""
    # Group messages into turns (each turn = user + assistant pair)
    turns = []
    current_turn = []
    for msg in messages:
        current_turn.append(msg)
        if msg["role"] == "assistant":
            turns.append(current_turn.copy())
            current_turn = []
    if current_turn:  # trailing user message without response
        turns.append(current_turn.copy())

    n_turns = min(len(turns), 5)
    turn_results = {}

    for turn_idx in range(n_turns):
        # Build context up to just before victim responds at this turn
        context = []
        for t in range(turn_idx):
            context.extend(turns[t])
        # Add the adversary's message for this turn (the user message)
        for msg in turns[turn_idx]:
            if msg["role"] == "user":
                context.append(msg)
                break

        if not context:
            continue

        layer_results = get_logit_lens_at_turn(
            model, tokenizer, context, compliance_ids, refusal_ids, device
        )
        turn_results[turn_idx] = layer_results

    return turn_results


def load_conversations(data_dir, n_convos=40):
    """Load conversations, split into wins and losses."""
    data_dir = Path(data_dir)
    wins = []
    losses = []

    for round_file in sorted(data_dir.glob("round_*.jsonl")):
        with open(round_file) as f:
            for line in f:
                conv = json.loads(line)
                msgs = conv.get("conversation", conv if isinstance(conv, list) else [])
                is_win = conv.get("unsafe", False) or conv.get("win", False)

                if is_win and len(wins) < n_convos:
                    wins.append(msgs)
                elif not is_win and len(losses) < n_convos:
                    losses.append(msgs)

                if len(wins) >= n_convos and len(losses) >= n_convos:
                    break
        if len(wins) >= n_convos and len(losses) >= n_convos:
            break

    print(f"Loaded {len(wins)} wins, {len(losses)} losses")
    return wins, losses


def aggregate_results(all_results):
    """Average logit lens results across conversations."""
    # Collect all (layer, turn) pairs
    layer_turn_deltas = {}
    for conv_results in all_results:
        for turn_idx, layer_results in conv_results.items():
            for layer_idx, metrics in layer_results.items():
                key = (layer_idx, turn_idx)
                if key not in layer_turn_deltas:
                    layer_turn_deltas[key] = []
                layer_turn_deltas[key].append(metrics["delta"])

    # Compute means
    agg = {}
    for (layer, turn), deltas in layer_turn_deltas.items():
        agg[(layer, turn)] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "n": len(deltas),
        }
    return agg


def plot_heatmap(agg, title, output_path):
    """Plot 2D heatmap: layer x turn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(set(k[0] for k in agg.keys()))
    turns = sorted(set(k[1] for k in agg.keys()))

    grid = np.full((len(layers), len(turns)), np.nan)
    for i, layer in enumerate(layers):
        for j, turn in enumerate(turns):
            if (layer, turn) in agg:
                grid[i, j] = agg[(layer, turn)]["mean_delta"]

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", interpolation="nearest")

    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels([f"Turn {t}" for t in turns])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Model Layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(comply) - P(refuse)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../figures")
    parser.add_argument("--n_convos", type=int, default=40)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, compliance_ids, refusal_ids = load_model(args.model, args.device)
    wins, losses = load_conversations(args.data_dir, args.n_convos)

    # Process wins
    print(f"\nProcessing {len(wins)} wins...")
    win_results = []
    for i, msgs in enumerate(wins):
        result = process_conversation(model, tokenizer, msgs, compliance_ids, refusal_ids, args.device)
        win_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(wins)} wins done")

    # Process losses
    print(f"\nProcessing {len(losses)} losses...")
    loss_results = []
    for i, msgs in enumerate(losses):
        result = process_conversation(model, tokenizer, msgs, compliance_ids, refusal_ids, args.device)
        loss_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(losses)} losses done")

    # Aggregate
    win_agg = aggregate_results(win_results)
    loss_agg = aggregate_results(loss_results)

    # Save raw results
    results_path = Path(args.output_dir) / "logit_lens_results.json"
    serializable = {
        "wins": {f"{k[0]}_{k[1]}": v for k, v in win_agg.items()},
        "losses": {f"{k[0]}_{k[1]}": v for k, v in loss_agg.items()},
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_heatmap(win_agg, "Logit Lens: Successful Jailbreaks (Wins)", Path(args.output_dir) / "logit_lens_heatmap_wins.png")
    plot_heatmap(loss_agg, "Logit Lens: Failed Attacks (Losses)", Path(args.output_dir) / "logit_lens_heatmap_losses.png")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    layers = sorted(set(k[0] for k in win_agg.keys()))
    turns = sorted(set(k[1] for k in win_agg.keys()))
    for turn in turns:
        print(f"\nTurn {turn}:")
        for layer in layers:
            w = win_agg.get((layer, turn), {}).get("mean_delta", float("nan"))
            l = loss_agg.get((layer, turn), {}).get("mean_delta", float("nan"))
            print(f"  Layer {layer:2d}: wins={w:+.4f}  losses={l:+.4f}  gap={w-l:+.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
