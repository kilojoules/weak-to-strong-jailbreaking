"""
Extract victim hidden states at multiple layers for the layer sweep analysis.

Replays Turnstile conversations through Llama-3.1-8B-Instruct and saves
hidden states at every 4th layer for each turn of each conversation.

Usage:
    python extract_hidden_states.py \
        --data_dir ../data/stealth_hard_s42/rounds \
        --output_dir ../data/stealth_hard_s42/hidden_states \
        --layers 0 4 8 12 16 20 24 28 31

Outputs:
    One .pt file per round: hidden_states_round_N.pt
    Each contains a list of dicts with keys:
        - round: int
        - conv_idx: int
        - turn: int (0-4)
        - label: int (0=safe, 1=unsafe based on JBB judge)
        - hidden_states: dict[int, tensor]  # layer -> [hidden_dim]
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
    """Load victim model and tokenizer."""
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
    print(f"Model loaded on {device}")
    return model, tokenizer


def extract_turn_hidden_states(model, tokenizer, conversation, turn_idx, layers, device="cuda"):
    """
    Extract hidden states at the end of the victim's context at a given turn.

    For turn N, we feed all messages up to the adversary's message at turn N,
    then extract the hidden state at the last token (where the victim would
    start generating its response).
    """
    # Build context up to just before the victim responds at this turn
    # Conversation alternates: user (adversary), assistant (victim), user, assistant, ...
    # Turn 0 = first exchange, turn 1 = second exchange, etc.
    # We want the state after the adversary's message at turn_idx
    messages = []
    for i, msg in enumerate(conversation):
        # Each turn has a user + assistant pair
        # Turn 0 = messages[0:2], turn 1 = messages[2:4], etc.
        # We want up to and including the user message at turn_idx
        pair_idx = i // 2
        is_user = (i % 2 == 0)

        if pair_idx < turn_idx:
            messages.append(msg)
        elif pair_idx == turn_idx and is_user:
            messages.append(msg)
            break
        elif pair_idx == turn_idx and not is_user:
            # Include prior turns + this user message
            # but we already broke above
            break

    if not messages:
        return None

    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract hidden states at specified layers, last token position
    hidden_states = {}
    for layer_idx in layers:
        if layer_idx < len(outputs.hidden_states):
            # outputs.hidden_states[0] is embedding, [1] is layer 0, etc.
            hs = outputs.hidden_states[layer_idx + 1]  # +1 for embedding offset
            last_token_hs = hs[0, -1, :].cpu().float()  # [hidden_dim]
            hidden_states[layer_idx] = last_token_hs

    return hidden_states


def is_jailbreak_win(conversation):
    """
    Check if conversation is a jailbreak win.
    Uses the 'jbb_label' field if present, otherwise checks for long assistant
    responses (heuristic: >200 chars in last assistant turn suggests compliance).
    """
    # Check for explicit label
    if isinstance(conversation, dict) and "jbb_label" in conversation:
        return conversation["jbb_label"] == 1

    # Heuristic: last assistant message length
    msgs = conversation if isinstance(conversation, list) else conversation.get("conversation", [])
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    if assistant_msgs:
        last = assistant_msgs[-1].get("content", "")
        # Very rough heuristic — will be replaced by actual judge labels
        return len(last) > 300
    return False


def process_round(model, tokenizer, round_file, layers, device="cuda"):
    """Process all conversations in a round file."""
    results = []

    with open(round_file) as f:
        conversations = [json.loads(line) for line in f]

    round_num = int(Path(round_file).stem.split("_")[-1])
    print(f"  Processing round {round_num}: {len(conversations)} conversations")

    for conv_idx, conv_data in enumerate(conversations):
        # Get conversation messages
        if isinstance(conv_data, dict) and "conversation" in conv_data:
            msgs = conv_data["conversation"]
            label = 1 if conv_data.get("unsafe", False) or conv_data.get("win", False) else 0
        elif isinstance(conv_data, list):
            msgs = conv_data
            label = 0  # unknown
        else:
            continue

        # Count turns (each turn = one user+assistant exchange)
        n_turns = len([m for m in msgs if m.get("role") == "user"])

        for turn in range(min(n_turns, 5)):
            hs = extract_turn_hidden_states(model, tokenizer, msgs, turn, layers, device)
            if hs is not None:
                results.append({
                    "round": round_num,
                    "conv_idx": conv_idx,
                    "turn": turn,
                    "label": label,
                    "hidden_states": hs,
                })

        if (conv_idx + 1) % 20 == 0:
            print(f"    {conv_idx + 1}/{len(conversations)} conversations done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states for layer sweep")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with round_N.jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28, 31])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rounds", type=int, nargs="*", default=None, help="Specific rounds to process (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Find round files
    data_dir = Path(args.data_dir)
    round_files = sorted(data_dir.glob("round_*.jsonl"))

    if args.rounds is not None:
        round_files = [f for f in round_files if int(f.stem.split("_")[-1]) in args.rounds]

    print(f"Found {len(round_files)} round files")
    print(f"Extracting layers: {args.layers}")

    for round_file in round_files:
        round_num = int(round_file.stem.split("_")[-1])
        output_file = Path(args.output_dir) / f"hidden_states_round_{round_num}.pt"

        if output_file.exists():
            print(f"  Skipping round {round_num} (already exists)")
            continue

        results = process_round(model, tokenizer, round_file, args.layers, args.device)
        torch.save(results, output_file)
        print(f"  Saved {len(results)} samples to {output_file}")

    print("Done!")


if __name__ == "__main__":
    main()
