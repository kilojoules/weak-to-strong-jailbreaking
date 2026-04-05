#!/bin/bash
# Setup script for Vast.ai instance
# Usage: ssh -p PORT root@HOST 'bash -s' < setup_remote.sh

set -e

echo "=== Installing dependencies ==="
pip install -q transformers accelerate torch scikit-learn huggingface_hub 2>&1 | tail -3

echo "=== Logging into HuggingFace ==="
# HF token will be passed via env or file
if [ -f /workspace/hf_token ]; then
    huggingface-cli login --token $(cat /workspace/hf_token) 2>&1 | tail -1
fi

echo "=== Creating workspace ==="
mkdir -p /workspace/w2s/data/stealth_hard_s42/rounds
mkdir -p /workspace/w2s/data/stealth_hard_s42/hidden_states
mkdir -p /workspace/w2s/analysis
mkdir -p /workspace/w2s/figures

echo "=== Setup complete ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
