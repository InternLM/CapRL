#!/bin/bash
set -e

echo "=== Step 1: Install key dependencies ==="
pip install torch==2.8.0
pip install numpy==1.26.4
pip install transformers==4.57.1
pip install vllm==0.11.0
pip install deepspeed==0.16.3
pip install ray==2.49.1
pip install accelerate==1.10.1
pip install qwen-vl-utils

echo "=== Step 2: Install flash-attn (requires --no-build-isolation) ==="
pip install flash-attn==2.8.1 --no-build-isolation

echo "=== Step 3: Install OpenRLHF training framework ==="
pip install -e .

echo "=== Setup complete! ==="

