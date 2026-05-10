#!/usr/bin/env bash
# vLLM launch for Qwen3-Embedding-8B FP8 (Phase 60 / Phase 60-B).
# Tuned for the 22 GB 4090.
#
# Phase 60-B: dropped --gpu-memory-utilization 0.80 → 0.50.
# Embed-only workloads don't use autoregressive KV cache (pooling output,
# no generation), so the 80% reservation was just sitting idle. At 0.50,
# vLLM reserves ~11 GB at startup (model weights ~9 GB FP8 + small
# per-batch buffers) — same throughput on our batch sizes, ~8 GB freed
# back to the rest of the GPU. Drop further to 0.40 if you observe
# steady-state vLLM memory <8 GB during typical reindex load.
#
# is_matryoshka was baked into config.json by the quantization step, so
# we don't need --hf-overrides at serve time (vLLM's overrides
# pass-through is finicky for embedding configs).
set -e
# Phase 60 audit fix: explicitly export CUDA env. ~/.bashrc returns early on
# non-interactive shells (`*) return;;`), so when invoked from Task Scheduler
# via `wsl -e bash -c '...'` the inherited env has no CUDA_HOME / nvcc on
# PATH and vLLM falls back to risky lazy-compile paths (or fails outright on
# kernels that need nvcc). Forcing them here makes the serve script
# self-contained regardless of how it was launched.
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ~/engines/vllm/bin/vllm serve ~/data/models/Qwen3-Embedding-8B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --runner pooling --convert embed \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.50 \
  --served-model-name Qwen3-Embedding-8B-FP8
