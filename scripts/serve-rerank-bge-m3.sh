#!/usr/bin/env bash
# vLLM launch for BAAI/bge-reranker-v2-m3 cross-encoder (Phase 60-C).
#
# Why this exists: previously each MCP server (4 concurrent Claude Code
# sessions = 4 MCPs) loaded its own copy of bge-reranker-v2-m3 into the
# 4090's VRAM via sentence-transformers — ~1.5 GB × 4 = 6 GB of pure
# duplication. Plus the watcher loaded one too. Centralizing here:
#
#   - one model load on the GPU (~1.1 GB FP16 weights + buffers)
#   - all MCPs HTTP-call the /v1/rerank endpoint
#   - net savings ≈ 4.5 GB VRAM
#
# vLLM's /v1/rerank endpoint (Jina-style) accepts:
#   {"model":"...", "query":"...", "documents":["...","..."], "top_n":N}
# and returns scored+ranked results. Compatible with code-rag-mcp's
# existing reranker.kind="lm_studio" backend (the name is historical;
# it's just an OpenAI-compatible /v1/rerank client).
#
# Sized small: 0.15 mem util ≈ 3.4 GB ceiling on a 22.5 GB GPU.
# bge-reranker-v2-m3 is 568M params XLM-RoBERTa-base; FP16 weights
# fit in ~1.1 GB. Top-K is small (<=100 pairs) so per-batch buffers
# are tiny. This is plenty.
set -e
# Phase 60 audit fix: explicit CUDA env (see serve-qwen3-embed-8b.sh for why).
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ~/engines/vllm/bin/vllm serve BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 --port 8001 \
  --runner pooling --convert classify \
  --max-model-len 1024 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.15 \
  --served-model-name bge-reranker-v2-m3
