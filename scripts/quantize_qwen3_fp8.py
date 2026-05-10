"""FP8_DYNAMIC quantization of Qwen3-Embedding-8B using llm-compressor.

Embedding model (encoder pooling), so we use the dynamic FP8 recipe — no
calibration data needed:
  - per-channel weight quantization
  - per-token dynamic activation quantization
  - skip lm_head (still present on Qwen3-Embedding even though it's pooled,
    quantizing it degrades retrieval quality)

After the save we patch config.json to add `is_matryoshka: true` so the
vLLM `dimensions` API param works without needing --hf-overrides at
serve time (vLLM's HF-overrides pass-through is finicky for embedding
configs — better to bake it into the saved config).

Usage (from inside the vLLM venv):
    python quantize_qwen3_fp8.py SRC DST

Env:
    CUDA_VISIBLE_DEVICES — set to your 4090 (default: 0)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize(src: Path, dst: Path) -> None:
    print(f"[quant] source: {src}", flush=True)
    print(f"[quant] dest:   {dst}", flush=True)

    # Load tokenizer first so we can copy it over later.
    print("[quant] loading tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(str(src))

    print("[quant] loading model (FP16) — this allocates ~16 GB VRAM", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype="auto",
        device_map="auto",
    )

    # FP8_DYNAMIC: per-channel weight (W8) + per-token dynamic activation
    # (A8). No calibration data needed.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    print("[quant] running oneshot quantization (FP8_DYNAMIC)", flush=True)
    oneshot(model=model, recipe=recipe)

    print(f"[quant] saving to {dst}", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), save_compressed=True)
    tok.save_pretrained(str(dst))

    # Patch config.json: add is_matryoshka=true so vLLM accepts the
    # `dimensions` API param without --hf-overrides at serve time.
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["is_matryoshka"] = True
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("[quant] patched config.json: is_matryoshka=true", flush=True)
    else:
        print("[quant] WARNING: no config.json found at dst, can't patch", flush=True)

    print(f"[quant] done. dst contents:", flush=True)
    for f in sorted(dst.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<60} {size_mb:>8.1f} MB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path,
                    help="HF model dir (e.g. ~/data/models/Qwen3-Embedding-8B)")
    ap.add_argument("dst", type=Path,
                    help="output dir (e.g. ~/data/models/Qwen3-Embedding-8B-FP8)")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    if not src.exists():
        print(f"ERROR: source dir not found: {src}", file=sys.stderr)
        return 1
    if dst.exists() and any(dst.iterdir()):
        print(f"ERROR: destination already exists and is non-empty: {dst}",
              file=sys.stderr)
        return 1

    try:
        quantize(src, dst)
    except Exception as e:
        print(f"ERROR: quantization failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
