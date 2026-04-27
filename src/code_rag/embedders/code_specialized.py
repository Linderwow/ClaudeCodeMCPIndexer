"""Phase 17 scaffold: pre-wired adapters for code-specialized embedders.

The `LMStudioEmbedder` is generic — it talks to whatever model is loaded
at the configured `model` id. To run an A/B between code-tuned alternatives
(BGE-Code-v1, CodeSage-large-v2, Qwen3-Embedding-4B baseline), we provide
named factory presets here.

Switching is a one-line config change:

    [embedder]
    kind  = "lm_studio"
    model = "bge-code-v1"          # or "codesage-large-v2", or current Qwen

Tonight: this scaffold wires the names and validates the config; you run

    lms get <huggingface-repo>          # download the GGUF
    lms load bge-code-v1                # load the new model
    rm -rf data/{chroma,graph,fts.db}   # mismatched dim → fresh index
    code-rag index                       # re-embed everything

after waking up. The IndexMeta guard refuses to query if the embedder
identity changes mid-flight, so this can never silently mix vector spaces.
"""
from __future__ import annotations

from dataclasses import dataclass

from code_rag.embedders.lm_studio import LMStudioEmbedder


@dataclass(frozen=True)
class CodeEmbedderPreset:
    """Curated entry for an A/B candidate. `lms_id` is the name `lms load`
    reports in `lms ls` — verify with `lms ls` before flipping."""
    name: str                   # short identifier used in config
    lms_id: str                 # exact id LM Studio exposes via /v1/models
    expected_dim: int | None    # None = probe at runtime
    huggingface_repo: str       # for `lms get`
    notes: str


CODE_EMBEDDER_PRESETS: dict[str, CodeEmbedderPreset] = {
    "qwen3-embedding-4b": CodeEmbedderPreset(
        name="qwen3-embedding-4b",
        lms_id="text-embedding-qwen3-embedding-4b",
        expected_dim=2560,
        huggingface_repo="Qwen/Qwen3-Embedding-4B-GGUF",
        notes=(
            "Current baseline. General-purpose multilingual; not code-tuned. "
            "Top-5 MTEB Code among open models. 2.5 GB GGUF."
        ),
    ),
    "bge-code-v1": CodeEmbedderPreset(
        name="bge-code-v1",
        lms_id="bge-code-v1",
        expected_dim=1536,                  # corrected: HF reports 1536, not 1024
        huggingface_repo="BAAI/bge-code-v1",
        notes=(
            "BAAI's code-specialized embedder. Top-3 MTEB Code among free "
            "models as of late 2025. Smaller (~1.3 GB) than the Qwen baseline "
            "and ~2-4 pp better Recall@10 on code-search benchmarks. "
            "GGUF-free — load with `kind=\"sentence_transformers\"`."
        ),
    ),
    "codesage-large-v2": CodeEmbedderPreset(
        name="codesage-large-v2",
        lms_id="codesage-large-v2",
        expected_dim=1024,
        huggingface_repo="codesage/codesage-large-v2",
        notes=(
            "AWS open-released code embedder. Comparable to Voyage Code 3 on "
            "many code benchmarks; fully open weights and runnable locally. "
            "Run `lms get codesage/codesage-large-v2` to download."
        ),
    ),
}


def build_code_embedder(
    preset_name: str,
    *,
    base_url: str = "http://localhost:1234/v1",
    timeout_s: float = 60.0,
    batch: int = 32,
) -> LMStudioEmbedder:
    """Construct an LMStudioEmbedder pointed at one of the curated presets.

    Raises if the preset name is unknown — keeps typos out of the index
    metadata (which would force a wipe to recover).
    """
    if preset_name not in CODE_EMBEDDER_PRESETS:
        names = sorted(CODE_EMBEDDER_PRESETS)
        raise ValueError(
            f"unknown code embedder preset {preset_name!r}; "
            f"choose from {names}"
        )
    preset = CODE_EMBEDDER_PRESETS[preset_name]
    return LMStudioEmbedder(
        base_url=base_url,
        model=preset.lms_id,
        dim=preset.expected_dim or 0,
        timeout_s=timeout_s,
        batch=batch,
    )


def list_presets() -> list[CodeEmbedderPreset]:
    """For CLI / docs: return all known A/B candidates."""
    return list(CODE_EMBEDDER_PRESETS.values())
