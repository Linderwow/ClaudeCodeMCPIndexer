# Roadmap

This file tracks the world-class buildout. Phases marked **shipped** are in
`main`; **scaffolded** means the code exists but the integration requires
either a model download or a system-binary install you should supervise.

## Status

| # | Phase | Status | Notes |
|---|---|---|---|
| 13 | Eval harness + telemetry baseline | ✅ shipped | 200 mined queries + 40 manual; Recall/MRR/NDCG@10/p50-99/per-tag; `code-rag eval --baseline` diffs |
| 14 | Per-chunk hash; skip unchanged | ✅ shipped | `data/file_hashes.db`; idempotent re-runs are now ~30 s instead of 4 h |
| 15 | Parallel pipeline | ✅ shipped | `[indexer].parallel_workers=4`; bounded `asyncio.gather` + store-write lock |
| 16 | Tree-sitter tag queries | ✅ shipped | Vendored `.scm` files for C# / TS / JS; runner supports any grammar that ships `tags.scm` |
| 17 | A/B 3 free code embedders | 🛠 scaffolded | Adapters wired (Qwen baseline / BGE-Code-v1 / CodeSage-v2); `lms get` + reindex to flip |
| 18 | SCIP semantic indexing | 🛠 scaffolded | Loader contract + Kuzu mapping; `pip install scip-python` to enable |
| 19 | HyDE + intent classifier | ✅ shipped | Local LM Studio LLM; intent-gated so identifier queries skip the LLM call |
| 20 | ColBERT v2 late-interaction | 🛠 scaffolded | `LateInteractionRetriever` interface + `NullLateInteraction` no-op; `pip install colbert-ai` to enable |
| 21 | Graph-augmented retrieval | ✅ shipped | New MCP tool `search_code_anchored`; BFS radius + score boost |
| 22 | Diff-aware retrieval (git-log) | ✅ shipped | `code-rag git-log-index`; one chunk per (commit × file) diff hunk |
| 23 | Local Prometheus + fsck | ✅ shipped | `code-rag metrics` (OpenMetrics) + `code-rag fsck --fix` |

## To enable each scaffolded phase

### Phase 17 — code-specialized embedder

Pick one (BGE recommended for size/quality balance):

```powershell
lms get BAAI/bge-code-v1            # 1.3 GB
# OR
lms get codesage/codesage-large-v2  # ~2 GB
```

Edit `config.toml`:

```toml
[embedder]
model = "bge-code-v1"   # OR "codesage-large-v2"
```

Wipe + reindex (vector dim changes; existing index is incompatible):

```powershell
rm -rf data/{chroma,fts.db,graph.kz,index_meta.json,file_hashes.db}
code-rag index
```

Then run the eval baseline diff:

```powershell
code-rag eval src/code_rag/eval/fixtures/manual_eval.json `
  --label "after-bge-code-v1" `
  --baseline data/eval/baseline_manual.json `
  --json-out data/eval/after-bge.json
```

Expected lift: +3 to +6 pp Recall@10 over Qwen baseline on natural-language queries.

### Phase 18 — SCIP semantic indexing

```powershell
pip install scip-python                                # protobuf parser
npm install -g @sourcegraph/scip-typescript            # TS indexer
dotnet tool install -g Sourcegraph.Scip.CSharp         # C# indexer
```

Run the indexers per repo (writes `index.scip` files), then enable our loader by removing the `NotImplementedError` guard in `src/code_rag/graph/scip.py:load_scip_index`. Validate against a small repo first.

### Phase 20 — ColBERT v2 late-interaction

```powershell
pip install "colbert-ai[torch,faiss-gpu]"   # ~3-5 GB
```

Then build the ColBERT index alongside Chroma, and enable in `config.toml`:

```toml
[retrieval]
use_colbert = true
colbert_index_dir = "data/colbert"
```

Replace the `NotImplementedError` in `src/code_rag/retrieval/colbert.py:build_colbert_retriever` with the real loader. Quality lift: +5 to +10 pp Recall@10 on long, multi-concept natural-language queries.

## Tier 4 — production hardening (ongoing)

- Periodic `code-rag fsck --fix` from a scheduled task; emit metrics on issue counts
- RSS-watchdog inside the indexer that gracefully restarts if peak memory > 4 GB
- Fault-injection tests for embedder dropouts mid-batch
- `prometheus_client` + Grafana dashboard JSONs
- Cross-language symbol resolution (Python → TypeScript via FFI / API)

These don't fit a single overnight session and don't add user-facing capability — they harden what's already there.
