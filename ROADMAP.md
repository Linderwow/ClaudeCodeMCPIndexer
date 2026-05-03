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
| 17 | A/B 3 free code embedders | ✅ shipped | `code-rag embedder switch <preset>` does `lms get` + config edit + wipe automatically. Then `code-rag index` to rebuild. |
| 18 | SCIP semantic indexing | 🛠 scaffolded | Loader contract + Kuzu mapping; `pip install scip-python` to enable |
| 19 | HyDE + intent classifier | ✅ shipped | Local LM Studio LLM; intent-gated so identifier queries skip the LLM call |
| 20 | ColBERT v2 late-interaction | ❌ removed | Scaffold deleted (commit `f03d1b9`) — would need 3-5 GB weights and the cross-encoder reranker covers the same lift on this corpus. |
| 21 | Graph-augmented retrieval | ✅ shipped | New MCP tool `search_code_anchored`; BFS radius + score boost |
| 22 | Diff-aware retrieval (git-log) | ✅ shipped | `code-rag git-log-index`; one chunk per (commit × file) diff hunk |
| 23 | Local Prometheus + fsck | ✅ shipped | `code-rag metrics` (OpenMetrics) + `code-rag fsck --fix` |
| 27 | MMR diversity | ✅ shipped | `[search] mmr_lambda=0.7` between fusion and rerank |
| 28 | Identifier exact-match boost | ✅ shipped | `+50% per matched identifier capped at 3` in `boost_exact_matches` |
| 29 | Cross-encoder reranker | ✅ shipped | sentence-transformers `BAAI/bge-reranker-v2-m3`; sigmoid-normalized scores |
| 30 | Query rewriter (snake/camel/spaced + LLM) | ✅ shipped | Local rewrites always; LLM gated, currently off by config |
| 31 | File-summary chunks | ✅ shipped | One synthetic ToC chunk per file for high-level natural-language queries |
| 32 | Process hygiene + singleton lockfiles | ✅ shipped | `proc_hygiene` + `code-rag reap`; PID-rollover check pending |
| 33 | LM Studio per-model parallel/context | ✅ shipped | `_LMS_LOAD_SETTINGS` map keeps KV cache lean |
| 34 | Sentence-transformers embedder backend | ✅ shipped (dormant) | Backend code lives; active config reverted to LM Studio embedder |
| 35 | Streaming indexer + .tmp-event ignore | ✅ shipped | producer/consumer; ignores `.tmp/.swp/.bak/.#*` |
| 36 | Self-healing infrastructure | ✅ shipped | chroma-heal, lms-enforce, chroma-defrag, reap, MCP retry, store-lock fixes |
| 37-A | Query decomposition (NVIDIA-inspired) | ✅ shipped | Multi-part question splitter via qwen2.5-coder-7b |
| 37-B | Reflection (post-rerank LLM check) | ✅ shipped | Listwise relevance scoring; sigmoid-blended with rerank |
| 37-C | Continuous telemetry | ✅ shipped | `data/eval/history.jsonl` + dashboard `/api/eval-history` |
| 37-D | PDF image OCR (multimodal) | ✅ shipped | Tesseract via pytesseract; opt-in via `setup.ps1 -InstallTesseract` |
| 37-I | Eval-gate on cron + drift detection | ✅ shipped | Daily 02:30 eval-gate, 02:45 median-window drift check |
| 37-J | Dashboard degraded-state alerter | ✅ shipped | 5-min `/api/health` poll; atomic state writes; optional BurntToast |
| 37-K | Recent-files-only + prefer-root MCP filters | ✅ shipped | `recent_files_only_days` + `prefer_root` SearchParams |
| 37-L | Hands-off auto-redeploy | ✅ shipped | Daily 03:00 `git pull --ff-only` + restart watcher/dashboard if HEAD changed |
| 38 | Monster audit fixes | ✅ shipped | Cross-encoder sigmoid; store_lock reassignment; atomic stamp/state writes; ensure_workspace_indexed singleton; chroma-watchdog parent-watch; etc. |

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

### Phase 20 — ColBERT v2 late-interaction (REMOVED)

The Phase 20 scaffold was deleted in commit `f03d1b9` after measuring that
the Phase 29 cross-encoder reranker (BAAI/bge-reranker-v2-m3) closes the
gap on this corpus, and the 3-5 GB weight footprint isn't worth it for a
single-machine setup. If you want late-interaction back, restore from git
history; the loader contract is intentionally orthogonal to the rest of
the retrieval pipeline.

## Tier 4 — production hardening (ongoing)

- Periodic `code-rag fsck --fix` from a scheduled task; emit metrics on issue counts
- RSS-watchdog inside the indexer that gracefully restarts if peak memory > 4 GB
- Fault-injection tests for embedder dropouts mid-batch
- `prometheus_client` + Grafana dashboard JSONs
- Cross-language symbol resolution (Python → TypeScript via FFI / API)

These don't fit a single overnight session and don't add user-facing capability — they harden what's already there.
