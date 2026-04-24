# code-rag-mcp

Local code + doc RAG, exposed as an MCP server for Claude Code.

**Stack:** Python 3.11 · tree-sitter · Chroma · Kuzu · SQLite FTS5 · LM Studio (qwen3-embedding-4b / qwen3-reranker-4b) · watchdog · MCP stdio.

**Languages indexed:** Python, C#, TypeScript (+ TSX), JavaScript (+ JSX, MJS, CJS). Markdown / PDF / DOCX / HTML / CSS / SCSS as docs.

Private — proprietary. No code leaves the machine.

---

## What you get

| | |
|---|---|
| **Hybrid retrieval** | Vector (Chroma) + BM25 (SQLite FTS5), RRF fusion, cross-encoder rerank (optional) |
| **AST-aware chunks** | Tree-sitter per language; symbol-hierarchical names like `UserService.getById` |
| **Call graph** | Kuzu stores `defines` / `calls` / `imports` edges — `get_callers`, `get_callees` in one hop |
| **Live watcher** | watchdog with 500 ms debounce; incremental reindex per file edit |
| **MCP server** | stdio transport; 7 tools registered for Claude Code |
| **Autostart (Windows)** | windowless `pythonw` via Task Scheduler; survives logon |
| **Eval harness** | Recall@1/3/10, MRR, p50/p95 latency over a JSON fixture |

## MCP tools exposed

| Tool | Purpose |
|---|---|
| `search_code` | Hybrid search (vector + BM25 + rerank) over code + docs |
| `get_chunk_text` | Full text of a chunk by id (after a truncated `search_code` hit) |
| `get_symbol` | Find a symbol by exact name in the graph index |
| `get_callers` | Who calls this symbol — 1-hop reverse call edge |
| `get_callees` | What does this symbol call — 1-hop forward call edge |
| `get_file_context` | A file's symbols + 1-hop graph neighborhood |
| `index_stats` | Metadata + chunk counts + last-updated |

The MCP server is **read-only**. Writes (reindex, doc ingestion) happen via the CLI or the autostart watcher — keeps lock contention on Kuzu's single writer clean.

---

## One-time setup

```powershell
git clone <repo> code-rag-mcp
cd code-rag-mcp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Copy the example config and point it at YOUR codebases:
copy config.example.toml config.toml
#   → edit [paths].roots = ["C:/path/to/your/repo", ...]
```

Start LM Studio and load **text-embedding-qwen3-embedding-4b** (and optionally **qwen3-reranker-4b**) on the local server (default `http://localhost:1234`).

Confirm the pipeline is live:

```powershell
code-rag doctor
```

A green `doctor` output stamps the embedder model + dim into the index on first build — any later model swap is refused until you delete `data\` and reindex.

## Build the index

```powershell
code-rag index
```

Reindex is idempotent — same content produces the same chunk id, so reruns are no-ops on unchanged files. The pipeline embeds **before** deleting old chunks, so a transient LM Studio outage never empties the index.

## Search from the CLI

```powershell
code-rag search "how is OnBarUpdate wired in the strategy" -k 8
code-rag search "UserService" --lang typescript
code-rag callers getById
code-rag get-symbol OnBarUpdate
code-rag index-stats
```

## Evaluate retrieval quality

Author a JSON fixture (`my_eval.json`):

```json
[
  {"query": "how is OnBarUpdate wired", "expected": [{"path": "MNQAlpha.cs"}]},
  {"query": "UserService.getById", "expected": [{"path": "user-service.ts", "symbol": "UserService.getById"}]}
]
```

Run:

```powershell
code-rag eval my_eval.json --json-out report.json
```

Output:

```
{ "n": 2, "recall@1": 0.5, "recall@3": 1.0, "recall@10": 1.0, "mrr": 0.75, "p50_latency_ms": 84.2, "p95_latency_ms": 91.1 }
```

## Register with Claude Code (MCP)

`code-rag install` auto-merges the MCP entry into every Claude config it finds (Claude Desktop + Claude Code user config). For a manual setup, add to your Claude Code `~/.claude.json`:

```json
{
  "mcpServers": {
    "code-rag": {
      "command": "<repo>/.venv/Scripts/pythonw.exe",
      "args": ["-m", "code_rag", "mcp"],
      "cwd": "<repo>"
    }
  }
}
```

Claude Code spawns this process over stdio on demand. No console window (pythonw).

## Live watcher on Windows startup (optional)

```powershell
# Run once from an elevated PowerShell:
.\scripts\install-autostart.ps1
```

Registers a Scheduled Task `code-rag-watch` that:
- Triggers at user logon.
- Runs `pythonw.exe -m code_rag.autostart_bootstrap` (**windowless** — no console).
- Auto-brings LM Studio up via the `lms` CLI if it's not already running.
- Auto-restarts on failure.
- Holds a single-writer lock on Kuzu; MCP server readers coexist in read-only mode.

To stop auto-start:

```powershell
.\scripts\uninstall-autostart.ps1
```

Boot log: `logs\autostart.log` (plain text) + `logs\code_rag.jsonl` (structured).

---

## Config

- `config.example.toml` — committed template. Copy to `config.toml` on first setup.
- `config.toml` — your per-machine config, gitignored so absolute paths never enter history.
- `CODE_RAG_CONFIG=path/to/other.toml` — override the active config file at runtime.
- Relative `data_dir` / `log_dir` resolve against the config file's directory, so `./data` always means `<repo>/data` regardless of CWD.

## Design notes

- **Content-addressed chunks.** `blake3(repo | path | symbol | text)` = chunk id. Reruns on unchanged files are no-ops.
- **Embedder-model namespacing.** The Chroma collection name is `<prefix>_<blake2b(model|dim)>`, so swapping the embedder never mixes incompatible vectors.
- **Index-meta guard.** On every `open()`, we compare the persisted embedder kind/model/dim to the current config. Mismatch → refuse to query. Delete `data\` and reindex to move forward.
- **Single writer.** The watcher holds Kuzu's exclusive write lock; MCP server readers open `read_only=True`. Reindex CLI also writes; run it only when the watcher is stopped.
- **FTS5 hardening.** User queries are stripped of meta-chars and each token wrapped in quotes, so FTS5 keywords (`AND`, `OR`, `NOT`, `NEAR`) pass through as literal phrases.

## Troubleshooting

- **`FAIL ConnectError`** on `doctor` — LM Studio isn't running on `:1234`, or the configured model name isn't loaded.
- **`Index metadata mismatch`** — you changed embedder model or dim in `config.toml`. Delete `data\` and run `code-rag index` again.
- **Watcher not picking up changes** — check `logs\code_rag.jsonl`; the watcher emits `watcher.reindexed` on every debounce flush. Check the scheduled task state: `Get-ScheduledTask code-rag-watch`.
- **Reranker unreachable** — search gracefully falls back to unreranked results after a 5 s probe. To re-enable, load the reranker model in LM Studio.

## Layout

```
src/code_rag/
  interfaces/         Embedder, Reranker, VectorStore, GraphStore, LexicalStore
  embedders/          lm_studio, fake (tests)
  rerankers/          lm_studio, noop
  stores/             chroma_vector, sqlite_lexical, kuzu_graph
  chunking/           treesitter (code), docs (MD/PDF/DOCX/HTML/CSS/SCSS)
  indexing/           walker, indexer
  graph/              extractor, ingest
  retrieval/          search (hybrid), fusion (RRF)
  watcher/            live (watchdog)
  mcp_server/         stdio server with 7 tools
  eval/               harness + fixtures
  install.py          one-shot installer (probe → index → wire Claude → autostart)
  autostart_bootstrap.py   Task Scheduler entry point
  lms_ctl.py          LM Studio CLI control (start server, load models)
  cli.py              Click CLI entry points
scripts/              install-autostart.ps1, uninstall-autostart.ps1
tests/                pytest suite — 74 tests, ~7 s full run
config.example.toml   template config (committed)
config.toml           your config (gitignored)
```
