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
| `ensure_workspace_indexed` | Auto-register + background-index a repo you just opened |

The MCP server is read-only on the three stores. `ensure_workspace_indexed` is the one exception: it appends to a dedicated `dynamic_roots.json` registry (separate from the stores) and spawns a detached `code-rag index --path <p>` subprocess for the actual writes — the live watcher remains the sole in-process writer.

---

## Auto-discovery: any repo Claude opens gets indexed automatically

You don't have to pre-register every codebase in `config.toml`. The first time Claude Code searches in a new repo, its system prompt (built from the MCP tool descriptions) tells it to call `ensure_workspace_indexed(<repo-root>)`. That:

1. Appends the path to `data/dynamic_roots.json` (persistent across reboots).
2. Spawns a detached `code-rag index --path <path>` process that embeds the new repo in the background.
3. Returns immediately so `search_code` keeps serving whatever's already indexed.

On the next logon, the autostart watcher reads both `config.toml` roots **and** `dynamic_roots.json` and watches them all — so edits in any auto-discovered repo trigger incremental reindex, just like curated roots.

**Manage dynamic roots from the CLI:**

```powershell
code-rag roots list                           # show config + dynamic roots
code-rag roots remove C:/path/to/old-project  # drop a dynamic root
code-rag roots prune --days 30                # drop any dynamic root not used in 30 days
```

**Opt-out:** don't want auto-discovery? The feature is tool-driven — Claude only calls `ensure_workspace_indexed` if its reasoning decides the current workspace isn't indexed. If you never want it to fire, you can remove the tool from `TOOLS` in `src/code_rag/mcp_server/server.py` (or ignore the registry entries with `roots remove`). No opt-out flag — by design, "auto" is the default.

---

## One-command bootstrap

**Windows:**
```powershell
git clone <repo> code-rag-mcp
cd code-rag-mcp
.\scripts\setup.ps1
```

**macOS / Linux:**
```bash
git clone <repo> code-rag-mcp
cd code-rag-mcp
./scripts/setup.sh
```

What the bootstrap does (idempotent — safe to rerun):
1. Verifies Python 3.11+ on PATH
2. Creates `.venv/` and `pip install -e .[dev]`
3. Copies `config.example.toml` -> `config.toml`
4. Detects LM Studio CLI; downloads + loads the embedder via `lms get` + `lms load`
5. Best-effort loads the reranker (search degrades gracefully if missing)
6. Runs `code-rag install` (probes LM Studio, builds initial index, wires Claude Code MCP entry, registers autostart)
7. macOS: installs a `launchd` LaunchAgent. Linux: enables a `systemd --user` unit. Windows: registers a Task Scheduler entry.

After it finishes you should be able to:
- Open Claude Code in any repo and ask a question -> the MCP server auto-spawns and `ensure_workspace_indexed` registers the new repo
- Reboot once -> the watcher fires automatically and stays running

If you skipped the bootstrap or want to do it by hand:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Copy the example config — NO edits required if you'll rely on auto-discovery:
copy config.example.toml config.toml
```

That's it for config — `[paths].roots = []` (the default) is valid. As soon as Claude Code asks about a repo, the `ensure_workspace_indexed` MCP tool fires and registers it. Pure pull-based.

If you want **pre-pinned** codebases (so an initial bulk index runs at install time, before Claude touches them), edit `config.toml` and use `~`-relative paths so the same file works on every machine:

```toml
[paths]
roots = [
    "~/RiderProjects",
    "~/Documents/NinjaTrader 8/bin/Custom/Strategies",
]
```

`~` and `${VAR}` are expanded at load time, so the config file is portable across machines / accounts. You can also override the active config file via the `CODE_RAG_CONFIG` env var.

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

## Health & ops commands

```powershell
code-rag doctor               # full readiness check (config, LM Studio, index, drift, autostart)
code-rag fsck --fix           # repair index inconsistencies (orphan rows, dead dynamic roots)
code-rag metrics              # OpenMetrics snapshot (Prometheus / Grafana JSON datasource ready)
code-rag git-log-index        # one-shot git history -> diff chunks (Phase 22)
code-rag embedder list        # list code-specialized embedder presets
code-rag embedder switch bge-code-v1   # auto-download + config edit + wipe (Phase 17 A/B)
```

## Evaluate retrieval quality

The eval harness reports **Recall@1/3/10**, **MRR**, **NDCG@10**, **p50/p95/p99 latency**, and per-tag breakdowns — enough to cleanly diff every embedder swap, rerank tweak, or chunker change.

### Mine real queries from your Claude history

Get a FREE 200-pair ground-truth fixture from your past sessions:

```powershell
code-rag eval-mine --output data/eval/mined.json --max-pairs 500
```

Walks every transcript under `~/.claude/projects/`, finds user questions followed by Claude's first `Read`/`Edit` on a file, and emits the `(query, expected_path)` pairs as JSON. Heuristics filter out commands ("commit this", "run tests") and conversational noise — only real retrieval-style questions land in the fixture.

### Author cases by hand (for known-hard queries)

Each case can declare a `tag` so you can see which kinds of queries regress:

```json
[
  {"query": "OnBarUpdate", "expected": [{"path": "MNQAlpha.cs"}], "tag": "identifier"},
  {"query": "where does the strategy size positions", "expected": [{"path": "MNQAlpha.cs"}], "tag": "natural_language"},
  {"query": "UserService.getById", "expected": [{"path": "user-service.ts", "symbol": "UserService.getById"}], "tag": "graph_walk"}
]
```

A starter set is at `src/code_rag/eval/fixtures/manual_eval.json` (40 hand-authored cases covering MNQAlpha, signals/, RiderProjects).

### Run, save, and diff against a baseline

```powershell
# Establish a baseline.
code-rag eval src/code_rag/eval/fixtures/manual_eval.json `
  --label "baseline-qwen3-embed-4b" `
  --json-out data/eval/baseline.json

# After swapping embedders / rerankers / chunker, compare:
code-rag eval src/code_rag/eval/fixtures/manual_eval.json `
  --label "after-bge-code-v1" `
  --baseline data/eval/baseline.json `
  --fail-on-regression 2.0 `   # Exit 2 if any metric drops > 2 pp
  --json-out data/eval/after-bge.json
```

Output:

```json
{
  "n": 40,
  "recall@1": 0.425, "recall@3": 0.675, "recall@10": 0.875,
  "mrr": 0.569, "ndcg@10": 0.6434,
  "p50_latency_ms": 47.0, "p95_latency_ms": 188.0, "p99_latency_ms": 328.0
}

per-tag:
{
  "identifier":       {"n": 17, "recall@10": 0.82, "ndcg@10": 0.65, ...},
  "natural_language": {"n": 23, "recall@10": 0.91, "ndcg@10": 0.64, ...}
}

diff vs baseline:
{
  "deltas_pp": {"recall@1": +5.0, "recall@3": +3.5, "recall@10": +1.2, ...},
  "deltas_ms": {"p50_latency_ms": -8.0, ...}
}
```

### CI gate

`--fail-on-regression 2.0` exits non-zero if any headline metric drops by more than 2 percentage points vs the baseline — drop it into a GitHub Action to block commits that quietly hurt retrieval quality.

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
