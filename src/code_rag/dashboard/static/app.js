// code-rag dashboard — client.
// No framework. Plain DOM. Polls /api/status every 2s; suspends polling
// while a long operation is in flight to avoid stale UI churn.

(() => {
  const $ = (id) => document.getElementById(id);
  const POLL_MS = 2000;

  // ---- log pane ----
  const logEl = $('log');
  function logLine(text, level = 'info') {
    const line = document.createElement('span');
    line.className = `line ${level}`;
    const ts = new Date().toLocaleTimeString();
    line.innerHTML = `<span class="ts">${ts}</span>${escapeHtml(text)}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
    while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
  }
  function escapeHtml(s) {
    return String(s)
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  }

  // ---- HTTP helpers ----
  async function getJSON(path) {
    const r = await fetch(path, { method: 'GET' });
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
  }
  async function postJSON(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
  }

  // ---- toggle button rendering --------------------------------------------
  // A "toggle button" is one button per pair (all / lms / watcher) that
  // flips between Start vs Stop based on current state. The button carries
  // its current intended action in `data-action` so the click handler
  // dispatches to the right endpoint without separate handlers per direction.

  // Render config: each toggle target has classes for both states.
  const TOGGLE_CLASSES = {
    all:     { start: 'btn btn-primary',    stop: 'btn btn-danger' },
    lms:     { start: 'btn btn-ghost',      stop: 'btn btn-ghost-danger' },
    watcher: { start: 'btn btn-ghost',      stop: 'btn btn-ghost-danger' },
  };
  // Phase 55: 'all' controls the code-rag stack (LM Studio + watcher +
  // MCP servers), not the union of all 3 projects. Renamed so the
  // button doesn't imply 'start everything' — that would OOM the GPU
  // when YouTubeBot and code-rag both want VRAM.
  const TOGGLE_LABELS = {
    all:     { start: 'Start code-rag',     stop: 'Stop code-rag' },
    lms:     { start: 'Start server',       stop: 'Stop server' },
    watcher: { start: 'Start watcher',      stop: 'Stop watcher' },
  };

  function renderToggle(id, target, currentlyUp) {
    const btn = $(id);
    if (!btn) return;
    const action = currentlyUp ? 'stop' : 'start';
    btn.dataset.action = action;
    btn.dataset.target = target;
    btn.className = TOGGLE_CLASSES[target][action];
    // Topbar buttons keep the leading dot; cards have plain labels.
    const labelEl = btn.querySelector('.btn-label');
    if (labelEl) {
      labelEl.textContent = TOGGLE_LABELS[target][action];
    } else {
      btn.textContent = TOGGLE_LABELS[target][action];
    }
  }

  function busy(btn, on) {
    if (!btn) return;
    btn.classList.toggle('is-busy', on);
    btn.disabled = on;
  }
  function withBusy(btn, fn) {
    return async () => {
      busy(btn, true);
      pausePolling();
      try {
        await fn();
      } catch (e) {
        logLine(`error: ${e.message || e}`, 'err');
      } finally {
        busy(btn, false);
        await refreshStatus();
        resumePolling();
      }
    };
  }

  function logSteps(label, payload) {
    if (Array.isArray(payload?.steps)) {
      for (const s of payload.steps) {
        const lvl = s.ok ? 'ok' : 'err';
        const ms = s.duration_ms != null ? ` (${s.duration_ms.toFixed(0)} ms)` : '';
        logLine(`${label} :: ${s.name} -- ${s.detail || (s.ok ? 'ok' : 'fail')}${ms}`, lvl);
      }
    } else if (payload?.name) {
      const lvl = payload.ok ? 'ok' : 'err';
      const ms = payload.duration_ms != null ? ` (${payload.duration_ms.toFixed(0)} ms)` : '';
      logLine(`${label} :: ${payload.name} -- ${payload.detail || (payload.ok ? 'ok' : 'fail')}${ms}`, lvl);
    } else {
      logLine(`${label} :: ${JSON.stringify(payload)}`, 'info');
    }
  }

  // ---- status render ----
  let lastStatus = null;

  async function refreshStatus() {
    try {
      const s = await getJSON('/api/status');
      lastStatus = s;
      renderStatus(s);
      // Phase 60-A3: reset the Resources pill back to LIVE on success.
      // Previously a single failed fetch left "UNREACHABLE" on the pill
      // forever, even after subsequent fetches succeeded.
      const rs = $('resources-status');
      if (rs) { rs.textContent = 'live'; rs.className = 'status-pill ok'; }
    } catch (e) {
      // Network blip. /api/status powers the topbar Start/Stop button +
      // the Resources card; degraded but not catastrophic.
      const rs = $('resources-status');
      if (rs) { rs.textContent = 'unreachable'; rs.className = 'status-pill err'; }
    }
  }

  function renderStatus(s) {
    // Phase 50 simplification: the per-component cards (LM Studio, Watcher,
    // Index) were removed in favor of the unified All Projects panel below.
    // We still consume /api/status for two cross-cutting things: the topbar
    // Start/Stop button (needs to know if the code-rag stack is up) and
    // the Resources card (RAM/VRAM bars).

    // ---- topbar Start all / Stop all ----
    // Phase 60-A3: previously required watcher.task_state === 'running'
    // for the topbar to flip to "Stop". That worked in the LM Studio era
    // when a long-lived watcher process was the heartbeat. In the vLLM
    // era the embedder server IS the heartbeat — the Windows-scheduled
    // watcher task spends most of its life in 'Ready' (idle, armed) and
    // we'd misreport code-rag as "down" whenever it wasn't actively
    // crawling files. Now we treat code-rag as "up" iff the embedder
    // server is reachable; the watcher is incidental.
    const lms = s.lm_studio || {};
    const w = s.watcher || {};
    const lmsUp = !!lms.server_up;
    const watcherRunning = (w.task_state || '').toLowerCase() === 'running';
    const allUp = lmsUp;
    renderToggle('btn-toggle-all', 'all', allUp);

    // ---- Resources ----
    const res = s.resources || {};
    const ram = res.ram || {};
    if (ram.total_gb) {
      const pct = (ram.used_gb / ram.total_gb) * 100;
      $('ram-bar').style.width = `${pct.toFixed(1)}%`;
      $('ram-bar').className = `bar-fill${pct > 90 ? ' err' : pct > 75 ? ' warn' : ''}`;
      $('ram-label').textContent = `${ram.used_gb.toFixed(1)} / ${ram.total_gb.toFixed(1)} GB`;
      $('ram-pct').textContent = `${pct.toFixed(1)}%`;
    }
    const gpu = res.gpu;
    if (gpu) {
      const gpct = (gpu.vram_used_gb / gpu.vram_total_gb) * 100;
      $('gpu-bar').style.width = `${gpct.toFixed(1)}%`;
      $('gpu-bar').className = `bar-fill gpu${gpct > 92 ? ' err' : gpct > 80 ? ' warn' : ''}`;
      $('gpu-label').textContent =
        `${gpu.name} · ${gpu.vram_used_gb.toFixed(1)} / ${gpu.vram_total_gb.toFixed(1)} GB`;
      $('gpu-pct').textContent = `${gpct.toFixed(1)}%`;
      $('gpu-util').textContent = `${gpu.util_pct}% · ${gpu.temp_c}°C`;
    } else {
      $('gpu-label').textContent = 'GPU info unavailable';
      $('gpu-pct').textContent = '';
      $('gpu-util').textContent = '—';
    }

    // ---- Phase 60-A2 + A3: reindex-in-progress card ----
    // Server returns `index.vectors`, `index.chunks`, `index.reindex_pct`
    // (vectors / chunks * 100). Show the card only when reindex_pct < 99.
    //
    // Phase 60-A3: also compute rate + ETA from a sliding window of
    // previous samples. Without rate the card is a snapshot — you can't
    // tell if it's stuck or moving. With rate the user gets a useful
    // "9.2 vec/sec • ~38 min remaining" signal that updates in real time.
    //
    // Phase 60 audit fix: declare pct/chunks/vectors at function scope so
    // the topbar status section below can read them safely. Previously they
    // were inside `if (reindexEl)` and would ReferenceError if that block
    // was ever skipped.
    const idx = s.index || {};
    const pct = (typeof idx.reindex_pct === 'number') ? idx.reindex_pct : null;
    const vectors = (typeof idx.vectors === 'number') ? idx.vectors : null;
    const chunks = (typeof idx.chunks === 'number') ? idx.chunks : null;

    // ---- Phase 60-I: Indexing card (always visible) ----
    // Permanent at-a-glance summary of the index state. Shows coverage,
    // embedder, and the current state pill (live / catching up / idle /
    // auto-stopped / user-stopped). The transient `.reindex` card below
    // handles the live rate+ETA when actively catching up.
    const idxStatusEl = $('index-status');
    const idxBar = $('index-bar');
    if (idxBar && chunks !== null && chunks > 0 && vectors !== null) {
      const coveragePct = pct !== null ? pct : (100 * vectors / chunks);
      idxBar.style.width = `${coveragePct.toFixed(1)}%`;
      idxBar.className =
        `bar-fill${coveragePct < 50 ? ' err' : coveragePct < 95 ? ' warn' : ''}`;
      $('index-label').textContent =
        `${vectors.toLocaleString()} / ${chunks.toLocaleString()} vectors`;
      $('index-pct').textContent = `${coveragePct.toFixed(1)}%`;
    } else if (idxBar) {
      idxBar.style.width = '0%';
      $('index-label').textContent = chunks ? `0 / ${chunks.toLocaleString()}` : 'no index yet';
      $('index-pct').textContent = '—';
    }
    $('index-embedder-name').textContent = idx.embedder_model || '—';
    $('index-embedder-dim').textContent =
      idx.embedder_dim ? `dim ${idx.embedder_dim}` : '';

    // State pill: live / catching up / wipe-recovery / down. Determined
    // from lm_studio.server_up + index.state (Phase 60-K) + reindex_pct.
    const lmsUp_idx = !!(s.lm_studio && s.lm_studio.server_up);
    let stateLabel = 'live';
    let stateClass = 'status-pill ok';
    let stateDetail = 'embedder reachable';
    const indexState = idx.state;  // Phase 60-K: 'wipe_recovery' | 'catching_up' | 'live'
    if (!lmsUp_idx) {
      stateLabel = 'down';
      stateClass = 'status-pill err';
      stateDetail = 'embedder unreachable';
    } else if (indexState === 'wipe_recovery') {
      stateLabel = 'wipe recovery';
      stateClass = 'status-pill err';
      stateDetail = 'chroma was wiped — re-vectorizing all chunks';
    } else if (pct !== null && pct < 99.0) {
      stateLabel = 'catching up';
      stateClass = 'status-pill warn';
      stateDetail = `dense lagging fts by ${(100 - pct).toFixed(1)} pp`;
    }
    if (idxStatusEl) {
      idxStatusEl.textContent = stateLabel;
      idxStatusEl.className = stateClass;
    }
    $('index-state-detail').textContent = stateDetail;
    // Last-activity: relative time since updated_at (when index_meta
    // was last touched) — proxy for "when did the indexer last run".
    const updatedAt = idx.updated_at;
    if (updatedAt) {
      try {
        const d = new Date(updatedAt);
        if (!isNaN(d.getTime())) {
          const ageS = Math.max(0, (Date.now() - d.getTime()) / 1000);
          let ageStr;
          if (ageS < 60) ageStr = `${Math.floor(ageS)}s ago`;
          else if (ageS < 3600) ageStr = `${Math.floor(ageS / 60)}m ago`;
          else if (ageS < 86400) ageStr = `${Math.floor(ageS / 3600)}h ago`;
          else ageStr = `${Math.floor(ageS / 86400)}d ago`;
          $('index-last-activity').textContent = `updated ${ageStr}`;
        } else {
          $('index-last-activity').textContent = '';
        }
      } catch { $('index-last-activity').textContent = ''; }
    } else {
      $('index-last-activity').textContent = '';
    }

    const reindexEl = $('reindex');
    if (reindexEl) {
      // Phase 60 audit fix: also guard `vectors !== null` so we never call
      // .toLocaleString() on a null/undefined.
      const inProgress = (chunks && vectors !== null && pct !== null && pct < 99.0);
      reindexEl.style.display = inProgress ? '' : 'none';
      if (inProgress) {
        const bar = $('reindex-bar');
        bar.style.width = `${pct.toFixed(1)}%`;
        bar.className = `bar-fill${pct < 50 ? ' err' : pct < 75 ? ' warn' : ''}`;
        $('reindex-label').textContent =
          `${vectors.toLocaleString()} / ${chunks.toLocaleString()} vectors`;
        $('reindex-pct').textContent = `${pct.toFixed(1)}%`;
        $('reindex-embedder').textContent =
          `${idx.embedder_model || '—'} (dim ${idx.embedder_dim || '—'})`;
        const sp = $('reindex-status');
        sp.textContent = 'catching up';
        sp.className = 'status-pill warn';

        // Rate / ETA from sliding window. Keep ~last 60 samples (≈ 2 min
        // at the dashboard's 2 s poll cadence) so the rate doesn't whip
        // around on per-sample noise.
        if (typeof window._reindexHist === 'undefined') window._reindexHist = [];
        const hist = window._reindexHist;
        const nowMs = Date.now();
        hist.push({ t: nowMs, v: vectors });
        while (hist.length > 60) hist.shift();
        let rateLine = '—';
        let etaLine = '—';
        if (hist.length >= 2) {
          const first = hist[0], last = hist[hist.length - 1];
          const dv = last.v - first.v;
          const dt = (last.t - first.t) / 1000;  // seconds
          if (dt > 5 && dv > 0) {
            const rate = dv / dt;
            rateLine = `${rate.toFixed(1)} vec/sec`;
            const remaining = chunks - vectors;
            const etaSec = remaining / rate;
            if (etaSec < 60)        etaLine = `~${Math.ceil(etaSec)}s remaining`;
            else if (etaSec < 3600) etaLine = `~${Math.ceil(etaSec / 60)} min remaining`;
            else                    etaLine = `~${(etaSec / 3600).toFixed(1)} h remaining`;
          } else {
            // Phase 60 audit fix: stall detection now looks at the LAST 30s
            // of the window, not the entire (up-to-120s) window. Old code
            // used dv across the whole window and triggered "stalled" any
            // time a saturated 120s window had no progress in the last few
            // seconds — false positives on slow-but-not-stuck reindexes.
            const recent = hist.filter(h => last.t - h.t <= 30000);
            if (recent.length >= 2) {
              const recentDv = recent[recent.length - 1].v - recent[0].v;
              const recentDt = (recent[recent.length - 1].t - recent[0].t) / 1000;
              if (recentDt > 25 && recentDv === 0) {
                rateLine = 'stalled (no progress in 30s)';
                etaLine = '—';
              }
            }
          }
        }
        const rateEl = $('reindex-rate');
        const etaEl = $('reindex-eta');
        if (rateEl) rateEl.textContent = rateLine;
        if (etaEl) etaEl.textContent = etaLine;
      } else {
        // Reset history once reindex completes so the next one starts clean.
        window._reindexHist = [];
      }
    }

    // ---- Phase 60-A3: topbar status line ----
    // Compact, glanceable: "vLLM ✓ • indexer running • dashboard ✓".
    // Tells the user WHY the Start/Stop button shows what it shows.
    const statusLine = $('topbar-status');
    if (statusLine) {
      const parts = [];
      const vLabel = lmsUp ? '✓' : '✗';
      const vClass = lmsUp ? 'ok' : 'err';
      parts.push(`<span class="ts-${vClass}">vLLM ${vLabel}</span>`);
      // Reindex / indexer activity — derived from the same heuristic as
      // the reindex card.
      if (chunks && pct !== null && pct < 99.0) {
        parts.push(`<span class="ts-warn">indexer running</span>`);
      } else if (chunks) {
        parts.push(`<span class="ts-ok">indexer idle</span>`);
      }
      // Dashboard is implicitly up (we wouldn't be rendering otherwise).
      parts.push(`<span class="ts-ok">dashboard ✓</span>`);
      statusLine.innerHTML = parts.join(' · ');
    }
  }

  function formatTs(s) {
    if (!s) return '—';
    try {
      const d = new Date(s);
      if (isNaN(d.getTime())) return s;
      return d.toLocaleString();
    } catch { return s; }
  }
  function formatTaskResult(r) {
    if (r == null) return '—';
    // 0x41301 = "task currently running" — friendliest decode for the common values.
    const codes = {
      0:        'success',
      1:        'general error',
      267009:   'currently running',
      267010:   'task ready',
      267011:   'task disabled',
    };
    return codes[r] != null ? `${codes[r]} (${r})` : `${r}`;
  }

  // ---- polling ----
  let pollTimer = null;
  let paused = false;
  function startPolling() {
    refreshStatus();
    pollTimer = setInterval(() => { if (!paused) refreshStatus(); }, POLL_MS);
  }
  function pausePolling() { paused = true; }
  function resumePolling() { paused = false; }

  // ---- Phase 50: Unified Command Center — projects panel ----
  // Polls /api/projects every 5s (heavier than /api/status — runs ~6
  // PowerShell cmdlets to enumerate scheduled tasks across 3 projects).
  // Renders three project cards: tasks table + processes table.
  const PROJECTS_POLL_MS = 5000;
  let projectsTimer = null;

  function fmtRepeat(r) {
    // Phase 53: backend now sends rich schedule strings like
    // 'PT5M', 'PT1H', 'P1D', 'daily 03:00', 'weekly Mon 05:00',
    // 'at boot', 'at logon', 'monthly', 'one-shot', or combos like
    // 'daily 03:00 / +PT30M'. Format ISO durations to human; pass
    // human strings through as-is.
    if (!r) return '—';
    const s = String(r);
    // Split combo like "daily 03:00 / +PT30M" — format each part.
    const parts = s.split('/').map(p => p.trim());
    const fmtOne = (x) => {
      const m = x.match(/^(\+?)PT(\d+)([HMS])$/);
      if (m) {
        const sign = m[1], v = m[2], u = m[3];
        if (u === 'M') return `${sign}${v} min`;
        if (u === 'H') return `${sign}${v} h`;
        if (u === 'S') return `${sign}${v} s`;
      }
      const m2 = x.match(/^PT(\d+)M(\d+)S$/);
      if (m2) return `${m2[1]}m${m2[2]}s`;
      if (x === 'P1D') return 'daily';
      return x;
    };
    return parts.map(fmtOne).join(' · ');
  }

  function fmtAge(s) {
    if (!s || s < 60) return `${s|0}s`;
    if (s < 3600) return `${(s/60)|0}m`;
    if (s < 86400) return `${(s/3600).toFixed(1)}h`;
    return `${(s/86400).toFixed(1)}d`;
  }

  function fmtLastRun(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      const now = new Date();
      const ageS = (now - d) / 1000;
      if (ageS < 60) return `${ageS|0}s ago`;
      if (ageS < 3600) return `${(ageS/60)|0}m ago`;
      if (ageS < 86400) return `${(ageS/3600).toFixed(1)}h ago`;
      return d.toLocaleDateString();
    } catch { return '—'; }
  }

  function projectCard(project) {
    const card = document.createElement('article');
    card.className = 'card project-card';
    card.id = `project-${project.id}`;

    const taskCount = project.tasks.length;
    const procCount = project.processes.length;
    const runningProcs = project.processes.filter(p => p.age_s > 0).length;
    const runningTasks = project.tasks.filter(t => t.state === 'Running').length;
    const headerSummary = `${runningTasks} task running · ${runningProcs} process${runningProcs === 1 ? '' : 'es'}`;

    let html = `
      <header class="card-header">
        <h2>${escapeHtml(project.label)}</h2>
        <span class="status-pill">${headerSummary}</span>
      </header>
      <div class="card-body">
    `;

    if (project.error) {
      html += `<div class="kv error">probe error: ${escapeHtml(project.error)}</div>`;
    }

    // Tasks. Phase 54: skip the section entirely when empty.
    // Phase 56: a one-line note clarifies tasks fire autonomously on schedule.
    if (taskCount) {
      html += `<h3 class="subhead">Scheduled tasks (${taskCount})
        <span class="subhead-note">— autonomous on schedule; <strong>run</strong> = force-fire now</span>
      </h3>`;
      html += `<table class="proj-table proj-table-tasks"><thead><tr><th>Name</th><th>State</th><th>H</th><th>Repeat</th><th>Last run</th><th>Exit</th><th></th></tr></thead><tbody>`;
      for (const t of project.tasks) {
        const stateCls = t.state === 'Running' ? 'state-running'
                       : t.state === 'Disabled' ? 'state-disabled' : '';
        const hiddenIcon = t.hidden ? '✓' : '⚠';
        const hiddenCls = t.hidden ? 'hidden-ok' : 'hidden-warn';
        // Phase 57: huge Win32 status codes (e.g. 3221225786 = 0xC000013A
        // STATUS_CONTROL_C_EXIT) overflow the column. Render small ints
        // as-is; render >9999 as 0xHEX so the column stays narrow. Full
        // decimal is in the tooltip for the curious.
        const rawResult = t.last_result;
        let lastResult, lastResultTip;
        if (rawResult === 0) { lastResult = '0'; lastResultTip = 'success (exit 0)'; }
        else if (rawResult == null) { lastResult = '—'; lastResultTip = 'never run'; }
        else if (Math.abs(rawResult) > 9999) {
          lastResult = '0x' + (rawResult >>> 0).toString(16).toUpperCase();
          lastResultTip = `exit ${rawResult}`;
        } else {
          lastResult = String(rawResult);
          lastResultTip = `exit ${rawResult}`;
        }
        const lastResultCls = rawResult === 0 ? '' : (rawResult == null ? '' : 'state-warn');
        // Phase 51 → 56: per-row action button. Stop stays bold (urgent —
        // user needs to see it). Run is dim-by-default + visible on hover
        // since 99% of tasks fire autonomously on schedule and the button
        // is only useful for "force-fire NOW" (e.g. retry a failed daily
        // task without waiting 24 hours). Tasks with a schedule are
        // explicitly tagged so CSS can dim them differently from
        // schedule-less manual tasks.
        let actionBtn = '';
        if (t.state === 'Running') {
          actionBtn = `<button class="row-btn row-btn-stop" data-task-action="stop" data-task-name="${escapeHtml(t.name)}" title="Stop this running task">stop</button>`;
        } else if (t.state !== 'Disabled') {
          const scheduledClass = t.repeat ? ' row-btn-run-scheduled' : '';
          const tipText = t.repeat
            ? `Force-fire NOW (manual override). Task is already on schedule: ${t.repeat}.`
            : 'Run NOW (this task has no automatic schedule).';
          actionBtn = `<button class="row-btn row-btn-run${scheduledClass}" data-task-action="run" data-task-name="${escapeHtml(t.name)}" title="${escapeHtml(tipText)}">run</button>`;
        }
        html += `<tr>
          <td title="${escapeHtml(t.exe)}">${escapeHtml(t.name)}</td>
          <td class="${stateCls}">${escapeHtml(t.state)}</td>
          <td class="${hiddenCls}">${hiddenIcon}</td>
          <td>${escapeHtml(fmtRepeat(t.repeat))}</td>
          <td>${escapeHtml(fmtLastRun(t.last_run))}</td>
          <td class="${lastResultCls}" title="${escapeHtml(lastResultTip)}">${escapeHtml(lastResult)}</td>
          <td class="row-actions">${actionBtn}</td>
        </tr>`;
      }
      html += `</tbody></table>`;
    }

    // Processes. Phase 54: skip when empty.
    if (procCount) {
      html += `<h3 class="subhead">Live processes (${procCount})</h3>`;
      html += `<table class="proj-table proj-table-procs"><thead><tr><th>PID</th><th>Name</th><th>RAM</th><th>Age</th><th>Cmd</th><th></th></tr></thead><tbody>`;
      for (const p of project.processes) {
        // Kill button for every process row. Clicked button confirms first
        // (kill is destructive, no undo). Backend whitelist verifies the
        // PID's cmdline matches the project pattern at action time so a
        // PID-reuse race can't trick us into killing the wrong process.
        const killBtn = `<button class="row-btn row-btn-kill" data-proc-pid="${p.pid}" data-proc-name="${escapeHtml(p.name)}" title="Force-terminate this process (Stop-Process -Force, no undo)">kill</button>`;
        html += `<tr>
          <td>${p.pid}</td>
          <td>${escapeHtml(p.name)}</td>
          <td>${p.ram_mb} MB</td>
          <td>${escapeHtml(fmtAge(p.age_s))}</td>
          <td class="cmd-cell" title="${escapeHtml(p.cmd_preview)}">${escapeHtml(p.cmd_preview)}</td>
          <td class="row-actions">${killBtn}</td>
        </tr>`;
      }
      html += `</tbody></table>`;
    }

    html += '</div>';
    card.innerHTML = html;
    return card;
  }

  async function refreshProjects() {
    try {
      const data = await getJSON('/api/projects');
      const grid = $('projects-grid');
      if (!grid) return;
      grid.innerHTML = '';
      for (const p of (data.projects || [])) {
        grid.appendChild(projectCard(p));
      }
      const status = $('projects-status');
      if (status) {
        const total = (data.projects || []).reduce(
          (acc, p) => acc + p.processes.length, 0,
        );
        status.textContent = `${total} live process${total === 1 ? '' : 'es'}`;
        status.className = 'status-pill ok';
      }
    } catch (e) {
      const status = $('projects-status');
      if (status) {
        status.textContent = 'probe failed';
        status.className = 'status-pill err';
      }
      logLine(`projects probe failed: ${e.message}`, 'warn');
    }
  }

  function startProjectsPolling() {
    refreshProjects();
    projectsTimer = setInterval(refreshProjects, PROJECTS_POLL_MS);
  }

  // ---- Phase 52: Resource Budget panel ----
  // Polls /api/budget every 5s alongside projects. Renders per-project
  // can-start verdicts: green ✓ if there's headroom, red ✗ if start
  // would OOM (with the suggestion text inline so the user knows what
  // to free).
  let budgetTimer = null;

  function budgetRow(verdict, current, isRunning) {
    const div = document.createElement('div');
    // Phase 60-A3: a project that's already running shouldn't read as
    // "BLOCKED" — that's pre-flight language and visually alarming.
    // When isRunning=true, render as a calm "RUNNING" row regardless
    // of what the budget arbitrator says (it's gating new starts, not
    // saying the running project is broken).
    let cls, icon, status;
    if (isRunning) {
      cls = 'running';
      icon = '●';
      status = 'RUNNING';
    } else if (verdict.ok) {
      cls = 'ok';
      icon = '✓';
      status = 'can start';
    } else {
      cls = 'blocked';
      icon = '✗';
      status = 'BLOCKED';
    }
    div.className = 'budget-row ' + cls;
    const cost = `cost: ${verdict.cost_ram_gb} GB RAM · ${verdict.cost_vram_gb} GB VRAM`;
    const avail = `available: ${verdict.available_ram_gb} GB RAM · ${verdict.available_vram_gb} GB VRAM`;
    let html = `
      <div class="budget-row-head">
        <span class="budget-icon">${icon}</span>
        <strong>${escapeHtml(verdict.label)}</strong>
        <span class="budget-status-text">${status}</span>
      </div>
      <div class="budget-meta">${escapeHtml(cost)} · ${escapeHtml(avail)}</div>
    `;
    // Suppress the "refuse start" suggestion when the project is
    // actually running — it's confusing in that context.
    if (!isRunning && !verdict.ok && verdict.suggestion) {
      html += `<div class="budget-suggestion">${escapeHtml(verdict.suggestion)}</div>`;
    }
    if (verdict.notes) {
      html += `<div class="budget-notes" title="${escapeHtml(verdict.notes)}">notes ↗</div>`;
    }
    div.innerHTML = html;
    return div;
  }

  // Phase 60-A3: per-project liveness derived from lastStatus.
  // Currently only code-rag has a probe (the embedder server's reachability);
  // YouTubeBot + MNQAlpha would need their own (process scan, Task Scheduler
  // state) to participate. Returns false for unknown projects so they keep
  // their pre-flight ok/blocked rendering.
  function projectIsRunning(projectId) {
    if (!lastStatus) return false;
    if (projectId === 'code-rag') {
      return !!(lastStatus.lm_studio && lastStatus.lm_studio.server_up);
    }
    return false;
  }

  async function refreshBudget() {
    try {
      const data = await getJSON('/api/budget');
      const rows = $('budget-rows');
      const status = $('budget-status');
      if (!rows) return;
      rows.innerHTML = '';
      const cur = data.current || {};
      // Top summary line: current usage vs total minus reserves.
      const reserves = data.reserves || {};
      const summary = document.createElement('div');
      summary.className = 'budget-summary';
      summary.innerHTML = `
        <div>RAM: <strong>${cur.ram_used_gb} / ${cur.ram_total_gb} GB</strong>
             <span class="dim">(reserve ${reserves.ram_reserve_gb} GB)</span></div>
        <div>VRAM: <strong>${cur.vram_used_gb} / ${cur.vram_total_gb} GB</strong>
             <span class="dim">(reserve ${reserves.vram_reserve_gb} GB)</span></div>
      `;
      rows.appendChild(summary);
      // Phase 60-A3: tag each verdict as running/ok/blocked so the
      // panel header counts only TRUE blocks (not running projects).
      const verdicts = data.verdicts || [];
      const tagged = verdicts.map(v => ({
        verdict: v,
        running: projectIsRunning(v.project_id),
      }));
      for (const t of tagged) {
        rows.appendChild(budgetRow(t.verdict, cur, t.running));
      }
      if (status) {
        const running = tagged.filter(t => t.running).length;
        const blocked = tagged.filter(t => !t.running && !t.verdict.ok).length;
        const total = tagged.length;
        if (blocked === 0 && running === total) {
          status.textContent = `${total} running`;
          status.className = 'status-pill ok';
        } else if (blocked === 0) {
          status.textContent = `${running} running, ${total - running} ready`;
          status.className = 'status-pill ok';
        } else if (running > 0) {
          status.textContent = `${running} running, ${blocked} blocked`;
          status.className = 'status-pill warn';
        } else {
          status.textContent = `${blocked} of ${total} blocked`;
          status.className = 'status-pill warn';
        }
      }

      // Phase 58: gate the topbar 'Start code-rag' button on the budget
      // verdict. Phase 52 already refuses the action server-side if the
      // verdict says BLOCKED, but the button visually pretends it's a
      // happy path. When code-rag's verdict is BLOCKED AND the button
      // is currently in 'start' mode, disable it + surface the
      // suggestion as the tooltip. When code-rag is already running
      // (button is in 'stop' mode) leave it enabled — stopping is
      // never resource-constrained.
      const codeRagVerdict = (data.verdicts || []).find(
        v => v.project_id === 'code-rag',
      );
      const btn = $('btn-toggle-all');
      if (btn && codeRagVerdict) {
        const isStart = btn.dataset.action === 'start';
        if (isStart && !codeRagVerdict.ok) {
          btn.disabled = true;
          btn.classList.add('btn-blocked');
          btn.title = `BLOCKED: ${codeRagVerdict.suggestion}`;
        } else {
          btn.disabled = false;
          btn.classList.remove('btn-blocked');
          // Restore the default tooltip set in index.html.
          if (isStart) {
            btn.title = 'Start the code-rag stack only (LM Studio + watcher + MCP servers, ~14 GB RAM + 12 GB VRAM). Refused by the budget guard if there isn\'t enough headroom.';
          } else {
            btn.title = 'Stop the code-rag stack';
          }
        }
      }
    } catch (e) {
      const status = $('budget-status');
      if (status) {
        status.textContent = 'probe failed';
        status.className = 'status-pill err';
      }
    }
  }

  function startBudgetPolling() {
    refreshBudget();
    budgetTimer = setInterval(refreshBudget, PROJECTS_POLL_MS);
  }

  // Phase 51: per-row action delegation. Rows are re-rendered every 5s so
  // we can't bind a listener per button — single delegated listener on the
  // grid catches clicks for whichever row the user hit.
  function wireProjectActions() {
    const grid = $('projects-grid');
    if (!grid) return;
    grid.addEventListener('click', async (ev) => {
      const btn = ev.target.closest('button.row-btn');
      if (!btn) return;
      ev.preventDefault();
      // Brief disable to prevent double-click duplicates while in flight.
      btn.disabled = true;
      try {
        if (btn.dataset.taskAction) {
          const action = btn.dataset.taskAction;   // 'run' | 'stop'
          const name = btn.dataset.taskName;
          if (action === 'stop'
              && !confirm(`Stop scheduled task "${name}"?`)) return;
          logLine(`tasks.${action}(${name}) -> running…`, 'info');
          const r = await postJSON(`/api/tasks/${action}`, { name });
          logLine(
            `tasks.${action}(${name}): ${r.detail || (r.ok ? 'ok' : 'failed')}`,
            r.ok ? 'ok' : 'err',
          );
          refreshProjects();   // immediate refresh — show state flip without waiting 5s
        } else if (btn.dataset.procPid) {
          const pid = btn.dataset.procPid;
          const pname = btn.dataset.procName || 'process';
          if (!confirm(`Kill ${pname} (PID ${pid})?\n\nThis is destructive — no undo.`)) return;
          logLine(`processes.kill(${pid}) -> running…`, 'info');
          const r = await postJSON('/api/processes/kill', { pid: Number(pid) });
          logLine(
            `processes.kill(${pid}): ${r.detail || (r.ok ? 'ok' : 'failed')}`,
            r.ok ? 'ok' : 'err',
          );
          refreshProjects();
        }
      } catch (e) {
        logLine(`action failed: ${e.message}`, 'err');
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---- click wiring ----
  // Each toggle button carries its current intent in data-action ("start"|
  // "stop") and its target in data-target ("all"|"lms"|"watcher"). One
  // delegated handler per button dispatches to the right endpoint based on
  // those attrs at click time, so the wiring doesn't need to be re-bound
  // when the toggle flips.
  function wireToggle(id) {
    const btn = $(id);
    if (!btn) return;
    btn.addEventListener('click', withBusy(btn, async () => {
      const action = btn.dataset.action;
      const target = btn.dataset.target;
      const path = `/api/${action}/${target}`;
      // Stop all = stop EVERYTHING including the LM Studio server. If the
      // server stayed up, any background request (Claude Code MCP subprocess,
      // a stray search) would JIT-reload the embedder and waste the click.
      // The per-card "Stop server" button still exists for granular control.
      const body = (target === 'all' && action === 'stop')
        ? { stop_lm_studio: true } : {};
      logLine(`${action}_${target} -> running…`, 'info');
      const r = await postJSON(path, body);
      logSteps(`${action}_${target}`, r);
    }));
  }

  function wireButtons() {
    wireToggle('btn-toggle-all');
    // Phase 50: per-card Start server / Start watcher buttons removed
    // along with the LM Studio + Watcher cards. The topbar Start/Stop
    // all is the only stack-control button now; granular control lives
    // in the CLI / direct task scheduler.
    const clearBtn = $('btn-clear-log');
    if (clearBtn) clearBtn.addEventListener('click', () => { logEl.innerHTML = ''; });
  }

  document.addEventListener('DOMContentLoaded', () => {
    wireButtons();
    logLine('dashboard ready', 'info');
    startPolling();
    startProjectsPolling();   // Phase 50: unified command center poll
    wireProjectActions();     // Phase 51: per-row run/stop/kill buttons
    startBudgetPolling();     // Phase 52: resource budget panel
  });
})();
