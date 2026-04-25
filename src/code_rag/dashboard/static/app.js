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
  const TOGGLE_LABELS = {
    all:     { start: 'Start all',          stop: 'Stop all' },
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
    } catch (e) {
      // Network blip. Show degraded indicators but don't spam the log.
      $('lms-status').textContent = 'unreachable';
      $('lms-status').className = 'status-pill err';
    }
  }

  function renderStatus(s) {
    // ---- LM Studio ----
    const lms = s.lm_studio || {};
    const lmsServer = $('lms-server');
    const lmsPill = $('lms-status');
    const lmsUp = !!lms.server_up;
    if (lmsUp) {
      lmsServer.textContent = 'running';
      lmsPill.textContent = 'up';
      lmsPill.className = 'status-pill ok';
    } else {
      lmsServer.textContent = 'stopped';
      lmsPill.textContent = 'down';
      lmsPill.className = 'status-pill err';
    }
    $('lms-base').textContent = lms.base_url || '—';
    renderToggle('btn-toggle-lms', 'lms', lmsUp);

    // Loaded models
    const modelsEl = $('lms-models');
    modelsEl.innerHTML = '';
    const loaded = lms.models_loaded || [];
    if (loaded.length === 0) {
      modelsEl.innerHTML = '<div class="muted small">none loaded</div>';
    } else {
      for (const m of loaded) {
        const row = document.createElement('div');
        row.className = 'model-row';
        const sizeStr = m.size_mb >= 1024
          ? `${(m.size_mb / 1024).toFixed(2)} GB`
          : `${Math.round(m.size_mb)} MB`;
        const ttl = m.ttl ? ` · TTL ${m.ttl}` : '';
        row.innerHTML = `
          <div>
            <div class="model-id">${escapeHtml(m.id)}</div>
            <div class="model-meta">${escapeHtml(m.status || '')} · ${sizeStr}${escapeHtml(ttl)}</div>
          </div>
          <span class="model-meta">${escapeHtml(m.device || '')}</span>
          <button class="btn btn-ghost-danger" data-unload="${escapeHtml(m.id)}">unload</button>
        `;
        modelsEl.appendChild(row);
      }
    }
    // wire per-row unload buttons
    modelsEl.querySelectorAll('button[data-unload]').forEach(b => {
      b.addEventListener('click', withBusy(b, async () => {
        const r = await postJSON('/api/models/unload', { model: b.dataset.unload });
        logSteps(`unload(${b.dataset.unload})`, r);
      }));
    });

    // Available (downloaded) — comma-separated, soft text
    $('lms-available').textContent = (lms.models_available || []).join(', ') || '—';

    // ---- Watcher ----
    const w = s.watcher || {};
    const wpill = $('watcher-status');
    const ws = (w.task_state || '').toLowerCase();
    const watcherRunning = ws === 'running';
    if (watcherRunning) {
      wpill.textContent = 'running';
      wpill.className = 'status-pill ok';
    } else if (ws === 'ready') {
      wpill.textContent = 'idle';
      wpill.className = 'status-pill warn';
    } else if (ws === 'notregistered') {
      wpill.textContent = 'not registered';
      wpill.className = 'status-pill err';
    } else {
      wpill.textContent = w.task_state || '—';
      wpill.className = 'status-pill';
    }
    $('watcher-lastrun').textContent = formatTs(w.last_run);
    $('watcher-lastresult').textContent = formatTaskResult(w.last_result);
    $('watcher-pids').textContent = (w.pythonw_pids || []).join(', ') || '—';
    renderToggle('btn-toggle-watcher', 'watcher', watcherRunning);

    // ---- topbar Start all / Stop all ----
    // "All" is "up" iff BOTH LM Studio AND the watcher are up. Any partial
    // state renders as "Start all" so the click normalizes everything to up.
    // (Stopping a partial state is the user's call via the per-card buttons.)
    const allUp = lmsUp && watcherRunning;
    renderToggle('btn-toggle-all', 'all', allUp);

    // ---- Index ----
    const idx = s.index || {};
    const ipill = $('index-status');
    if (idx.present) {
      ipill.textContent = 'present';
      ipill.className = 'status-pill ok';
    } else {
      ipill.textContent = 'missing';
      ipill.className = 'status-pill err';
    }
    $('index-chunks').textContent = idx.chunks != null ? idx.chunks.toLocaleString() : '—';
    $('index-embedder').textContent = idx.embedder_model || '—';
    $('index-dim').textContent = idx.embedder_dim != null ? idx.embedder_dim : '—';
    $('index-schema').textContent = idx.schema_version != null ? `v${idx.schema_version}` : '—';
    $('index-updated').textContent = formatTs(idx.updated_at);

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
    wireToggle('btn-toggle-lms');
    wireToggle('btn-toggle-watcher');
    $('btn-clear-log').addEventListener('click', () => { logEl.innerHTML = ''; });
  }

  document.addEventListener('DOMContentLoaded', () => {
    wireButtons();
    logLine('dashboard ready', 'info');
    startPolling();
  });
})();
