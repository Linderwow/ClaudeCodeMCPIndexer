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
    const lms = s.lm_studio || {};
    const w = s.watcher || {};
    const lmsUp = !!lms.server_up;
    const watcherRunning = (w.task_state || '').toLowerCase() === 'running';
    // "All" is "up" iff BOTH LM Studio AND the watcher are up. Any partial
    // state renders as "Start all" so the click normalizes everything to up.
    const allUp = lmsUp && watcherRunning;
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

    // Tasks. Phase 54: skip the section entirely when empty (saves ~28px
    // of dead space per project card for the common no-tasks-here case).
    if (taskCount) {
      html += `<h3 class="subhead">Scheduled tasks (${taskCount})</h3>`;
      html += `<table class="proj-table"><thead><tr><th>Name</th><th>State</th><th>Hidden</th><th>Repeat</th><th>Last run</th><th>Exit</th><th></th></tr></thead><tbody>`;
      for (const t of project.tasks) {
        const stateCls = t.state === 'Running' ? 'state-running'
                       : t.state === 'Disabled' ? 'state-disabled' : '';
        const hiddenIcon = t.hidden ? '✓' : '⚠';
        const hiddenCls = t.hidden ? 'hidden-ok' : 'hidden-warn';
        const lastResult = t.last_result === 0 ? '0' : (t.last_result == null ? '—' : String(t.last_result));
        const lastResultCls = t.last_result === 0 ? '' : (t.last_result == null ? '' : 'state-warn');
        // Phase 51: per-row action button. Run if Ready/Disabled, Stop if Running.
        // Disabled tasks render Run too — Start-ScheduledTask refuses on disabled,
        // we surface that error via the result toast rather than hide the button.
        let actionBtn = '';
        if (t.state === 'Running') {
          actionBtn = `<button class="row-btn row-btn-stop" data-task-action="stop" data-task-name="${escapeHtml(t.name)}">stop</button>`;
        } else if (t.state !== 'Disabled') {
          actionBtn = `<button class="row-btn row-btn-run" data-task-action="run" data-task-name="${escapeHtml(t.name)}">run</button>`;
        }
        html += `<tr>
          <td title="${escapeHtml(t.exe)}">${escapeHtml(t.name)}</td>
          <td class="${stateCls}">${escapeHtml(t.state)}</td>
          <td class="${hiddenCls}">${hiddenIcon}</td>
          <td>${escapeHtml(fmtRepeat(t.repeat))}</td>
          <td>${escapeHtml(fmtLastRun(t.last_run))}</td>
          <td class="${lastResultCls}">${escapeHtml(lastResult)}</td>
          <td class="row-actions">${actionBtn}</td>
        </tr>`;
      }
      html += `</tbody></table>`;
    }

    // Processes. Phase 54: skip when empty.
    if (procCount) {
      html += `<h3 class="subhead">Live processes (${procCount})</h3>`;
      html += `<table class="proj-table"><thead><tr><th>PID</th><th>Name</th><th>RAM</th><th>Age</th><th>Cmd</th><th></th></tr></thead><tbody>`;
      for (const p of project.processes) {
        // Kill button for every process row. Clicked button confirms first
        // (kill is destructive, no undo). Backend whitelist verifies the
        // PID's cmdline matches the project pattern at action time so a
        // PID-reuse race can't trick us into killing the wrong process.
        const killBtn = `<button class="row-btn row-btn-kill" data-proc-pid="${p.pid}" data-proc-name="${escapeHtml(p.name)}">kill</button>`;
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

  function budgetRow(verdict, current) {
    const div = document.createElement('div');
    div.className = 'budget-row ' + (verdict.ok ? 'ok' : 'blocked');
    const icon = verdict.ok ? '✓' : '✗';
    const status = verdict.ok ? 'can start' : 'BLOCKED';
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
    if (!verdict.ok && verdict.suggestion) {
      html += `<div class="budget-suggestion">${escapeHtml(verdict.suggestion)}</div>`;
    }
    if (verdict.notes) {
      html += `<div class="budget-notes" title="${escapeHtml(verdict.notes)}">notes ↗</div>`;
    }
    div.innerHTML = html;
    return div;
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
      for (const v of (data.verdicts || [])) {
        rows.appendChild(budgetRow(v, cur));
      }
      if (status) {
        const blocked = (data.verdicts || []).filter(v => !v.ok).length;
        const total = (data.verdicts || []).length;
        if (blocked === 0) {
          status.textContent = `${total} ready`;
          status.className = 'status-pill ok';
        } else {
          status.textContent = `${blocked} of ${total} blocked`;
          status.className = 'status-pill warn';
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
