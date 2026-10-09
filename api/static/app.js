/* quantlab frontend — vanilla, no build step.
   Submits a backtest, polls the job, renders the result. */

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status} ${r.statusText}`);
  return body;
};

const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const pct = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
const money = (v) => v.toLocaleString(undefined, {
  style: "currency", currency: "USD", maximumFractionDigits: 0 });

let CATALOG = {};
let chart = null;

/* ── Form ──────────────────────────────────────────────────────────────── */

async function loadStrategies() {
  const { strategies } = await api("/strategies");
  CATALOG = Object.fromEntries(strategies.map((s) => [s.kind, s.params]));
  $("kind").innerHTML = strategies
    .map((s) => `<option value="${s.kind}">${s.kind.replace(/_/g, " ")}</option>`)
    .join("");
  $("kind").value = "trend_following";
  renderParams();
}

/* Parameter inputs are generated from the server's catalogue rather than
   hardcoded, so adding a strategy server-side needs no frontend change. */
function renderParams() {
  const params = CATALOG[$("kind").value] || {};
  $("params").innerHTML = Object.entries(params).map(([key, def]) => {
    const step = Number.isInteger(def) ? "1" : "0.1";
    return `<div>
      <label for="p-${key}">${key.replace(/_/g, " ")}</label>
      <input id="p-${key}" data-param="${key}" type="number"
             value="${def}" step="${step}">
    </div>`;
  }).join("");
}

function collectParams() {
  const out = {};
  document.querySelectorAll("#params input[data-param]").forEach((el) => {
    const v = parseFloat(el.value);
    if (!Number.isNaN(v)) out[el.dataset.param] = v;
  });
  return out;
}

/* ── Run ───────────────────────────────────────────────────────────────── */

async function run(event) {
  event.preventDefault();
  const btn = $("run");
  btn.disabled = true;
  $("status").className = "status";
  $("status").textContent = "Submitting…";

  const payload = {
    symbol: $("symbol").value.trim(),
    kind: $("kind").value,
    params: collectParams(),
    cash: parseFloat($("cash").value) || 1000,
    cost_model: { slippage_bps: parseFloat($("slippage").value) || 0 },
  };

  try {
    const job = await api("/backtest", {
      method: "POST",
      headers: { "content-type": "application/json",
                 ...(await Account.headers()) },
      body: JSON.stringify(payload),
    });
    const done = await poll(job.job_id);
    if (done.status === "failed") throw new Error(done.error);
    $("status").textContent = `Done in ${done.duration_sec}s.`;
    await show(done);
    if (Account.user()) loadMyRuns();
  } catch (err) {
    $("status").className = "status err";
    $("status").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function poll(jobId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  let wait = 150;
  while (Date.now() < deadline) {
    const job = await api(`/backtest/${jobId}`);
    if (job.status === "done" || job.status === "failed") return job;
    $("status").textContent = `${job.status}…`;
    await new Promise((r) => setTimeout(r, wait));
    wait = Math.min(wait * 1.4, 1500);   // back off; most finish fast
  }
  throw new Error("Timed out waiting for the backtest.");
}

/* ── Shareable results ─────────────────────────────────────────────────────
   Finished runs are kept server-side, so a run id in the URL is enough to
   bring a result back: after a restart, on another device, for someone else.
   The id is 64 random bits; the link is the only way to find a run. */

async function show(job) {
  render(job.result);
  // Reveal before charting: Chart.js measures its container, and a hidden
  // element is 0x0, which it sizes the canvas to and never recovers from.
  $("results").hidden = false;
  await drawChart(job.job_id);
  history.replaceState(null, "", `?run=${encodeURIComponent(job.job_id)}`);
}

/* Put the form back the way the run was submitted, so a shared link is
   reproducible — change one number and run again. */
function fillForm(req) {
  if (!req) return;
  if (req.symbol) $("symbol").value = req.symbol;
  if (req.kind && CATALOG[req.kind]) { $("kind").value = req.kind; renderParams(); }
  if (req.cash) $("cash").value = req.cash;
  if (req.cost_model) $("slippage").value = req.cost_model.slippage_bps;
  for (const [k, v] of Object.entries(req.params || {})) {
    const el = document.querySelector(`#params input[data-param="${k}"]`);
    if (el) el.value = v;
  }
}

async function openSharedRun() {
  const q = new URLSearchParams(location.search);
  if (q.get("run")) await openRun(q.get("run"));
  else if (q.get("research")) await watchResearch(q.get("research"));
}

async function openRun(id) {
  $("status").className = "status";
  $("status").textContent = "Loading saved result…";
  try {
    const job = await api(`/backtest/${encodeURIComponent(id)}`);
    fillForm(job.meta && job.meta.request);
    if (job.status === "failed") throw new Error(job.error);
    const done = job.status === "done" ? job : await poll(id);
    if (done.status === "failed") throw new Error(done.error);
    $("status").textContent = "";
    await show(done);
  } catch (err) {
    $("status").className = "status err";
    $("status").textContent = `Could not load that result: ${err.message}`;
  }
}

async function copyLink() {
  const btn = $("share");
  try {
    await navigator.clipboard.writeText(location.href);
    btn.textContent = "Copied";
  } catch {
    btn.textContent = "Copy from the address bar";
  }
  setTimeout(() => { btn.textContent = "Copy link"; }, 1800);
}

/* ── Your runs (signed in only) ───────────────────────────────────────── */

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function loadMyRuns() {
  try {
    const runs = await api("/me/runs", { headers: await Account.headers() });
    $("runs-empty").hidden = runs.length > 0;
    $("runs-table").hidden = runs.length === 0;
    $("runs-table").querySelector("tbody").innerHTML = runs.map((r) => {
      const when = new Date(r.submitted_at).toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
      const verdict = r.status === "failed" ? `<span class="no">ERROR</span>`
        : `<span class="${r.passed ? "ok" : "no"}">${r.passed ? "PASS" : "FAIL"}</span>`;
      const num = (v) => (typeof v === "number" ? pct(v) : "—");
      return `<tr data-run="${esc(r.job_id)}">
        <td>${esc(when)}</td><td>${esc(r.symbol)}</td>
        <td>${esc(r.kind.replace(/_/g, " "))}</td><td>${verdict}</td>
        <td class="num">${num(r.total_return_pct)}</td>
        <td class="num">${num(r.excess_return_pct)}</td>
        <td class="num"><button class="ghost" type="button" data-delete="${esc(r.job_id)}"
            title="Delete this run and its link">Delete</button></td></tr>`;
    }).join("");
  } catch (err) {
    $("runs-empty").hidden = false;
    $("runs-empty").textContent = `Could not load your runs: ${err.message}`;
  }
}

$("runs-table").addEventListener("click", async (event) => {
  const del = event.target.closest("[data-delete]");
  if (del) {
    if (!confirm("Delete this run? Anyone you shared its link with will lose it too.")) return;
    try {
      const r = await fetch(`/runs/${encodeURIComponent(del.dataset.delete)}`, {
        method: "DELETE", headers: await Account.headers() });
      if (!r.ok && r.status !== 404) throw new Error(`${r.status} ${r.statusText}`);
      if (new URLSearchParams(location.search).get("run") === del.dataset.delete) {
        $("results").hidden = true;
        history.replaceState(null, "", location.pathname);
      }
      loadMyRuns();
    } catch (err) {
      alert(`Could not delete: ${err.message}`);
    }
    return;
  }
  const row = event.target.closest("tr[data-run]");
  if (row) {
    await openRun(row.dataset.run);
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

Account.onChange((user) => {
  $("my-runs").hidden = !user;
  if (user) loadMyRuns();
  // Anyone with a research link can view it; only signed-in users with
  // research enabled get the form.
  const canRun = !!(user && Account.config().research);
  const viewing = new URLSearchParams(location.search).has("research");
  $("research-form").hidden = !canRun;
  $("r-quota").hidden = !canRun;
  $("research").hidden = !(canRun || viewing);
  if (canRun) loadResearchQuota();
});

/* ── AI research ───────────────────────────────────────────────────────────
   Everything the model wrote (hypotheses, notes) is untrusted text: it can
   echo whatever a user typed as a goal. It only ever reaches the page
   escaped or through textContent. Numbers come from the engine. */

const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");
const params = (p) => Object.entries(p || {})
  .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ");

async function loadResearchQuota() {
  try {
    const me = await api("/me", { headers: await Account.headers() });
    showQuota(me.research);
  } catch { /* the card still works; the server enforces the limit */ }
}

function showQuota(q) {
  if (!q) return;
  $("r-quota").textContent =
    `${q.remaining} of ${q.limit} research sessions left in the last 24 hours.`;
  $("r-run").disabled = q.remaining <= 0;
}

async function startResearch(event) {
  event.preventDefault();
  $("r-run").disabled = true;
  $("r-status").className = "status";
  $("r-status").textContent = "Starting…";
  try {
    const job = await api("/research", {
      method: "POST",
      headers: { "content-type": "application/json", ...(await Account.headers()) },
      body: JSON.stringify({
        symbol: $("r-symbol").value.trim(),
        goal: $("r-goal").value.trim(),
        trials: parseInt($("r-trials").value, 10) || 6,
      }),
    });
    showQuota(job.quota);
    await watchResearch(job.job_id);
  } catch (err) {
    $("r-status").className = "status err";
    $("r-status").textContent = err.message;
  } finally {
    loadResearchQuota();
  }
}

async function watchResearch(id, timeoutMs = 600000) {
  $("research").hidden = false;
  if (!Account.user()) { $("research-form").hidden = true; $("r-quota").hidden = true; }
  history.replaceState(null, "", `?research=${encodeURIComponent(id)}`);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const s = await api(`/research/${encodeURIComponent(id)}`);
    renderResearch(s);
    if (s.status === "done" || s.status === "failed") return s;
    await new Promise((r) => setTimeout(r, 1500));
  }
  throw new Error("Stopped waiting; reload the page to check again.");
}

function renderResearch(s) {
  const st = s.state || {};
  const trials = st.trials || [];
  const status = $("r-status");
  status.className = s.status === "failed" ? "status err" : "status";
  status.textContent = {
    queued: "Queued…",
    running: `Running — ${trials.length} trial${trials.length === 1 ? "" : "s"} so far…`,
    done: `Finished: ${trials.length} trials on ${s.symbol}.`,
    failed: s.error || "The session failed.",
  }[s.status] || s.status;
  if (s.symbol && !$("r-symbol").value) $("r-symbol").value = s.symbol;

  $("r-output").hidden = trials.length === 0;
  const picked = st.final && st.final.trial;
  $("r-trials-table").querySelector("tbody").innerHTML = trials.map((t) => {
    const m = t.summary;
    const gate = !m ? `<span class="no" title="${esc(t.error)}">ERROR</span>`
      : `<span class="${m.passed ? "ok" : "no"}">${m.passed ? "PASS" : "FAIL"}</span>`;
    return `<tr class="${t.n === picked ? "picked" : ""}">
      <td>${t.n}${t.n === picked ? " ★" : ""}</td>
      <td>${esc(String(t.kind || "").replace(/_/g, " "))}</td>
      <td>${esc(params(t.params))}</td>
      <td class="hyp">${esc(t.hypothesis)}</td>
      <td class="num">${m ? fmt(m.sharpe) : "—"}</td>
      <td class="num">${m ? fmt(m.oos_sharpe) : "—"}</td>
      <td class="num">${m && typeof m.excess_return_pct === "number" ? pct(m.excess_return_pct) : "—"}</td>
      <td>${gate}</td></tr>`;
  }).join("");

  const f = st.final;
  $("r-final").hidden = !(s.status === "done" && f);
  if (s.status === "done" && !f) {
    status.textContent = "Finished, but no trial was valid, so nothing was tested on the holdout.";
  }
  if (!f || s.status !== "done") return;

  const [from, to] = st.holdout_window || [];
  $("r-window").textContent = from ? `— ${from} to ${to}, never seen by the model` : "";
  $("r-badge").textContent = f.passed ? "PASS" : "FAIL";
  $("r-badge").className = `badge ${f.passed ? "pass" : "fail"}`;
  $("r-final-sub").textContent =
    `Trial ${f.trial}, ${f.kind.replace(/_/g, " ")} (${params(f.params)}), on unseen data.`;
  $("r-stats").innerHTML = [
    ["Return", pct(f.total_return_pct)], ["Buy & hold", pct(f.benchmark_return_pct)],
    ["Excess", pct(f.excess_return_pct)], ["Sharpe", fmt(f.sharpe)],
    ["Max drawdown", `${fmt(f.max_drawdown_pct, 1)}%`], ["Trades", String(f.trades)],
  ].map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  $("r-gates").querySelector("tbody").innerHTML = f.checks.map((c) => `
    <tr><td class="${c.passed ? "ok" : "no"}">${c.passed ? "PASS" : "FAIL"}</td>
        <td>${esc(c.name)}</td><td class="num">${esc(c.detail)}</td></tr>`).join("");
  $("r-notes").textContent = st.notes || "The model left no notes.";
  const who = st.picked_by === "model" ? "the model"
    : "the engine (the model did not nominate a valid trial)";
  $("r-disclosure").textContent =
    `Picked by ${who} from ${f.trials_tried} trials. Every trial searched the ` +
    `same research window, so the best of them is flattered by the search; ` +
    `the holdout result above is the one that counts. ` +
    `${st.input_tokens + st.output_tokens} tokens used.`;
  $("r-load").onclick = () => {
    fillForm({ symbol: s.symbol, kind: f.kind, params: f.params });
    $("form").scrollIntoView({ behavior: "smooth" });
  };
}

$("research-form").addEventListener("submit", startResearch);

/* ── Render ────────────────────────────────────────────────────────────── */

function render(r) {
  $("verdict-badge").textContent = r.passed ? "PASS" : "FAIL";
  $("verdict-badge").className = `badge ${r.passed ? "pass" : "fail"}`;
  const failed = r.checks.filter((c) => !c.passed).length;
  $("verdict-sub").textContent = r.passed
    ? `${r.symbol} cleared every gate.`
    : `${r.symbol} failed ${failed} of ${r.checks.length} gate${failed === 1 ? "" : "s"}.`;
  $("overfit").hidden = !r.likely_overfit;
  $("lg-strategy").textContent = r.symbol;

  const tiles = [
    ["Return", pct(r.total_return_pct)],
    ["Benchmark", pct(r.benchmark_return_pct)],
    ["Excess", pct(r.excess_return_pct)],
    ["Sharpe", r.sharpe.toFixed(2)],
    ["Out-of-sample", r.oos_sharpe.toFixed(2)],
    ["Max drawdown", `${r.max_drawdown_pct.toFixed(1)}%`],
    ["Trades", String(r.trades)],
    ["Win rate", `${r.win_rate.toFixed(0)}%`],
  ];
  $("stats").innerHTML = tiles
    .map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");
  $("cost-hint").textContent =
    `Net of costs. ${r.bars} bars via ${r.provider}. ` +
    `Profit factor ${Number.isFinite(r.profit_factor) ? r.profit_factor.toFixed(2) : "∞"}.`;

  $("gates").querySelector("tbody").innerHTML = r.checks.map((c) => `
    <tr><td class="${c.passed ? "ok" : "no"}">${c.passed ? "PASS" : "FAIL"}</td>
        <td>${c.name}</td><td class="num">${c.detail}</td></tr>`).join("");

  const is_ = r.in_sample, oos = r.out_of_sample;
  $("robust").querySelector("tbody").innerHTML = `
    <tr><td>In-sample Sharpe</td><td class="num">${is_.sharpe.toFixed(2)}</td></tr>
    <tr><td>Out-of-sample Sharpe</td><td class="num">${oos.sharpe.toFixed(2)}</td></tr>
    <tr><td>In-sample return</td><td class="num">${pct(is_.total_return_pct)}</td></tr>
    <tr><td>Out-of-sample return</td><td class="num">${pct(oos.total_return_pct)}</td></tr>
    <tr><td>Folds positive</td><td class="num">${r.consistency}</td></tr>`;

  $("folds").querySelector("tbody").innerHTML = r.folds.map((f, i) => `
    <tr><td>${i + 1}</td>
        <td class="num">${pct(f.total_return_pct)}</td>
        <td class="num">${f.sharpe.toFixed(2)}</td>
        <td class="num">${f.max_drawdown_pct.toFixed(1)}%</td>
        <td class="num">${f.trades}</td></tr>`).join("");
}

async function drawChart(jobId) {
  const s = await api(`/backtest/${jobId}/equity`);
  if (chart) chart.destroy();

  const line = (label, data, color) => ({
    label, data, borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.05,
  });

  chart = new Chart($("chart"), {
    type: "line",
    data: {
      labels: s.t,
      datasets: [
        line($("symbol").value.trim() || "Strategy", s.strategy, css("--series-strategy")),
        line("Buy & hold", s.benchmark, css("--series-benchmark")),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },          // rendered in HTML, above
        tooltip: {
          callbacks: { label: (c) => `${c.dataset.label}: ${money(c.parsed.y)}` },
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8, color: css("--text-muted") },
             grid: { display: false } },
        y: { ticks: { callback: money, color: css("--text-muted") },
             grid: { color: css("--border") } },
      },
    },
  });
}

/* Offer only symbols the server can actually serve, and open on one known to
   produce a meaningful result. The form used to default to SPY, which the
   bundled sample data does not contain, so a first click always errored. */
async function loadSymbols() {
  const { symbols, default: fallback } = await api("/symbols");
  if (symbols && symbols.length) {
    $("symbol-list").innerHTML = symbols
      .map((s) => `<option value="${s}"></option>`).join("");
  }
  if (!$("symbol").value) $("symbol").value = fallback || "SPY";
}

$("form").addEventListener("submit", run);
$("kind").addEventListener("change", renderParams);
$("share").addEventListener("click", copyLink);
Promise.all([loadStrategies(), loadSymbols()]).then(openSharedRun, (e) => {
  $("status").className = "status err";
  $("status").textContent = `Could not reach the API: ${e.message}`;
});
// Separate from the above: if sign-in cannot load, backtests still work.
Account.init().catch((e) => console.warn("Accounts unavailable:", e.message));
