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
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const done = await poll(job.job_id);
    if (done.status === "failed") throw new Error(done.error);
    $("status").textContent = `Done in ${done.duration_sec}s.`;
    render(done.result);
    // Reveal before charting: Chart.js measures its container, and a hidden
    // element is 0x0, which it sizes the canvas to and never recovers from.
    $("results").hidden = false;
    await drawChart(job.job_id);
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

$("form").addEventListener("submit", run);
$("kind").addEventListener("change", renderParams);
loadStrategies().catch((e) => {
  $("status").className = "status err";
  $("status").textContent = `Could not reach the API: ${e.message}`;
});
