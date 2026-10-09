# quantlab

Strategy backtesting with honest overfit detection.

Most retail backtesters will happily show you an in-sample curve fit and call
it a strategy. This one splits the data, tests out-of-sample, walks forward,
and tells you when a result is fitted noise.

Phase 1 of a planned web platform. Right now it is a dependency-free Python
package with no web layer — that comes next.

## Install

```bash
pip install -e ".[dev]"
pytest
```

The engine itself has no dependencies beyond the standard library; the web
layer adds FastAPI, uvicorn and pydantic.

## Use

```python
from quantlab import backtest, StrategySpec
from quantlab.providers import get_provider

bars = get_provider("local", root="tests/fixtures").bars("SPY")

result = backtest(bars,
                  StrategySpec("trend_following", {"fast": 10, "slow": 30}),
                  cost_model={"slippage_bps": 10})

print(result["sharpe"], result["excess_return_pct"], result["passed"])
for name, ok, detail in result["checks"]:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")
```

## What it guarantees

**No look-ahead.** A signal function sees only `bars[:i+1]`, and fills happen
at the *next* bar's open — never the close that produced the signal. Both are
enforced by tests, because every other number is meaningless if either breaks.

**Costs are real.** Slippage applies to both sides of a trade; commission is
charged per fill. A strategy that only works at zero cost fails here.

**Benchmarked.** Every result carries buy-and-hold over the same window.
Absolute return says nothing on its own — a strategy can post a 2.0 Sharpe
and still lose badly to doing nothing.

## Robustness

| Check | What it catches |
|---|---|
| In-sample / out-of-sample split | Parameters fitted to the whole history |
| Walk-forward (4 folds) | Edge that exists in one regime only |
| Parameter sensitivity sweep | A lone peak that collapses either side |
| `likely_overfit` flag | In-sample pass with out-of-sample failure |

Ranking uses **out-of-sample** Sharpe, since in-sample performance is the part
most easily curve-fitted.

## Strategies

| Kind | Parameters |
|---|---|
| `momentum` | `lookback`, `move_pct` |
| `mean_reversion` | `rsi_period`, `rsi_low`, `rsi_high` |
| `trend_following` | `fast`, `slow` |
| `breakout` | `lookback` |
| `volatility` | `fast`, `slow`, `spread_pct` |

Each parameter is named for the strategy that uses it. An earlier version
shared one `threshold` field across all of them, which meant a percent move
for momentum and an RSI level for mean reversion — at `threshold=2` that
produced 68 momentum signals and zero mean-reversion signals.

## Providers

The engine never talks to a vendor. It asks a `DataProvider` for bars.

| Provider | Needs | Notes |
|---|---|---|
| `local` | Nothing | CSV and JSONL on disk. Used by every test. |
| `alphavantage` | `ALPHAVANTAGE_API_KEY` | Free tier is daily bars only — intraday and options are premium, verified by probe. |

Adding one means implementing a single method:

```python
class MyProvider(DataProvider):
    def bars(self, symbol, interval="day", limit=None):
        ...
        return self._finish(rows, limit)
```

## HTTP API

```bash
pip install -e ".[web]"
QUANTLAB_DATA_ROOT=tests/fixtures uvicorn api.main:app --reload --workers 1
```

Interactive docs at `/docs`.

| Route | Purpose |
|---|---|
| `GET /health` | Version, active provider, cache size, jobs in flight |
| `GET /strategies` | Every kind with parameter names and defaults |
| `GET /providers` | Which providers this server can actually serve |
| `POST /backtest` | Queue a backtest, returns `202` and a job id |
| `GET /backtest/{id}` | Poll for status and result |
| `GET /backtest/{id}/equity` | Equity + benchmark series, for charting |
| `GET /symbols` | Symbols the active provider can serve, plus a sensible default |
| `GET /jobs` | Recent jobs — **admin** |
| `GET`/`DELETE /cache` | Inspect or clear cached bars — **admin** |

Backtests run as background jobs because they are CPU-bound — a walk-forward
plus a sweep takes seconds to minutes, which is far too long for a request
handler. Submit, get an id, poll.

```bash
curl -X POST localhost:8000/backtest -H 'content-type: application/json'   -d '{"symbol":"SPY","kind":"breakout","params":{"lookback":20}}'
# {"job_id":"fad81883b3304c63","status":"queued",...}

curl localhost:8000/backtest/fad81883b3304c63
```

**Run one worker.** Jobs live in process memory, so a second uvicorn worker
would accept a submission on one process and be asked for it on another,
returning 404 for a job running perfectly well next door.

### Why a thread pool and not Redis

The plan called for arq plus Redis. The part that is painful to retrofit is
the *submit -> poll* API shape, not whatever executes the work, and requiring
a broker to run the dev server is real friction for no present benefit.

`JobStore` is the seam. Moving to arq, Celery or RQ means writing one class
with the same four methods and changing the wiring in `api/deps.py`. The API
and any frontend never notice.

### Caching

Provider bars are cached to disk before anything else touches the API,
because development alone will exhaust a free vendor tier in an afternoon —
the same backtest re-run with different parameters asks for the same bars
every time. `CachedProvider` satisfies the provider interface, so the engine
cannot tell the difference.

A stale entry is served when the provider fails. Yesterday's bars beat an
error page.

### Saved runs and result links

Every finished backtest — passed, failed or errored — is written to a
database (`api/store.py`). `GET /backtest/{id}` and `/equity` read memory
first, where a job in flight is the only copy, then the database. A result
therefore survives restarts, redeploys and the in-memory history cap.

The web UI puts the run id in the address bar (`/?run=<id>`) and has a
**Copy link** button. Opening a link restores the form exactly as it was
submitted, so a shared result is reproducible, not just viewable. The id is
64 random bits and nothing lists runs publicly: the link is the only way in.

`DATABASE_URL` picks the database. Unset means SQLite in `quantlab.db`, which
is right for development and wrong for an ephemeral-disk host, where it is
wiped on every restart. Any `postgres://` URL works (Neon, Supabase, Render
Postgres). Runs older than `QUANTLAB_RUN_RETENTION_DAYS` are pruned.

Saving is best effort. If the database is down the job still completes and
its result is still served from memory; the failure is logged.

### Accounts

Optional, through Supabase Auth. Anonymous visitors keep everything above
under the per-IP limit. Signing in adds:

- **Your runs**: every backtest started while signed in, newest first, with
  a delete button. Deleting a run also kills its link.
- **A per-account limit** (`QUANTLAB_USER_RATE_LIMIT`, default 60 per window)
  in place of the per-IP one. An account is a better identity than an
  address: an office shares one IP, and one person can hop between several.

The browser signs in with supabase-js and sends its access token as a Bearer
header. The server verifies it locally (`api/auth.py`) against the project's
published signing keys, fetched once and cached: signature, expiry, issuer,
audience, and that it is a user session rather than the anon or service key.
A bad or expired token is a 401, never a silent fall back to anonymous.

Who submitted a run is stored but never returned: a link is shareable, the
account behind it is not. Deleting someone else's run and deleting one that
does not exist both return 404, so ids cannot be probed.

On startup against Postgres the server enables row-level security on its
table. Supabase serves every `public` table over its REST API to anyone
holding the publishable key, which ships to every browser; RLS with no
policies closes that, while the server, as the table owner, is unaffected.

#### Setting up Supabase

1. Create a project at supabase.com.
2. **Connect → Session pooler**: copy the URI, put your database password
   in it, and set it as `DATABASE_URL`. Not the direct connection — it is
   IPv6-only and Render cannot reach it.
3. **Project Settings → API**: set `SUPABASE_URL` to the project URL and
   `SUPABASE_PUBLISHABLE_KEY` to the publishable key (`sb_publishable_…`;
   the legacy anon key also works).
4. **Authentication → URL Configuration**: set the Site URL to the deployed
   address, so confirmation and password-reset emails link back to it.
5. Redeploy. `/health` should report `postgresql`, and a **Sign in** button
   appears top right.

Supabase's built-in email sender is rate-limited and meant for testing. For
real sign-ups, add an SMTP provider under Authentication → Emails.

Free Supabase projects pause after a week without activity; a paused
project takes sign-in and saved runs down with it until restored from the
dashboard. Older projects that still sign sessions with the legacy shared
secret need `SUPABASE_JWT_SECRET` as well.

## Web UI

```bash
pip install -e ".[dev]"
QUANTLAB_DATA_ROOT=tests/fixtures uvicorn api.main:app --reload --workers 1
# open http://127.0.0.1:8000
```

Served from the same process as the API, so there is one thing to run and no
CORS in development. No build step, no framework, no node — plain HTML, CSS
and one script, with Chart.js from a CDN for the equity curve.

Four panels, in the order the question gets answered:

1. **Verdict** — pass or fail, and how many gates failed. Flags a result that
   is profitable in-sample but not out-of-sample as a likely fitted curve.
2. **Performance** — return against benchmark, excess, Sharpe, out-of-sample
   Sharpe, drawdown, trades, win rate.
3. **Equity curve** — strategy against buy-and-hold on one axis, both after
   costs. Beating the benchmark line is the bar, so they are drawn together.
4. **Gates, robustness and walk-forward folds** — every criterion with its
   measured value, in-sample against out-of-sample, and each fold separately.

Strategy parameter inputs are generated from `GET /strategies`, so adding a
strategy server-side needs no frontend change.

The two series use slots 1 and 2 of a validated categorical palette: ΔE 33.6
normal vision and 24.7 worst-case colour-vision deficiency, both modes
checked against their own surface. Light and dark are separate selections
rather than an automatic flip.

## Deploying

This app needs a **persistent process**. It will not work on serverless
platforms as built, and that is not a configuration gap:

- Jobs live in a thread pool and an in-memory dict. On serverless, `POST`
  returns a job id from one instance and `GET` asks a different one, so the
  poll 404s forever — and the function is frozen when it responds, so the
  backtest never finishes anyway.
- The bar cache writes to disk. Serverless filesystems are read-only outside
  `/tmp`, which does not persist between invocations.

Render, Railway and Fly all run a real process and work unchanged. A
`Procfile`, `requirements.txt` and `render.yaml` are included.

```bash
# whatever the host runs reduces to this
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

**One worker.** Jobs are in process memory, so a second worker would 404 a job
running perfectly well in its sibling. When that limit starts to bite, the fix
is swapping `JobStore` for a real queue — one class, same four methods — not
adding workers.

On a host with an ephemeral filesystem, point `QUANTLAB_CACHE_ROOT` at `/tmp`
or accept that each run refetches, and set `DATABASE_URL` to a Postgres
instance or saved runs vanish on each deploy. Render's free Postgres expires
after 30 days; Neon's free tier does not. On Render, set it under the
service's Environment tab — `render.yaml` declares it without a value.

### Sample data

`sampledata/` holds synthetic bars in three regimes, so a fresh clone or
deploy works with no API key and no provider account.

Synthetic is deliberate. Real market data carries redistribution restrictions
that would make committing it to a public repository a licensing problem;
generated data carries none. It is also not real prices — do not read anything
into a backtest against it.

## Access and limits

**Admin endpoints** — `GET /jobs`, `GET /cache`, `DELETE /cache` — require
`Authorization: Bearer $QUANTLAB_ADMIN_TOKEN`. They reveal what other people
queried and can wipe state. With no token configured they are **disabled**,
so a fresh deploy is closed by default. `render.yaml` has Render generate the
token; read it from the dashboard.

Individual jobs stay readable by id without a token. Ids are 64 random bits,
so holding one means you submitted it or were given it.

**Submissions are throttled two ways:**

- A **global cap** on jobs queued or running (default 8). This is what
  protects the instance, and it holds no matter who is asking.
- A **per-client rate** (default 20 per 60s), returning `429` with
  `Retry-After`. This is fairness, not protection: a determined client can
  rotate addresses.

Identifying the client behind proxies is where rate limits usually fail
open. On Render, requests pass Cloudflare and then a load balancer, so
`X-Forwarded-For` arrives as `client, edge, lb`. The left end is
client-written; the right end comes from rotating infrastructure pools.
Trusting either lets a client escape the limit — both were demonstrated
against the live deploy. The app instead reads `CF-Connecting-IP`
(`QUANTLAB_CLIENT_IP_HEADER`), which Cloudflare overwrites on every request.
Elsewhere, set `QUANTLAB_TRUST_PROXY_HOPS` to the exact number of appending
proxies, or leave both unset to use the socket address.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `QUANTLAB_PROVIDER` | `local` | Default data provider |
| `QUANTLAB_DATA_ROOT` | `sampledata` | Where `local` reads files |
| `QUANTLAB_CACHE_ROOT` | `cache` | Cached bars |
| `QUANTLAB_CACHE_TTL` | `43200` | Cache lifetime, seconds |
| `QUANTLAB_WORKERS` | `2` | Concurrent backtest jobs |
| `QUANTLAB_MIN_BARS` | `60` | Refuse to backtest less than this |
| `QUANTLAB_MAX_BARS` | `5000` | Cap bars per run |
| `QUANTLAB_ADMIN_TOKEN` | *(unset)* | Enables admin endpoints; unset disables them |
| `QUANTLAB_RATE_LIMIT` | `20` | Submissions per client per window |
| `QUANTLAB_RATE_WINDOW` | `60` | Window, seconds |
| `QUANTLAB_MAX_INFLIGHT` | `8` | Queued + running jobs, all clients |
| `QUANTLAB_CLIENT_IP_HEADER` | *(unset)* | Edge-set client IP header, e.g. `cf-connecting-ip` |
| `QUANTLAB_TRUST_PROXY_HOPS` | `0` | Trusted appending proxies; 0 uses the socket address |
| `DATABASE_URL` | `sqlite:///quantlab.db` | Where finished runs are kept; Postgres in production |
| `QUANTLAB_RUN_RETENTION_DAYS` | `30` | Delete older runs; 0 keeps them forever |
| `SUPABASE_URL` | *(unset)* | Enables accounts; unset means everyone is anonymous |
| `SUPABASE_PUBLISHABLE_KEY` | *(unset)* | Browser key (`SUPABASE_ANON_KEY` also read) |
| `SUPABASE_JWT_SECRET` | *(unset)* | Only for projects on the legacy HS256 secret |
| `QUANTLAB_USER_RATE_LIMIT` | `60` | Submissions per signed-in user per window |

## Known issues

*Resolved:* the fixed `min_trades: 30` gate rejected every daily-bar strategy,
since they make 2–14 trades a year. It is replaced by a one-sided t-test on
per-trade returns, whose bar rises automatically as the sample shrinks.

*Resolved:* a position still open at the last bar appended its close-out as
an extra equity point, so the strategy curve was one point longer than the
dates and the benchmark. The close-out now replaces the last bar's mark.

## Licensing note

Market data redistribution is restricted by every major vendor. Free API tiers
permit personal use, not display to third parties. Options data is stricter
still: OPRA charges a redistribution fee independent of user count, with an
exemption for historical-only products.

`data/` and `cache/` are gitignored. Do not commit vendor data.

## Layout

```
quantlab/
  engine.py            metrics, simulation, robustness, grading
  strategies.py        specs, indicators, the compiler
  providers/
    base.py            DataProvider interface
    local.py           CSV / JSONL
    alphavantage.py    daily bars
  cache.py             disk cache + CachedProvider wrapper
  jobs.py              background job store
api/
  main.py              FastAPI routes
  schemas.py           request/response models and validation
  deps.py              settings, provider wiring, singletons
  store.py             saved runs (SQLite / Postgres)
  auth.py              Supabase token verification, current user
  static/, templates/  web UI — no build step
tests/                 114 tests, hermetic, no network
```
