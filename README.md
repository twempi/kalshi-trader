# kalshi-trader

A locally hosted, **multi-strategy** quantitative research and **paper-trading** platform for
Kalshi prediction markets.

It discovers markets, streams their order books (and, for tennis, live point-by-point scores),
lets several independent strategies propose opportunities, evaluates every proposal at its
*actual executable* size and cost, ranks them centrally, and executes the survivors in a
**simulated** paper account against real, live Kalshi prices. Every observation and every
decision (including every rejection and its reason) is recorded so strategies can be replayed,
compared and either validated or rejected on evidence.

No strategy in this repository has demonstrated profitability. The platform exists to find out
whether any can, without assuming it. Zero trades is an acceptable outcome when nothing clears
the requirements.

> **No real money.** The application never places real orders. The REST client refuses any
> non-`GET` request (`ReadOnlyViolation`), no order endpoints exist in the code base, and a test
> asserts that. API credentials are optional and only needed for the real-time WebSocket feed.
> All P&L shown anywhere is **simulated**.

---

## Contents

1. [What works today](#what-works-today)
2. [Architecture](#architecture)
3. [Strategies and operating modes](#strategies-and-operating-modes)
4. [How opportunities are evaluated, ranked and allocated](#how-opportunities-are-evaluated-ranked-and-allocated)
5. [The tennis probability model](#the-tennis-probability-model)
6. [Installation on NixOS](#installation-on-nixos)
7. [Configuration and API credentials](#configuration-and-api-credentials)
8. [Running the application](#running-the-application)
9. [The dashboard](#the-dashboard)
10. [Starting, pausing and stopping paper trading](#starting-pausing-and-stopping-paper-trading)
11. [Running continuously (systemd user service)](#running-continuously-systemd-user-service)
12. [Running on a NixOS server (nix-inputs module)](#running-on-a-nixos-server-nix-inputs-module)
13. [Backtesting and research](#backtesting-and-research)
14. [Strategy evaluation and selection](#strategy-evaluation-and-selection)
15. [Understanding the performance statistics](#understanding-the-performance-statistics)
16. [Running the tests](#running-the-tests)
17. [Logs and data files](#logs-and-data-files)
18. [Data sources: free vs paid](#data-sources-free-vs-paid)
19. [Known limitations and missing data](#known-limitations-and-missing-data)
20. [Results to date](#results-to-date)
21. [Milestone status](#milestone-status)
22. [Troubleshooting](#troubleshooting)

---

## What works today

| Capability | Status |
| --- | --- |
| Tennis market discovery (ATP, WTA, Challenger, ITF match-winner series) | Verified live 2026-11-02: 45 matches / 90 markets |
| Generic event discovery for cross-market research (`GET /events?with_nested_markets`) | Implemented and unit-tested; opt-in per series (`strategies.cross_market.series_tickers`), not yet run live |
| Order books over REST (public) / WebSocket (API key) | REST verified live; WebSocket tested against a local fake server only |
| Live tennis scores (Kalshi Live Data) with freshness checks | Verified live; CDN `Age` header captured, out-of-order responses rejected |
| Shared strategy framework (Opportunity, evaluator, risk, allocator, diagnostics) | Unit- and integration-tested; run live 2026-11-02 |
| Strategy A/C `tennis_model` (statistical mispricing, all match stages) | Paper mode; run live, no qualifying opportunity yet |
| Strategy B `cross_market` (verified settlement relationships, multi-leg) | Paper mode for tennis pairs; generic relationships unit-tested only |
| Strategy D `market_making` | **Research simulation only** (pessimistic fill model); never paper capital |
| Baselines: `late_match` (original rules), `price_threshold`, market-implied benchmark | `late_match` observe mode; `price_threshold` off by default; benchmark in model evaluation |
| Per-strategy shadow ledgers + evidence-based strategy lifecycle | Unit-tested; run live |
| Paper broker: book-walking fills, exact fees, partial fills, basket unwinds, settlement | Unit-tested with hand-checked numbers |
| Event-driven replay with chronological settlement | Tested (capital-recycling scenario) and run on the recorded database |
| Model evaluation vs market, sensitivity analysis | Implemented; 894 settled matches recorded on the server by 2026-11-04 |
| Probability calibration of the tennis model (held-out fit, decision check, `research calibrate`) | Implemented and unit-tested; not yet fitted on the server's data, so off in the shipped config |
| Dashboard: overview, opportunities, strategies, diagnostics, research, markets, analytics, health | Verified in a browser against live-recorded data |
| NixOS service module (nix-inputs) / systemd user unit | Unchanged; NixOS module tested in VMs earlier |

## Architecture

```
 Kalshi REST (public)         Kalshi WebSocket (API key)        Kalshi Live Data (public)
 markets, events, books,      order-book deltas,                tennis scores
 status, settlement           lifecycle events                  (+ HTTP Age of the cache)
        │                             │                                 │
        ▼                             ▼                                 ▼
 TennisDiscovery / GenericDiscovery ─► MarketDataHub ◄──┐        TennisTracker
 (tracked markets, related groups)    (valid books)    │   (regression guard, freshness,
                                              │         │    pre-match prior, Markov model)
                                              ▼         │                │
                         ScanContext (one instant: markets, books, matches, groups)
                                              ▼
              Strategies ──► Opportunity proposals (candidates + filtered/unsupported notes)
   tennis_model · cross_market · late_match · price_threshold     market_making (research sim)
                                              ▼
 TradingCore (shared by live engine and replay; clock injected)
   market checks ─► portfolio checks ─► risk budget ─► ExecutionEvaluator ─► allocator rank
   ─► re-check on current state ─► PaperBroker (main account)   +   per-strategy shadow ledgers
                                              ▼
 SQLite (WAL): snapshots · opportunities · diagnostics · predictions · orders · trades · equity
               · strategy state · safety flags · backtest runs (with config, universe, ledger)
                                              ▼
                     Dashboard (FastAPI, 127.0.0.1) · CLI · replay · research
```

| Path | Responsibility |
| --- | --- |
| `src/kalshi_trader/core/opportunity.py` | Standard `Opportunity` / `Leg` / `FairValue` types and decision statuses |
| `src/kalshi_trader/core/evaluator.py` | Execution-aware sizing: walks the book for the final size, exact fees, EV at the final fill, basket legging risk |
| `src/kalshi_trader/core/allocator.py` | Ranking objective (documented below) |
| `src/kalshi_trader/core/pipeline.py` | `TradingCore`: the single decision path, main + shadow ledgers, basket execution |
| `src/kalshi_trader/core/registry.py` | Strategy lifecycle and evidence requirements |
| `src/kalshi_trader/core/diagnostics.py` | Per-strategy outcome/reason counts and the opportunity log |
| `src/kalshi_trader/core/tennis_tracker.py` | Tennis state shared by live and replay: freshness, regression guard, priors, estimates |
| `src/kalshi_trader/strategies/` | `stat_mispricing.py` (A/C), `cross_market.py` (B), `market_making.py` (D), `late_match.py`, `price_threshold.py` |
| `src/kalshi_trader/risk/manager.py` | Market/portfolio checks, exposure caps, risk budget, persistent safety state |
| `src/kalshi_trader/execution/` | Fee formula, fill simulation, paper broker (main + shadow accounts) |
| `src/kalshi_trader/backtesting/` | Replay (event-driven), historical price-only backtest, metrics, significance |
| `src/kalshi_trader/research/` | Event/time splits, model evaluation vs market, probability calibration fit, sensitivity analysis |
| `src/kalshi_trader/market_data/` | Tennis + generic discovery, order books, recorder, pre-match capture |
| `src/kalshi_trader/api/`, `tennis/`, `models/` | Kalshi clients and schemas, tennis scoring and parsing, Markov model (unchanged design) |
| `src/kalshi_trader/engine.py` | Async loops, task supervision/restart, settlement monitor |
| `src/kalshi_trader/dashboard/` | FastAPI app and static front end (no external CDNs) |

All components run in one asyncio process (`kalshi-trader run`). Each loop survives its own
errors; a supervisor restarts any background task that ends unexpectedly (with backoff), and
repeated strategy/settlement errors auto-pause paper trading. Existing databases are migrated
additively on start-up (new tables and nullable columns only).

Market data and score feeds are treated as **untrusted input**: tickers are validated, prices
must lie in `[0, 1]`, sizes must be non-negative, malformed messages are rejected and counted,
sequence gaps mark books out of sync until a fresh snapshot arrives, and internally inconsistent
score payloads never reach a decision.

## Strategies and operating modes

Each strategy has a `mode` in `config/settings.toml`:

| Mode | Behaviour |
| --- | --- |
| `off` | Not loaded |
| `observe` | Scanned and fully evaluated (size, fees, risk) against the main account but ends as `observed`; trades only its own **shadow ledger** |
| `paper` | Eligible for the **main paper account** through the central allocator (and also trades its shadow ledger) |

| Strategy | What it does | Default | Status of the evidence |
| --- | --- | --- | --- |
| `tennis_model` (A/C) | Buys a tennis contract when the Markov model's **conservative** probability beats the full executable cost by `min_edge` after fees. Works at any configured stage: `early` (first set), `mid`, `late` (a player within 3 games of winning). `prematch` is always *unsupported*: the only pre-match information is the market price itself, which cannot prove the market wrong. Requires a pre-match price prior by default; other categories are reported as unsupported ("no independent probability model"). | `paper` | Unvalidated: model calibration vs market not yet measurable (see Results) |
| `cross_market` (B) | Buys baskets whose combined payoff is fixed by **verifiable** settlement rules: tennis pairs (after the first ball), `mutually_exclusive` NO-baskets (pays ≥ N−1), exhaustive bracket YES-baskets (strict coverage check, no integer-outcome assumption), threshold ladders. Requires the mispricing to persist (2 scans, 2 s), includes fees for every leg, budgets legging risk, executes least-liquid leg first and unwinds excess legs into the bid. Never recorded as risk-free. | `paper` | No opportunity observed so far |
| `late_match` (baseline 2) | The original late-match favourite rules, unchanged, for comparison | `observe` | Price-only version lost money historically |
| `price_threshold` (baseline 1) | Buy the favourite when its ask is in a band; no model, not edge-gated | `off` | Lost money historically (see Results) |
| market-implied benchmark (baseline 3) | Market mid used as a probability forecast; every model is scored against it | always on in evaluation | – |
| `market_making` (D) | Hypothetical one-tick-improved quotes on wide markets; simulated maker fills only when the market trades **through** the quote; maker fee always charged; inventory limits; adverse selection measured. **Cannot** be set to `paper`. | `observe` (research) | Insufficient data; fill model is deliberately pessimistic |

Change modes in `config/settings.toml` or from the environment, e.g.
`KT_STRATEGIES__LATE_MATCH__MODE=paper`. Changing a strategy's parameters changes its version
tag (`version+confighash`), which resets its evidence status: results earned under different
rules never count. Operators can also pause/resume a single strategy at runtime (persisted):

```bash
uv run kalshi-trader strategies                      # modes, statuses, evidence
uv run kalshi-trader strategies pause tennis_model   # stop main-account entries for one strategy
uv run kalshi-trader strategies resume tennis_model
```

## How opportunities are evaluated, ranked and allocated

Every proposal passes, in order (`core/pipeline.py`):

1. **Strategy filters.** Failures are kept as `filtered` / `unsupported` with the reason.
2. **Market checks** per leg (`risk/manager.py`): exchange and market open, order book younger
   than 10 s, for score strategies a score younger than 30 s **including the source's cache age**
   (HTTP `Age`, or 10 s when unknown), score not frozen for > 300 s, market mid not moved > 10¢
   since the last score change, no score-feed anomaly; spread ≤ 5¢, ≥ 10 contracts at the limit,
   trade-count limits per market/event, not already exposed to the event.
3. **Portfolio checks**: emergency stop, operator pause, max open positions, daily loss, drawdown.
4. **Risk budget**: the maximum *loss* this trade may add = the smallest of per-trade cap
   (2% of equity, ≤ $25), cash, and the remaining headroom of the aggregate (30%), per-event (3%),
   per-strategy (15%) and correlation-group (8%; same tournament or series) caps. Exposure is
   always measured as **maximum possible loss**, not win probability.
5. **Execution evaluator** (`core/evaluator.py`): finds the largest whole-contract order that fits
   the budget (× `risk.unvalidated_size_multiplier` for strategies without validated evidence:
   1.0 in the shipped config, so the main account sizes like the shadow ledgers; 0.25 is the
   cautious setting), fits 10% fractional Kelly on
   the conservative probability at the *actual* average cost, is fillable from displayed depth
   (× 50% liquidity haircut) within the limit price, and still clears `min_edge` **after** the
   walked average price and the exact Kalshi fee for that order size:

   ```
   expected_profit_low = n × q_low − (walked notional + fees)    must be ≥ n × min_edge
   expected_profit     = n × p     − (walked notional + fees)
   max_loss            = notional + fees                          (directional)
   baskets: guaranteed = n × G − Σ legs (notional + fees);  max_loss = max(0, −guaranteed, legging risk)
   ```

   NO contracts use the complementary probability (NO on player A pays when player B wins).
6. **Ranking** (`core/allocator.py`). Objective: maximise total conservative expected profit
   subject to the capital and exposure caps. With one binding capital constraint this is a
   knapsack whose LP relaxation is solved exactly by ordering items by value per unit of
   capital, so candidates are ranked by `expected_profit_low / max_loss`. Uncertainty enters
   through `q_low` (a wider posterior lowers it) and evidence through the size multiplier; no
   arbitrary weighted score is used. Baselines (not edge-gated) rank after edge-gated
   candidates.
7. **Allocation**: in rank order, each candidate is re-checked and re-sized against the account
   *as it is after earlier allocations in the same cycle*, on the current book, immediately
   before simulated execution. Two correlated candidates on one event cannot both take capital.

Each order has a deterministic client order id with a database UNIQUE constraint, plus
trade-count limits and a cooldown, so restarts and repeated signals cannot duplicate positions.
Sizing depends only on current equity, exposure and edge: there is **no** martingale, revenge
trading or loss chasing. Daily-loss and drawdown limits block *new* entries; they are **not** a
loss ceiling, because positions already open can still lose their full cost.

**Fees** follow Kalshi's schedule and rounding: `0.07 × multiplier × C × P × (1 − P)` per
order, rounded up to $0.000001, cash debit rounded up to the cent (so small orders pay more per
contract). Event-level fee overrides are read for generic events. The maker fee
(`0.0175 × …`) is used only by the market-making research simulation.

**Paper execution**: IOC limit orders walk the real displayed book, take whole contracts, can
fill partially, and update cash, orders and trades in one transaction. Each trade stores its
ex-ante expected P&L *at the actual fill*, maximum loss and slippage versus the best ask, so
expected and realised results can be compared. Multi-leg baskets: legs after the first may only
take `paper.basket_later_leg_liquidity_fraction` (50%) of their displayed depth, because real
books move between legs; mismatched legs are unwound into the bid and anything unsellable keeps
its full cost as maximum loss. Settlement uses Kalshi's own `result` / `settlement_value`.

## The tennis probability model

Unchanged in design (`src/kalshi_trader/models/`), now with explicit validation tooling:

- *Markov chain*: exact recursion point → game → tiebreak → set → match for all common formats.
- *Player strength*: `p1 = μ + δ/2`, `p2 = μ − δ/2`; `μ` from tour-level serve priors; `δ`
  anchored on the **pre-match market price** (captured live ≥ 60 s before the first live score,
  or backfilled from public candles that predate it), updated with in-match serve points.
- *Uncertainty*: posterior draws; decisions use the 10th percentile, minimised over all scores
  reachable within `latency_points` (1) further points (latency guard).
- *Retirements*: a small per-game hazard pulls probabilities toward ½.
- *Calibration* (optional, `[strategies.tennis_model.calibration]`): before the strategy trades,
  `logit(p) = slope · logit(p_model) + market_weight · logit(p_market)`, applied to the point
  estimate and the conservative bound alike. `slope` > 1 corrects a model that is too timid;
  `market_weight` > 0 shrinks the estimate toward the live mid, so only larger disagreements
  clear the edge. The shipped values (1, 0) are the raw model. Fit them with
  `research calibrate` (below); never set them by hand. Recorded predictions stay raw, so
  refits always start from the model itself. Changing them changes the strategy's version tag,
  so its evidence starts again.

The independent information versus the live market is the in-play state (score, server, serve
statistics) propagated through the Markov chain; the market enters only through the pre-match
prior. Whether that adds accuracy over the live market is an **open empirical question**; the
model is not described as accurate. `kalshi-trader research model-eval` and
`research sensitivity` measure calibration, Brier score and log loss overall and by stage, tour
and probability band, always against the market-implied benchmark, plus sensitivity to serve
priors, prior width, retirement hazard, latency guard and stale (delayed) scores.

## Installation on NixOS

Requirements: Nix with flakes enabled and Git. Nothing is installed system-wide; the flake
provides Python 3.13, `uv` and SQLite in a development shell.

```bash
cd ~/Projects/kalshi-trader
nix develop              # enter the dev shell (Python 3.13, uv, sqlite)
uv sync                  # create .venv from uv.lock (first time and after dependency changes)
uv run pytest            # optional: run the test suite
uv run kalshi-trader scan --live-only
```

If flakes are not enabled on your system, you do not need to change your NixOS configuration.
Pass the feature flag per command instead:

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

Notes:

- Run project commands **inside** `nix develop`. The shell sets `LD_LIBRARY_PATH` so that the
  prebuilt NumPy/pandas wheels can find `libstdc++` on NixOS. It also stops `uv` from downloading
  its own Python.
- Nix only sees files tracked by Git. `flake.nix` and `flake.lock` must stay committed or staged
  (`git add flake.nix flake.lock`). The "Git tree is dirty" warning is harmless.
- One-off commands without an interactive shell:
  `nix develop -c uv run kalshi-trader status`.

## Configuration and API credentials

- `config/settings.toml`: every non-secret setting, with comments (risk limits, strategy
  thresholds, model priors, polling intervals, paper bankroll, dashboard port).
- Environment variables override any setting with the `KT_` prefix and `__` for nesting, e.g.
  `KT_RISK__MAX_OPEN_POSITIONS=3` or `KT_PAPER__STARTING_BALANCE=500`.
- `.env` (git-ignored) holds secrets. Copy `.env.example` to start. Secrets placed in the TOML
  file are **rejected** at start-up.

**No credentials are needed** for scanning, REST order books, live scores, paper trading, the
dashboard or backtests. All of that uses Kalshi's public endpoints.

Optional: to use the lower-latency WebSocket order-book feed, Kalshi requires an authenticated
handshake even for public channels.

1. In your Kalshi account's API key settings, create a key. Choose **read-only** access if
   scopes are offered; this application never needs write/trading permission.
2. Save the private key inside the project and restrict it:
   ```bash
   mkdir -p secrets
   mv ~/Downloads/<your-key-file> secrets/kalshi-read-only.key
   chmod 600 secrets/kalshi-read-only.key
   ```
   (`secrets/`, `*.key` and `*.pem` are git-ignored.)
3. Create `.env`:
   ```bash
   cp .env.example .env
   # then edit .env and uncomment/fill:
   # KT_KALSHI__API_KEY_ID=<your key id>
   # KT_KALSHI__PRIVATE_KEY_PATH=secrets/kalshi-read-only.key
   ```
4. Verify, using read-only calls only:
   ```bash
   uv run kalshi-trader check-auth
   ```
   It prints the key's scopes and warns if the key can trade.

Credentials are never logged. A log processor redacts any field that looks like a key,
signature, secret or authorization header.

## Running the application

All commands are run from `~/Projects/kalshi-trader` inside `nix develop`.

```bash
uv run kalshi-trader scan                 # one-shot table of live + upcoming tennis markets
uv run kalshi-trader scan --live-only     # only matches in progress (score, prices, model)
uv run kalshi-trader scan --json          # machine-readable

uv run kalshi-trader run                  # engine: record data + all strategies + dashboard
uv run kalshi-trader run --record-only    # record + evaluate strategies, never simulate any trade
uv run kalshi-trader run --no-dashboard   # engine without the web server

uv run kalshi-trader dashboard            # dashboard only (reads the database)
uv run kalshi-trader status               # account, safety state, strategies, health, row counts
uv run kalshi-trader strategies           # strategy modes, evidence status, version tags
uv run kalshi-trader diagnostics          # why each strategy is (not) trading today, by reason

uv run kalshi-trader backtest replay      # event-driven replay of your recordings (see below)
uv run kalshi-trader research model-eval --source live    # forward model vs market calibration
uv run kalshi-trader research model-eval --source replay  # same, on replayed recordings
uv run kalshi-trader research calibrate                   # fit + held-out check of a probability calibration
uv run kalshi-trader research sensitivity                 # assumption sensitivity (slow: ~12 replays)
```

Stop `run` with `Ctrl+C` (SIGINT) or SIGTERM. Shutdown is graceful: loops stop, the recorder
flushes, equity is snapshotted, and connections are closed.

`scan` output and the dashboard are labelled with their data source. Live prices and scores are
**real**; model probabilities and EV are **estimates**; account figures are **simulated**.

## The dashboard

Start `uv run kalshi-trader run` (or `uv run kalshi-trader dashboard`) and open
**http://127.0.0.1:8787**. It binds to localhost only. Change the port with
`KT_DASHBOARD__PORT=9000`.

| Tab | Contents |
| --- | --- |
| Overview | Simulated equity, total / realised / unrealised / daily P&L, drawdown, available capital, open positions, active strategies, system health, equity chart, recent activity |
| Opportunities | Scanner of every opportunity evaluated in the last 15 minutes: strategy and mode, market or basket legs, bid/ask, fair value (conservative), size and average fill, fees, expected net profit, expected return on capital, maximum risk, data ages, decision status and rejection reason. Filter by strategy, status and category |
| Strategies | Strategy comparison (main account vs standalone shadow ledgers: P&L, trades, win rate, profit factor, drawdown, capital in use, fees, expected vs realised, evidence status, today's outcome counts) with per-strategy Pause/Resume, and exposure by strategy/event/correlation group against the caps |
| Why no trades? | Today's outcome counts and most common normalised rejection reasons per strategy, plus the last one-second cycle |
| Research | Forward model-vs-market calibration (Brier, log loss, reliability chart, by stage), the probability-calibration fit (candidates scored on held-out matches, whether the model adds anything to the market price, a decision check, and the suggested settings) and the market-making research simulation (hypothetical quote cycles vs simulated fills, adverse selection, P&L) |
| Live tennis markets | One row per player contract: score, server, games to win, bid/ask, spread, depth, market vs model vs robust-low probability, conservative EV, flagged opportunities, data age. Filter by text, tour, live matches, opportunities; sort by EV, match progress, price, liquidity or start time |
| Trading activity | Main-account open positions, cumulative P&L, completed simulated trades with CSV export, and recent decisions that reached the risk/allocation stage |

Raw data for your own analysis (GET, no login, same host restrictions as the dashboard):
`/api/export/predictions.csv` (every settled forward prediction with its outcome),
`/api/export/trades.csv?ledger=tennis_model` and `/api/trades?ledger=tennis_model` (a
strategy's shadow ledger; without `ledger` they return the main account).
| Analytics | Equity curve, P&L over time, drawdown, win rate by entry price vs break-even, outcome distribution, performance by strategy, a "skill or luck?" significance panel, stored backtest runs |
| System health | Recorded-data counts and per-component status (Kalshi REST and WebSocket, score feed, exchange status, engine loops): last update and success, last error, details such as request counts, failures and latency; components that stop updating are marked stale. Recent warnings and errors |

Every chart has a "Table view" toggle. Light and dark themes are supported. Data kinds are kept
apart everywhere: `live_paper` (main account, simulated fills on real books), `shadow`
(per-strategy ledgers), `replay` (backtests), research simulations, and synthetic test data
(tests only, never displayed as results).

The header has **Pause / Resume / Emergency stop / Clear stop** buttons. They work from this
machine and, if you use Tailscale (below), from devices on your tailnet. Every request must carry
a custom header, and the page is served with a strict Content-Security-Policy.

### Viewing it from your phone or laptop (Tailscale)

The dashboard only listens on `127.0.0.1`, so your LAN and the internet cannot reach it. To open
it from any device on your tailnet, let Tailscale Serve share it over HTTPS. Run this once on
the machine that runs the engine:

```bash
sudo tailscale serve --bg 8787
tailscale serve status    # shows the address, e.g. https://<machine>.<your-tailnet>.ts.net/
```

Open that address on any device signed in to your tailnet.

- This puts the dashboard on HTTPS port 443 of the machine's tailnet name, in front of anything
  else serving there. If that port is taken (Caddy uses it on the T480s), give the dashboard its
  own port with `sudo tailscale serve --bg --https=8787 8787` and open
  `https://<machine>.<your-tailnet>.ts.net:8787/`. On a NixOS server, use the
  [module](#running-on-a-nixos-server-nix-inputs-module) instead.
- `--bg` makes it persistent: it keeps working across reboots and engine restarts.
- The first run may print a link to enable HTTPS certificates for your tailnet. Approve it once
  in the Tailscale admin console.
- To skip `sudo` for Tailscale commands, run `sudo tailscale set --operator=$USER` once.
- To stop sharing: `sudo tailscale serve --https=443 off` (or `--https=8787 off`). Avoid
  `sudo tailscale serve reset` on a server: it removes every share on the machine, including
  those of other services.

Security notes:

- Only devices on your tailnet can reach it, plus anyone you share the machine with in Tailscale.
- **Never** use `tailscale funnel` for this dashboard. Funnel publishes to the public internet, and
  the dashboard has no login. As a safeguard it refuses any request that arrives through Funnel.
- Tailnet devices can also use the control buttons. This is set by `dashboard.control_networks`
  in `config/settings.toml`, which lists Tailscale's address ranges. Set it to `[]` to make the
  buttons work only on the machine itself.

## Starting, pausing and stopping paper trading

Paper trading starts automatically with `uv run kalshi-trader run` (unless `--record-only`).

```bash
uv run kalshi-trader trading pause        # stop opening new simulated positions
uv run kalshi-trader trading resume
uv run kalshi-trader trading stop         # emergency stop (persists across restarts)
uv run kalshi-trader trading clear-stop   # clear the emergency stop, then `resume`
uv run kalshi-trader status

uv run kalshi-trader export --out paper_trades.csv     # export simulated trades
uv run kalshi-trader reset-account --yes               # wipe simulated history, restore bankroll
```

The same controls are on the dashboard. A paused engine keeps recording data, evaluating
strategies (shadow ledgers keep running) and settling existing positions. Individual strategies
are paused with `uv run kalshi-trader strategies pause <name>` or the Strategies tab. The emergency stop is also triggered automatically when the drawdown limit
is hit. Change the bankroll with `KT_PAPER__STARTING_BALANCE` before the first run, or with
`reset-account` afterwards.

## Running continuously (systemd user service)

Optional. Nothing below is done automatically. It writes to `~/.config/systemd/user`, outside
this project, so it is left for you to run. Both options use `scripts/run-service.sh`, which runs
the engine inside the project's Nix shell. For an always-on NixOS server, the
[NixOS module](#running-on-a-nixos-server-nix-inputs-module) is the better fit.

**Plain systemd user unit:**

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/kalshi-trader.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kalshi-trader
systemctl --user status kalshi-trader
journalctl --user -u kalshi-trader -f
systemctl --user stop kalshi-trader          # graceful (SIGTERM)
systemctl --user disable --now kalshi-trader # uninstall step 1; then delete the unit file
```

To keep it running while you are logged out: `loginctl enable-linger "$USER"`. The declarative
NixOS equivalent is `users.users.<you>.linger = true;`. That is a system configuration change, so
it is your call.

**Home Manager:** import `deploy/home-manager/kalshi-trader.nix` in your Home Manager
configuration and run `home-manager switch`. The file's header has the details. It is not
imported anywhere by default.

## Running on a NixOS server (nix-inputs module)

[twempi/nix-inputs](https://github.com/twempi/nix-inputs) builds this repository as a Nix package
(with uv2nix, from `uv.lock`) and provides a NixOS module, `services.kalshi-trader`. It pins this
repository from GitHub, so a change here reaches the server only after it is pushed (see
[Updating](#updating-the-server) below).

The module runs `kalshi-trader run` as a system service:

- as its own `kalshi-trader` system user, with systemd sandboxing;
- with the database at `/var/lib/kalshi-trader/kalshi_trader.db` and JSON logs in
  `/var/lib/kalshi-trader/logs/`. Console logs go to the journal;
- with the dashboard on `127.0.0.1` only;
- without credentials unless you add them. It then polls public REST data, as it does without `.env`.

Host configuration, set up like the other services on the T480s (a Tailscale Service in front):

```nix
{ inputs, pkgs, ... }: let
  kalshiTraderPort = 8787;
in {
  imports = [ inputs.twempi.nixosModules.kalshi-trader ];

  services.kalshi-trader = {
    enable = true;
    port = kalshiTraderPort;
  };

  systemd.services.tailscale-serve-kalshi-trader = import ./_tailscale-serve-service.nix {
    inherit pkgs;
    name = "kalshi-trader";
    description = "Expose kalshi-trader as a Tailscale Service";
    port = kalshiTraderPort;
  };
}
```

The Tailscale Service also needs a one-time setup in the Tailscale admin console:

1. Define a service named `kalshi-trader` with endpoint `tcp:8787`.
2. Approve the server as its host.
3. Make sure an access rule lets your devices reach it.

The dashboard is then at `https://kalshi-trader.<your-tailnet>.ts.net:8787/`.

| Option | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Run the engine and the dashboard |
| `port` | `8787` | Dashboard port on `127.0.0.1` |
| `configFile` | this repository's `config/settings.toml` | Non-secret settings. The module always sets `db_path`, `log_dir` and the dashboard port itself |
| `environmentFile` | `null` | Optional `KT_*` secrets kept out of the Nix store, e.g. `KT_KALSHI__API_KEY_ID` and `KT_KALSHI__PRIVATE_KEY_PATH`. The key path must be absolute and readable by the `kalshi-trader` user. Use a read-only key |

The module also installs a `kalshi-trader` command on the server that works on the service's
database. It re-runs itself through `sudo` as the `kalshi-trader` user, so sudo may ask for
your password:

```bash
kalshi-trader status
kalshi-trader trading pause
kalshi-trader backtest historical --days 30 --cache-dir /var/lib/kalshi-trader/cache
kalshi-trader export --out /var/lib/kalshi-trader/paper_trades.csv
journalctl -u kalshi-trader -f
```

Files it writes belong to `kalshi-trader`, so put output paths under `/var/lib/kalshi-trader`.

To keep the history recorded on another machine, stop the engine there, copy its
`data/kalshi_trader.db` to the server, and install it while the service is stopped:

```bash
sudo systemctl stop kalshi-trader
sudo install -o kalshi-trader -g kalshi-trader -m 640 kalshi_trader.db /var/lib/kalshi-trader/
sudo systemctl start kalshi-trader
```

### Updating the server

1. Commit and push this repository.
2. In nix-inputs: `nix flake update kalshi-trader-src`, then commit and push.
3. In the system flake: `nix flake update twempi`, then rebuild the server.

## Backtesting and research

Nothing here fabricates data: when the data does not exist, the tools say so.

### 1. Historical, price-only (real Kalshi candles)

Kalshi does not publish historical scores, so score-aware strategies cannot be backtested on
history. What can be tested is the price-only favourite baseline: "buy the favourite when its
ask enters a price band during play".

```bash
uv run kalshi-trader backtest historical --days 30 --max-events 400 \
    --cache-dir data/cache/historical
```

Decisions see only closed 1-minute candles, fills use the **next** minute's opening ask, nothing
after market close is fillable, one entry per match, 10 contracts, taker fees, chronological
out-of-sample split. Limitations: no depth data and minute granularity.

### 2. Event-driven replay of your recordings (all strategies)

`kalshi-trader run` (or `run --record-only`) records order books and scores. The replay merges
book snapshots, score snapshots **and market settlements** into one chronological stream and runs
the **same** `TennisTracker` + `TradingCore` used live, with a simulated clock:

- positions settle when their result became known, so cash, equity, open positions, exposure,
  daily-loss and drawdown limits reflect the simulated state at every moment;
- market status comes from the recorded snapshots and nothing trades after close;
- a recorded book/score counts as continuously confirmed until its next record only if the gap
  is within the recorder heartbeat, otherwise it ages exactly like stale live data;
- pre-match priors are captured from replayed books before the first live score; stored priors
  are used only if observed before it (no in-play leakage);
- events (not observations) are split chronologically into train / validation / test with an
  embargo, so one match never appears in two groups.

```bash
uv run kalshi-trader backtest replay                       # train / validation / test
uv run kalshi-trader backtest replay --no-test             # keep the test split untouched
uv run kalshi-trader backtest replay --grid                # choose params on VALIDATION only
uv run kalshi-trader backtest replay --embargo-hours 0 --market-making
```

The report shows the main account, every shadow ledger, model evaluation vs market, the
diagnostics and the split sizes. Every run is stored with its full configuration, market
universe, input fingerprint, code version and trade ledger (`backtest_runs`), and the report
warns how many times the test split has already been evaluated on the same data, so repeated
peeking is visible. Results need recorded matches that have since **settled**; weeks of
recordings are required for meaningful sample sizes.

### 3. Model evaluation and sensitivity

```bash
uv run kalshi-trader research model-eval --source live     # forward predictions (model_predictions table)
uv run kalshi-trader research model-eval --source replay
uv run kalshi-trader research sensitivity
```

Statistics use one market per match (the two complementary markets are not double-counted),
weigh every observation the same, and are reported next to the market-implied benchmark, with
a CI of the paired Brier difference that resamples whole matches. Fewer than 30 settled events
are flagged as insufficient.

Until 2026-11-04 every match had a total weight of 1 instead (weight 1/n for a match with n
observations). That is biased because n depends on how the match ends: matches the favourite
wins quickly have few observations, so each of them counts more. In a simulation of 40,000
matches with exact Markov probabilities (a perfectly calibrated forecaster), the old
weighting showed underdogs at 10–40% winning 2–3 points less often than predicted and
favourites at 60–90% 2–3 points more often. Equal observation weights showed no bias (within
±0.5 points). Calibration tables from before that date overstate how timid the model is.

### 4. Probability calibration

```bash
uv run kalshi-trader research calibrate                    # forward predictions recorded live
uv run kalshi-trader research calibrate --test-fraction 0.3 --embargo-hours 1
```

The same report is on the dashboard's Research tab (`/api/research/calibration`, recomputed
at most every 10 minutes), so it can be read on the server without `sudo`. It uses the
observations the strategy could trade on: its stages and tours, the pre-match prior, spread
≤ `max_spread`, and a mid inside the buyable band. It fits three logistic recalibrations on the
earlier 70% of matches and scores them on the later 30%:

- `model_slope`: `logit(p) = slope · logit(p_model)`;
- `market_anchored`: `slope · logit(p_model) + market_weight · logit(p_market)`;
- `market_slope`: a recalibrated market. This is the benchmark, not a candidate: it never
  disagrees with the market.

"Model adds information" compares `market_anchored` with `market_slope` on the held-out
matches. If the CI does not exclude zero, the model has shown nothing the price did not
already contain. The decision check buys one contract at the first observation per held-out
match that clears the entry rule (conservative probability − ask − taker fee ≥ `min_edge`, ask
inside the price band) and holds it to settlement, for the raw model and each candidate.
A candidate is recommended only if it beats the raw model on held-out log loss with a CI that
excludes zero. Its parameters are then refitted on all matches and printed as a
`[strategies.tennis_model.calibration]` block to paste into `config/settings.toml`.

## Strategy evaluation and selection

Each strategy's status (`kalshi-trader strategies`, Strategies tab):

| Status | Meaning |
| --- | --- |
| `experimental` | New, or its rules changed (version or config hash); no evidence |
| `collecting_data` | Observe mode, accumulating forward shadow-ledger results |
| `ready_for_paper` | Forward evidence meets every requirement; an operator *may* set `mode = "paper"` (never automatic) |
| `paper_trading` | `mode = "paper"`; sized at `risk.unvalidated_size_multiplier` (1.0 in the shipped config, 0.25 cautious) until validated |
| `paused` | Paused by an operator (persists) |
| `rejected` | Forward evidence significantly negative; main-account entries blocked until the rules change |

Evidence = settled trades in the strategy's shadow ledger under its current version tag
(genuinely out-of-sample: nothing was tuned on them). Requirements (`[evidence]`): ≥ 50 trades
across ≥ 30 independent events, calibration-null p ≤ 0.05, cluster-bootstrap CI of mean P&L
above zero, drawdown ≤ 20%. Backtests never change a status, so a strategy cannot be enabled
because it looked good in historical optimisation. Metrics reported per strategy: net P&L,
return on capital, EV per trade, average winner/loser, win rate, profit factor, max drawdown,
capital utilisation, fees, slippage, trade count, expected vs realised P&L, significance.

## Understanding the performance statistics

| Metric | Meaning |
| --- | --- |
| Total P&L | Realised simulated profit after fees |
| Return on bankroll | Total P&L ÷ starting balance |
| ROI on capital | Total P&L ÷ total cost of all entries |
| Win rate | Share of trades that settled in the money. **High by construction** for favourites; meaningless on its own |
| Average win / loss | Mean P&L of winners / losers. Buying at 93¢ wins about 7¢ and loses about 93¢ per contract |
| Profit factor | Gross wins ÷ gross losses. Above 1 means profitable |
| EV per trade | Mean realised P&L per trade |
| Max drawdown | Largest peak-to-trough fall in equity ($ and %) |
| Bankroll utilisation | Average and peak share of equity at risk |
| Win rate by entry price | Empirical win rate per price bucket vs the average entry price (the break-even line before fees). A real edge shows bars consistently **above** the ticks |

**Is it skill or luck?** Every report includes a significance check against the null hypothesis
that each contract wins with probability equal to its entry price (the market is calibrated):

- a Monte Carlo p-value: how often a calibrated market would produce at least the observed P&L
  (near 0 suggests an edge, near 1 means worse than fair prices);
- a bootstrap confidence interval for mean P&L per trade, resampling whole matches;
- a Wilson interval for the win rate vs the average price paid.

Reports turn these into a plain-language verdict, e.g. "consistent with fair prices", "loses money
after fees", or "significantly worse than fair prices".

Favourite-buying needs **large** samples. At a 93.5¢ break-even, distinguishing a true win rate of
95% from 93.5% at about two standard errors takes roughly 850 independent trades. Treat fewer than
a few hundred trades as anecdotal, whatever the P&L says. When several configurations are compared,
the best in-sample result is optimistic; only the out-of-sample numbers count.

## Running the tests

```bash
nix develop
uv run pytest               # full suite (no network access needed)
uv run pytest -k risk       # a subset
uv run pytest -m live       # opt-in smoke tests against the real Kalshi API (network, read-only)
uv run ruff check src tests && uv run ruff format --check src tests
```

Highlights of what is verified (all with synthetic data unless noted): execution-aware EV
with hand-checked numbers against an independent fee implementation (multi-level walks, per-order
fee rounding, edges that vanish after fees), risk budgets under aggregate/event/strategy/
correlated caps, allocation of correlated same-event opportunities, ranking under a binding
position limit, observe vs paper modes, restart duplicate prevention, basket legging failures
and unwinds, cross-market relationship verification (bracket gaps, MECNET, ladders, tennis
pairs + persistence), market-making trade-through fills, score regression/staleness guards,
event splits, observation-weighted calibration (exact on a calibrated forecaster), the
probability-calibration map, fit and report, schema migration, and a replay where chronological
settlement recycles capital that end-of-replay settlement would have blocked.

The suite also covers API parsing (with **real captured Kalshi responses** in `tests/fixtures/`),
ticker validation, request signing, retries and rate limiting, WebSocket sequence gaps and
reconnects (against a local fake server), the tennis state machine and format inference, the Markov
model (including a martingale consistency check), EV and fees, sizing, fill simulation, settlement,
every risk limit, persistence, duplicate prevention, the engine cycle, dashboard endpoints and
controls, and look-ahead protection in both backtests. Test data other than the captured fixtures
is synthetic and is never shown as trading results.

## Logs and data files

| Path | Contents |
| --- | --- |
| `data/kalshi_trader.db` | SQLite database (WAL): markets, events, snapshots, opportunities and rejection reasons, daily diagnostics, model predictions, simulated orders and trades (main + shadow accounts), equity, strategy state and history, safety state, health, market-making research fills, backtest runs (with config, universe and ledger) |
| `data/logs/kalshi-trader.jsonl` | Structured JSON logs from `kalshi-trader run`, rotated at 20 MB × 10 files (one-shot commands such as `scan` and `backtest` log to the console only) |
| `data/cache/` | Optional backtest download cache |

```bash
tail -f data/logs/kalshi-trader.jsonl                         # follow the log
grep '"level": "warning"' data/logs/kalshi-trader.jsonl | tail  # recent warnings
journalctl --user -u kalshi-trader -f                         # when run as a service
nix develop -c sqlite3 data/kalshi_trader.db '.tables'        # inspect the database
```

The console shows human-readable logs. Recent warnings and system events also appear on the
dashboard's System health tab. Everything under `data/` is git-ignored. Under the NixOS module
the database and log files are in `/var/lib/kalshi-trader/` instead, and
`journalctl -u kalshi-trader` shows the console log.

## Data sources: free vs paid

**Used (free):**

- **Kalshi Trade API v2.** Markets, events, order books, trades, candlesticks, historical data
  (public REST); real-time order books over WebSocket (requires a free API key).
- **Kalshi Live Data API** (`/milestones`, `/live_data`). Free point-level tennis scores (sets,
  games, points, server, status, retirements) for the matches Kalshi lists. These are the scores
  Kalshi's own site displays. Responses are CDN-cached (about 10 s) and may lag television.

**Paid services (not integrated, documented for evaluation only):** lower-latency, independent
point-by-point feeds would make the latency guard less necessary and allow cross-checking scores.
Candidates found during research:

- Sportradar Tennis API: enterprise licensing; contact sales.
- Goalserve tennis feed: listed at roughly US$150/month at the time of research.
- API-Tennis (api-tennis.com): paid tiers.

The provider interface (`src/kalshi_trader/tennis/providers/base.py`) is designed for adding one.
None is required. Scraping sites in breach of their terms is deliberately not done.

**Not used, possibly useful:** Jeff Sackmann's ATP/WTA point and match datasets (free, CC BY-NC-SA
4.0, non-commercial) could provide player-specific serve priors in future.

## Known limitations and missing data

- **Not enough settled, recorded data.** As of 2026-11-02 the database holds about 9 hours of
  recordings with 6 settled markets (3 settled matches with model predictions). No statistical
  claim about any strategy or about the model is possible yet. Run the engine for weeks.
- **Score latency remains the central risk** for the tennis model. The CDN age, frozen-score,
  regression and market-moved guards reduce it; they cannot remove it. Kalshi's live data has no
  source timestamp, so the true source age is unknowable (the HTTP `Age` header is a lower bound).
- **No independent probability models outside tennis.** Other categories are marked
  `unsupported` rather than guessed. Pre-match tennis is unsupported for the same reason.
- **Cross-market residual risks**: voided markets or fair-price settlements can pay other than
  0/1; legging is stressed but simulated on one book snapshot per leg; generic bracket and
  ladder relationships have only been unit-tested (no series configured by default).
- **Market making cannot be simulated credibly** from public books: queue position and trades at
  our price are unobservable, so fills are only counted when the market trades through the quote
  (pessimistic, adverse-selection-heavy). It stays research-only.
- **Paper fills are not validated against real fills** (no real orders exist). They take
  displayed liquidity with a 50% haircut and ignore hidden liquidity and other traders.
- **Replay fidelity**: market metadata (fee multiplier, tick size) is the latest recorded value;
  recorded depth is limited to the configured top levels; periods when the engine was off are
  missing.
- **Fees**: Kalshi's published taker formula with API multipliers and event overrides; `flat`
  fee-type series are not modelled specially. Check the current schedule before trusting small edges.
- **WebSocket** implemented and tested against a fake server only (no API key available).
- **Single machine**, SQLite, one process.

## Results to date

**Forward paper trading on the server (trades settled 2026-11-02 10:23 → 2026-11-04 03:15 UTC, ~41 h):**

| Ledger | Trades | P&L | Notes |
| --- | ---: | ---: | --- |
| `tennis_model`, main account (× 0.25 size) | 57 | +$12.32 | ~$1.90 per trade; 25 wins where the market mid implied 19.1 |
| `tennis_model`, shadow ledger (full size) | 57 | +$75.11 | 16% return on capital deployed; calibration-null p = 0.14 (not significant) |
| `cross_market`, main | 2 baskets | +$0.02 | two tennis-pair baskets, +1¢ each after fees |
| `late_match`, shadow | 5 | −$10.81 | 4 small wins and one −$19.24 loss (average entry 89¢) |

- Contracts bought under 20¢ (21 of the 57) lost 18% of their cost; trades at 20¢ or more made
  money. This is why `tennis_model.min_price` is now 0.20. The calibration table then also
  showed the model overrating 10–30% underdogs by 3–5 points, but that table weighted every
  match equally. That weighting alone makes a calibrated forecaster look about 3 points too
  timid (see [Model evaluation](#3-model-evaluation-and-sensitivity)), and the market's own
  table showed a similar gap. Only part of it is the model.
- Model vs market over 557 settled matches (forward predictions, the old match weighting):
  Brier 0.1201 (model) vs 0.1179 (market mid). Paired difference +0.0022, 95% CI [+0.0001,
  +0.0043], so the market is still slightly more accurate on average. By 2026-11-04 (894
  matches) it was 0.1175 vs 0.1153, CI [+0.0005, +0.0039].
- Fees plus half the spread cost 1.85¢ per contract on these trades, 8% of the average price.
  At about 33 trades a day that hurdle dominates: with no real edge, shadow-size trading loses
  about $14 a day on $1,000.

The main account's size multiplier is now 1.0 (it was 0.25), so the main account trades at
shadow-ledger size: about $8 per trade on $1,000, at most 2% of equity.

**Paper trading, first live check:** in a 2.5-minute
live verification run on 2026-11-02 (3 matches in progress), every opportunity was evaluated
and none cleared its requirements: the tennis model's best conservative edge stayed below 3¢
after fees, tennis pairs never crossed after fees, and the late-match baseline's leader
probabilities were below 0.90.

**Replay of all recordings (2026-11-02, 67 events, embargo 0):** 0 trades in every split and
every ledger; 5–8 out-of-order score responses per split were rejected; the most frequent
rejection reasons were insufficient conservative edge, prices outside bands, and (new) the
market moving ≥ 10¢ without a score change. Model vs market on the 3 settled events: Brier
0.0176 (model) vs 0.0151 (market mid), i.e. the market was more accurate, but 3 events cannot
support any conclusion.

**Historical price-only backtest on real Kalshi data** (run 2026-10-29):

```bash
uv run kalshi-trader backtest historical --days 30 --max-events 1500 --cache-dir data/cache/historical
```

- Data: 1,500 of the 7,587 tennis matches that settled in the 30 days to 2026-10-29 (an evenly
  spaced sample across the ATP, WTA, Challenger and ITF series), with 1-minute candles for both
  players' markets. No match lacked a start time and no market lacked candles.
- Rule: buy 10 YES contracts of a player in the first in-play minute their ask is inside the
  band, hold to settlement, at most one trade per match.
- In-sample: matches starting before 2026-10-21 14:01 UTC. Out-of-sample: the remaining 30%
  (the last 8 days).
- p-value: the probability that fair prices (each contract winning exactly as often as its price
  implies, fees included) would have done at least this well. Values near 0 would suggest an
  edge; values near 1 mean the trades did **worse** than fair prices.

| Ask band | In-sample trades | In-sample P&L | Out-of-sample trades | Out-of-sample P&L | Out-of-sample win rate vs avg price paid | Out-of-sample p-value |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 80–85¢ | 871 | +$8.16 | 378 | −$203.21 | 77.5% vs 81.8¢ | 0.98 |
| 85–90¢ | 928 | −$10.44 | 383 | −$131.45 | 84.1% vs 86.7¢ | 0.94 |
| 90–93¢ | 898 | −$16.60 | 385 | −$209.42 | 86.2% vs 91.0¢ | 0.999 |
| 93–95¢ | 900 | −$32.68 | 367 | −$145.11 | 90.2% vs 93.7¢ | 0.997 |
| 95–97¢ | 873 | −$9.04 | 373 | −$147.66 | 92.0% vs 95.6¢ | 0.999 |
| 97–99¢ | 857 | −$37.32 | 373 | −$114.41 | 94.6% vs 97.5¢ | 0.999 |

Default band (90–97¢) over all 1,406 trades: favourites won 90.5% of the time, against the 92.2%
needed to break even after fees. P&L was −$238.10 after $83.10 of fees, or −1.7¢ per contract
(95% confidence interval −3.2¢ to −0.2¢). The band chosen on the in-sample period (80–85¢) lost
$203.21 out of sample on 378 trades.

**Interpretation:** there is no evidence that Kalshi tennis favourites are underpriced. In-sample,
every band was within half a cent per contract of break-even after fees. In the last 8 days every
band lost: favourites won 2.6–4.8 points less often than their prices implied, which is worse than
fair prices (p > 0.99 for four of the six bands). The two periods differ, so results cluster in
time, and short windows can mislead in either direction. The bands overlap (one match can appear
in several), so they are not independent tests.

This is the price-only baseline. It does not test the score-aware late-match model, because Kalshi
publishes no historical scores. That needs weeks of recordings (`kalshi-trader run`) followed by
`kalshi-trader backtest replay`.

## Milestone status

| Milestone | Status |
| --- | --- |
| 1. Review of the existing implementation and audit findings | Done (findings 1–6 fixed; 7–10 are data/evidence gaps, documented) |
| 2. Correctness repairs: replay, execution-cost EV, data freshness, accounting | Done |
| 3. Shared strategy framework, diagnostics, central risk allocation | Done |
| 4. Statistical mispricing (`tennis_model`) | Done for tennis; other categories correctly *unsupported* |
| 5. Cross-market inconsistencies | Done (tennis pairs live in paper mode; generic relationships unit-tested, no series configured) |
| 6. Tennis strategy improvements: stages, tours, bands, validation tooling | Done; validation awaits settled data |
| 7. Market making | Research simulation only (data cannot support credible maker fills) |
| 8. Baselines | Done (`late_match`, `price_threshold`, market-implied benchmark) |
| 9. Multi-strategy paper execution through the allocator | Done |
| 10. Dashboard and analytics | Done |
| 11. Validation: tests, live integration run, documentation | Done; profitability **not** demonstrated |

### Audit findings

| # | Finding | Resolution |
| --- | --- | --- |
| 1 | Replay settled positions only at the end | Fixed: chronological event stream; test proves capital recycling |
| 2 | Replay did not reproduce live risk/lifecycle | Fixed: replay runs `TradingCore` + `TennisTracker`; recorded status, close times, marks, day limits |
| 3 | Locally fresh responses could hold stale scores | Mitigated: HTTP `Age`/assumed cache age added, regressions rejected, frozen-score and market-moved guards |
| 4 | EV computed for a hypothetical 10-contract order | Fixed: EV, fees and Kelly at the final simulated fill |
| 5 | Filter failures were not recorded | Fixed: every opportunity outcome + reason counted and logged |
| 6 | Metadata/priors not as known at decision time | Fixed for priors (observation time must predate the first live score) and status; fee/tick metadata documented as latest-known |
| 7 | Paper fills not validated against real fills | Open (impossible without real orders); conservative assumptions documented |
| 8 | No profitable forward paper trading | Open; forward shadow ledgers now collect the evidence |
| 9 | −$238 over 1,406 price-only trades | Confirmed; the price-band baseline stays off / observe |
| 10 | Model accuracy beyond market unproven | Open; forward predictions now recorded and scored against the market |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: libstdc++.so.6` (NumPy/pandas) | Run inside `nix develop` |
| `error: experimental Nix feature 'nix-command' is disabled` | `nix --extra-experimental-features 'nix-command flakes' develop` |
| `path '.../flake.nix' does not exist` inside the repo | `git add flake.nix flake.lock` (Nix only sees tracked files) |
| Dashboard port in use | `KT_DASHBOARD__PORT=8790 uv run kalshi-trader run` |
| "kalshi_ws disabled" on the health tab | Expected without API credentials; REST polling is used |
| No live matches in `scan --live-only` | Tennis is seasonal and time-zone dependent; try `scan` for upcoming matches |
| Trading never happens | Usually correct behaviour. Open the **Why no trades?** tab or run `kalshi-trader diagnostics`: every evaluated opportunity is counted with its rejection reason. Check `kalshi-trader strategies` for modes (`observe` never uses the main account) |
| Many `kalshi_request_retry ... status=429` log lines | Another process (e.g. a backtest) is sharing the request budget. The client halves its rate and retries automatically |
| `https://kalshi-trader.<tailnet>.ts.net:8787` does not load | `systemctl status kalshi-trader tailscale-serve-kalshi-trader` on the server, and check that the service is defined and the host approved in the Tailscale admin console |
