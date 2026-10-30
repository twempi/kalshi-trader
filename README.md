# kalshi-trader

A locally hosted research and **paper-trading** platform for Kalshi tennis match-winner markets.

It discovers live and upcoming tennis markets, streams their order books and live point-by-point
scores, estimates each player's win probability with an explainable tennis model, and runs a
fully automated **simulated** trading engine against real, live Kalshi prices. Everything it
observes is recorded, so strategies can be replayed and evaluated later.

The initial research question is whether, late in a match, the contract on the player who is
about to win is **underpriced after fees and execution costs**. The platform is built to *test*
that hypothesis, not to assume it.

> **No real money.** The application never places real orders. The REST client refuses any
> non-`GET` request (`ReadOnlyViolation`), no order endpoints exist in the code base, and a test
> asserts that. API credentials are optional and only needed for the real-time WebSocket feed.
> All P&L shown anywhere is **simulated**.

---

## Contents

1. [What works today](#what-works-today)
2. [Architecture](#architecture)
3. [How a trading decision is made](#how-a-trading-decision-is-made)
4. [Installation on NixOS](#installation-on-nixos)
5. [Configuration and API credentials](#configuration-and-api-credentials)
6. [Running the application](#running-the-application)
7. [The dashboard](#the-dashboard)
8. [Starting, pausing and stopping paper trading](#starting-pausing-and-stopping-paper-trading)
9. [Running continuously (systemd user service)](#running-continuously-systemd-user-service)
10. [Running on a NixOS server (nix-inputs module)](#running-on-a-nixos-server-nix-inputs-module)
11. [Backtesting](#backtesting)
12. [Understanding the performance statistics](#understanding-the-performance-statistics)
13. [Running the tests](#running-the-tests)
14. [Logs and data files](#logs-and-data-files)
15. [Data sources: free vs paid](#data-sources-free-vs-paid)
16. [Known limitations and blockers](#known-limitations-and-blockers)
17. [Results to date](#results-to-date)
18. [Milestone status](#milestone-status)
19. [Troubleshooting](#troubleshooting)

---

## What works today

Each item below was run against **real Kalshi data** on 2026-10-29 unless marked otherwise.

| Capability | Status |
| --- | --- |
| Tennis market scanner (ATP, WTA, Challenger, ITF match-winner series) | Verified live: 39 matches / 78 markets discovered |
| Order-book monitoring over REST (public, no credentials) | Verified live: 4-minute engine run, 0 failed requests |
| Order-book monitoring over WebSocket (sequence checks, gap recovery, reconnects) | Tested against a local fake server only; needs an API key for the real feed |
| Live scores from Kalshi's free Live Data API | Verified live: point-level scores parsed for the matches in progress |
| Markov tennis model + uncertainty + EV after fees | Unit-tested; tracks market prices closely on live matches |
| Late-match strategy and score-agnostic control strategy | Unit-tested; run live (no qualifying trade occurred during the test run) |
| Paper broker: fills against real books, fees, partial fills, settlement | Unit-tested; engine cycle tested end-to-end with fakes |
| Centralized risk manager, persistent emergency stop, auto-pause | Unit-tested; pause/stop/resume verified from the CLI on the real database and through the dashboard API in tests |
| SQLite persistence (market/score snapshots, signals, orders, trades, equity) | Verified live: 793 market and 59 score snapshots recorded in 4 minutes |
| Historical backtest on real Kalshi 1-minute candles | Run on real data (see [Results to date](#results-to-date)) |
| Replay backtest on your own recorded data | Tested with synthetic recordings; needs recorded, settled matches to be meaningful |
| Local web dashboard | Verified in a browser against live data |
| systemd user service / Home Manager module | Provided as examples; **not installed** (that would touch files outside the project) |
| NixOS service module (in [nix-inputs](https://github.com/twempi/nix-inputs)) | Tested in NixOS VMs on 2026-10-30: starts under systemd sandboxing, dashboard on localhost only, server CLI, graceful stop, and real Kalshi data recorded when the VM had internet. Not yet running on a real server |

## Architecture

```
 Kalshi REST (public)          Kalshi WebSocket (API key)      Kalshi Live Data (public)
 discovery, order books,       order-book deltas,              point-by-point
 status, settlement            lifecycle events                tennis scores
        │                              │                               │
        ▼                              ▼                               ▼
 TennisDiscovery ──────────► MarketDataHub                       ScoreProvider
 (matches, markets)          (validated order books)             → MatchState (validated)
        │                              │                               │
        └──────────────────────────────┼───────────────────────────────┘
                                       ▼
                 WinProbabilityEstimator (Markov model + uncertainty)
                                       ▼
                 Strategies (late_match; price_threshold control)
                                       ▼
                 RiskManager ──► PaperBroker (simulated fills, fees, settlement)
                                       ▼
      SQLite (WAL): snapshots · signals · orders · trades · equity · health · safety flags
                                       ▼
                 Dashboard (FastAPI, 127.0.0.1) · CLI · backtests
```

All components run in one asyncio process (`kalshi-trader run`). The engine loops are
independent: discovery (60 s), exchange status (30 s), order books, scores (5 s), settlement,
strategy evaluation (1 s), recorder flush and equity snapshots. A failure in one loop is
logged, counted and retried without taking down the others; repeated failures auto-pause
paper trading.

| Path | Responsibility |
| --- | --- |
| `src/kalshi_trader/api/` | Kalshi REST client (GET-only, retries, backoff, adaptive rate limit, pagination), WebSocket client (signed handshake, per-subscription sequence tracking, reconnects), request signing, validated response schemas |
| `src/kalshi_trader/market_data/` | Market discovery, local order books (snapshot + delta), REST/WS ownership hand-off, pre-match price capture, recorder |
| `src/kalshi_trader/tennis/` | Tennis scoring state machine, match-format inference, Kalshi Live Data parser |
| `src/kalshi_trader/models/` | Hierarchical Markov model, Bayesian strength estimator, EV calculations |
| `src/kalshi_trader/strategies/` | Strategy interface, late-match strategy, price-threshold control strategy |
| `src/kalshi_trader/risk/` | All risk limits, sizing, safety state |
| `src/kalshi_trader/execution/` | Fee formula, fill simulation, paper broker |
| `src/kalshi_trader/database/` | SQLAlchemy tables, repository |
| `src/kalshi_trader/backtesting/` | Historical (candles) and replay backtests, metrics, significance tests |
| `src/kalshi_trader/dashboard/` | FastAPI app and static front end (no external CDNs) |
| `src/kalshi_trader/engine.py` | Orchestration of all loops, graceful shutdown |
| `src/kalshi_trader/main.py` | Command-line interface |
| `config/settings.toml` | All non-secret settings |
| `deploy/` | Optional systemd user unit and Home Manager module |

Market data and score feeds are treated as **untrusted input**. Tickers are validated against a
strict pattern; prices must lie in `[0, 1]` and sizes must be non-negative. Malformed messages are
rejected and counted, and duplicate WebSocket messages are dropped. A sequence gap, or a delta that
would leave a price level negative, marks the affected books out of sync: REST polling covers
them until a fresh WebSocket snapshot (requested automatically) arrives. Score payloads that are
internally inconsistent (impossible point scores, unknown player ids, unknown server) are marked
invalid and never reach a trading decision.

## How a trading decision is made

The late-match strategy (`src/kalshi_trader/strategies/late_match.py`) only proposes a trade
when **every** condition holds:

1. **Fresh, valid data.** The order book is younger than `risk.max_market_data_age_s` (10 s),
   the score is younger than `risk.max_score_data_age_s` (30 s), the match format is known,
   the exchange and market are open.
2. **Late in the match.** The model's favourite needs at most `max_games_from_victory` (3) games.
3. **High model probability.** The point estimate is at least `min_model_prob` (0.90).
4. **Tradeable price.** The executable YES ask is within `[min_price, max_price]`
   (0.80–0.97) and the spread is at most `max_spread` (4¢).
5. **Positive edge after costs, on a conservative probability:**

   ```
   EV per contract = p_conservative − (ask + fee per contract)  ≥  min_edge  (2¢)
   ```

**The probability model** (`src/kalshi_trader/models/`):

- *Markov chain.* Exact recursion from point → game → tiebreak (with serve rotation) → set →
  match, for best-of-3/5 with standard tiebreaks, 10-point final-set tiebreaks, advantage final
  sets, and match tiebreaks. Its inputs are each player's probability of winning a point on serve.
- *Player strength.* Serve probabilities are parameterised as `p1 = μ + δ/2`, `p2 = μ − δ/2`.
  `μ` has a tour-level prior (e.g. ATP 0.64). `δ` is anchored on the **pre-match market price**
  when one was captured (inverted through the same model) and is otherwise wide. The prior is
  updated with the serve points observed in the match (sampling-importance-resampling).
- *Uncertainty.* Posterior draws give a distribution of win probabilities. The strategy uses the
  10th percentile (`credible_quantile`), not the mean.
- *Latency guard.* The live score may lag the market. The conservative probability is the
  minimum over every score reachable within `latency_points` (1) further points.
- *Retirements.* A small per-game retirement hazard pulls probabilities towards ½.

The dashboard's market table shows the market-implied probability, the model point estimate,
and the conservative probability side by side so the difference is always visible.

**Fees** follow Kalshi's published schedule and rounding rules:
`fee = 0.07 × fee multiplier × contracts × P × (1 − P)` per order, where the series' fee
multiplier comes from the API. The fee is rounded up to $0.000001, and the cash debit (cost plus
fee) is rounded up to the cent. The maker rate (0.0175) is configured but unused, because simulated
orders always take liquidity.

**Execution simulation** (`src/kalshi_trader/execution/`): an immediate-or-cancel limit buy at
most `limit_slippage` (1¢) above the ask, and never above the price at which EV would fall below
`min_edge`. It walks the real displayed order book but assumes only
`paper.displayed_liquidity_fraction` (50%) of the displayed size is actually available. Fills are
whole contracts and can be partial. Positions are held to settlement by default
(`paper.enable_exits = false`). Settlement uses Kalshi's own `result` / `settlement_value`.

**Risk checks** (`src/kalshi_trader/risk/manager.py`) run on every signal:

| Limit | Default |
| --- | --- |
| Max stake per trade | 2% of equity and at most $25 |
| Sizing | 10% fractional Kelly on the *conservative* probability, capped by the above |
| Max daily loss (UTC day, mark-to-bid) | $50, then no new entries until the next UTC day |
| Max drawdown from peak | 15%, then a **persistent emergency stop** |
| Max simultaneous positions | 5 |
| Min liquidity at the limit price | 10 contracts |
| Max spread | 5¢ |
| Max order-book / score age | 10 s / 30 s |
| Max trades per market / per match | 1 / 1 |
| Consecutive engine errors before auto-pause | 5 |

Sizing depends only on current equity and the conservative edge. It **never** increases after a
loss (no martingale, no loss chasing). Duplicate orders are prevented by a unique client order
id per attempt plus an entry cooldown. Safety flags (paused, emergency stop) are stored in the
database and survive restarts.

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

uv run kalshi-trader run                  # engine: record data + paper trade + dashboard
uv run kalshi-trader run --record-only    # record data + dashboard, never simulate trades
uv run kalshi-trader run --no-dashboard   # engine without the web server

uv run kalshi-trader dashboard            # dashboard only (reads the database)
uv run kalshi-trader status               # account, safety state, component health, row counts
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
| Overview | Simulated equity, total and daily P&L, open positions, completed trades, win rate, equity chart, recent activity (paper fills, settlements, operator actions, warnings) |
| Live tennis markets | One row per player contract: score, server, games to win, bid/ask, spread, depth, market vs model vs robust-low probability, conservative EV, flagged opportunities, data age. Filter by text, tour, live matches, opportunities; sort by EV, match progress, price, liquidity or start time |
| Trading activity | Open positions, cumulative P&L, completed simulated trades (timestamp, market, player, score at entry, price, model probability, EV, size, cost, fees, result, P&L) with CSV export, and every strategy signal with the risk manager's decision and reasons |
| Analytics | Equity curve, P&L over time, drawdown, win rate by entry price vs break-even, outcome distribution, performance by strategy, a "skill or luck?" significance panel, stored backtest runs |
| System health | Recorded-data counts and per-component status (Kalshi REST and WebSocket, score feed, exchange status, engine loops): last update and success, last error, details such as request counts, failures and latency; components that stop updating are marked stale. Recent warnings and errors |

Every chart has a "Table view" toggle. Light and dark themes are supported.

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

The same controls are on the dashboard. A paused engine keeps recording data and settling
existing positions. The emergency stop is also triggered automatically when the drawdown limit
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

## Backtesting

Two kinds of backtest are provided. Neither fabricates data: if the data does not exist, they
say so and stop.

### 1. Historical, price-only (real Kalshi candles)

Kalshi does not publish historical scores, so the late-match *model* strategy cannot be backtested
on history. What can be tested on history is its price-only core: "buy the favourite when its ask
enters a price band during play". That tests directly whether high-priced tennis contracts have
been underpriced after fees.

```bash
uv run kalshi-trader backtest historical --days 30 --max-events 400 \
    --cache-dir data/cache/historical
uv run kalshi-trader backtest historical --days 30 --min-price 0.85 --max-price 0.95 \
    --series KXATPMATCH KXWTAMATCH --cache-dir data/cache/historical
```

- Data: 1-minute YES bid/ask candles for settled match-winner markets. The match start comes from
  Kalshi's milestone `start_date`. Market `occurrence_datetime` is the *expected expiration* and
  is never used as a start time.
- No look-ahead: a decision only sees closed candles. The fill is at the **next** minute's opening
  ask (a one-minute latency); if that ask has left the band, there is no fill. Nothing after the
  market's close is fillable, and the outcome is used only to compute P&L.
- One entry per match, 10 contracts, taker fees, spread ≤ 3¢.
- Chronological out-of-sample split (`--test-fraction`, default 0.3). A sweep of six price bands
  is reported for train and test separately; a band chosen in-sample is evaluated once out of
  sample.
- Results are printed as JSON and stored, labelled with their data source, in the dashboard's
  "Backtest runs" table. Downloads are cached in `--cache-dir`, so re-runs are reproducible
  and fast.

### 2. Replay of your own recordings (score-aware)

`kalshi-trader run` (or `run --record-only`) records order books and scores. A replay feeds
those recordings, in time order, through the **same** estimator, strategy, risk manager and paper
broker used live.

```bash
uv run kalshi-trader backtest replay                 # current config, train/test split
uv run kalshi-trader backtest replay --grid          # 27-config sweep (edge, games, latency)
```

It needs recorded matches that have since settled. Start recording as early as possible: weeks
of recordings are needed for meaningful sample sizes.

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

The suite covers API parsing (with **real captured Kalshi responses** in `tests/fixtures/`),
ticker validation, request signing, retries and rate limiting, WebSocket sequence gaps and
reconnects (against a local fake server), the tennis state machine and format inference, the Markov
model (including a martingale consistency check), EV and fees, sizing, fill simulation, settlement,
every risk limit, persistence, duplicate prevention, the engine cycle, dashboard endpoints and
controls, and look-ahead protection in both backtests. Test data other than the captured fixtures
is synthetic and is never shown as trading results.

## Logs and data files

| Path | Contents |
| --- | --- |
| `data/kalshi_trader.db` | SQLite database (WAL): markets, events, snapshots, signals, simulated orders and trades, equity, safety state, health, backtest runs |
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

## Known limitations and blockers

- **Score latency is the central risk.** Market makers with faster data can react before the
  Kalshi score feed updates. An apparent late-match "edge" may just be the market already knowing
  the next point. The latency guard and conservative quantile reduce, but cannot remove, this risk.
- **No historical scores.** The score-aware strategy can only be evaluated on data recorded by
  this application going forward. The historical backtest is price-only.
- **Tour-level priors.** There are no player-specific serve statistics. The pre-match market
  price is the main source of player strength, so matches discovered after they started (with no
  pre-match price) get wider uncertainty and rarely qualify.
- **Match format** is inferred from the tournament, gender and Kalshi's scoring-type field.
  Unknown formats are never traded.
- **Fill realism.** Simulated IOC orders take displayed liquidity with a 50% haircut. They do not
  model queue position, hidden liquidity, or other traders reacting to the same information.
  Real fills could be worse.
- **Fees** use Kalshi's published taker formula and the API's per-series fee multiplier. Check the
  current fee schedule before relying on small edges.
- **WebSocket feed** is implemented and tested against a fake server, but has not been exercised
  against Kalshi's real WebSocket (no API key was available). Without a key the engine polls
  order books over REST every 3 s.
- **Weekly exchange maintenance** (Thursday early morning ET) and unscheduled pauses are detected
  through the exchange-status endpoint, and no simulated entries are made while the exchange is
  inactive.
- **Single machine.** SQLite and one process.

## Results to date

**Paper trading:** no simulated trades yet. The engine has only run for short verification
sessions, and no contract passed the late-match strategy's conservative-edge test during them.

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
| 1. Project foundation: Git, Nix flake, uv, configuration, logging, structure | Done |
| 2. Kalshi integration: discovery, live prices, real-time updates, storage, real data | Done (WebSocket verified against a fake server only) |
| 3. Paper trading: account, positions, fees, execution, risk, persistent history | Done |
| 4. Tennis analytics: live scores, late-match strategy, probability, EV, signals | Done |
| 5. Dashboard: real market data, paper performance, opportunities, analytics, health | Done |
| 6. Backtesting and reliability: replay, reproducible evaluations, recovery, NixOS service, tests, docs | Done. Replay awaits recorded data; NixOS module tested in VMs, not yet deployed |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: libstdc++.so.6` (NumPy/pandas) | Run inside `nix develop` |
| `error: experimental Nix feature 'nix-command' is disabled` | `nix --extra-experimental-features 'nix-command flakes' develop` |
| `path '.../flake.nix' does not exist` inside the repo | `git add flake.nix flake.lock` (Nix only sees tracked files) |
| Dashboard port in use | `KT_DASHBOARD__PORT=8790 uv run kalshi-trader run` |
| "kalshi_ws disabled" on the health tab | Expected without API credentials; REST polling is used |
| No live matches in `scan --live-only` | Tennis is seasonal and time-zone dependent; try `scan` for upcoming matches |
| Trading never happens | Usually correct behaviour. The markets tab shows the conservative EV of every contract (mostly negative); the signals table lists every proposal the risk manager rejected, with reasons (stale data, spread, liquidity, limits) |
| Many `kalshi_request_retry ... status=429` log lines | Another process (e.g. a backtest) is sharing the request budget. The client halves its rate and retries automatically |
| `https://kalshi-trader.<tailnet>.ts.net:8787` does not load | `systemctl status kalshi-trader tailscale-serve-kalshi-trader` on the server, and check that the service is defined and the host approved in the Tailscale admin console |
