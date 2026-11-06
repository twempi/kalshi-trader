# AGENTS.md

Kalshi trading bot, paper trading for now (currently on tennis match markets).

**Goal:** a profitable Kalshi trading bot. Maximise long-run profit after fees without risking
ruin: growth matters, but never at the cost of a drawdown that could wipe the account out. Any
method, market or strategy is fine. "Profitable" means the profit is real: it holds out of
sample, is too large to be luck, and survives realistic execution (delays, fees, liquidity).
Prove it on paper before real money.

## Rules
- **Paper only.** No code path may place a real order (the REST client is GET-only). Do not add
  one unless the user explicitly asks.
- **Evidence before belief.** Never call a strategy profitable without out-of-sample evidence
  (`/api/strategies` shows its status). Small samples are luck until shown otherwise.
- **Don't reset evidence casually.** Changing a strategy's settings changes its version tag and
  resets its evidence. Test ideas as observe-only variants
  (`[strategies.tennis_model_variants.<name>]`) and promote only clear winners.
- **Parameters come from held-out data, never guesses** (e.g. calibration via
  `/api/research/calibration`).
- **Sizing never depends on past wins or losses** (no martingale).

## Where things are
- `config/settings.toml`: every tunable. Comment each changed setting with what it does and why.
- `README.md` → "Results to date": the project's history of findings. Add a dated entry for
  every analysis or strategy change.
- `src/kalshi_trader/`: `strategies/` (signals), `core/` (evaluator, risk budget, allocator),
  `execution/paper.py` (simulated fills), `research/` (calibration, fill delay),
  `dashboard/app.py` (API).

## Working
- Tests: `nix develop -c uv run pytest -q`. Plain `uv run` fails on NixOS (missing libstdc++).
  Lint: `uv run ruff check src tests && uv run ruff format src tests`.
- The live bot runs on `t480s` (service `kalshi-trader`, login shell fish). Its database needs
  sudo, so read data through the dashboard API:
  `ssh t480s "curl -s 'http://127.0.0.1:8787/api/<endpoint>'"`.
- The user commits and deploys: push → `nix flake update kalshi-trader-src` in
  `~/Projects/nix-inputs` → rebuild t480s (needs sudo). Don't commit or push unless asked.
