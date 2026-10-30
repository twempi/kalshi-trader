/* Dashboard logic. Data from the API is untrusted: it is only ever inserted with
   textContent / createTextNode, never innerHTML. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { view: "overview", markets: [], timers: [], analyticsAt: 0 };

  // ------------------------------------------------------------ formatting
  const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const money = (v) => (v === null || v === undefined ? "–" : (v < 0 ? "−$" : "$") + Math.abs(v).toFixed(2));
  const signedMoney = (v) => (v === null || v === undefined ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + "$" + Math.abs(v).toFixed(2));
  const pct = (v, d = 1) => (v === null || v === undefined ? "–" : (v * 100).toFixed(d) + "%");
  const prob = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");
  const cents = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "¢");
  const ago = (s) => (s === null || s === undefined ? "–" : s < 90 ? Math.round(s) + "s" : s < 5400 ? Math.round(s / 60) + "m" : Math.round(s / 3600) + "h");
  const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
  const signClass = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  const compact = (v) => (v === null || v === undefined ? "–" : Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(1) + "K" : String(Math.round(v)));

  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function pill(kind, text) {
    const p = node("span", "pill " + kind);
    p.append(node("span", "dot"), node("span", null, text));
    return p;
  }
  function setPill(id, kind, text) { $(id).replaceWith(Object.assign(pill(kind, text), { id })); }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) throw new Error(path + " -> " + res.status);
    return res.json();
  }

  function table(container, columns, rows, opts) {
    const o = opts || {};
    container.replaceChildren();
    if (!rows.length) { container.appendChild(node("div", "empty", o.empty || "Nothing to show yet.")); return; }
    const t = node("table", "data");
    const head = node("tr");
    columns.forEach((c) => head.appendChild(node("th", c.num ? "num" : "", c.label)));
    const thead = node("thead"); thead.appendChild(head); t.appendChild(thead);
    const tbody = node("tbody");
    rows.forEach((row, i) => {
      const tr = node("tr", o.rowClass ? o.rowClass(row, i) : "");
      columns.forEach((c) => {
        const td = node("td", (c.num ? "num " : "") + (c.cls ? c.cls(row) : ""));
        const v = c.render ? c.render(row) : c.fmt ? c.fmt(row) : row[c.key];
        if (v instanceof Node) td.appendChild(v); else td.textContent = v === null || v === undefined ? "–" : String(v);
        if (c.title) td.title = c.title(row) || "";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    t.appendChild(tbody);
    container.appendChild(t);
  }

  function tile(label, value, sub, cls, hero) {
    const t = node("div", "tile" + (hero ? " hero" : ""));
    t.append(node("div", "label", label), node("div", "value " + (cls || ""), value));
    if (sub) t.appendChild(node("div", "sub", sub));
    return t;
  }

  // ------------------------------------------------------------- header
  async function refreshHeader() {
    try {
      const o = await api("/api/overview");
      setPill("engine-pill", o.engine_running ? "good" : "critical",
              o.engine_running ? (o.record_only ? "Engine running (record-only)" : "Engine running")
                               : "Engine stopped");
      if (o.emergency_stop) setPill("trading-pill", "critical", "Emergency stop");
      else if (!o.trading_enabled) setPill("trading-pill", "warning", "Paper trading paused");
      else setPill("trading-pill", "good", "Paper trading enabled");
      $("banner").title = o.pause_reason || "";
      return o;
    } catch (e) {
      setPill("engine-pill", "critical", "Dashboard API unreachable");
      return null;
    }
  }

  // ----------------------------------------------------------- overview
  async function renderOverview() {
    const o = await refreshHeader();
    if (!o) return;
    const k = $("kpis");
    k.replaceChildren(
      tile("Paper account equity", money(o.equity), `Cash ${money(o.cash)} · open ${money(o.open_value)}`, "", true),
      tile("Total P&L", signedMoney(o.total_pnl), `vs ${money(o.starting_balance)} start`, signClass(o.total_pnl)),
      tile("Today's P&L", signedMoney(o.daily_pnl), "UTC day, mark-to-bid", signClass(o.daily_pnl)),
      tile("Open positions", String(o.open_positions), "simulated"),
      tile("Completed trades", String(o.completed_trades), `fees paid ${money(o.fees_paid)}`),
      tile("Win rate", o.win_rate === null ? "–" : pct(o.win_rate), "settled/closed trades"),
    );
    const sys = node("div", "tile");
    sys.append(node("div", "label", "System status"));
    const holder = node("div", "value");
    holder.style.fontSize = "14px";
    holder.appendChild(o.emergency_stop ? pill("critical", "Emergency stop") : !o.trading_enabled ? pill("warning", "Paused") : pill("good", "Enabled"));
    sys.appendChild(holder);
    if (o.pause_reason) sys.appendChild(node("div", "sub", o.pause_reason));
    k.appendChild(sys);
    const a = await api("/api/analytics");
    Charts.line($("overview-equity"), a.equity_curve, { yFormat: (v) => "$" + v.toFixed(0), label: "Equity", area: false, empty: "Equity snapshots appear once the engine has run for a minute." });
    const h = await api("/api/health");
    const events = h.events.filter((e) => e.component === "paper" || e.component === "risk" || e.component === "dashboard" || e.level !== "info").slice(0, 12);
    table($("overview-events"), [
      { label: "Time", fmt: (e) => when(e.ts) },
      { label: "Level", render: (e) => pill(e.level === "error" || e.level === "critical" ? "critical" : e.level === "warning" ? "warning" : "neutral", e.level) },
      { label: "Message", fmt: (e) => e.message.slice(0, 160) },
    ], events, { empty: "No activity yet." });
  }

  // ------------------------------------------------------------ markets
  function marketRows() {
    const q = $("mk-search").value.trim().toLowerCase();
    const tour = $("mk-tour").value;
    const liveOnly = $("mk-live").checked, oppsOnly = $("mk-opps").checked;
    let rows = state.markets.filter((m) => {
      if (tour && m.tour !== tour) return false;
      if (liveOnly && m.match_status !== "live") return false;
      if (oppsOnly && !m.opportunity) return false;
      if (q && !`${m.player_name} ${m.opponent_name} ${m.match_title} ${m.competition}`.toLowerCase().includes(q)) return false;
      return true;
    });
    const key = $("mk-sort").value;
    const val = {
      ev: (m) => (m.ev_conservative === null ? -9 : m.ev_conservative),
      progress: (m) => (m.games_from_victory === null ? -99 : -m.games_from_victory),
      price: (m) => (m.yes_ask === null ? -1 : m.yes_ask),
      liquidity: (m) => (m.ask_depth_3c === null ? -1 : m.ask_depth_3c),
      start: (m) => -(m.scheduled_start_ts || 0),
    }[key];
    rows.sort((a, b) => val(b) - val(a));
    return rows;
  }

  function renderMarketsTable() {
    const rows = marketRows();
    $("mk-count").textContent = `${rows.length} of ${state.markets.length} markets`;
    table($("markets-table"), [
      { label: "Player (YES)", render: (m) => { const d = node("div"); d.append(node("div", null, m.player_name)); d.append(node("div", "muted small", "vs " + m.opponent_name)); return d; } },
      { label: "Competition", fmt: (m) => [m.competition || m.tour, m.round].filter(Boolean).join(" · ") },
      { label: "Status", render: (m) => m.match_status === "live" ? pill("good", "live") : m.match_status ? pill("neutral", m.match_status) : pill("neutral", m.market_status) },
      { label: "Score", render: (m) => { const d = node("div", null, m.score_text || "–"); if (m.score_issues && m.score_issues.length) d.title = m.score_issues.join("; "); if (m.score_quiet) d.appendChild(node("div", "muted small", "feed quiet")); return d; } },
      { label: "Srv", fmt: (m) => (m.serving === null ? "" : m.serving ? "●" : "") , title: () => "serving" },
      { label: "Games to win", num: true, fmt: (m) => (m.games_from_victory === null ? "–" : m.games_from_victory + (m.has_match_point ? " MP" : "")) },
      { label: "Bid / Ask", num: true, fmt: (m) => `${cents(m.yes_bid)} / ${cents(m.yes_ask)}` },
      { label: "Spread", num: true, fmt: (m) => cents(m.spread) },
      { label: "Ask depth ≤3¢", num: true, fmt: (m) => compact(m.ask_depth_3c), title: (m) => (m.ask_depth_3c === null ? "" : `${Math.round(m.ask_depth_3c)} contracts within 3¢ of the best ask`) },
      { label: "Market prob", num: true, fmt: (m) => prob(m.market_prob) },
      { label: "Model prob", num: true, fmt: (m) => prob(m.model_prob), title: (m) => m.prior_source ? `prior: ${m.prior_source}${m.pre_match_prob ? " (pre-match " + prob(m.pre_match_prob) + ")" : ""}` : "" },
      { label: "Robust low", num: true, fmt: (m) => prob(m.model_prob_robust) },
      { label: "EV/contract", num: true, fmt: (m) => (m.ev_conservative === null ? "–" : (m.ev_conservative >= 0 ? "+" : "−") + Math.abs(m.ev_conservative * 100).toFixed(1) + "¢"), cls: (m) => (m.ev_conservative > 0 ? "pos" : "") },
      { label: "Opportunity", render: (m) => (m.opportunity ? pill("good", `${m.opportunity_strategy}: ${m.opportunity_side.toUpperCase()}`) : node("span", "muted", "–")), title: (m) => m.opportunity_reason || "" },
      { label: "Data age", num: true, fmt: (m) => `${ago(m.book_age_s)} / ${ago(m.score_age_s)}`, title: (m) => `order book (${m.book_source || "?"}) / score feed; last score change ${ago(m.score_last_change_s)} ago` },
    ], rows, { empty: "No tracked tennis markets. Is the engine running?", rowClass: (m) => (m.opportunity ? "opportunity" : "") });
  }

  async function renderMarkets() {
    await refreshHeader();
    const data = await api("/api/markets");
    state.markets = data.markets;
    renderMarketsTable();
  }

  // ----------------------------------------------------------- activity
  async function renderActivity() {
    await refreshHeader();
    const [p, t, s] = await Promise.all([api("/api/positions"), api("/api/trades?limit=500"), api("/api/signals?limit=100")]);
    table($("positions-table"), [
      { label: "Opened", fmt: (r) => when(r.opened_ts) },
      { label: "Strategy", key: "strategy" },
      { label: "Match", fmt: (r) => r.match_title },
      { label: "Backing", fmt: (r) => `${r.player_name} (${r.side.toUpperCase()} ${r.market_ticker})` },
      { label: "Score at entry", key: "score_at_entry" },
      { label: "Contracts", num: true, fmt: (r) => num(r.contracts, 0) },
      { label: "Entry", num: true, fmt: (r) => cents(r.entry_price) },
      { label: "Cost + fee", num: true, fmt: (r) => money(r.entry_cost + r.entry_fee) },
      { label: "Model / robust", num: true, fmt: (r) => `${prob(r.model_prob)} / ${prob(r.robust_prob)}` },
      { label: "Mark", num: true, fmt: (r) => cents(r.mark) },
      { label: "Unrealised", num: true, fmt: (r) => signedMoney(r.unrealized_pnl), cls: (r) => signClass(r.unrealized_pnl) },
    ], p.positions, { empty: "No open simulated positions." });
    const trades = t.trades;
    table($("trades-table"), [
      { label: "Opened", fmt: (r) => when(r.opened_ts) },
      { label: "Closed", fmt: (r) => when(r.closed_ts) },
      { label: "Strategy", key: "strategy" },
      { label: "Match", fmt: (r) => r.match_title },
      { label: "Backing", fmt: (r) => `${r.player_name} (${r.side.toUpperCase()})` },
      { label: "Score at entry", key: "score_at_entry" },
      { label: "Contracts", num: true, fmt: (r) => num(r.entry_contracts, 0) },
      { label: "Entry", num: true, fmt: (r) => cents(r.entry_price) },
      { label: "Fees", num: true, fmt: (r) => money(r.entry_fee + r.exit_fee) },
      { label: "Model / robust", num: true, fmt: (r) => `${prob(r.model_prob)} / ${prob(r.robust_prob)}` },
      { label: "EV est.", num: true, fmt: (r) => (r.ev_per_contract === null ? "–" : (r.ev_per_contract * 100).toFixed(1) + "¢") },
      { label: "Result", render: (r) => pill((r.realized_pnl || 0) > 0 ? "good" : "critical", (r.realized_pnl || 0) > 0 ? "win" : "loss") },
      { label: "Settlement", fmt: (r) => r.status === "closed" ? "exited" : `${r.settlement_result} (${num(r.settlement_value, 2)})` },
      { label: "P&L", num: true, fmt: (r) => signedMoney(r.realized_pnl), cls: (r) => signClass(r.realized_pnl) },
    ], trades, { empty: "No completed simulated trades yet." });
    const cum = []; let total = 0;
    trades.slice().reverse().forEach((r) => { total += r.realized_pnl || 0; cum.push([r.closed_ts, total]); });
    Charts.line($("activity-cum"), cum, { yFormat: (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toFixed(2), label: "Cumulative P&L", zeroBaseline: true, empty: "Appears after two or more trades settle." });
    table($("signals-table"), [
      { label: "Time", fmt: (r) => when(r.ts) },
      { label: "Strategy", key: "strategy" },
      { label: "Market", fmt: (r) => `${r.side.toUpperCase()} ${r.market}` },
      { label: "Score", key: "score" },
      { label: "Ask", num: true, fmt: (r) => cents(r.ask) },
      { label: "Model / robust", num: true, fmt: (r) => `${prob(r.model_prob)} / ${prob(r.robust_prob)}` },
      { label: "EV", num: true, fmt: (r) => (r.ev === null ? "–" : (r.ev * 100).toFixed(1) + "¢") },
      { label: "Risk decision", render: (r) => pill(r.approved ? "good" : "warning", r.approved ? "approved" : "rejected") },
      { label: "Reasons", fmt: (r) => r.reasons.join("; ") },
    ], s.signals, { empty: "No signals yet. Strategies only fire under strict conditions." });
  }

  // ---------------------------------------------------------- analytics
  async function renderAnalytics() {
    await refreshHeader();
    const a = await api("/api/analytics");
    const m = a.metrics || {};
    $("analytics-kpis").replaceChildren(
      tile("Total P&L (simulated)", signedMoney(m.total_pnl), m.trades ? `${m.trades} trades` : "no trades yet", signClass(m.total_pnl), true),
      tile("Return on bankroll", m.return_on_bankroll == null ? "–" : pct(m.return_on_bankroll, 2), "after fees"),
      tile("ROI on capital", m.roi_on_capital_deployed == null ? "–" : pct(m.roi_on_capital_deployed, 2), "P&L ÷ capital deployed"),
      tile("Win rate", m.win_rate == null ? "–" : pct(m.win_rate), m.trades ? `${m.wins} W / ${m.losses} L` : ""),
      tile("Profit factor", m.profit_factor == null ? "–" : num(m.profit_factor, 2), "gross win ÷ gross loss"),
      tile("EV per trade", m.ev_per_trade == null ? "–" : signedMoney(m.ev_per_trade), `avg win ${money(m.avg_win)} · avg loss ${money(m.avg_loss)}`),
      tile("Max drawdown", m.max_drawdown == null ? "–" : money(m.max_drawdown), m.max_drawdown_fraction == null ? "" : pct(m.max_drawdown_fraction) + " of peak"),
      tile("Fees paid", m.total_fees == null ? "–" : money(m.total_fees), m.pnl_before_fees == null ? "" : "P&L before fees " + signedMoney(m.pnl_before_fees)),
      tile("Bankroll utilisation", m.avg_stake_fraction == null ? "–" : pct(m.avg_stake_fraction, 2), m.peak_exposure_fraction == null ? "" : "peak exposure " + pct(m.peak_exposure_fraction, 1)),
    );
    const sig = a.significance || {};
    const box = $("significance");
    box.replaceChildren(node("p", null, sig.verdict || "–"));
    if (sig.trades) {
      const facts = node("p", "muted small");
      facts.textContent = `Win rate ${pct(sig.win_rate)} (95% CI ${pct(sig.win_rate_ci95[0])}–${pct(sig.win_rate_ci95[1])}) vs break-even ${pct(sig.breakeven_win_rate)} · calibration-null p-value ${num(sig.calibration_null_pvalue, 4)}` +
        (sig.mean_pnl_ci95_cluster_bootstrap ? ` · mean P&L/trade 95% CI ${signedMoney(sig.mean_pnl_ci95_cluster_bootstrap[0])} to ${signedMoney(sig.mean_pnl_ci95_cluster_bootstrap[1])}` : "");
      box.appendChild(facts);
    }
    const dollars = (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toFixed(0);
    Charts.line($("an-equity"), a.equity_curve, { yFormat: dollars, label: "Equity", area: false, empty: "No equity snapshots yet." });
    Charts.line($("an-pnl"), a.cumulative_pnl, { yFormat: (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toFixed(2), label: "Cumulative P&L", zeroBaseline: true, empty: "Appears after two or more trades settle." });
    Charts.line($("an-dd"), a.drawdown, { yFormat: (v) => (v * 100).toFixed(1) + "%", label: "Drawdown", zeroBaseline: true, empty: "No equity history yet." });
    Charts.bars($("an-calib"), (a.win_rate_by_entry_price || []).map((b) => ({ label: b.bucket, short: b.bucket.split("-")[0], value: b.win_rate, ref: b.avg_entry_price, note: `${b.trades} trades · P&L ${signedMoney(b.pnl)}` })),
      { yFormat: (v) => (v * 100).toFixed(0) + "%", label: "Win rate", refLabel: "Avg entry price", max: 1, empty: "No settled trades yet.",
        extraCols: [{ label: "Trades", num: true, fmt: (r) => r.note.split(" ")[0] }] });
    Charts.bars($("an-hist"), (a.pnl_distribution || []).map((b) => ({ label: `${signedMoney(b.lo)} to ${signedMoney(b.hi)}`, short: signedMoney((b.lo + b.hi) / 2), value: b.count })),
      { yFormat: (v) => String(Math.round(v)), label: "Trades", gap: true, empty: "No settled trades yet." });
    const strategies = Object.entries(a.by_strategy || {}).map(([name, s]) => Object.assign({ name }, s));
    table($("an-strategy"), [
      { label: "Strategy", key: "name" },
      { label: "Trades", num: true, key: "trades" },
      { label: "Win rate", num: true, fmt: (s) => pct(s.win_rate) },
      { label: "P&L", num: true, fmt: (s) => signedMoney(s.total_pnl), cls: (s) => signClass(s.total_pnl) },
      { label: "EV/trade", num: true, fmt: (s) => signedMoney(s.ev_per_trade) },
      { label: "Profit factor", num: true, fmt: (s) => num(s.profit_factor, 2) },
      { label: "Max DD", num: true, fmt: (s) => money(s.max_drawdown) },
    ], strategies, { empty: "No completed trades yet." });
    const bt = await api("/api/backtests");
    table($("backtests-table"), [
      { label: "Run", fmt: (r) => `#${r.id} · ${when(r.created_ts)}` },
      { label: "Kind", key: "kind" },
      { label: "Data source", key: "data_source" },
      { label: "In-sample trades", num: true, fmt: (r) => (r.metrics.train || {}).trades ?? "–" },
      { label: "Out-of-sample trades", num: true, fmt: (r) => (r.metrics.test || {}).trades ?? "–" },
      { label: "Out-of-sample P&L", num: true, fmt: (r) => signedMoney((r.metrics.test || {}).total_pnl), cls: (r) => signClass((r.metrics.test || {}).total_pnl) },
      { label: "Out-of-sample verdict", fmt: (r) => ((r.metrics.significance_test || {}).verdict || "").slice(0, 120) },
    ], bt.runs, { empty: "No backtests stored. Run: uv run kalshi-trader backtest historical" });
  }

  // ------------------------------------------------------------- health
  async function renderHealth() {
    await refreshHeader();
    const h = await api("/api/health");
    const c = h.counts || {};
    $("health-kpis").replaceChildren(
      tile("Market snapshots", (c.market_snapshots || 0).toLocaleString(), "recorded order-book states"),
      tile("Score snapshots", (c.score_snapshots || 0).toLocaleString(), "recorded live score states"),
      tile("Markets tracked (all time)", (c.markets || 0).toLocaleString(), `${(c.events || 0).toLocaleString()} matches`),
      tile("Open simulated trades", String(c.open_trades || 0), ""),
    );
    const kind = (s) => (s === "ok" ? "good" : s === "disabled" || s === "starting" || s === "stopped" ? "neutral" : s === "paused" || s === "reconnecting" ? "warning" : "critical");
    table($("health-table"), [
      { label: "Component", key: "component" },
      { label: "Status", render: (r) => pill(r.age_s > 30 && r.status === "ok" ? "warning" : kind(r.status), r.age_s > 30 && r.status === "ok" ? "stale" : r.status) },
      { label: "Last update", fmt: (r) => ago(r.age_s) + " ago" },
      { label: "Last success", fmt: (r) => (r.last_ok_ts ? ago(h.now - r.last_ok_ts) + " ago" : "–") },
      { label: "Last error", fmt: (r) => (r.last_error ? `${ago(h.now - r.last_error_ts)} ago: ${r.last_error.slice(0, 120)}` : "–") },
      { label: "Details", fmt: (r) => Object.entries(r.details || {}).filter(([k, v]) => v !== null && typeof v !== "object").slice(0, 8).map(([k, v]) => `${k}=${typeof v === "number" && v > 1e9 ? ago(h.now - v) + " ago" : v}`).join("  ") },
    ], h.components, { empty: "No component health recorded yet. Start the engine." });
    table($("health-events"), [
      { label: "Time", fmt: (e) => when(e.ts) },
      { label: "Level", render: (e) => pill(e.level === "error" || e.level === "critical" ? "critical" : e.level === "warning" ? "warning" : "neutral", e.level) },
      { label: "Component", key: "component" },
      { label: "Message", fmt: (e) => e.message.slice(0, 200) },
    ], h.events.filter((e) => e.level !== "info").slice(0, 100), { empty: "No warnings or errors recorded." });
  }

  // -------------------------------------------------------------- shell
  const renderers = { overview: renderOverview, markets: renderMarkets, activity: renderActivity, analytics: renderAnalytics, health: renderHealth };
  const intervals = { overview: 5000, markets: 2000, activity: 5000, analytics: 15000, health: 5000 };

  async function refresh() {
    const view = $("view-" + state.view);
    try { view.classList.remove("stale"); await renderers[state.view](); }
    catch (e) { view.classList.add("stale"); console.error(e); }
  }
  function schedule() {
    state.timers.forEach(clearInterval);
    state.timers = [setInterval(refresh, intervals[state.view])];
  }
  document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => {
    state.view = b.dataset.view;
    document.querySelectorAll("nav.tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll("section.view").forEach((s) => { s.hidden = s.id !== "view-" + state.view; });
    refresh(); schedule();
  }));
  ["mk-search", "mk-tour", "mk-live", "mk-opps", "mk-sort"].forEach((id) => $(id).addEventListener("input", renderMarketsTable));

  async function control(action, confirmText) {
    if (confirmText && !window.confirm(confirmText)) return;
    try { await api("/api/control/" + action, { method: "POST", headers: { "X-Kalshi-Trader": "1" } }); }
    catch (e) { window.alert("Control action failed: " + e.message); }
    refresh();
  }
  $("btn-pause").addEventListener("click", () => control("pause"));
  $("btn-resume").addEventListener("click", () => control("resume"));
  $("btn-estop").addEventListener("click", () => control("emergency_stop", "Engage the emergency stop? Paper trading halts until you clear it."));
  $("btn-clear-estop").addEventListener("click", () => control("clear_emergency_stop", "Clear the emergency stop?"));
  $("btn-theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("kt-theme", root.dataset.theme);
  });
  const saved = localStorage.getItem("kt-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  refresh(); schedule();
})();
