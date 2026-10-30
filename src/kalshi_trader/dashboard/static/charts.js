/* Minimal dependency-free SVG charts. All labels are set with textContent (data is
   untrusted). Every chart has a table-view twin and a hover/focus tooltip. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const PLOT_H = 200, PAD_L = 56, PAD_R = 64, PAD_T = 12, AXIS_H = 28;

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function niceTicks(min, max, count) {
    if (!(isFinite(min) && isFinite(max))) return [0];
    if (min === max) { const d = Math.abs(min) || 1; min -= d * 0.5; max += d * 0.5; }
    const span = max - min, raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const start = Math.floor(min / step) * step, ticks = [];
    for (let v = start; v <= max + step * 0.5; v += step) ticks.push(Number(v.toFixed(10)));
    return ticks;
  }
  function timeLabel(ts, spanS) {
    const d = new Date(ts * 1000);
    if (spanS > 2 * 86400) return d.toLocaleDateString([], { month: "short", day: "numeric" });
    if (spanS < 900) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  /* Container with chart + table toggle. Returns {plot, table, setEmpty}. */
  function frame(container, tableRows, tableCols) {
    container.replaceChildren();
    const bar = html("div", "legend");
    const toggle = html("button", "btn toggle-table", "Table view");
    toggle.type = "button";
    const plot = html("div", "chart");
    const table = html("div", "table-wrap");
    table.hidden = true;
    toggle.addEventListener("click", () => {
      table.hidden = !table.hidden;
      plot.hidden = !table.hidden;
      toggle.textContent = table.hidden ? "Table view" : "Chart view";
    });
    const head = html("div");
    head.style.display = "flex";
    head.style.justifyContent = "space-between";
    head.style.alignItems = "center";
    head.append(bar, toggle);
    container.append(head, plot, table);
    const t = html("table", "data");
    const tr = html("tr");
    tableCols.forEach((c) => tr.appendChild(html("th", c.num ? "num" : "", c.label)));
    const thead = html("thead"); thead.appendChild(tr); t.appendChild(thead);
    const tbody = html("tbody");
    tableRows.forEach((row) => {
      const r = html("tr");
      tableCols.forEach((c) => r.appendChild(html("td", c.num ? "num" : "", c.fmt(row))));
      tbody.appendChild(r);
    });
    t.appendChild(tbody);
    table.appendChild(t);
    return { plot, legend: bar };
  }

  function tooltip(plot) {
    const tip = html("div", "tooltip");
    tip.hidden = true;
    plot.appendChild(tip);
    return {
      show(x, y, rows) {
        tip.replaceChildren();
        rows.forEach((r, i) => {
          const line = html("div");
          if (r.key) {
            const k = html("span", "key"); k.style.background = r.key; line.appendChild(k);
          }
          line.appendChild(html("span", i === 0 ? "tv" : "tl", r.text));
          tip.appendChild(line);
        });
        tip.hidden = false;
        const w = plot.clientWidth;
        const left = Math.min(Math.max(0, x + 12), Math.max(0, w - tip.offsetWidth - 4));
        tip.style.left = left + "px";
        tip.style.top = Math.max(0, y - 10) + "px";
      },
      hide() { tip.hidden = true; },
    };
  }

  /* Single-series time line (optional area wash). points: [[ts, value], ...] */
  function line(container, points, opts) {
    const o = Object.assign({ yFormat: (v) => v.toFixed(2), area: true, label: "value",
                              zeroBaseline: false }, opts || {});
    const cols = [
      { label: "Time", fmt: (p) => new Date(p[0] * 1000).toLocaleString() },
      { label: o.label, num: true, fmt: (p) => o.yFormat(p[1]) },
    ];
    const { plot } = frame(container, points.slice(-500), cols);
    if (!points || points.length < 2) {
      plot.appendChild(html("div", "chart-empty", o.empty || "Not enough data yet."));
      return;
    }
    function render() {
      plot.querySelectorAll("svg").forEach((n) => n.remove());
      const width = Math.max(320, plot.clientWidth);
      const W = width - PAD_L - PAD_R;
      const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs);
      let y0 = Math.min(...ys), y1 = Math.max(...ys);
      if (o.zeroBaseline) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
      const ticks = niceTicks(y0, y1, 4);
      y0 = Math.min(y0, ticks[0]); y1 = Math.max(y1, ticks[ticks.length - 1]);
      const sx = (t) => PAD_L + ((t - x0) / Math.max(1, x1 - x0)) * W;
      const sy = (v) => PAD_T + (1 - (v - y0) / Math.max(1e-12, y1 - y0)) * PLOT_H;
      const svg = el("svg", { viewBox: `0 0 ${width} ${PLOT_H + PAD_T + AXIS_H}`,
                              role: "img", "aria-label": o.label + " over time" });
      plot.prepend(svg);
      const grid = el("g", { class: "axis" }, svg);
      ticks.forEach((t) => {
        el("line", { class: "gridline", x1: PAD_L, x2: PAD_L + W, y1: sy(t), y2: sy(t) }, grid);
        const label = el("text", { x: PAD_L - 8, y: sy(t) + 4, "text-anchor": "end" }, grid);
        label.textContent = o.yFormat(t);
      });
      const nx = Math.max(2, Math.min(6, Math.floor(W / 120)));
      for (let i = 0; i <= nx; i++) {
        const t = x0 + ((x1 - x0) * i) / nx;
        const label = el("text", { x: sx(t), y: PAD_T + PLOT_H + 18, "text-anchor": "middle" }, grid);
        label.textContent = timeLabel(t, x1 - x0);
      }
      el("line", { class: "baseline", x1: PAD_L, x2: PAD_L + W, y1: PAD_T + PLOT_H,
                   y2: PAD_T + PLOT_H }, svg);
      const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      if (o.area) {
        const base = sy(o.zeroBaseline ? 0 : y0);
        el("path", { class: "series-area",
                     d: `${d}L${sx(x1).toFixed(1)},${base}L${sx(x0).toFixed(1)},${base}Z` }, svg);
      }
      el("path", { class: "series-line", d }, svg);
      const last = points[points.length - 1];
      el("circle", { class: "end-dot", cx: sx(last[0]), cy: sy(last[1]), r: 4 }, svg);
      const endLabel = el("text", { class: "end-label", x: sx(last[0]) + 8, y: sy(last[1]) + 4 }, svg);
      endLabel.textContent = o.yFormat(last[1]);
      const cross = el("line", { class: "crosshair", y1: PAD_T, y2: PAD_T + PLOT_H,
                                 visibility: "hidden" }, svg);
      const tip = tooltip(plot);
      const hit = el("rect", { x: PAD_L, y: PAD_T, width: W, height: PLOT_H, fill: "transparent",
                               tabindex: 0 }, svg);
      function at(clientX) {
        const rect = svg.getBoundingClientRect();
        const scale = width / rect.width;
        const px = (clientX - rect.left) * scale;
        const t = x0 + ((px - PAD_L) / W) * (x1 - x0);
        let best = 0;
        for (let i = 1; i < points.length; i++) {
          if (Math.abs(points[i][0] - t) < Math.abs(points[best][0] - t)) best = i;
        }
        const p = points[best];
        cross.setAttribute("x1", sx(p[0])); cross.setAttribute("x2", sx(p[0]));
        cross.setAttribute("visibility", "visible");
        tip.show(sx(p[0]) / scale, sy(p[1]) / scale, [
          { text: o.yFormat(p[1]), key: "var(--series-1)" },
          { text: new Date(p[0] * 1000).toLocaleString() },
        ]);
      }
      hit.addEventListener("pointermove", (e) => at(e.clientX));
      hit.addEventListener("pointerleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); });
      hit.addEventListener("focus", () => {
        const rect = svg.getBoundingClientRect();
        at(rect.right - 1);
      });
      hit.addEventListener("blur", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); });
    }
    render();
    new ResizeObserver(() => render()).observe(plot);
  }

  /* Columns. items: [{label, value, ref?}] ; values >= 0 on a shared 0..max axis. */
  function bars(container, items, opts) {
    const o = Object.assign({ yFormat: (v) => v.toFixed(2), label: "value", refLabel: null,
                              max: null, gap: false }, opts || {});
    const cols = [{ label: "Bucket", fmt: (r) => r.label },
                  { label: o.label, num: true, fmt: (r) => o.yFormat(r.value) }];
    if (o.refLabel) cols.push({ label: o.refLabel, num: true, fmt: (r) => o.yFormat(r.ref) });
    if (o.extraCols) cols.push(...o.extraCols);
    const { plot, legend } = frame(container, items, cols);
    if (!items || !items.length) {
      plot.appendChild(html("div", "chart-empty", o.empty || "No data yet."));
      return;
    }
    if (o.refLabel) {  // two encodings -> legend (rect for bars, line for the reference)
      const a = html("span"); const sw = html("span", "sw"); sw.style.background = "var(--series-1)";
      a.append(sw, document.createTextNode(o.label));
      const b = html("span"); const ln = html("span", "ln"); ln.style.background = "var(--series-2)";
      b.append(ln, document.createTextNode(o.refLabel));
      legend.append(a, b);
    }
    function render() {
      plot.querySelectorAll("svg").forEach((n) => n.remove());
      const width = Math.max(320, plot.clientWidth);
      const W = width - PAD_L - 16;
      const vmax = o.max != null ? o.max
        : Math.max(...items.map((i) => Math.max(i.value, i.ref || 0)), 1e-9);
      const ticks = niceTicks(0, vmax, 4);
      const top = Math.max(vmax, ticks[ticks.length - 1]);
      const sy = (v) => PAD_T + (1 - v / top) * PLOT_H;
      const svg = el("svg", { viewBox: `0 0 ${width} ${PLOT_H + PAD_T + AXIS_H}`, role: "img",
                              "aria-label": o.label });
      plot.prepend(svg);
      const g = el("g", { class: "axis" }, svg);
      ticks.forEach((t) => {
        el("line", { class: "gridline", x1: PAD_L, x2: PAD_L + W, y1: sy(t), y2: sy(t) }, g);
        const tx = el("text", { x: PAD_L - 8, y: sy(t) + 4, "text-anchor": "end" }, g);
        tx.textContent = o.yFormat(t);
      });
      const band = W / items.length;
      const bw = o.gap ? Math.max(2, band - 2) : Math.min(24, band * 0.6);
      const tip = tooltip(plot);
      items.forEach((item, i) => {
        const cx = PAD_L + band * i + band / 2;
        const x = cx - bw / 2, y = sy(item.value), base = sy(0), r = Math.min(4, bw / 2, base - y);
        const d = item.value > 0
          ? `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${base}Z`
          : `M${x},${base}H${x + bw}`;
        const bar = el("path", { class: "bar", d }, svg);
        if (item.ref != null) {
          el("line", { class: "ref-mark", x1: cx - bw / 2 - 4, x2: cx + bw / 2 + 4,
                       y1: sy(item.ref), y2: sy(item.ref) }, svg);
        }
        if (!o.gap || items.length <= 14) {
          const lbl = el("text", { x: cx, y: PAD_T + PLOT_H + 18, "text-anchor": "middle" }, g);
          lbl.textContent = item.short || item.label;
        }
        const hit = el("rect", { x: PAD_L + band * i, y: PAD_T, width: band, height: PLOT_H,
                                 fill: "transparent", tabindex: 0 }, svg);
        const rows = [{ text: o.yFormat(item.value), key: "var(--series-1)" }, { text: item.label }];
        if (item.ref != null && o.refLabel) rows.splice(1, 0, { text: `${o.refLabel}: ${o.yFormat(item.ref)}`, key: "var(--series-2)" });
        if (item.note) rows.push({ text: item.note });
        const show = () => { bar.classList.add("active"); tip.show((cx / width) * plot.clientWidth, (y / width) * plot.clientWidth, rows); };
        const hide = () => { bar.classList.remove("active"); tip.hide(); };
        hit.addEventListener("pointermove", show);
        hit.addEventListener("pointerleave", hide);
        hit.addEventListener("focus", show);
        hit.addEventListener("blur", hide);
      });
      el("line", { class: "baseline", x1: PAD_L, x2: PAD_L + W, y1: sy(0), y2: sy(0) }, svg);
    }
    render();
    new ResizeObserver(() => render()).observe(plot);
  }

  window.Charts = { line, bars };
})();
