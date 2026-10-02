"use strict";
/* Stage reports for HATI Watch: every test shows its own measurements, as numbers
   and as charts, from its live snapshot while it runs and from its saved result
   once it finishes. Display only; nothing here reaches the calculation. */
const HatiStages = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  // Series colours validated on the chart surface #0b1521 (dataviz checks, all pairs):
  // blue, orange, aqua. Grey is de-emphasis. Status colours always come with an icon and a word.
  const C = {s1: '#3987e5', s2: '#d95926', s3: '#199e70', other: '#5c6b7a', neg: '#e66767', surface: '#0b1521'};
  const CLASSES = [['rock_like', 'Rock-like', C.s2], ['relief_like', 'Relief-like', C.s3],
                   ['ambiguous', 'Ambiguous', C.s1], ['none', 'None', C.other]];
  const RAMP = ['#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef'];

  const ok = v => typeof v === 'number' && Number.isFinite(v);
  const fmt = (v, d = 2) => ok(v) ? v.toFixed(d) : '--';
  const pct = (v, d = 1) => ok(v) ? `${(100*v).toFixed(d)}%` : '--';
  const int = v => ok(v) ? Math.round(v).toLocaleString('en-GB') : '--';
  const of = (a, b) => `${int(a)} / ${int(b)}`;
  const human = v => v === null || v === undefined ? '--' : String(v).replaceAll('_', ' ');
  const median = values => {
    const s = values.filter(ok).sort((a, b) => a-b);
    if (!s.length) return null;
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
  };
  const share = (a, b) => ok(a) && ok(b) && b > 0 ? a/b : null;

  function node(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) e.setAttribute(k, String(v));
    if (parent) parent.append(e);
    return e;
  }
  function text(parent, x, y, value, attrs = {}) {
    const t = svg('text', {x, y, ...attrs}, parent);
    t.textContent = value;
    return t;
  }
  function tip(el, value) {
    el.dataset.tip = value;
    el.setAttribute('tabindex', '0');
    el.setAttribute('aria-label', value.replaceAll('\n', '. '));
    return el;
  }
  // Text set inside a filled mark takes dark or light ink by the fill's luminance (as a class,
  // because the stylesheet's text colour would override a fill attribute).
  function inkOn(hex) {
    const lum = [1, 3, 5].map(i => parseInt(hex.slice(i, i+2), 16)/255)
      .map(c => c <= .03928 ? c/12.92 : ((c+.055)/1.055)**2.4);
    return .2126*lum[0]+.7152*lum[1]+.0722*lum[2] > .2 ? 'on-light' : 'on-dark';
  }
  function ticks(lo, hi, count = 4) {
    if (!(hi > lo)) hi = lo+1;
    const raw = (hi-lo)/count, mag = 10**Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m*mag).find(s => s >= raw*.999) || 10*mag;
    const out = [];
    for (let v = Math.floor(lo/step)*step; v <= hi+step*.999; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  // Bars grow from one baseline; the data end is rounded, the base square.
  // Horizontal bars start at x and run w to the right (to the left when w is negative).
  function barPath(x, y, w, h, horizontal) {
    const len = horizontal ? Math.abs(w) : h, r = Math.min(4, len, horizontal ? h/2 : w/2);
    if (!horizontal) {
      if (r <= .5) return `M${x},${y}h${w}v${h}h${-w}z`;
      return `M${x},${y+h}v${-(h-r)}a${r},${r} 0 0 1 ${r},${-r}h${w-2*r}a${r},${r} 0 0 1 ${r},${r}v${h-r}z`;
    }
    if (r <= .5) return `M${Math.min(x, x+w)},${y}h${len}v${h}h${-len}z`;
    return w >= 0
      ? `M${x},${y}h${len-r}a${r},${r} 0 0 1 ${r},${r}v${h-2*r}a${r},${r} 0 0 1 ${-r},${r}h${-(len-r)}z`
      : `M${x},${y}h${-(len-r)}a${r},${r} 0 0 0 ${-r},${r}v${h-2*r}a${r},${r} 0 0 0 ${r},${r}h${len-r}z`;
  }
  function frame(host, width, height, title) {
    return svg('svg', {viewBox: `0 0 ${width} ${height}`, class: 'plot-svg', role: 'img', 'aria-label': title || 'chart'}, host);
  }

  /* Horizontal bars; negative values grow left of zero. rows: {label, value, text, tip, color} */
  function hbars(host, {rows, lo, hi, refs = [], format = v => fmt(v), width = 560, labelW = 150, tickFormat}) {
    const rowH = 26, top = refs.length ? 24 : 8, bottom = 24, height = top+rows.length*rowH+bottom;
    const s = frame(host, width, height);
    const values = rows.map(r => r.value).filter(ok).concat(refs.map(r => r.value));
    const t = ticks(Math.min(0, lo ?? Math.min(0, ...values)), hi ?? Math.max(1e-9, ...values));
    const a = t[0], b = t[t.length-1], x0 = labelW+10, x1 = width-70, X = v => x0+(x1-x0)*(v-a)/(b-a);
    for (const v of t) {
      svg('line', {x1: X(v), x2: X(v), y1: top-4, y2: height-bottom, class: v === 0 ? 'baseline' : 'grid'}, s);
      text(s, X(v), height-8, (tickFormat || format)(v), {class: 'tick', 'text-anchor': 'middle'});
    }
    rows.forEach((r, i) => {
      const y = top+i*rowH, g = svg('g', {class: 'mark'}, s);
      svg('rect', {x: 0, y, width, height: rowH, class: 'hit'}, g);
      text(g, labelW, y+rowH/2+4, r.label, {class: 'row-label', 'text-anchor': 'end'});
      if (ok(r.value)) {
        const w = X(r.value)-X(0);
        svg('path', {d: barPath(X(0), y+7, w, 12, true), fill: r.color || (r.value < 0 ? C.neg : C.s1)}, g);
        const end = r.value < 0 ? X(0)+6 : X(r.value)+6;
        text(g, Math.max(end, X(0)+6), y+rowH/2+4, r.text ?? format(r.value), {class: 'value'});
      } else text(g, X(0)+6, y+rowH/2+4, r.text ?? 'not measured', {class: 'value muted'});
      if (r.tip) tip(g, r.tip);
    });
    for (const ref of refs) {
      const x = X(ref.value);
      svg('line', {x1: x, x2: x, y1: top-6, y2: height-bottom, class: 'ref'}, s);
      text(s, x, top-10, ref.label, {class: 'ref-label', 'text-anchor': x > width-120 ? 'end' : 'middle'});
    }
  }

  /* Grouped columns. groups: {label, sub, values: [{value, text, tip}]}, series: [{name, color}] */
  function columns(host, {groups, series, hi, refs = [], format = v => fmt(v), width = 560, height = 230}) {
    const left = 46, right = 12, top = refs.length ? 24 : 16, bottom = 44;
    const s = frame(host, width, height);
    const values = groups.flatMap(g => g.values.map(v => v && v.value)).filter(ok).concat(refs.map(r => r.value));
    const t = ticks(0, hi ?? Math.max(1e-9, ...values)), b = t[t.length-1];
    const Y = v => height-bottom-(height-bottom-top)*Math.max(0, v)/b;
    for (const v of t) {
      svg('line', {x1: left, x2: width-right, y1: Y(v), y2: Y(v), class: v === 0 ? 'baseline' : 'grid'}, s);
      text(s, left-6, Y(v)+4, format(v), {class: 'tick', 'text-anchor': 'end'});
    }
    const band = (width-left-right)/groups.length, n = series.length;
    const bw = Math.max(6, Math.min(24, (band*.72-(n-1)*2)/n));
    groups.forEach((g, i) => {
      const cx = left+band*(i+.5), start = cx-(n*bw+(n-1)*2)/2;
      text(s, cx, height-bottom+16, g.label, {class: 'tick strong', 'text-anchor': 'middle'});
      if (g.sub) text(s, cx, height-bottom+30, g.sub, {class: 'tick', 'text-anchor': 'middle'});
      g.values.forEach((v, j) => {
        if (!v) return;
        const x = start+j*(bw+2), mark = svg('g', {class: 'mark'}, s);
        svg('rect', {x: x-1, y: top, width: bw+2, height: Y(0)-top, class: 'hit'}, mark);
        // Values ride on the caps only for a single series; with more, labels would collide,
        // so the tooltip and the table carry them.
        if (ok(v.value)) {
          svg('path', {d: barPath(x, Y(v.value), bw, Math.max(0, Y(0)-Y(v.value)), false), fill: series[j].color}, mark);
          if (v.text && n === 1) text(mark, x+bw/2, Y(v.value)-5, v.text, {class: 'value', 'text-anchor': 'middle'});
          // A zero has no height to see, so it says so.
          else if (v.value === 0) text(mark, x+bw/2, Y(0)-5, '0', {class: 'value small', 'text-anchor': 'middle'});
        } else text(mark, x+bw/2, Y(0)-5, 'n/a', {class: 'value small muted', 'text-anchor': 'middle'});
        if (v.tip) tip(mark, v.tip);
      });
    });
    for (const ref of refs) {
      const y = Y(ref.value);
      svg('line', {x1: left, x2: width-right, y1: y, y2: y, class: 'ref'}, s);
      text(s, width-right, y-5, ref.label, {class: 'ref-label', 'text-anchor': 'end'});
    }
  }

  /* 100% horizontal stacks. rows: {label, parts: {key: value}, n}; keys: [{key, name, color}] */
  function stack100(host, {rows, keys, width = 560, labelW = 170, unit = 'cells'}) {
    const rowH = 32, height = 8+rows.length*rowH, s = frame(host, width, height);
    const x0 = labelW+10, span = width-10-x0;
    rows.forEach((r, i) => {
      const y = 4+i*rowH, total = keys.reduce((m, k) => m+(ok(r.parts[k.key]) ? r.parts[k.key] : 0), 0);
      text(s, labelW, y+rowH/2+4, r.label, {class: 'row-label', 'text-anchor': 'end'});
      let x = x0;
      for (const k of keys) {
        const v = r.parts[k.key];
        if (!ok(v) || v <= 0 || !total) continue;
        const w = span*v/total, seg = Math.max(0, w-2), mark = svg('g', {class: 'mark'}, s);
        svg('rect', {x, y: y+6, width: seg, height: 18, fill: k.color}, mark);
        if (seg >= 36) text(mark, x+seg/2, y+rowH/2+4, pct(v/total, 0), {class: `inside ${inkOn(k.color)}`, 'text-anchor': 'middle'});
        tip(mark, `${pct(v/total)} ${k.name}\n${r.label}${ok(r.n) ? ` · ${int(r.n)} ${unit}` : ''}`);
        x += w;
      }
    });
  }

  /* Heat cells, one hue, lighter is more. cells[i][j]: {text, sub, t, tip} */
  function matrix(host, {rows, cols, cells, colTitle, width = 560, labelW = 110}) {
    const top = colTitle ? 42 : 26, cellH = 42, height = top+rows.length*cellH+6, s = frame(host, width, height);
    const x0 = labelW+10, cw = (width-x0-2)/cols.length;
    if (colTitle) text(s, x0+(width-x0)/2, 12, colTitle, {class: 'axis-title', 'text-anchor': 'middle'});
    cols.forEach((c, j) => text(s, x0+cw*(j+.5), top-8, c, {class: 'tick strong', 'text-anchor': 'middle'}));
    rows.forEach((r, i) => {
      const y = top+i*cellH;
      text(s, labelW, y+cellH/2+4, r, {class: 'row-label', 'text-anchor': 'end'});
      cols.forEach((c, j) => {
        const cell = cells[i][j];
        if (!cell) return;
        const mark = svg('g', {class: 'mark'}, s);
        const fill = ok(cell.t) ? RAMP[Math.max(0, Math.min(RAMP.length-1, Math.round(cell.t*(RAMP.length-1))))] : '#16222f';
        svg('rect', {x: x0+j*cw+1, y: y+1, width: cw-2, height: cellH-2, rx: 4, fill}, mark);
        const ink = ok(cell.t) ? inkOn(fill) : 'on-empty', cx = x0+cw*(j+.5);
        text(mark, cx, y+cellH/2+(cell.sub ? -1 : 4), cell.text, {class: `inside ${ink}`, 'text-anchor': 'middle'});
        if (cell.sub) text(mark, cx, y+cellH/2+13, cell.sub, {class: `inside small ${ink}`, 'text-anchor': 'middle'});
        if (cell.tip) tip(mark, cell.tip);
      });
    });
  }

  /* Measured against true, with the identity line. points: {x, y, key, tip}; keys: [{key, name, color, shape}] */
  // Tick labels without rounding a 0.25 step to 0.3.
  const exact = v => String(+v.toFixed(2));

  /* Points on two axes. identity (the default) shares one scale and draws measured = true;
     otherwise xHi and yHi set each axis. yDown puts zero at the top, as in an image. p.r sizes a point. */
  function scatter(host, {points, keys, hi, xHi, yHi, xLabel, yLabel, width = 400, height = 320, refs = [],
                          identity = true, square = identity, yDown = false, xTick = exact, yTick = exact}) {
    const left = 50, right = 14, top = 14, bottom = 42, s = frame(host, width, height);
    if (square) s.classList.add('square');
    const xs = points.map(p => p.x).filter(ok), ys = points.map(p => p.y).filter(ok);
    const tx = ticks(0, xHi ?? hi ?? Math.max(1, ...xs, ...(identity ? ys : []))), bx = tx[tx.length-1];
    const ty = identity ? tx : ticks(0, yHi ?? hi ?? Math.max(1, ...ys)), by = ty[ty.length-1];
    const X = v => left+(width-left-right)*v/bx;
    const Y = v => yDown ? top+(height-bottom-top)*v/by : height-bottom-(height-bottom-top)*v/by;
    for (const v of tx) {
      svg('line', {x1: X(v), x2: X(v), y1: top, y2: height-bottom, class: v ? 'grid' : 'baseline'}, s);
      text(s, X(v), height-bottom+15, xTick(v), {class: 'tick', 'text-anchor': 'middle'});
    }
    for (const v of ty) {
      svg('line', {x1: left, x2: width-right, y1: Y(v), y2: Y(v), class: v ? 'grid' : 'baseline'}, s);
      text(s, left-6, Y(v)+4, yTick(v), {class: 'tick', 'text-anchor': 'end'});
    }
    if (identity) {
      svg('line', {x1: X(0), y1: Y(0), x2: X(bx), y2: Y(bx), class: 'identity'}, s);
      text(s, X(bx)-4, Y(bx)+14, 'measured = true', {class: 'ref-label', 'text-anchor': 'end'});
    }
    for (const ref of refs) {
      svg('line', {x1: left, x2: width-right, y1: Y(ref.value), y2: Y(ref.value), class: 'ref'}, s);
      text(s, width-right, Y(ref.value)-5, ref.label, {class: 'ref-label', 'text-anchor': 'end'});
    }
    text(s, left+(width-left-right)/2, height-6, xLabel, {class: 'axis-title', 'text-anchor': 'middle'});
    const my = top+(height-top-bottom)/2;
    text(s, 13, my, yLabel, {class: 'axis-title', 'text-anchor': 'middle', transform: `rotate(-90 13 ${my})`});
    for (const p of points) {
      if (!ok(p.x) || !ok(p.y)) continue;
      const k = keys.find(q => q.key === p.key) || keys[0], x = X(p.x), y = Y(p.y), mark = svg('g', {class: 'mark'}, s);
      const r = p.r ?? 4.5, q = r/4.5;
      svg('circle', {cx: x, cy: y, r: Math.max(12, r+4), class: 'hit'}, mark);
      if (k.shape === 'triangle') svg('path', {d: `M${x},${y+6*q}l${5.6*q},${-9.6*q}h${-11.2*q}z`, fill: k.color, class: 'dot'}, mark);
      else if (k.shape === 'cross') svg('path', {d: `M${x-6*q},${y}h${12*q}M${x},${y-6*q}v${12*q}`, stroke: k.color, 'stroke-width': 2.5, fill: 'none', class: 'cross'}, mark);
      else svg('circle', {cx: x, cy: y, r, fill: k.color, class: 'dot'}, mark);
      if (p.tip) tip(mark, p.tip);
    }
  }

  /* A value against a reference distribution: band from lo to hi, whisker to max. */
  function ranges(host, {rows, hi = 1, format = v => pct(v, 0), width = 560, labelW = 170, refs = []}) {
    const rowH = 34, top = refs.length ? 24 : 8, bottom = 24, height = top+rows.length*rowH+bottom, s = frame(host, width, height);
    const t = ticks(0, hi), b = t[t.length-1], x0 = labelW+10, x1 = width-60, X = v => x0+(x1-x0)*v/b;
    for (const v of t) {
      svg('line', {x1: X(v), x2: X(v), y1: top-4, y2: height-bottom, class: v ? 'grid' : 'baseline'}, s);
      text(s, X(v), height-8, format(v), {class: 'tick', 'text-anchor': 'middle'});
    }
    rows.forEach((r, i) => {
      const y = top+i*rowH+rowH/2, mark = svg('g', {class: 'mark'}, s);
      svg('rect', {x: 0, y: y-rowH/2, width, height: rowH, class: 'hit'}, mark);
      text(mark, labelW, y+4, r.label, {class: 'row-label', 'text-anchor': 'end'});
      if (ok(r.lo) && ok(r.hi)) svg('rect', {x: X(r.lo), y: y-6, width: Math.max(2, X(r.hi)-X(r.lo)), height: 12, rx: 3, fill: C.other}, mark);
      if (ok(r.whisker) && ok(r.hi)) svg('line', {x1: X(r.hi), x2: X(r.whisker), y1: y, y2: y, class: 'whisker'}, mark);
      if (ok(r.whisker)) svg('line', {x1: X(r.whisker), x2: X(r.whisker), y1: y-5, y2: y+5, class: 'whisker'}, mark);
      if (ok(r.value)) {
        svg('circle', {cx: X(r.value), cy: y, r: 6, fill: C.s1, class: 'dot'}, mark);
        text(mark, X(r.value)+10, y-9, format(r.value), {class: 'value'});
      }
      if (r.tip) tip(mark, r.tip);
    });
    for (const ref of refs) {
      svg('line', {x1: X(ref.value), x2: X(ref.value), y1: top-6, y2: height-bottom, class: 'ref'}, s);
      text(s, X(ref.value), top-10, ref.label, {class: 'ref-label', 'text-anchor': 'middle'});
    }
  }

  /* Before and after on one scale. rows: {label, from, to, tip} */
  function dumbbell(host, {rows, hi = 1, format = v => pct(v, 0), width = 560, labelW = 190}) {
    const rowH = 30, top = 8, bottom = 24, height = top+rows.length*rowH+bottom, s = frame(host, width, height);
    const t = ticks(0, hi), b = t[t.length-1], x0 = labelW+10, x1 = width-60, X = v => x0+(x1-x0)*v/b;
    for (const v of t) {
      svg('line', {x1: X(v), x2: X(v), y1: top, y2: height-bottom, class: v ? 'grid' : 'baseline'}, s);
      text(s, X(v), height-8, format(v), {class: 'tick', 'text-anchor': 'middle'});
    }
    rows.forEach((r, i) => {
      const y = top+i*rowH+rowH/2, mark = svg('g', {class: 'mark'}, s);
      svg('rect', {x: 0, y: y-rowH/2, width, height: rowH, class: 'hit'}, mark);
      text(mark, labelW, y+4, r.label, {class: 'row-label', 'text-anchor': 'end'});
      if (ok(r.from) && ok(r.to)) svg('line', {x1: X(r.from), x2: X(r.to), y1: y, y2: y, class: 'connector'}, mark);
      if (ok(r.from)) svg('circle', {cx: X(r.from), cy: y, r: 5, fill: C.other, class: 'dot'}, mark);
      if (ok(r.to)) {
        svg('circle', {cx: X(r.to), cy: y, r: 5, fill: C.s1, class: 'dot'}, mark);
        text(mark, Math.max(X(r.to), ok(r.from) ? X(r.from) : 0)+10, y+4, format(r.to), {class: 'value'});
      }
      if (r.tip) tip(mark, r.tip);
    });
  }

  /* Every trial as a dot, with the median as a tick. groups: {label, values, tip} */
  function strip(host, {groups, hi, refs = [], width = 560, labelW = 150, format = v => fmt(v, 0)}) {
    const rowH = 30, top = refs.length ? 24 : 8, bottom = 24, height = top+groups.length*rowH+bottom, s = frame(host, width, height);
    const values = groups.flatMap(g => g.values).filter(ok).concat(refs.map(r => r.value));
    const t = ticks(0, hi ?? Math.max(1, ...values)), b = t[t.length-1], x0 = labelW+10, x1 = width-40;
    const X = v => x0+(x1-x0)*Math.min(v, b)/b;
    for (const v of t) {
      svg('line', {x1: X(v), x2: X(v), y1: top-4, y2: height-bottom, class: v ? 'grid' : 'baseline'}, s);
      text(s, X(v), height-8, format(v), {class: 'tick', 'text-anchor': 'middle'});
    }
    groups.forEach((g, i) => {
      const y = top+i*rowH+rowH/2, mark = svg('g', {class: 'mark'}, s);
      svg('rect', {x: 0, y: y-rowH/2, width, height: rowH, class: 'hit'}, mark);
      text(mark, labelW, y+4, g.label, {class: 'row-label', 'text-anchor': 'end'});
      g.values.filter(ok).forEach((v, k) => svg('circle', {cx: X(v), cy: y+((k*7919)%11-5)*.9, r: 3, fill: C.s1, class: 'speck'}, mark));
      const m = median(g.values);
      if (ok(m)) svg('line', {x1: X(m), x2: X(m), y1: y-9, y2: y+9, class: 'median'}, mark);
      if (g.tip) tip(mark, g.tip);
    });
    for (const ref of refs) {
      svg('line', {x1: X(ref.value), x2: X(ref.value), y1: top-6, y2: height-bottom, class: 'ref'}, s);
      text(s, X(ref.value), top-10, ref.label, {class: 'ref-label', 'text-anchor': 'middle'});
    }
  }

  const CHARTS = {hbars, columns, stack100, matrix, scatter, ranges, dumbbell, strip};

  /* ---- Page pieces ---- */
  function kpi(k) {
    const d = node('div', 'kpi'+(k.state ? ' '+k.state : '')+(k.wide ? ' wide' : ''));
    d.append(node('span', 'kpi-label', k.label), node('strong', 'kpi-value', k.value));
    if (k.state) d.append(node('span', 'kpi-state', `${{good: '✓', warn: '!', bad: '✕'}[k.state]} ${k.stateText || {good: 'as expected', warn: 'check', bad: 'not met'}[k.state]}`));
    if (k.sub) d.append(node('small', 'kpi-sub', k.sub));
    if (ok(k.progress)) {
      const p = node('progress');
      p.max = 1; p.value = Math.max(0, Math.min(1, k.progress));
      d.append(p);
    }
    return d;
  }
  function legend(items) {
    const d = node('div', 'legend');
    for (const it of items) {
      const item = node('span', 'legend-item'), sw = node('i', 'swatch'+(it.shape ? ' '+it.shape : ''));
      sw.style.background = it.color;
      item.append(sw, document.createTextNode(it.name));
      d.append(item);
    }
    return d;
  }
  function tableView(table) {
    const d = node('details', 'table-view'), t = node('table', 'data-table'), head = node('tr');
    d.append(node('summary', null, 'Table'));
    for (const h of table.head) head.append(node('th', null, h));
    const thead = node('thead'), body = node('tbody');
    thead.append(head);
    for (const r of table.rows) {
      const tr = node('tr');
      for (const v of r) tr.append(node('td', null, v === null || v === undefined ? '--' : String(v)));
      body.append(tr);
    }
    t.append(thead, body);
    d.append(t);
    return d;
  }
  function chart(spec) {
    const fig = node('figure', 'chart'+(spec.wide ? ' wide' : '')), cap = node('figcaption');
    cap.append(node('strong', null, spec.title));
    if (spec.subtitle) cap.append(node('span', null, spec.subtitle));
    fig.append(cap);
    if (spec.legend && spec.legend.length > 1) fig.append(legend(spec.legend));
    const host = node('div', 'plot');
    fig.append(host);
    try {
      CHARTS[spec.type](host, spec);
    } catch (error) {
      host.replaceChildren(node('p', 'hint', `Chart unavailable: ${error.message}`));
    }
    if (spec.note) fig.append(node('p', 'chart-note', spec.note));
    if (spec.table) fig.append(tableView(spec.table));
    return fig;
  }

  /* ---- The views: what each test asks, its numbers and its charts ---- */
  const QUESTIONS = {
    maps: 'The three maps replayed from the saved run: where each module is qualified, and what each says at the touchdown.',
    T1: 'Does the detector find planted shadows on the real Sun geometry, stay quiet on static ground, and how does it react to changing backgrounds and real relief?',
    T2: 'Do the warnings hold up when single frames, the right-camera frame or the poorly registered frames are left out?',
    T3: 'Do the warnings follow the real Sun, or do they persist when the Sun directions are handed to the wrong frames?',
    T5: 'Does letting the detector choose wider shadows change which roots warn?',
    T6: 'How tightly does a single cell\'s height profile constrain the caster?',
    T7: 'How well do the frames line up locally, and can planted shifts be measured?',
    T9: 'What happens to each warning cell when it gets more image context?',
    T10: 'On synthetic 3D rocks of known height, does the adaptive sizing recover the truth?',
    T11: 'Does a fitted object predict a frame it never saw, better than a static scene or a wrong Sun?',
    T12: 'How noisy are the images really, and is the excess noise relief that follows the Sun?',
    T13: 'Is each warning a compact rock or extended relief? Calibrated on simulated scenes, then applied to the real cells.',
    T14: 'Which real rocks does the detector find once shape from shading removes the relief, and how well does it find and size planted rocks of known shape?',
    T16: 'Do terrain-only scenes get through the adaptive gate? They should not, while real casters should.',
  };

  function viewMaps(r) {
    const cf = r.counterfactual || {}, maps = cf.maps || {}, q = r.qualification || {};
    const kpis = ['terrain', 'shadow', 'fused'].map(name => {
      const m = maps[name] || {};
      return {label: `${name[0].toUpperCase()+name.slice(1)} index at the touchdown`, value: fmt(m.index, 3),
              state: m.at_or_above_configured_threshold ? 'warn' : undefined, stateText: 'at or above the threshold',
              sub: ok(m.descriptive_percentile_in_available_module_area) ? `${fmt(m.descriptive_percentile_in_available_module_area, 0)}th percentile of its module's area` : undefined};
    });
    kpis.push({label: 'Qualified at the touchdown', value: cf.qualified_at_touchdown ? 'Yes' : 'No', sub: human(cf.assessment)});
    const names = {0: 'Unknown', 1: 'Both modules qualified', 2: 'High evidence, incomplete'};
    const levels = Object.entries(q).sort();
    return {kpis, charts: levels.length ? [{type: 'hbars', title: 'Fused map: share of the window by status', subtitle: 'Where the terrain and shadow modules are both qualified, and where evidence is high but incomplete',
      rows: levels.map(([k, v]) => ({label: names[k] || `Status ${k}`, value: v, text: pct(v), tip: `${names[k] || `Status ${k}`}: ${pct(v)} of the window`})), hi: 1, format: v => pct(v, 0), labelW: 190,
      table: {head: ['Status', 'Share'], rows: levels.map(([k, v]) => [names[k] || k, pct(v)])}}] : [],
      note: cf.limitation};
  }

  function viewT1(r) {
    const controls = (r.controls || []).filter(c => c.kind && c.status === 'assessed');
    const groups = [['static', null, 'Static scene', 'quiet'], ['structured_null', null, 'Changing background', 'quiet'],
                    ['resolved_ridge', null, 'Resolved ridge', 'relief'], ['caster', .3, '0.3 m caster', 'found'],
                    ['caster', .6, '0.6 m caster', 'found'], ['caster', 1.2, '1.2 m caster', 'found']].map(([kind, h, name, expect]) => {
      const set = controls.filter(c => c.kind === kind && (h === null || Math.abs((c.height_m ?? -1)-h) < 1e-6));
      const hits = expect === 'found' ? set.filter(c => c.recovered_within_2px).length : set.filter(c => c.warning_roots > 0).length;
      return {name, expect, n: set.length, hits, scores: set.map(c => c.maximum_score)};
    }).filter(g => g.n);
    const verb = g => g.expect === 'found' ? 'found within 2 px' : 'warned';
    const kpis = groups.map(g => {
      const f = share(g.hits, g.n);
      const state = g.expect === 'found' ? (f >= .9 ? 'good' : f >= .5 ? 'warn' : 'bad') : g.expect === 'quiet' ? (f <= .1 ? 'good' : 'bad') : undefined;
      return {label: g.name, value: of(g.hits, g.n), sub: `trials ${verb(g)}`, state,
              stateText: g.expect === 'found' ? (state === 'good' ? 'found' : 'missed often') : g.expect === 'quiet' ? (state === 'good' ? 'stays quiet' : 'warns on a null') : undefined};
    });
    const frames = r.frames || [];
    return {kpis, charts: [
      {type: 'hbars', title: 'What each control does', subtitle: 'Casters: share found within 2 px. Nulls and the ridge: share of trials with a warning.',
       rows: groups.map(g => ({label: g.name, value: share(g.hits, g.n), text: of(g.hits, g.n),
         tip: `${g.name}: ${of(g.hits, g.n)} trials ${verb(g)}\nExpected: ${g.expect === 'found' ? 'found' : g.expect === 'quiet' ? 'no warning' : 'real relief, a warning is not a false alarm'}`})),
       hi: 1, format: v => pct(v, 0), table: {head: ['Control', 'Trials', 'Outcome', 'Expected'], rows: groups.map(g => [g.name, g.n, `${g.hits} ${verb(g)}`, g.expect])}},
      {type: 'strip', title: 'Highest score in each trial', subtitle: 'One dot per trial; the tick is the median. The dashed line is the warning score.',
       groups: groups.map(g => ({label: g.name, values: g.scores, tip: `${g.name}: median highest score ${fmt(median(g.scores), 1)} over ${g.n} trials`})),
       refs: [{value: 8, label: 'warning 8'}],
       table: {head: ['Control', 'Median', 'Minimum', 'Maximum'], rows: groups.map(g => [g.name, fmt(median(g.scores), 1), fmt(Math.min(...g.scores.filter(ok)), 1), fmt(Math.max(...g.scores.filter(ok)), 1)])}},
      ...(frames.length ? [{type: 'hbars', wide: true, title: 'Signed contribution by frame', subtitle: 'Median signed fit improvement each frame adds at the eligible roots; negative frames argue against a shadow.',
        rows: frames.map(f => ({label: `${f.frame+1} · ${String(f.pid).replace('nac.', '')}`, value: f.signed_median, text: fmt(f.signed_median, 1),
          tip: `Frame ${f.frame+1} (${f.pid})\nSun ${fmt(f.azimuth_deg, 1)}° azimuth, ${fmt(f.elevation_deg, 2)}° high, cot e ${fmt(f.cot_e, 1)}\nMedian signed contribution ${fmt(f.signed_median, 2)}; negative at ${int(f.negative_count)} of ${int(f.eligible_selected_roots)} roots`})),
        format: v => fmt(v, 0), labelW: 190,
        table: {head: ['Frame', 'Sun az', 'Sun el', 'Median contribution', 'Negative roots'], rows: frames.map(f => [f.pid, fmt(f.azimuth_deg, 1), fmt(f.elevation_deg, 2), fmt(f.signed_median, 2), `${int(f.negative_count)} / ${int(f.eligible_selected_roots)}`])}}] : [])]};
  }

  function groupName(group) {
    const m = /^without_frame_(\d+)$/.exec(group || '');
    if (m) return `Without frame ${Number(m[1])+1}`;
    return {without_RE: 'Without the right-camera frame', without_low_NCC: 'Without the low-NCC frames'}[group] || human(group);
  }

  function viewT2(r) {
    const rows = (r.comparisons || []).map(c => ({label: groupName(c.group), from: c.first_exceedance_on_matched, to: c.second_exceedance_on_matched,
      tip: `${groupName(c.group)}: warnings ${pct(c.first_exceedance_on_matched)} with all frames, ${pct(c.second_exceedance_on_matched)} without\nCasters found ${of(c.recovered_caster_trials, c.assessed_caster_trials)} · nulls warning ${of(c.null_trials_with_warnings, c.assessed_null_trials)}`}));
    return {kpis: [{label: 'Groups tested', value: String(rows.length)}], charts: [
      {type: 'dumbbell', wide: true, title: 'Share of roots warning, with all frames and without the group', subtitle: 'Grey: all frames. Blue: the group left out.',
       rows, legend: [{name: 'All frames', color: C.other, shape: 'circle'}, {name: 'Group left out', color: C.s1, shape: 'circle'}],
       hi: Math.max(.5, ...rows.flatMap(x => [x.from, x.to]).filter(ok)),
       labelW: 230,
       table: {head: ['Group', 'All frames', 'Without', 'Casters found', 'Nulls warning'], rows: (r.comparisons || []).map(c => [groupName(c.group), pct(c.first_exceedance_on_matched), pct(c.second_exceedance_on_matched), of(c.recovered_caster_trials, c.assessed_caster_trials), of(c.null_trials_with_warnings, c.assessed_null_trials)])}}]};
  }

  function viewT3(r) {
    const comps = r.comparisons || [];
    return {kpis: [{label: 'Wrong-Sun assignments', value: String(comps.length)},
                   {label: 'Warning share, real Sun', value: pct(comps[0]?.first_exceedance_on_matched)},
                   {label: 'Median with the wrong Sun', value: pct(median(comps.map(c => c.second_exceedance_on_matched)))}],
      charts: [{type: 'dumbbell', wide: true, title: 'Share of roots warning: real Sun against each cyclic reassignment', subtitle: 'Grey: real geometry. Blue: Sun directions shifted by the given number of frames.',
        rows: comps.map(c => ({label: `shift ${c.shift}`, from: c.first_exceedance_on_matched, to: c.second_exceedance_on_matched,
          tip: `Shift ${c.shift}: ${pct(c.first_exceedance_on_matched)} with the real Sun, ${pct(c.second_exceedance_on_matched)} reassigned\n${pct(c.fraction_second_lower)} of roots score lower`})),
        legend: [{name: 'Real Sun', color: C.other, shape: 'circle'}, {name: 'Reassigned', color: C.s1, shape: 'circle'}], hi: .6,
        table: {head: ['Shift', 'Real Sun', 'Reassigned', 'Roots lower'], rows: comps.map(c => [c.shift, pct(c.first_exceedance_on_matched), pct(c.second_exceedance_on_matched), pct(c.fraction_second_lower)])}}]};
  }

  function viewT5(r) {
    const c = r.comparison || {};
    return {kpis: [{label: 'Roots compared', value: int(c.matched_roots)},
                   {label: 'Widest width chosen', value: pct(r.widest_selected_fraction), sub: 'share of roots whose best fit takes the widest template'},
                   {label: 'Roots scoring lower', value: pct(c.fraction_second_lower), sub: 'with the expanded width bank'}],
      charts: [{type: 'dumbbell', title: 'Share of roots warning', subtitle: 'Grey: the declared widths. Blue: with the expanded width bank.',
        rows: [{label: 'All matched roots', from: c.first_exceedance_on_matched, to: c.second_exceedance_on_matched,
          tip: `Declared widths: ${pct(c.first_exceedance_on_matched)} of roots warn\nExpanded widths: ${pct(c.second_exceedance_on_matched)}`}],
        legend: [{name: 'Declared widths', color: C.other, shape: 'circle'}, {name: 'Expanded widths', color: C.s1, shape: 'circle'}],
        hi: Math.max(.5, ...[c.first_exceedance_on_matched, c.second_exceedance_on_matched].filter(ok)),
        table: {head: ['Width bank', 'Share of roots warning'], rows: [['Declared', pct(c.first_exceedance_on_matched)], ['Expanded', pct(c.second_exceedance_on_matched)]]}}]};
  }

  function viewT9(r) {
    const counts = Object.entries(r.state_counts || {}).sort((a, b) => b[1]-a[1]);
    const total = counts.reduce((m, [, v]) => m+v, 0);
    return {kpis: [{label: 'Cells refined', value: of(r.processed, r.requested)},
                   {label: 'Context supported', value: int((r.state_counts || {}).context_supported_unvalidated), sub: 'stable, supported dimensions'}],
      charts: [{type: 'hbars', wide: true, title: 'Where each warning cell ended up', subtitle: 'Final state after the context expansion',
        rows: counts.map(([k, v]) => ({label: human(k), value: v, text: `${int(v)} (${pct(share(v, total), 0)})`, tip: `${human(k)}: ${int(v)} cells, ${pct(share(v, total))}`})),
        format: v => int(v), labelW: 220, table: {head: ['State', 'Cells', 'Share'], rows: counts.map(([k, v]) => [human(k), int(v), pct(share(v, total))])}}]};
  }

  function viewT10(r) {
    const controls = r.controls || [];
    return {kpis: controls.filter(c => ok(c.median_absolute_height_error_m)).map(c => ({label: human(c.kind), value: `${fmt(c.median_absolute_height_error_m, 2)} m`, sub: `median height error · bias ${fmt(c.median_height_bias_m, 2)} m`})),
      charts: [{type: 'hbars', wide: true, title: 'Trials recovered within 2 px, by scene', subtitle: 'Share of trials; rocks should be found, nulls should not warn',
        rows: controls.map(c => ({label: human(c.kind), value: share(c.recovered ?? c.trials_with_warning, c.trials), text: `${int(c.recovered ?? c.trials_with_warning)} / ${int(c.trials)}`,
          tip: `${human(c.kind)}: ${int(c.trials_with_warning)} of ${int(c.trials)} trials warned, ${int(c.trials_with_context_supported)} context supported`})),
        hi: 1, format: v => pct(v, 0), labelW: 170,
        table: {head: ['Scene', 'Trials', 'Warned', 'Recovered', 'Context supported', 'Median |height error|'], rows: controls.map(c => [human(c.kind), c.trials, c.trials_with_warning, c.recovered ?? '--', c.trials_with_context_supported, ok(c.median_absolute_height_error_m) ? `${fmt(c.median_absolute_height_error_m, 2)} m` : '--'])}}]};
  }

  function viewT11(r) {
    const p = r.predictions || [];
    const groups = p.map(x => ({label: `scale ${x.scale}`, sub: `degree ${x.spatial_degree}`, values: [
      {value: share(x.correct_better_than_static, x.assessed), text: of(x.correct_better_than_static, x.assessed), tip: `Scale ${x.scale}, degree ${x.spatial_degree}: the fitted object beats a static scene in ${of(x.correct_better_than_static, x.assessed)} held-out frames`},
      {value: share(x.correct_better_than_wrong, x.wrong_assessed), text: of(x.correct_better_than_wrong, x.wrong_assessed), tip: `Scale ${x.scale}, degree ${x.spatial_degree}: the true Sun beats a Sun rotated 90° in ${of(x.correct_better_than_wrong, x.wrong_assessed)} held-out frames`}]}));
    return {kpis: [{label: 'Held-out trials', value: int(p.reduce((m, x) => m+(x.assessed || 0), 0))}],
      charts: [{type: 'columns', wide: true, title: 'Held-out frames predicted better', subtitle: 'Share of assessed trials; 50% is a coin toss',
        groups, series: [{name: 'Beats a static scene', color: C.s1}, {name: 'Beats a wrong Sun', color: C.s2}],
        legend: [{name: 'Beats a static scene', color: C.s1}, {name: 'Beats a wrong Sun', color: C.s2}], hi: 1, format: v => pct(v, 0), refs: [{value: .5, label: 'coin toss'}],
        table: {head: ['Scale', 'Degree', 'Assessed', 'Beats static', 'Beats wrong Sun'], rows: p.map(x => [x.scale, x.spatial_degree, x.assessed, of(x.correct_better_than_static, x.assessed), of(x.correct_better_than_wrong, x.wrong_assessed)])}}]};
  }

  function viewT12(r) {
    const rc = r.relief_consistency || {}, rb = r.rescaled_baseline || {}, st = r.structure || {}, ref = r.reference_noise_structure || {};
    const kpis = [
      {label: 'Measured noise', value: fmt(r.measured_pooled_sigma, 4), sub: `${fmt(r.ratio_to_assumed, 2)}× the assumed ${fmt(r.assumed_sigma, 2)} · ${int(r.patches)} patches`, state: r.ratio_to_assumed > 2 ? 'bad' : r.ratio_to_assumed > 1.3 ? 'warn' : 'good', stateText: r.ratio_to_assumed > 1.3 ? 'above the assumption' : 'as assumed'},
      {label: 'After relief correction', value: fmt(r.relief_corrected_sigma ?? rc.relief_corrected_sigma, 4), sub: `${fmt(share(r.relief_corrected_sigma ?? rc.relief_corrected_sigma, r.assumed_sigma), 2)}× the assumed noise`},
      {label: 'Spatial structure', value: `lag-1 ${fmt(st.lag1_correlation, 2)}`, sub: `white-noise reference ${fmt(ref.lag1_correlation, 2)}`, state: st.lag1_correlation > 2*(ref.lag1_correlation || 0) ? 'warn' : undefined, stateText: 'structured, not white'},
      {label: 'Follows the Sun', value: pct(rc.explained_true_geometry), sub: `true geometry; shuffled median ${pct(rc.explained_shuffled_median)}`, state: rc.fraction_shuffled_at_or_above_true < .01 ? 'good' : 'warn', stateText: rc.fraction_shuffled_at_or_above_true < .01 ? 'relief, not noise' : 'not clearly relief'},
      {label: 'Warnings, rescaled', value: `${pct(rb.fraction_above_threshold_assumed, 0)} → ${pct(rb.fraction_above_threshold_rescaled, 0)}`, sub: 'at the assumed noise → at the measured noise (upper bound)'},
    ];
    const frames = r.per_frame || [];
    const passes = r.control_passes || [];
    const scenarios = passes[0]?.scenarios || [];
    const passName = {render_assumed_model_assumed: 'Assumed', render_measured_model_assumed: 'Measured, fit assumed', render_measured_model_measured: 'Measured', render_relief_corrected_model_relief_corrected: 'Relief corrected'};
    const scenarioName = s => s.kind === 'caster' ? `${s.height_m} m caster` : s.kind === 'structured_null' ? 'Changing background' : human(s.kind)[0].toUpperCase()+human(s.kind).slice(1);
    const charts = [
      {type: 'hbars', title: 'Noise in each frame', subtitle: 'Residual scale on flat patches; the dashed lines are the assumed and the pooled noise',
       rows: frames.map(f => ({label: `${f.frame+1} · ${String(f.pid).replace('nac.', '')}`, value: f.sigma_plane, text: fmt(f.sigma_plane, 3),
         tip: `Frame ${f.frame+1} (${f.pid})\nNoise ${fmt(f.sigma_plane, 4)}, ${fmt(f.ratio_to_assumed, 1)}× the assumed\nSun ${fmt(f.azimuth_deg, 1)}°, ${fmt(f.elevation_deg, 2)}° high\nT1 signed contribution ${fmt(f.t1_signed_median_contribution, 1)}`})),
       refs: [{value: r.assumed_sigma, label: 'assumed'}, {value: r.measured_pooled_sigma, label: 'pooled'}], format: v => fmt(v, 2), labelW: 190,
       note: ok(r.frame_scale_vs_evidence?.spearman_sigma_vs_signed_median) ? `Noisier frames carry more of the warnings: Spearman ${fmt(r.frame_scale_vs_evidence.spearman_sigma_vs_signed_median, 2)} between frame noise and T1 contribution.` : undefined,
       table: {head: ['Frame', 'Noise (plane)', 'Noise (quadratic)', '× assumed', 'T1 contribution'], rows: frames.map(f => [f.pid, fmt(f.sigma_plane, 4), fmt(f.sigma_quadratic, 4), fmt(f.ratio_to_assumed, 2), fmt(f.t1_signed_median_contribution, 1)])}},
      {type: 'ranges', title: 'Is the excess noise relief?', subtitle: `Variance a shading model explains with the true Sun (dot) against ${int(rc.shuffled_assignments)} shuffled Sun assignments (band: median to 95th percentile, whisker: highest)`,
       rows: [{label: 'Shading model', lo: rc.explained_shuffled_median, hi: rc.explained_shuffled_p95, whisker: rc.explained_shuffled_max, value: rc.explained_true_geometry,
         tip: `True geometry explains ${pct(rc.explained_true_geometry)}\nShuffled: median ${pct(rc.explained_shuffled_median)}, 95th percentile ${pct(rc.explained_shuffled_p95)}, highest ${pct(rc.explained_shuffled_max)}\n${pct(rc.fraction_shuffled_at_or_above_true, 3)} of assignments reach the true value`}],
       refs: ok(rc.explained_best_rank2) ? [{value: rc.explained_best_rank2, label: 'best rank-2'}] : [], hi: 1,
       table: {head: ['Quantity', 'Value'], rows: [['True geometry', pct(rc.explained_true_geometry)], ['Shuffled median', pct(rc.explained_shuffled_median)], ['Shuffled 95th percentile', pct(rc.explained_shuffled_p95)], ['Shuffled highest', pct(rc.explained_shuffled_max)], ['Best rank-2 model', pct(rc.explained_best_rank2)], ['Indicative slope, median', `${fmt(rc.indicative_slope_deg?.median, 2)}°`]]}},
    ];
    if (passes.length && scenarios.length) {
      const cells = scenarios.map((sc, i) => passes.map(p => {
        const x = p.scenarios[i] || {}, found = sc.kind === 'caster', hits = found ? x.recovered_within_2px : x.trials_with_warning_roots;
        return {text: of(hits, x.assessed), sub: found ? 'found' : 'warned', t: share(hits, x.assessed),
          tip: `${scenarioName(sc)}, ${passName[p.name] || human(p.name)} noise\n${of(hits, x.assessed)} trials ${found ? 'found within 2 px' : 'with a warning'}\nMedian highest score ${fmt(x.median_maximum_score, 1)}`};
      }));
      charts.push({type: 'matrix', wide: true, title: 'Controls rendered and fitted at each noise level', subtitle: 'Casters should be found; the other scenes should stay quiet. Lighter is more trials.',
        rows: scenarios.map(scenarioName), cols: passes.map(p => passName[p.name] || human(p.name)), cells, colTitle: 'Noise used to render and to fit', labelW: 150,
        table: {head: ['Scene', ...passes.map(p => passName[p.name] || human(p.name))], rows: scenarios.map((sc, i) => [scenarioName(sc), ...cells[i].map(c => `${c.text} ${c.sub}`)])}});
    }
    return {kpis, charts};
  }

  function viewT13(r) {
    const conf = r.evaluation_confusion || {}, a = r.athena_summary || {}, td = r.touchdown_cell || {}, margins = r.margins || {};
    const truth = [['rock', 'Rock'], ['relief', 'Relief'], ['none', 'Blank'], ['stripes', 'Stripes']].filter(([k]) => conf[k]);
    const rowTotal = k => CLASSES.reduce((m, [c]) => m+(conf[k]?.[c] || 0), 0);
    const call = (k, labels) => labels.reduce((m, c) => m+(conf[k]?.[c] || 0), 0);
    const targets = r.targets || {};
    const checks = [
      ['Rocks called relief', call('rock', ['relief_like']), rowTotal('rock'), targets.rock_called_relief],
      ['Relief called rock', call('relief', ['rock_like']), rowTotal('relief'), targets.relief_called_rock],
      // As in the calibration: a blank given any label but none counts as called a signal.
      ['Blanks called a signal', call('none', ['rock_like', 'relief_like', 'ambiguous']), rowTotal('none'), targets.blank_called_signal],
      ['Stripes called relief', call('stripes', ['relief_like']), rowTotal('stripes'), targets.stripes_called_relief]].filter(c => c[2]);
    const all = a.all_sampled || {}, near = a.near_touchdown || {};
    const kpis = [
      {label: 'Real cells classified', value: int(all.cells), sub: `${pct(all.rock_like, 0)} rock-like · ${pct(all.relief_like, 0)} relief-like · ${pct(all.ambiguous, 0)} ambiguous`},
      {label: 'Within 22 m of the touchdown', value: int(near.cells), sub: `${pct(near.rock_like, 0)} rock-like · ${pct(near.relief_like, 0)} relief-like`},
      {label: 'Touchdown cell', value: (CLASSES.find(c => c[0] === td.label) || [null, human(td.label)])[1], sub: td.relief_sign ? `${human(td.relief_sign)} · relief gain ${fmt(td.gain_relief, 2)} vs rock ${fmt(td.gain_rock, 2)}` : undefined},
      ...checks.map(([name, k, n, target]) => ({label: name, value: of(k, n), sub: ok(target) ? `target at most ${pct(target, 0)}` : undefined,
        state: ok(target) ? (share(k, n) <= target ? 'good' : 'bad') : undefined, stateText: ok(target) ? (share(k, n) <= target ? 'within target' : 'misses target') : undefined})),
    ];
    const cells = truth.map(([k, name]) => CLASSES.map(([c, cname]) => {
      const n = conf[k]?.[c] || 0, total = rowTotal(k);
      return {text: int(n), sub: pct(share(n, total), 0), t: share(n, total), tip: `${name} scenes labelled ${cname.toLowerCase()}: ${int(n)} of ${int(total)} (${pct(share(n, total))})`};
    }));
    const rows = [['all_sampled', 'All sampled cells'], ['baseline_warning', 'Cells warning before'], ['near_touchdown', 'Within 22 m of the touchdown']]
      .filter(([k]) => a[k]).map(([k, label]) => ({label, n: a[k].cells, parts: Object.fromEntries(CLASSES.map(([c]) => [c, a[k][c]]))}));
    const signs = rows.length ? [['all_sampled', 'All relief-like'], ['near_touchdown', 'Near the touchdown']].filter(([k]) => a[k]?.relief_signs)
      .map(([k, label]) => ({label, parts: a[k].relief_signs})) : [];
    const charts = [
      {type: 'matrix', title: 'Held-out simulated scenes: truth against label', subtitle: 'Each row is one kind of scene; lighter means a larger share of that row. The diagonal should be light.',
       rows: truth.map(t => t[1]), cols: CLASSES.map(c => c[1]), cells, colTitle: 'Label given', labelW: 80,
       table: {head: ['Truth', ...CLASSES.map(c => c[1])], rows: truth.map(([k, name]) => [name, ...CLASSES.map(([c]) => int(conf[k]?.[c] || 0))])}},
      {type: 'stack100', title: 'Labels on the real cells', subtitle: 'Share of cells in each class',
       rows, keys: CLASSES.map(([key, name, color]) => ({key, name, color})), legend: CLASSES.map(([, name, color]) => ({name, color})), labelW: 190,
       table: {head: ['Cells', 'Count', ...CLASSES.map(c => c[1])], rows: rows.map(x => [x.label, int(x.n), ...CLASSES.map(([c]) => pct(x.parts[c]))])}},
    ];
    if (signs.length) charts.push({type: 'stack100', title: 'Relief-like cells: raised or sunken?', subtitle: 'Protrusions put bright before dark along the Sun; depressions the reverse',
      rows: signs, keys: [{key: 'protrusion', name: 'Protrusion', color: C.s2}, {key: 'depression', name: 'Depression', color: C.s1}, {key: 'undetermined', name: 'Undetermined', color: C.other}],
      legend: [{name: 'Protrusion', color: C.s2}, {name: 'Depression', color: C.s1}, {name: 'Undetermined', color: C.other}], labelW: 190,
      table: {head: ['Cells', 'Protrusion', 'Depression', 'Undetermined'], rows: signs.map(x => [x.label, pct(x.parts.protrusion), pct(x.parts.depression), pct(x.parts.undetermined)])}});
    return {kpis, charts, note: `Margins calibrated at noise ${fmt(r.competition_sigma, 4)} (${human(r.competition_sigma_source)}): rock ${fmt(margins.rock, 3)}, relief ${fmt(margins.relief, 3)}, none ${fmt(margins.none, 4)}, Sun ${fmt(margins.sun, 3)}.`};
  }

  function viewT14(r, extras) {
    const sfs = r.sfs || {}, sl = r.slope_deg || {}, ex = r.exceedance || {}, rs = r.residual_scale_after || {}, sub = r.subpixel_casters || {};
    const rec = r.injection_recovery || {}, sizing = sub.injected_rock_sizing || {};
    const pop = r.planted_population || {}, real = r.real_rocks;
    const population = pop.geometry === 'population';
    const heights = Object.keys(rec);
    const holds = Object.entries(sizing).map(([h, v]) => ({h, ...v}));
    const body = k => k === 'procedural' ? 'procedural' : `NASA Apollo ${String(k).split('_')[0]}`;
    const total = (key, a, b) => Object.values(key).reduce((s, v) => [s[0]+(v?.[a] || 0), s[1]+(v?.[b] || 0)], [0, 0]);
    const found = total(Object.fromEntries(heights.map(h => [h, rec[h].corrected_measured])), 'recovered', 'quiet_sites');
    const bounded = total(sizing, 'lower_bound_holds', 'with_warning_evidence');
    const nearest = real?.nearest_at_or_above_clearance;
    const m = r.measurable, cal = r.bound_calibration, ecal = r.estimate_calibration;
    const span = v => Array.isArray(v) ? `${fmt(v[0], 2)} to ${fmt(v[1], 2)} m` : '--';
    const frames = r.noise_per_frame?.enabled ? r.noise_per_frame.per_frame_sigma : null;
    const kpis = [
      ...(m ? [{label: 'Measurable from', value: ok(m.measurable_from_m) ? `${fmt(m.measurable_from_m, 2)} m` : 'not yet established',
                sub: `${pct(m.detection_target, 0)} of planted rocks found and ${pct(m.coverage_target, 0)} of their height bounds holding, at ${pct(m.confidence, 0)} confidence`,
                state: ok(m.measurable_from_m) ? 'good' : 'warn', stateText: ok(m.measurable_from_m) ? 'established' : 'more planted rocks or less noise needed'}] : []),
      ...(real ? [{label: 'Real candidate rocks', value: int(real.objects),
                   sub: `${int(real.by_label?.rock_like)} rock-like · ${int(real.by_label?.ambiguous)} ambiguous · from ${int(real.cells)} sized cells`},
                  {label: `Real candidates at least ${fmt(sub.clearance_m, 1)} m`, value: int(real.at_or_above_clearance_calibrated ?? real.at_or_above_clearance),
                   sub: ok(real.at_or_above_clearance_calibrated)
                     ? `by calibrated bounds; ${int(real.at_or_above_clearance)} by fitted bounds${nearest ? ` · nearest ${fmt(nearest.distance_to_touchdown_m, 0)} m from the touchdown` : ''}`
                     : (nearest ? `nearest ${fmt(nearest.distance_to_touchdown_m, 0)} m from the touchdown` : 'none sized that tall')}] : []),
      ...(cal && cal.kind !== 'none' ? [{label: 'Height bound margin', value: ok(cal.margin) ? (cal.kind === 'ratio' ? `÷ ${fmt(cal.margin, 2)}` : `${fmt(cal.margin, 2)} m`) : 'too few planted rocks',
                   sub: `split conformal from ${int(cal.planted_bounds)} planted bounds, so ${pct(cal.coverage, 0)} of rocks like them hold`,
                   state: ok(cal.margin) ? 'good' : 'warn', stateText: ok(cal.margin) ? 'calibrated' : 'not calibrated'}] : []),
      ...(ecal && ecal.kind !== 'none' ? [{label: 'Height estimate interval', value: Array.isArray(ecal.factors) ? `×${fmt(ecal.factors[0], 2)} to ×${fmt(ecal.factors[1], 2)}` : 'too few planted estimates',
                   sub: `split conformal from ${int(ecal.planted_estimates)} planted estimates, so ${pct(ecal.coverage, 0)} of rocks like them fall inside`,
                   state: Array.isArray(ecal.factors) ? 'good' : 'warn', stateText: Array.isArray(ecal.factors) ? 'calibrated' : 'not calibrated'}] : []),
      {label: 'Planted calibration rocks', value: int(pop.planted ?? r.injected_sites),
       sub: population ? `${Object.entries(pop.shapes || {}).map(([k, n]) => `${int(n)} ${body(k)}`).join(' · ')} · height/diameter median ${fmt(pop.height_over_diameter_median, 2)}`
                       : (pop.body || 'fixed heights')},
      {label: 'Shading explained', value: pct(sfs.explained_fraction), sub: `frame-to-frame ratio RMS ${fmt(sfs.ratio_rms_before, 3)} → ${fmt(sfs.ratio_rms_after, 3)}`},
      {label: 'Metre-scale slopes', value: `${fmt(sl.median, 1)}° median`, sub: `90th percentile ${fmt(sl.p90, 1)}° · ${pct(sl.fraction_above_limit)} above the ${fmt(sl.limit, 0)}° limit`},
      {label: 'Noise after correction', value: fmt(rs.pooled_sigma, 4), sub: `${int(rs.patches)} patches · lag-1 ${fmt(rs.structure?.lag1_correlation, 2)}${frames ? ` · frames weighted by their own, ${fmt(Math.min(...frames), 3)} to ${fmt(Math.max(...frames), 3)}` : ''}`},
      {label: 'Warnings', value: `${pct(ex.before_assumed_sigma, 0)} → ${pct(frames ? ex.after_frame_noise : ex.after_measured_sigma, 0)}`, sub: frames ? `before correction → after it, each frame at its own noise (${pct(ex.after_measured_sigma, 0)} at the pooled, ${pct(ex.after_assumed_sigma, 0)} at the assumed)` : `before correction → after it, at the measured noise (${pct(ex.after_assumed_sigma, 0)} at the assumed)`},
      ...(population
        ? [{label: 'Planted rocks found', value: of(found[0], found[1]), sub: 'all heights, after correction, at the measured noise, on quiet sites'},
           ...(bounded[1] ? [{label: 'Height bound holds', value: of(bounded[0], bounded[1]), sub: 'every planted rock against its own true height',
                              state: share(...bounded) >= .9 ? 'good' : 'bad', stateText: share(...bounded) >= .9 ? 'bound holds' : 'bound overshoots'}] : [])]
        : [...heights.map(h => ({label: `Planted ${h} rocks found`, value: of(rec[h].corrected_measured?.recovered, rec[h].corrected_measured?.quiet_sites), sub: 'after correction, at the measured noise, on quiet sites'})),
           ...holds.filter(x => x.with_warning_evidence).map(x => ({label: `Height bound holds, ${x.h} rocks`, value: of(x.lower_bound_holds, x.with_warning_evidence),
             sub: `median bound ${fmt(x.lower_bound_median_m, 2)} m${ok(x.estimate_median_m) ? ` · estimate ${fmt(x.estimate_median_m, 2)} m` : ''}`,
             state: share(x.lower_bound_holds, x.with_warning_evidence) >= .9 ? 'good' : 'bad', stateText: share(x.lower_bound_holds, x.with_warning_evidence) >= .9 ? 'bound holds' : 'bound overshoots'}))]),
    ];
    const conditions = [['original_assumed', 'Original, assumed noise', C.other], ['corrected_assumed', 'Corrected, assumed noise', C.s1], ['corrected_measured', 'Corrected, measured noise', C.s2]];
    const charts = [
      {type: 'columns', title: 'Share of assessed cells warning', subtitle: 'Before and after the relief correction',
       groups: [{label: 'Before', sub: 'assumed noise', values: [{value: ex.before_assumed_sigma, text: pct(ex.before_assumed_sigma, 0), tip: `Before correction, assumed noise: ${pct(ex.before_assumed_sigma)} warn; median score ${fmt(ex.median_score_before, 1)}`}]},
                {label: 'After', sub: 'assumed noise', values: [{value: ex.after_assumed_sigma, text: pct(ex.after_assumed_sigma, 0), tip: `After correction, assumed noise: ${pct(ex.after_assumed_sigma)} warn; median score ${fmt(ex.median_score_after, 1)}`}]},
                {label: 'After', sub: 'measured noise', values: [{value: ex.after_measured_sigma, text: pct(ex.after_measured_sigma, 0), tip: `After correction, at the measured noise ${fmt(rs.pooled_sigma, 4)}: ${pct(ex.after_measured_sigma)} warn`}]}],
       series: [{name: 'Warning share', color: C.s1}], hi: 1, format: v => pct(v, 0),
       table: {head: ['Stage', 'Warning share'], rows: [['Before, assumed noise', pct(ex.before_assumed_sigma)], ['After, assumed noise', pct(ex.after_assumed_sigma)], ['After, measured noise', pct(ex.after_measured_sigma)]]}},
      {type: 'columns', title: population ? 'Calibration: planted rocks found within 2 px, by height' : 'Planted rocks found within 2 px',
       subtitle: population ? 'Rocks drawn at random heights, grouped; only sites whose background was quiet count' : 'Only sites whose background was quiet count',
       groups: heights.map(h => ({label: h, values: conditions.map(([k, name]) => {
         const v = rec[h][k] || {};
         return {value: share(v.recovered, v.quiet_sites), text: of(v.recovered, v.quiet_sites), tip: `${h} rocks, ${name.toLowerCase()}: ${of(v.recovered, v.quiet_sites)} found`};
       })})), series: conditions.map(([, name, color]) => ({name, color})), legend: conditions.map(([, name, color]) => ({name, color})), hi: 1, format: v => pct(v, 0),
       table: {head: ['Height', ...conditions.map(c => c[1])], rows: heights.map(h => [h, ...conditions.map(([k]) => of(rec[h][k]?.recovered, rec[h][k]?.quiet_sites))])}},
    ];
    const inj = Array.isArray(extras?.injection) ? extras.injection : [];
    if (population && inj.length) {
      const state = p => p.recovered_corrected_measured === true ? 'found' : p.recovered_corrected_measured === false ? 'missed' : 'busy';
      const nasa = p => p.shape && p.shape !== 'procedural';
      const said = {found: 'Found after the correction', missed: 'Missed after the correction', busy: 'Background already warning: not scored'};
      const points = inj.filter(p => ok(p.height_over_diameter)).map(p => ({x: p.height_m, y: p.height_over_diameter, key: state(p)+(nasa(p) ? '_nasa' : ''),
        tip: `Planted rock, site ${p.site}\n${fmt(p.height_m, 2)} m tall, ${fmt(p.width_m, 2)} × ${fmt(p.length_m, 2)} m across\nHeight / diameter ${fmt(p.height_over_diameter, 2)} · burial ${pct(p.burial, 0)} · yaw ${fmt(p.yaw_deg, 0)}°\nBody: ${body(p.shape)}\n${said[state(p)]}`}));
      const keys = [['found', 'Found', C.s3], ['missed', 'Missed', C.s2], ['busy', 'Not scored', C.other]]
        .flatMap(([key, name, color]) => [{key, name, color}, {key: key+'_nasa', name: `${name}, NASA body`, color, shape: 'triangle'}]);
      charts.push({type: 'scatter', title: 'Planted rocks: the bodies drawn', subtitle: 'Each rock has its own height, proportions, burial and yaw. Triangles are NASA Apollo bodies, circles procedural.',
        points, keys, identity: false, xHi: Math.max(...points.map(p => p.x)), yHi: 1, xLabel: 'true height (m)', yLabel: 'height / diameter',
        legend: [{name: 'Found', color: C.s3, shape: 'circle'}, {name: 'Missed', color: C.s2, shape: 'circle'}, {name: 'Not scored', color: C.other, shape: 'circle'}, {name: 'NASA Apollo body', color: C.other, shape: 'triangle'}],
        note: (pop.sources || []).join(' · '),
        table: {head: ['Site', 'Height', 'Across', 'Height / diameter', 'Burial', 'Body', 'After correction'],
                rows: inj.map(p => [p.site, `${fmt(p.height_m, 2)} m`, `${fmt(p.width_m, 2)} × ${fmt(p.length_m, 2)} m`, fmt(p.height_over_diameter, 2), pct(p.burial, 0), body(p.shape), said[state(p)]])}});
    }
    const sized = inj.filter(p => ok(p.sized_height_lower_bound_m) || ok(p.sized_height_m));
    if (sized.length) {
      const jitter = population ? (p => p.height_m) : (p, k) => p.height_m+((p.site*37+k*13)%9-4)*.012;
      const points = sized.flatMap(p => [
        ok(p.sized_height_lower_bound_m) && {x: jitter(p, 0), y: p.sized_height_lower_bound_m, key: 'bound', tip: `Planted ${fmt(p.height_m, 2)} m rock, site ${p.site}\nLower bound ${fmt(p.sized_height_lower_bound_m, 2)} m${p.sized_height_lower_bound_m > p.height_m+.05 ? ' (overshoots)' : ' (holds)'}\nState: ${human(p.sized_state)}`},
        ok(p.sized_height_m) && {x: jitter(p, 1), y: p.sized_height_m, key: 'estimate', tip: `Planted ${fmt(p.height_m, 2)} m rock, site ${p.site}\nHeight estimate ${fmt(p.sized_height_m, 2)} m${Array.isArray(p.sized_height_interval_m) ? `\nCalibrated interval ${span(p.sized_height_interval_m)} (${p.sized_height_interval_m[0] <= p.height_m && p.height_m <= p.sized_height_interval_m[1] ? 'holds the truth' : 'misses the truth'})` : ''}`}].filter(Boolean));
      charts.push({type: 'scatter', title: 'Planted rocks: measured height against true height', subtitle: 'A lower bound above the line overshoots the truth',
        points, keys: [{key: 'bound', name: 'Lower bound', color: C.s1}, {key: 'estimate', name: 'Estimate', color: C.s2, shape: 'triangle'}],
        legend: [{name: 'Lower bound', color: C.s1, shape: 'circle'}, {name: 'Estimate', color: C.s2, shape: 'triangle'}],
        xLabel: 'true height (m)', yLabel: 'measured (m)', hi: Math.max(1.6, ...points.map(p => p.y)),
        table: {head: ['Site', 'True', 'Lower bound', 'Estimate', 'Calibrated interval', 'State'], rows: sized.map(p => [p.site, `${fmt(p.height_m, 2)} m`, ok(p.sized_height_lower_bound_m) ? `${fmt(p.sized_height_lower_bound_m, 2)} m` : '--', ok(p.sized_height_m) ? `${fmt(p.sized_height_m, 2)} m` : '--', span(p.sized_height_interval_m), human(p.sized_state)])}});
    }
    const rocks = Array.isArray(extras?.real_rocks) ? extras.real_rocks : [];
    if (real && rocks.length) {
      const px = real.pixel_m || 1, td = real.touchdown_px, size = real.image_shape;
      const height = o => o.height_m ?? o.height_lower_bound_m;
      const describe = o => ok(o.height_m) ? `${fmt(o.height_m, 2)} m (context-supported${Array.isArray(o.height_interval_m) ? `; ${pct(ecal?.coverage, 0)} interval ${span(o.height_interval_m)}` : ''})`
        : ok(o.height_lower_bound_calibrated_m) ? `at least ${fmt(o.height_lower_bound_calibrated_m, 2)} m (calibrated; fitted ${fmt(o.height_lower_bound_m, 2)} m)`
        : ok(o.height_lower_bound_m) ? `at least ${fmt(o.height_lower_bound_m, 2)} m` : 'not sized';
      const calib = o => o.calibration ? `${of(o.calibration.found, o.calibration.of)} of planted ${o.calibration.group} rocks found` : 'below the planted range';
      const points = rocks.map(o => ({x: o.col_px*px, y: o.row_px*px, key: o.label, r: 2.2+3.3*Math.min(1, (height(o) || 0)/1.5),
        tip: `Candidate ${o.object}: ${human(o.label)}, ${int(o.cells)} cell${o.cells === 1 ? '' : 's'}\nHeight ${describe(o)}\n${fmt(o.distance_to_touchdown_m, 0)} m from the touchdown\nCalibration: ${calib(o)}`}));
      if (td) points.push({x: td[1]*px, y: td[0]*px, key: 'touchdown', r: 6, tip: 'Published touchdown point'});
      const keys = [{key: 'rock_like', name: 'Rock-like', color: C.s2}, {key: 'ambiguous', name: 'Ambiguous', color: C.s1},
                    {key: 'unchecked', name: 'Not checked for relief', color: C.other}, {key: 'touchdown', name: 'Touchdown', color: '#ffffff', shape: 'cross'}];
      const top = [...rocks].filter(o => ok(height(o))).sort((a, b) => height(b)-height(a)).slice(0, 30);
      charts.push({type: 'scatter', title: 'Real rocks HATI found', wide: true,
        subtitle: `${int(real.objects)} candidate objects (touching warning cells merged) after the relief correction${real.examined_share < 1 ? `; ${pct(real.examined_share, 0)} of warning cells examined` : ''}. Larger dots are taller.`,
        points, keys, identity: false, square: true, yDown: true, xHi: size ? size[1]*px : undefined, yHi: size ? size[0]*px : undefined,
        xLabel: 'image column (m)', yLabel: 'image row (m)', xTick: v => fmt(v, 0), yTick: v => fmt(v, 0),
        legend: keys.map(k => ({name: k.name, color: k.color, shape: k.shape || 'circle'})),
        note: real.note,
        table: {head: ['Candidate', 'Label', 'Height', 'Cells', 'From touchdown', 'Calibration'],
                rows: top.map(o => [o.object, human(o.label), describe(o), int(o.cells), `${fmt(o.distance_to_touchdown_m, 0)} m`, calib(o)])}});
      const groupRows = Object.entries(real.by_height_group || {}).map(([g, n]) => {
        const cal = rec[g]?.corrected_measured;
        return {label: g, value: n, text: int(n), tip: `${int(n)} real candidates of ${g}\nPlanted rocks of that height found: ${of(cal?.recovered, cal?.quiet_sites)}`};
      });
      if (real.below_calibrated_range) groupRows.push({label: 'below the planted range', value: real.below_calibrated_range, text: int(real.below_calibrated_range),
        tip: `${int(real.below_calibrated_range)} candidates shorter than any planted rock: no calibration`});
      charts.push({type: 'hbars', title: 'Real candidates by height', subtitle: 'Context-supported estimate where available, otherwise the lower bound',
        rows: groupRows, format: v => int(v), labelW: 170,
        table: {head: ['Height', 'Real candidates', 'Planted rocks found at that height'],
                rows: Object.entries(real.by_height_group || {}).map(([g, n]) => [g, int(n), of(rec[g]?.corrected_measured?.recovered, rec[g]?.corrected_measured?.quiet_sites)])}});
    }
    const bySlope = r.residual_by_slope;
    if (bySlope?.after) {
      const classes = Object.keys(bySlope.after);
      const deg = x => Math.round(Math.atan(Number(x))*180/Math.PI);
      const label = k => { const [lo, hi] = k.split('-'); return `${deg(lo)} to ${deg(hi)}°`; };
      const prior = sfs.dem_prior ? 'DEM shading subtracted first, then shape from shading' : 'shape from shading';
      charts.push({type: 'columns', title: 'Noise left by ground slope', subtitle: `Residual of the detector's null in 24 px patches, by DEM slope; after the correction (${prior})`,
        groups: classes.map(k => ({label: label(k), sub: `${int(bySlope.after[k]?.patches)} patches`, values: [['before', 'Before'], ['after', 'After']].map(([w, name]) => ({
          value: bySlope[w]?.[k]?.pooled_sigma, text: fmt(bySlope[w]?.[k]?.pooled_sigma, 3),
          tip: `${label(k)}, ${name.toLowerCase()} the correction: ${fmt(bySlope[w]?.[k]?.pooled_sigma, 4)} over ${int(bySlope[w]?.[k]?.patches)} patches`}))})),
        series: [{name: 'Before', color: C.other}, {name: 'After', color: C.s1}], legend: [{name: 'Before the correction', color: C.other}, {name: 'After it', color: C.s1}],
        format: v => fmt(v, 2),
        table: {head: ['DEM slope', 'Patches', 'Before', 'After'], rows: classes.map(k => [label(k), int(bySlope.after[k]?.patches), fmt(bySlope.before?.[k]?.pooled_sigma, 4), fmt(bySlope.after[k]?.pooled_sigma, 4)])}});
    }
    const pf = rs.per_frame_sigma || [];
    if (pf.length) charts.push({type: 'hbars', title: 'Noise in each frame after the correction', subtitle: 'The dashed line is the pooled value the detector uses',
      rows: pf.map((v, i) => ({label: `Frame ${i+1}`, value: v, text: fmt(v, 3), tip: `Frame ${i+1}: residual scale ${fmt(v, 4)} after the relief correction`})),
      refs: [{value: rs.pooled_sigma, label: 'pooled'}], format: v => fmt(v, 2), labelW: 90,
      table: {head: ['Frame', 'Noise after correction'], rows: pf.map((v, i) => [i+1, fmt(v, 4)])}});
    if (sub.sized_cells) {
      const hb = sub.height_lower_bound_m || {};
      charts.push({type: 'hbars', title: 'Every detection sized', subtitle: `${int(sub.candidate_cells)} warning cells after the correction, checked for relief, then sized`,
        rows: [['Checked', sub.examined_cells], ['Sized (rock-like or ambiguous)', sub.sized_cells], ['With warning evidence', sub.with_warning_evidence], ['Lower bound at least ' + fmt(sub.clearance_m, 1) + ' m', sub.exceeding_clearance], ['Height estimated', sub.context_supported]]
          .map(([label, v]) => ({label, value: v, text: int(v), tip: `${label}: ${int(v)} cells`})), format: v => int(v), labelW: 200,
        note: `Height lower bounds: 10th percentile ${fmt(hb.p10, 2)} m, median ${fmt(hb.median, 2)} m, 90th ${fmt(hb.p90, 2)} m. Cells are 3.6 m warning cells, not rock counts.`,
        table: {head: ['Step', 'Cells'], rows: [['Candidates', int(sub.candidate_cells)], ['Checked', int(sub.examined_cells)], ['Sized', int(sub.sized_cells)], ['With warning evidence', int(sub.with_warning_evidence)], ['Bound at or above clearance', int(sub.exceeding_clearance)], ['Height estimated', int(sub.context_supported)]]}});
    }
    if (m) {
      const interval = v => v ? `${pct(v[0], 0)} to ${pct(v[1], 0)}` : 'no rocks';
      const rows = Object.entries(m.groups || {}).flatMap(([g, v]) => [
        {label: `${g}: found`, lo: v.found_interval?.[0], hi: v.found_interval?.[1], value: share(v.found, v.quiet_sites),
         tip: `${g}: ${of(v.found, v.quiet_sites)} planted rocks found on quiet sites\n${pct(m.confidence, 0)} interval ${interval(v.found_interval)}`},
        {label: `${g}: bounds hold`, lo: v.holds_interval?.[0], hi: v.holds_interval?.[1], value: share(v.bound_holds, v.bounded),
         tip: `${g}: height lower bound holds for ${of(v.bound_holds, v.bounded)}\n${pct(m.confidence, 0)} interval ${interval(v.holds_interval)}${v.established ? '\nEstablished' : ''}`}]);
      const targets = [...new Set([m.detection_target, m.coverage_target])];
      charts.unshift({type: 'ranges', wide: true, title: 'What this run can vouch for',
        subtitle: `Share of planted rocks found and share of height bounds that hold (${m.bounds_judged || 'as fitted'}), with ${pct(m.confidence, 0)} intervals. A height is established when both intervals clear the dashed target.`,
        rows, refs: targets.map(t => ({value: t, label: `target ${pct(t, 0)}`})), labelW: 220, width: 1000, note: m.note,
        table: {head: ['Height', 'Found', `${pct(m.confidence, 0)} interval`, 'Bounds hold', `${pct(m.confidence, 0)} interval`, 'Estimates', 'Median error', 'Estimate intervals hold', 'Established'],
                rows: Object.entries(m.groups || {}).map(([g, v]) => [g, of(v.found, v.quiet_sites), interval(v.found_interval), of(v.bound_holds, v.bounded),
                  interval(v.holds_interval), int(v.estimates), ok(v.median_error_m) ? `${fmt(v.median_error_m, 2)} m` : '--',
                  v.estimate_intervals ? of(v.estimate_interval_holds, v.estimate_intervals) : '--', v.established ? 'yes' : 'no'])}});
    }
    return {kpis, charts};
  }

  function viewT16(r) {
    const rows = (r.summaries || []).map(s => {
      const rock = s.kind === 'procedural';
      const noise = s.noise_pass === 'render_measured' ? 'measured noise' : 'assumed noise';
      // The saved fraction is set for the nulls the gate applies to; the counts hold for every scene.
      const value = s.context_supported_trial_fraction ?? share(s.trials_with_context_supported, s.trials);
      const passed = rock ? value >= .5 : s.within_declared_gate;
      return {label: `${human(s.kind)}${ok(s.height_m) ? ` ${s.height_m} m` : ''} · ${noise}`, value,
        text: `${int(s.trials_with_context_supported)} / ${int(s.trials)} ${passed ? '✓' : '✕'}`,
        tip: `${human(s.kind)}${ok(s.height_m) ? ` ${s.height_m} m` : ''}, ${noise}\n${int(s.trials_with_context_supported)} of ${int(s.trials)} trials pass the gate (context supported)\n${rock ? 'A real caster: passing is right.' : `A null: at most ${pct(s.declared_gate_max_fraction, 0)} may pass. ${s.within_declared_gate ? 'Within' : 'Outside'} the gate.`}`,
        color: rock ? C.s3 : C.s1};
    });
    return {kpis: [{label: 'Null scenes within the gate', value: of(r.null_scenarios_within_gate, r.null_scenarios), state: r.null_scenarios_within_gate === r.null_scenarios ? 'good' : 'bad', stateText: r.null_scenarios_within_gate === r.null_scenarios ? 'all within' : 'some fail'},
                   {label: 'Gate', value: `at most ${pct(r.declared_gate_max_fraction, 0)}`, sub: 'of null trials may reach a supported size'}],
      charts: [{type: 'hbars', wide: true, title: 'Trials that reach a supported size', subtitle: 'Nulls (blue) must stay under the dashed gate; casters (aqua) should pass. ✓ and ✕ mark the verdict.',
        rows, refs: [{value: r.declared_gate_max_fraction, label: `gate ${pct(r.declared_gate_max_fraction, 0)}`}], hi: 1, format: v => pct(v, 0), labelW: 250,
        legend: [{name: 'Null scene', color: C.s1}, {name: 'Real caster', color: C.s3}],
        table: {head: ['Scene', 'Passing trials', 'Share', 'Verdict'], rows: (r.summaries || []).map(s => [`${human(s.kind)}${ok(s.height_m) ? ` ${s.height_m} m` : ''} · ${human(s.noise_pass)}`, of(s.trials_with_context_supported, s.trials), pct(share(s.trials_with_context_supported, s.trials)), s.kind === 'procedural' ? 'caster' : s.within_declared_gate ? 'within gate' : 'outside gate'])}}]};
  }

  function viewGeneric(r) {
    const skip = new Set(['test', 'status', 'reason', 'input_demo', 'interpretation', 'limitations', 'limitation']);
    const kpis = [];
    const walk = (obj, prefix) => {
      for (const [k, v] of Object.entries(obj || {})) {
        if (skip.has(k) || kpis.length >= 12) continue;
        if (ok(v)) kpis.push({label: human(prefix ? `${prefix} ${k}` : k), value: Math.abs(v) < 1 && v !== 0 ? fmt(v, 3) : int(v)});
        else if (v && typeof v === 'object' && !Array.isArray(v) && !prefix) walk(v, k);
      }
    };
    walk(r, '');
    return {kpis, charts: []};
  }

  const VIEWS = {maps: viewMaps, T1: viewT1, T2: viewT2, T3: viewT3, T5: viewT5, T9: viewT9, T10: viewT10, T11: viewT11,
                 T12: viewT12, T13: viewT13, T14: viewT14, T16: viewT16};

  /* While a stage runs: what it is doing now, from its live snapshot. */
  function liveView(snap) {
    const kpis = [];
    if (!snap) return {kpis: [{label: 'Now', value: 'Waiting for the first snapshot', wide: true}], charts: []};
    const age = snap.updated ? Math.max(0, (Date.now()-Date.parse(snap.updated))/1000) : null;
    kpis.push({label: 'Now', value: snap.message || human(snap.kind), sub: [snap.subrun, ok(age) ? `snapshot ${Math.round(age)} s ago` : null].filter(Boolean).join(' · '), wide: true});
    if (snap.solver) {
      const s = snap.solver, done = ((s.pass_index || 0)+(s.iterations ? s.iteration/s.iterations : 0))/Math.max(1, s.passes || 1);
      kpis.push({label: 'Shape-from-shading solve', value: `pass ${Math.min((s.pass_index || 0)+1, s.passes || 1)} of ${s.passes || 1}`, sub: `iteration ${int(s.iteration)} of ${int(s.iterations)}`, progress: done});
    }
    if (ok(snap.cells_total) && snap.cells_total > 0) kpis.push({label: 'Regional scan', value: of(snap.cells_visited, snap.cells_total), sub: `cells visited · ${int(snap.cells_assessed)} assessed`, progress: snap.cells_visited/snap.cells_total});
    if (snap.sizing) kpis.push({label: 'Rock sizing', value: of(snap.sizing.done, snap.sizing.total), sub: human(snap.sizing.phase), progress: share(snap.sizing.done, snap.sizing.total)});
    if (snap.metrics) for (const [k, v] of Object.entries(snap.metrics)) kpis.push({label: human(k)[0].toUpperCase()+human(k).slice(1), value: ok(v) ? (Math.abs(v) < 10 ? fmt(v, 4) : int(v)) : String(v)});
    if (snap.control) {
      const c = snap.control;
      kpis.push({label: 'Latest control', value: `${human(c.kind)}${ok(c.height_m) ? ` ${c.height_m} m` : ''}`,
        sub: [human(c.status), ok(c.maximum_score) ? `highest score ${fmt(c.maximum_score, 1)}` : null, c.recovered === true ? 'found' : c.recovered === false ? 'missed' : null].filter(Boolean).join(' · ')});
    }
    if (snap.adaptive) {
      const a = snap.adaptive;
      kpis.push({label: 'Adaptive pass', value: `${a.scale}× at row ${a.centre?.[0]}, col ${a.centre?.[1]}`, sub: `${human(a.status)} · heights ${a.height_range_m ? a.height_range_m.map(v => fmt(v, 1)).join('–')+' m' : 'not yet'}`});
    }
    if (snap.fit && !snap.adaptive) kpis.push({label: 'Last fit', value: `score ${fmt(snap.fit.score, 1)}`, sub: `row ${snap.fit.row_px}, col ${snap.fit.col_px} · ${fmt(snap.fit.height_m, 2)} m bank height`});
    return {kpis, charts: []};
  }

  /* ---- Rendering into the report panel ---- */
  const STATUS_CLASS = {COMPLETE: 'complete', PARTIAL: 'partial', RUNNING: 'live', FAILED: 'failed', BLOCKED: 'partial', STALE: 'stale', PENDING: 'quiet'};
  function render(els, {stageId, row, snapshot, report}) {
    const status = row?.status || 'PENDING';
    els.status.textContent = human(status).toLowerCase();
    els.status.className = 'badge '+(STATUS_CLASS[status] || '');
    els.title.textContent = !stageId ? 'Waiting for a stage' : stageId === 'maps' ? 'Three-map replay' : `${stageId} · ${row?.title || ''}`;
    els.question.textContent = QUESTIONS[stageId] || row?.reason || 'This stage reports its measurements when it finishes.';
    let view;
    const running = status === 'RUNNING';
    if (stageId?.startsWith('software-')) {
      // Software checks run before the science: a pass or a failure, and what it covered.
      const passed = status === 'COMPLETE', failed = status === 'FAILED';
      els.title.textContent = `Software check · ${human(stageId.slice(9))}`;
      els.question.textContent = `Automated tests (${row?.title || 'test file'}) that run before the science stages, so no result comes from broken code.`;
      view = {kpis: [{label: 'Result', value: passed ? 'Passed' : failed ? 'Failed' : human(status), sub: row?.reason,
                      state: passed ? 'good' : failed ? 'bad' : undefined, stateText: passed ? 'all checks passed' : 'see the log below'}], charts: []};
      els.eyebrow.textContent = '00 / STAGE REPORT';
    } else if (report && !running) {
      try {
        view = (VIEWS[stageId] || viewGeneric)(report.result, report.extras || {});
      } catch (error) {
        view = {kpis: [], charts: [], note: `This report could not be drawn: ${error.message}`};
      }
      els.eyebrow.textContent = '00 / STAGE REPORT · SAVED RESULT';
      view.note = [report.result?.reason, view.note].filter(Boolean).join(' ');
    } else if (running) {
      view = liveView(snapshot);
      els.eyebrow.textContent = '00 / STAGE REPORT · LIVE';
      view.note = 'The full report with every chart appears when the stage finishes.';
      if (report?.result?.interim) {
        // A run that saves its summary as it goes (the sizing bench, after every round) shows it under the live view.
        try {
          const saved = (VIEWS[stageId] || viewGeneric)(report.result, report.extras || {});
          view = {kpis: [...view.kpis, ...saved.kpis], charts: [...view.charts, ...saved.charts],
                  note: ['Live, with the summary saved so far.', report.result.reason].filter(Boolean).join(' ')};
        } catch (error) {
          view.note += ` The summary saved so far could not be drawn: ${error.message}`;
        }
      }
    } else {
      view = {kpis: [], charts: [], note: status === 'PENDING' ? 'Not started yet.' : 'No saved result for this stage.'};
      els.eyebrow.textContent = '00 / STAGE REPORT';
    }
    els.kpis.replaceChildren(...view.kpis.map(kpi));
    els.charts.replaceChildren(...view.charts.map(chart));
    els.note.textContent = view.note || '';
  }

  /* One tooltip for every chart mark: hover, or focus with the keyboard. */
  function bindTips(panel, box) {
    const show = (target, x, y) => {
      box.textContent = target.dataset.tip;
      box.hidden = false;
      const rect = panel.getBoundingClientRect();
      box.style.left = `${Math.min(x-rect.left+14, rect.width-box.offsetWidth-8)}px`;
      box.style.top = `${Math.min(y-rect.top+14, rect.height-box.offsetHeight-8)}px`;
    };
    panel.addEventListener('pointermove', e => {
      const target = e.target.closest?.('[data-tip]');
      if (target && panel.contains(target)) show(target, e.clientX, e.clientY);
      else box.hidden = true;
    });
    panel.addEventListener('pointerleave', () => { box.hidden = true; });
    panel.addEventListener('focusin', e => {
      const target = e.target.closest?.('[data-tip]');
      if (!target) return;
      const r = target.getBoundingClientRect();
      show(target, r.left+r.width/2, r.top+r.height/2);
    });
    panel.addEventListener('focusout', () => { box.hidden = true; });
  }

  return {render, bindTips};
})();
