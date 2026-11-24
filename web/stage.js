// 객석 렌더러: Canvas 2D. 좌석 1,500개와 사람 600명을 DOM 없이 그린다.
(function () {
  'use strict';
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v, lerp = (a, b, t) => a + (b - a) * t;
  const easeIO = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const easeOut = t => 1 - Math.pow(1 - t, 3);
  const BG = '#0a0a0a';
  const FONT = '"Pretendard Variable", Pretendard, -apple-system, sans-serif';
  const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const mix = (a, b, t) => { const A = hex(a), B = hex(b); return `rgb(${Math.round(lerp(B[0], A[0], t))},${Math.round(lerp(B[1], A[1], t))},${Math.round(lerp(B[2], A[2], t))})`; };
  const CAMS = {
    horseshoe: { x: 0, y: 1.65, z: -4.5, yaw: 0, pitch: 0.06, fov: 66 },
    fan: { x: 0, y: 1.65, z: -4.5, yaw: 0, pitch: 0.16, fov: 66 },
    cylinder: { x: 0, y: 1.65, z: -3.5, yaw: 0, pitch: 0.22, fov: 80 }
  };

  // 줄(row) 정의: 중심(cx,cz)에서 반지름 r, 각도 a0~a1(+z 방향이 0). sections = 통로로 나뉜 구역 수
  function rowsFor(kind) {
    const R = [];
    if (kind === 'fan') {
      for (let i = 0; i < 30; i++) R.push({ cx: 0, cz: -1, r: 4 + i * 0.8, y: -1.2 + i * 0.5, a0: -0.95, a1: 0.95, tier: 0, g: i, sec: 4 });
    } else if (kind === 'cylinder') {
      for (let t = 0; t < 7; t++) for (let j = 0; j < 2; j++) R.push({ cx: 0, cz: -3.5, r: 8.5 + j * 0.6, y: -1.6 + t * 2.1 + j * 0.35, a0: -2.55, a1: 2.55, tier: t + 1, g: t * 2 + j, sec: 8, front: j === 0 });
    } else {
      for (let i = 0; i < 11; i++) R.push({ cx: 0, cz: -1, r: 4 + i * 0.95, y: -0.85 + i * 0.16, a0: -0.8, a1: 0.8, tier: 0, g: i, sec: 3 });
      for (let t = 1; t <= 4; t++) for (let j = 0; j < 3; j++) R.push({ cx: 0, cz: 6.5, r: 10.8 + t * 0.6 + j * 0.85, y: 1.6 + (t - 1) * 2.8 + j * 0.42, a0: -2.0, a1: 2.0, tier: t, g: 11 + (t - 1) * 3 + j, sec: 7, front: j === 0 });
    }
    return R;
  }
  function buildSeats(kind, total) {
    const rows = rowsFor(kind), L = rows.map(r => r.r * (r.a1 - r.a0)), sum = L.reduce((a, b) => a + b, 0);
    const counts = L.map(l => Math.max(4, Math.round(l / sum * total)));
    let diff = total - counts.reduce((a, b) => a + b, 0);
    const byLen = rows.map((r, i) => i).sort((a, b) => L[b] - L[a]);
    for (let k = 0; diff !== 0; k++) { const i = byLen[k % byLen.length]; counts[i] += diff > 0 ? 1 : -1; diff += diff > 0 ? -1 : 1; }
    const seats = [], rn = window.MM ? window.MM.mulberry(9) : Math.random;
    rows.forEach((rw, ri) => {
      const n = counts[ri], span = rw.a1 - rw.a0, S = rw.sec, gap = 0.035;
      for (let k = 0; k < n; k++) {
        const u = (k + 0.5) / n, idx = Math.min(S - 1, Math.floor(u * S)), loc = u * S - idx;
        const a = rw.a0 + span * (idx / S + gap / 2 + loc * (1 / S - gap));
        seats.push({ x: rw.cx + rw.r * Math.sin(a), y: rw.y, z: rw.cz + rw.r * Math.cos(a), a, tier: rw.tier, g: rw.g, key: rw.g + Math.abs(a) * 1.4 + rn() * 0.4 });
      }
    });
    const order = seats.map((s, i) => i).sort((a, b) => seats[a].key - seats[b].key);
    return { seats, rows, order };
  }
  function makeProj(cam, W, H) {
    const f = (H / 2) / Math.tan(cam.fov * Math.PI / 360);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const out = { x: 0, y: 0, z: 0, v: false, f };
    out.p = (x, y, z) => {
      const dx = x - cam.x, dy = y - cam.y, dz = z - cam.z;
      const x1 = cy * dx - sy * dz, z1 = sy * dx + cy * dz, y1 = cp * dy - sp * z1, z2 = sp * dy + cp * z1;
      if (z2 < 0.25) { out.v = false; return out; }
      out.x = W / 2 + f * x1 / z2; out.y = H / 2 - f * y1 / z2; out.z = z2; out.v = true; return out;
    };
    return out;
  }
  function shapePath(ctx, shape, x, y, r) {
    if (shape === 'square') { const s = r * 0.9; ctx.rect(x - s, y - s, s * 2, s * 2); }
    else if (shape === 'diamond') { const s = r * 1.25; ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); }
    else if (shape === 'triangle') { const s = r * 1.3; ctx.moveTo(x, y - s); ctx.lineTo(x + s * 0.9, y + s * 0.6); ctx.lineTo(x - s * 0.9, y + s * 0.6); ctx.closePath(); }
    else { ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, Math.PI * 2); }
  }
  // 무대 바닥·난간·빈 좌석(어둠과 빛만으로)
  function drawHouse(ctx, W, H, P, rows, seats, alpha, occ, kind) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha = alpha;
    const lip = [P.p(-9, 0, 0)].map(o => ({ x: o.x, y: o.y, v: o.v }))[0];
    const lip2 = (() => { const o = P.p(9, 0, 0); return { x: o.x, y: o.y, v: o.v }; })();
    const mid = (() => { const o = P.p(0, 0, 0); return { x: o.x, y: o.y, v: o.v }; })();
    if (mid.v) {
      const g = ctx.createLinearGradient(0, mid.y, 0, H);
      g.addColorStop(0, '#1a1918'); g.addColorStop(1, '#0d0d0d');
      ctx.fillStyle = g; ctx.fillRect(0, Math.max(0, mid.y), W, H - mid.y);
      const pool = ctx.createRadialGradient(W / 2, H * 1.05, 10, W / 2, H * 1.05, W * 0.6);
      pool.addColorStop(0, 'rgba(255,248,236,0.10)'); pool.addColorStop(1, 'rgba(255,248,236,0)');
      ctx.fillStyle = pool; ctx.fillRect(0, mid.y, W, H - mid.y);
      const spill = ctx.createRadialGradient(W / 2, mid.y, 4, W / 2, mid.y, W * 0.5);
      spill.addColorStop(0, 'rgba(255,248,236,0.045)'); spill.addColorStop(1, 'rgba(255,248,236,0)');
      ctx.fillStyle = spill; ctx.fillRect(0, 0, W, mid.y);
      if (lip.v && lip2.v) { ctx.strokeStyle = 'rgba(255,255,255,0.22)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(lip.x, lip.y); ctx.lineTo(lip2.x, lip2.y); ctx.stroke(); }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = 1;
    rows.forEach(rw => {
      if (!rw.front) return;
      ctx.beginPath(); let pen = false;
      for (let k = 0; k <= 48; k++) {
        const a = rw.a0 - 0.03 + (rw.a1 - rw.a0 + 0.06) * k / 48, o = P.p(rw.cx + (rw.r - 0.45) * Math.sin(a), rw.y - 0.55, rw.cz + (rw.r - 0.45) * Math.cos(a));
        if (!o.v) { pen = false; continue; }
        if (pen) ctx.lineTo(o.x, o.y); else { ctx.moveTo(o.x, o.y); pen = true; }
      }
      ctx.stroke();
    });
    const lv = ['#3a3936', '#2b2a29', '#1f1f1e'], paths = [[], [], []];
    for (let i = 0; i < seats.length; i++) {
      if (occ && occ[i]) continue;
      const s = seats[i], o = P.p(s.x, s.y, s.z); if (!o.v) continue;
      const r = clamp(P.f * 0.075 / o.z, 0.6, 7), d = s.z + s.y * 0.6;
      paths[d < 9 ? 0 : d < 17 ? 1 : 2].push(o.x, o.y, r);
    }
    paths.forEach((arr, k) => { ctx.fillStyle = lv[k]; ctx.beginPath(); for (let j = 0; j < arr.length; j += 3) ctx.rect(arr[j] - arr[j + 2], arr[j + 1] - arr[j + 2] * 0.8, arr[j + 2] * 2, arr[j + 2] * 1.6); ctx.fill(); });
    ctx.restore();
  }

  class TheaterStage {
    constructor(canvas, o) {
      this.c = canvas; this.ctx = canvas.getContext('2d'); this.o = o || {}; this.D = o.data;
      const b = buildSeats('horseshoe', 1500); this.seats = b.seats; this.rows = b.rows; this.waveOrder = b.order;
      this.camHome = Object.assign({}, CAMS.horseshoe); this.cam = Object.assign({}, this.camHome);
      this.pIds = this.D.products.map(p => p.id); this.metric = 'buy'; this.mods = {};
      this.view = 'theater'; this.prevView = 'theater'; this.viewT0 = -1e9; this.viewDur = 900;
      this.insets = { top: 0, right: 0, bottom: 0, left: 0 };
      this.mode = 'stage'; this.hover = -1; this.focus = -1; this.visited = -1; this.lensOpen = 0;
      this.reduced = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.fade = 0; this.fadeAnim = null; this.fly = null; this.sortCond = null;
      this.w = 0; this.h = 0; this.dpr = Math.min(2, window.devicePixelRatio || 1);
      this.prodIdx = {}; this.D.products.forEach((p, i) => this.prodIdx[p.id] = i);
      this.colors = this.D.products.map(p => [0.42, 0.6, 0.8, 1].map(t => mix(p.color, BG, t)));
      this.shapes = this.D.products.map((p, i) => p.shape || ['circle', 'square', 'diamond', 'triangle'][i % 4]);
      const conds = this.D.conditions, W = this.D.worlds;
      this.pct = {}; conds.forEach(c => { const idx = W.map((w, i) => i).sort((a, b) => W[a].cond[c] - W[b].cond[c]); const arr = new Float32Array(W.length); idx.forEach((wi, r) => arr[wi] = (r + 0.5) / W.length); this.pct[c] = arr; });
      this._bind(); this.resize(); this.setWorlds(1500);
      this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas);
      this._raf = requestAnimationFrame(t => this._frame(t));
    }
    destroy() { cancelAnimationFrame(this._raf); this.ro.disconnect(); this._unbind(); }
    resize() {
      const r = this.c.getBoundingClientRect(); this.w = r.width; this.h = r.height;
      this.c.width = Math.max(1, Math.round(r.width * this.dpr)); this.c.height = Math.max(1, Math.round(r.height * this.dpr));
      const mob = r.width > 0 && r.width < 700;
      if (mob !== this.mobile) { this.mobile = mob; this.camHome = Object.assign({}, CAMS.horseshoe, mob ? { z: -7.5, fov: 74, pitch: 0.12 } : {}); if (this.mode === 'stage' && !this.fly) this.cam = Object.assign({}, this.camHome); }
      this.barsDirty = true;
    }
    setInsets(ins) { Object.assign(this.insets, ins); this.barsDirty = true; }
    area() { const i = this.insets; return { x: i.left, y: i.top, w: Math.max(50, this.w - i.left - i.right), h: Math.max(50, this.h - i.top - i.bottom) }; }

    // ---------- 데이터 ----------
    setWorlds(N) {
      this.N = N; const step = 1500 / N;
      this.active = []; for (let i = 0; i < N; i++) this.active.push(this.waveOrder[Math.floor(i * step)]);
      this.slotOf = Int32Array.from(this.active);
      this.p3 = new Float32Array(N * 3); this.from3 = new Float32Array(N * 3); this.moveT0 = new Float64Array(N).fill(-1e9);
      for (let i = 0; i < N; i++) { const s = this.seats[this.slotOf[i]]; this.p3[i * 3] = s.x; this.p3[i * 3 + 1] = s.y; this.p3[i * 3 + 2] = s.z; }
      this.litAt = new Float64Array(N).fill(-1); this.litCount = 0; this.running = false;
      this.sx = new Float32Array(N); this.sy = new Float32Array(N); this.ss = new Float32Array(N); this.vis = new Uint8Array(N);
      this.graph = null; this.sortCond = null; this.visited = -1; this.hover = -1; this.focus = -1;
      this.computeWinners();
    }
    setData(D, N) {
      this.D = D; const W = D.worlds; this.pct = {};
      D.conditions.forEach(c => { const arr = new Float32Array(W.length); W.forEach((w, i) => { arr[i] = w.cond[c]; }); this.pct[c] = arr; });
      this.codeCache = {}; this.socLayouts = {}; this.setWorlds(Math.min(N, W.length));
    }
    setProducts(ids) { this.pIds = ids.slice(); this.computeWinners(); }
    setMods(m) { this.mods = m || {}; this.computeWinners(); }
    setMetric(m) { this.metric = m; this.computeWinners(); }
    val(w, p) { return window.MM.value(w, p, this.metric, this.mods); }
    computeWinners() {
      const N = this.N, W = this.D.worlds;
      this.win = new Int8Array(N); this.bright = new Float32Array(N); this.clar = new Uint8Array(N); this.margin = new Float32Array(N);
      const bests = [];
      for (let i = 0; i < N; i++) {
        let b = -1, s = -1, bp = null;
        this.pIds.forEach(p => { const v = this.val(W[i], p); if (v > b) { s = b; b = v; bp = p; } else if (v > s) s = v; });
        this.win[i] = this.prodIdx[bp]; bests.push(b); const m = s > 0 ? (b - s) / b : 1; this.margin[i] = m;
        this.clar[i] = m < 0.12 ? 0 : m < 0.35 ? 1 : 2;
      }
      const sorted = bests.slice().sort((a, b) => a - b), lo = sorted[Math.floor(N * 0.05)], hi = sorted[Math.floor(N * 0.95)] || 1;
      for (let i = 0; i < N; i++) this.bright[i] = clamp((bests[i] - lo) / (hi - lo || 1), 0, 1);
      this.barsDirty = true; this._emitProgress(true);
    }
    tallies(n) { const t = {}; this.pIds.forEach(p => t[p] = 0); for (let i = 0; i < n; i++) t[this.D.products[this.win[i]].id]++; return t; }

    // ---------- 장면 2: 세상이 생겨난다 ----------
    run() {
      const now = performance.now();
      this.litAt.fill(-1); this.litCount = 0; this.hist = []; this.mode = 'stage';
      if (this.reduced) { this.litAt.fill(now - 2000); this.litCount = this.N; this._buildHist(); this._emitProgress(true); return; }
      this.running = true; this.runT0 = now; this.runDur = (3.2 + 4.3 * this.N / 1500) * 1000;
    }
    reset() { this.litAt.fill(-1); this.litCount = 0; this.running = false; this.hist = []; this.visited = -1; this.setSort(null, true); this.setView('theater', true); this.cam = Object.assign({}, this.camHome); }
    _wilson(k, n) { const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [p, Math.max(0, c - h), Math.min(1, c + h)]; }
    _buildHist() {
      this.hist = []; const cnt = {}; let lead = null;
      const stepN = Math.max(1, Math.floor(this.litCount / 120));
      for (let i = 0; i < this.litCount; i++) {
        const id = this.D.products[this.win[i]].id; cnt[id] = (cnt[id] || 0) + 1;
        if (!lead || cnt[id] > cnt[lead]) lead = id;
        if ((i + 1) % stepN === 0 || i === this.litCount - 1) { const w = this._wilson(cnt[lead], i + 1); this.hist.push({ n: i + 1, lead, p: w[0], lo: w[1], hi: w[2] }); }
      }
    }
    _emitProgress(force) {
      if (!this.o.onProgress || !this.litAt) return;
      const now = performance.now();
      if (!force && now - (this._lastEmit || 0) < 70) return;
      this._lastEmit = now; this._buildHist();
      this.o.onProgress({ n: this.litCount, N: this.N, tallies: this.tallies(this.litCount), hist: this.hist, done: this.litCount >= this.N && !this.running });
    }

    // ---------- 장면 3: 둘러보기 ----------
    setView(v, instant) {
      if (v === this.view) return;
      this.prevView = this.view; this.view = v; this.viewT0 = (instant || this.reduced) ? -1e9 : performance.now();
      if (v === 'graph') this._ensureGraph();
      if (v === 'bars') this.barsDirty = true;
    }
    setSort(cond, instant) {
      this.sortCond = cond; const N = this.N, now = performance.now();
      let assign;
      if (!cond) assign = Array.from(this.active);
      else {
        const cz = this.camHome.z;
        const slots = Array.from(this.active).sort((a, b) => { const A = this.seats[a], B = this.seats[b]; return Math.atan2(A.x, A.z - cz) - Math.atan2(B.x, B.z - cz); });
        const ws = []; for (let i = 0; i < N; i++) ws.push(i); ws.sort((a, b) => this.D.worlds[a].cond[cond] - this.D.worlds[b].cond[cond]);
        assign = new Array(N); ws.forEach((wi, r) => assign[wi] = slots[r]);
      }
      const rank = {}; if (cond) { const cz = this.camHome.z; const s = Array.from(this.active).sort((a, b) => Math.atan2(this.seats[a].x, this.seats[a].z - cz) - Math.atan2(this.seats[b].x, this.seats[b].z - cz)); s.forEach((sl, r) => rank[sl] = r / N); }
      for (let i = 0; i < N; i++) {
        this.from3[i * 3] = this.p3[i * 3]; this.from3[i * 3 + 1] = this.p3[i * 3 + 1]; this.from3[i * 3 + 2] = this.p3[i * 3 + 2];
        this.slotOf[i] = assign[i]; this.moveT0[i] = (instant || this.reduced) ? -1e9 : now + (rank[assign[i]] || this.D.worlds[i].id / N) * 500;
      }
    }
    _ensureGraph() {
      const N = this.N; if (this.graph && this.graph.N === N) { this.graph.alpha = Math.max(this.graph.alpha, 0.3); return; }
      const W = this.D.worlds, C = this.D.conditions, vec = [];
      for (let i = 0; i < N; i++) vec.push(C.map(c => W[i].cond[c]));
      const edges = [], seen = new Set();
      for (let i = 0; i < N; i++) {
        const best = [[1e9, -1], [1e9, -1], [1e9, -1]];
        for (let j = 0; j < N; j++) { if (j === i) continue; let d = 0; const a = vec[i], b = vec[j]; for (let k = 0; k < a.length; k++) { const e = a[k] - b[k]; d += e * e; } if (d < best[2][0]) { best[2] = [d, j]; best.sort((x, y) => x[0] - y[0]); } }
        best.forEach(([d, j]) => { const key = i < j ? i * 4096 + j : j * 4096 + i; if (j >= 0 && !seen.has(key)) { seen.add(key); edges.push(i, j); } });
      }
      const x = new Float32Array(N), y = new Float32Array(N), vx = new Float32Array(N), vy = new Float32Array(N);
      const fi = C.indexOf('유행 지속'), ai = C.indexOf('광고 도달'), rn = window.MM.mulberry(3);
      for (let i = 0; i < N; i++) { x[i] = (vec[i][fi] - 0.5) * 1.6 + (rn() - .5) * .1; y[i] = -(vec[i][ai] - 0.5) * 1.6 + (rn() - .5) * .1; }
      this.graph = { N, edges: Int32Array.from(edges), x, y, vx, vy, alpha: 1, drag: -1, fit: null };
      for (let t = 0; t < 90; t++) this._tickGraph(this.graph, 0.03, 0.05);
    }
    _tickGraph(g, rest, rep) {
      const { x, y, vx, vy, edges, N } = g, a = g.alpha;
      for (let e = 0; e < edges.length; e += 2) {
        const i = edges[e], j = edges[e + 1], dx = x[j] - x[i], dy = y[j] - y[i], d = Math.sqrt(dx * dx + dy * dy) || 1e-4, f = (d - rest) / d * 0.1 * a;
        vx[i] += dx * f; vy[i] += dy * f; vx[j] -= dx * f; vy[j] -= dy * f;
      }
      const cell = rep, grid = new Map();
      for (let i = 0; i < N; i++) { const k = Math.floor(x[i] / cell) * 100003 + Math.floor(y[i] / cell); let b = grid.get(k); if (!b) grid.set(k, b = []); b.push(i); }
      for (let i = 0; i < N; i++) {
        const gx = Math.floor(x[i] / cell), gy = Math.floor(y[i] / cell);
        for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
          const b = grid.get((gx + ox) * 100003 + gy + oy); if (!b) continue;
          for (let q = 0; q < b.length; q++) { const j = b[q]; if (j <= i) continue; const dx = x[j] - x[i], dy = y[j] - y[i], d2 = dx * dx + dy * dy; if (d2 < cell * cell && d2 > 1e-9) { const d = Math.sqrt(d2), f = (cell - d) / d * 0.25 * a; vx[i] -= dx * f; vy[i] -= dy * f; vx[j] += dx * f; vy[j] += dy * f; } }
        }
        vx[i] -= x[i] * 0.004 * a; vy[i] -= y[i] * 0.004 * a;
      }
      for (let i = 0; i < N; i++) { if (i === g.drag) { vx[i] = vy[i] = 0; continue; } vx[i] *= 0.55; vy[i] *= 0.55; x[i] += vx[i]; y[i] += vy[i]; }
      g.alpha = Math.max(0.02, a * 0.985);
    }
    _graphFit(g) {
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      for (let i = 0; i < g.N; i++) { if (g.x[i] < x0) x0 = g.x[i]; if (g.x[i] > x1) x1 = g.x[i]; if (g.y[i] < y0) y0 = g.y[i]; if (g.y[i] > y1) y1 = g.y[i]; }
      const A = this.area(), s = Math.min(A.w / (x1 - x0 || 1), A.h / (y1 - y0 || 1)) * 0.86;
      const t = { s, cx: A.x + A.w / 2 - (x0 + x1) / 2 * s, cy: A.y + A.h / 2 - (y0 + y1) / 2 * s };
      if (!g.fit || g.drag >= 0) g.fit = g.fit || t; else { g.fit.s = lerp(g.fit.s, t.s, 0.08); g.fit.cx = lerp(g.fit.cx, t.cx, 0.08); g.fit.cy = lerp(g.fit.cy, t.cy, 0.08); }
      return g.fit;
    }
    _layoutBars() {
      const N = this.N, A = this.area(), ids = this.pIds.slice();
      const tal = this.tallies(N); ids.sort((a, b) => tal[b] - tal[a]);
      const n = ids.length, colW = Math.min(300, A.w / n * 0.62), gapX = Math.min(120, (A.w - colW * n) / (n + 1));
      const C = 24, maxR = Math.ceil(Math.max(1, ...ids.map(p => tal[p])) / C);
      const pitch = Math.max(2, Math.min(colW / C, (A.h - 96) / maxR)), cw = pitch * C;
      const baseY = A.y + A.h - 40, cols = {};
      ids.forEach((p, k) => cols[p] = { x: A.x + gapX + k * (cw + gapX), n: tal[p], k: 0 });
      const order = []; for (let i = 0; i < N; i++) order.push(i); order.sort((a, b) => this.margin[b] - this.margin[a]);
      this.bx = new Float32Array(N); this.by = new Float32Array(N);
      order.forEach(i => { const col = cols[this.D.products[this.win[i]].id], k = col.k++; this.bx[i] = col.x + (k % C) * pitch + pitch / 2; this.by[i] = baseY - Math.floor(k / C) * pitch - pitch / 2; });
      this.bars = { cols, ids, pitch, cw, baseY, tal }; this.barsDirty = false;
    }

    // ---------- 장면 4: 세상 안으로 ----------
    enterWorld(i, pid) {
      if (i < 0 || i >= this.N || this.fly) return;
      pid = pid || this.D.products[this.win[i]].id;
      this.visited = i; this.hover = -1; this.focus = i;
      const done = () => this._openSociety(i, pid);
      if (this.reduced || this.view !== 'theater') { done(); return; }
      const s = { x: this.p3[i * 3], y: this.p3[i * 3 + 1], z: this.p3[i * 3 + 2] }, c = this.cam;
      const to = { x: lerp(c.x, s.x, .78), y: lerp(c.y, s.y, .78), z: lerp(c.z, s.z, .78), yaw: Math.atan2(s.x - c.x, s.z - c.z), pitch: Math.atan2(s.y - c.y, Math.hypot(s.x - c.x, s.z - c.z)), fov: c.fov * 0.75 };
      this.flyBack = to;
      this.fly = { t0: performance.now(), dur: 1150, from: Object.assign({}, c), to, fadeOut: true, done };
    }
    _getSociety(i, pid) {
      const S = this.D.society;
      if (S && S.worldId === i && S.product === pid && !Object.keys(this.mods).some(k => Math.abs(this.mods[k] - 1) > 1e-6)) return S;
      return window.MM.society(i, pid, this.mods);
    }
    _openSociety(i, pid) {
      const soc = this._getSociety(i, pid);
      this.socLayouts = this.socLayouts || {};
      if (!this.socLayouts[i]) this.socLayouts[i] = this._layoutSociety(soc, i);
      this.soc = soc; this.soc2 = null; this.socL = this.socLayouts[i]; this.socWorld = i; this.person = null;
      this.mode = 'society'; this.socT0 = performance.now(); this.day = 1;
      this.playing = !this.reduced; this.playT0 = performance.now(); if (this.reduced) this.day = 28.999;
      this.fade = 1; this.fadeAnim = { from: 1, to: 0, t0: performance.now(), dur: this.reduced ? 1 : 450 };
      this._cum(soc); this.o.onEnter && this.o.onEnter({ worldId: i, product: pid });
      this._emitDay(true);
    }
    _cum(soc) {
      if (soc.cum) return; soc.cum = [];
      for (let d = 0; d <= 28; d++) soc.cum.push({ exposed: 0, bought: 0, positive: 0, negative: 0 });
      soc.nodes.forEach(n => { for (let d = 1; d <= 28; d++) { const c = soc.cum[d]; if (n.exposed && n.exposed <= d) c.exposed++; if (n.bought && n.bought <= d) c.bought++; if (n.posted && n.posted <= d) { if (n.positive) c.positive++; else c.negative++; } } });
      soc.adj = Array.from({ length: soc.nodes.length }, () => []); soc.links.forEach(([a, b]) => { soc.adj[a].push(b); soc.adj[b].push(a); });
    }
    _layoutSociety(soc, seed) {
      const N = soc.nodes.length, rn = window.MM.mulberry(400 + seed), x = new Float32Array(N), y = new Float32Array(N);
      for (let i = 0; i < N; i++) { const a = rn() * Math.PI * 2, r = Math.sqrt(rn()); x[i] = Math.cos(a) * r; y[i] = Math.sin(a) * r; }
      const g = { N, x, y, vx: new Float32Array(N), vy: new Float32Array(N), edges: Int32Array.from(soc.links.flat()), alpha: 1, drag: -1 };
      for (let t = 0; t < 260; t++) this._tickGraph(g, 0.04, 0.15);
      g.alpha = 0.05; return g;
    }
    exitWorld() {
      if (this.mode !== 'society') return;
      this.mode = 'stage'; this.soc = null; this.soc2 = null; this.person = null;
      this.fade = 1; this.fadeAnim = { from: 1, to: 0, t0: performance.now(), dur: this.reduced ? 1 : 500 };
      if (!this.reduced && this.flyBack && this.view === 'theater') { this.cam = Object.assign({}, this.flyBack); this.fly = { t0: performance.now(), dur: 1000, from: Object.assign({}, this.flyBack), to: Object.assign({}, this.camHome), fadeOut: false }; }
      else this.cam = Object.assign({}, this.camHome);
      this.focus = this.visited;
    }
    setCompare(pid) { if (!this.soc) return; this.soc2 = pid ? window.MM.society(this.socWorld, pid, this.mods) : null; if (this.soc2) this._cum(this.soc2); this.person = null; this._emitDay(true); }
    play() { if (this.day >= 28.9) this.day = 1; this.playing = true; this.playT0 = performance.now() - (this.day - 1) * 520; this._emitDay(true); }
    pause() { this.playing = false; this._emitDay(true); }
    setDay(d) { this.playing = false; this.day = clamp(d, 1, 28) + 0.999; this._emitDay(true); }
    _emitDay(force) {
      const d = Math.min(28, Math.floor(this.day));
      if (!force && d === this._lastDay) return; this._lastDay = d;
      this.o.onDay && this.o.onDay({ day: d, playing: this.playing, a: this.soc && this.soc.cum[d], b: this.soc2 && this.soc2.cum[d], aProduct: this.soc && this.soc.product, bProduct: this.soc2 && this.soc2.product });
    }

    // ---------- 키보드 ----------
    moveFocus(dx, dy) {
      const N = this.N; let cur = this.focus >= 0 ? this.focus : this.hover;
      if (cur < 0 || !this.vis[cur]) { let best = -1, bd = 1e9; for (let i = 0; i < N; i++) if (this.vis[i] && this.litAt[i] >= 0) { const d = Math.hypot(this.sx[i] - this.w / 2, this.sy[i] - this.h / 2); if (d < bd) { bd = d; best = i; } } this.focus = best; return this._emitHover(true); }
      let best = -1, bs = 1e9;
      for (let i = 0; i < N; i++) {
        if (i === cur || !this.vis[i] || this.litAt[i] < 0) continue;
        const ex = this.sx[i] - this.sx[cur], ey = this.sy[i] - this.sy[cur], along = ex * dx + ey * dy; if (along < 2) continue;
        const perp = Math.abs(ex * dy - ey * dx), sc = along + perp * 2.5; if (sc < bs) { bs = sc; best = i; }
      }
      if (best >= 0) this.focus = best; this._emitHover(true);
    }
    selectFocus() { const i = this.focus >= 0 ? this.focus : this.hover; if (i >= 0 && this.o.onSelect) this.o.onSelect(i); }
    focusWorld(i) { this.focus = i; this.hover = -1; this._emitHover(true); }
    info(i) {
      const w = this.D.worlds[i], C = this.D.conditions, L = this.D.condLabels;
      const diffs = C.map(c => ({ c, p: this.pct[c][i] })).sort((a, b) => Math.abs(b.p - .5) - Math.abs(a.p - .5)).slice(0, 3)
        .map(d => ({ label: L[d.c][d.p < .5 ? 0 : 1], pct: d.p < .5 ? `하위 ${Math.max(1, Math.round(d.p * 100))}%` : `상위 ${Math.max(1, Math.round((1 - d.p) * 100))}%` }));
      const vals = this.pIds.map(p => ({ id: p, v: this.val(w, p) }));
      // 이 세상에서 실제로 계산된 누적 구매경험률(한 구간 = 13일)
      const SD = this.D.step_days, curves = this.pIds.map(p => ({ id: p, pts: w.curve[p] }));
      const wp = this.D.products[this.win[i]], wc = w.curve[wp.id]; let peak = 1, pv = -1; for (let d = 1; d <= 28; d++) { const dv = wc[d] - wc[d - 1]; if (dv > pv) { pv = dv; peak = d; } }
      const sortedV = vals.slice().sort((a, b) => b.v - a.v);
      const events = [
        { d: '12개월', t: window.MM.eul(wp.name) + ' 알게 된 사람 ' + Math.round(100 * w.seen[wp.id]) + '%' },
        { d: peak * SD + '일째', t: '처음 산 사람이 가장 많았던 때' },
        { d: '결과', t: wp.name + ' 승, 2등보다 ' + (sortedV[1] && sortedV[1].v > 0 ? (sortedV[0].v / sortedV[1].v).toFixed(1) + '배' : '크게 앞섬') }
      ];
      return { id: i, winner: wp.id, vals, diffs, margin: this.margin[i], curves, events };
    }
    _emitHover(force) {
      const i = this.focus >= 0 && this.hover < 0 ? this.focus : this.hover;
      if (!this.o.onHover) return;
      if (i < 0 || this.mode !== 'stage' || !this.vis[i] || this.litAt[i] < 0) { if (this._lastHover !== -1) { this._lastHover = -1; this.o.onHover(null); } return; }
      const x = this.sx[i], y = this.sy[i];
      if (!force && this._lastHover === i && Math.abs(x - this._lhx) < 1 && Math.abs(y - this._lhy) < 1) return;
      this._lastHover = i; this._lhx = x; this._lhy = y;
      const r = 70, side = x + r + 300 > this.w ? 'left' : 'right';
      this.o.onHover(Object.assign(this.info(i), { x, y, r, side, keyboard: this.hover < 0 }));
    }

    // ---------- 입력 ----------
    _bind() {
      const c = this.c; this._h = {};
      const pos = e => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      this._h.down = e => {
        const [x, y] = pos(e); this.drag = { x, y, x0: x, y0: y, moved: false, node: -1 };
        if (this.mode === 'society') { const h = this._hitPerson(x, y); if (h) { this.drag.node = h.id; this.socL.drag = h.id; this.socL.alpha = 0.3; } }
        else if (this.view === 'graph' && this.hover >= 0) { this.drag.node = this.hover; this.graph.drag = this.hover; this.graph.alpha = 0.35; }
        c.setPointerCapture && c.setPointerCapture(e.pointerId);
      };
      this._h.move = e => {
        const [x, y] = pos(e), d = this.drag;
        if (d) {
          const dx = x - d.x, dy = y - d.y; d.x = x; d.y = y;
          if (Math.hypot(x - d.x0, y - d.y0) > 4) d.moved = true;
          if (d.node >= 0 && this.mode === 'society') { const m = this._socMap(this.socL, 0); this.socL.x[d.node] = (x - m.cx) / m.s; this.socL.y[d.node] = (y - m.cy) / m.s; this.socL.alpha = 0.3; }
          else if (d.node >= 0 && this.view === 'graph') { const f = this.graph.fit; this.graph.x[d.node] = (x - f.cx) / f.s; this.graph.y[d.node] = (y - f.cy) / f.s; this.graph.alpha = 0.35; }
          else if (this.mode === 'stage' && this.view === 'theater' && !this.mobile && !this.fly && d.moved) {
            this.cam.yaw = clamp(this.cam.yaw - dx * 0.0035, -1.25, 1.25); this.cam.pitch = clamp(this.cam.pitch + dy * 0.0035, -0.35, 0.62);
          }
        }
        if (this.mode === 'stage' && !(d && d.moved && d.node < 0)) this._hoverAt(x, y);
        else if (this.mode === 'society') { const h = this._hitPerson(x, y); c.style.cursor = h ? 'pointer' : 'default'; }
      };
      this._h.up = e => {
        const d = this.drag; this.drag = null;
        if (this.graph) this.graph.drag = -1; if (this.socL) this.socL.drag = -1;
        if (!d || d.moved) return;
        const [x, y] = pos(e);
        if (this.mode === 'society') { const h = this._hitPerson(x, y); this.person = h; this.o.onPerson && this.o.onPerson(h); }
        else { this._hoverAt(x, y); if (this.hover >= 0 && this.o.onSelect) this.o.onSelect(this.hover); }
      };
      this._h.leave = () => { if (!this.drag) { this.hover = -1; this._emitHover(); } };
      this._h.wheel = e => {
        if (this.mode !== 'stage' || this.view !== 'theater' || this.mobile || this.fly) return;
        e.preventDefault(); this.cam.z = clamp(this.cam.z - e.deltaY * 0.012, -4.5, 6.5);
      };
      this._h.key = e => {
        if (this.mode === 'society') { if (e.key === 'Escape') { e.preventDefault(); this.o.onExit && this.o.onExit(); } return; }
        const map = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (map[e.key]) { e.preventDefault(); this.hover = -1; this.moveFocus(...map[e.key]); }
        else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.selectFocus(); }
        else if (e.key === 'Escape') { this.focus = -1; this._emitHover(true); }
      };
      c.addEventListener('pointerdown', this._h.down); c.addEventListener('pointermove', this._h.move); c.addEventListener('pointerup', this._h.up);
      c.addEventListener('pointerleave', this._h.leave); c.addEventListener('wheel', this._h.wheel, { passive: false }); c.addEventListener('keydown', this._h.key);
    }
    _unbind() { const c = this.c, h = this._h; c.removeEventListener('pointerdown', h.down); c.removeEventListener('pointermove', h.move); c.removeEventListener('pointerup', h.up); c.removeEventListener('pointerleave', h.leave); c.removeEventListener('wheel', h.wheel); c.removeEventListener('keydown', h.key); }
    _hoverAt(x, y) {
      let best = -1, bd = 1e9;
      for (let i = 0; i < this.N; i++) { if (!this.vis[i] || this.litAt[i] < 0) continue; const d = Math.hypot(this.sx[i] - x, this.sy[i] - y), lim = Math.max(9, this.ss[i] + 4); if (d < lim && d < bd) { bd = d; best = i; } }
      if (best !== this.hover) { this.hover = best; if (best >= 0) this.focus = -1; }
      this.c.style.cursor = best >= 0 ? 'pointer' : (this.view === 'theater' && !this.mobile ? 'grab' : 'default');
      this._emitHover();
    }

    // ---------- 그리기 ----------
    _frame(now) {
      this._raf = requestAnimationFrame(t => this._frame(t));
      if (!this.w || !this.h) return;
      const ctx = this.ctx; ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      if (this.fly) {
        const f = this.fly, k = clamp((now - f.t0) / f.dur, 0, 1), e = easeIO(k);
        for (const key of ['x', 'y', 'z', 'yaw', 'pitch', 'fov']) this.cam[key] = lerp(f.from[key], f.to[key], e);
        if (f.fadeOut) this.fade = clamp((k - 0.55) / 0.45, 0, 1);
        if (k >= 1) { this.fly = null; f.done && f.done(); }
      }
      if (this.fadeAnim) { const a = this.fadeAnim, k = clamp((now - a.t0) / a.dur, 0, 1); this.fade = lerp(a.from, a.to, k); if (k >= 1) this.fadeAnim = null; }
      ctx.fillStyle = BG; ctx.fillRect(0, 0, this.w, this.h);
      if (this.mode === 'society') this._drawSociety(now); else this._drawStage(now);
      if (this.fade > 0.001) { ctx.fillStyle = `rgba(10,10,10,${this.fade})`; ctx.fillRect(0, 0, this.w, this.h); }
    }
    _target(view, i, P, now, out) {
      if (view === 'theater') {
        const t0 = this.moveT0[i], s = this.seats[this.slotOf[i]];
        let x = s.x, y = s.y, z = s.z;
        if (now < t0 + 900) {
          const k = easeIO(clamp((now - t0) / 900, 0, 1));
          x = lerp(this.from3[i * 3], s.x, k); y = lerp(this.from3[i * 3 + 1], s.y, k) + Math.sin(Math.PI * k) * 1.4; z = lerp(this.from3[i * 3 + 2], s.z, k);
        }
        this.p3[i * 3] = x; this.p3[i * 3 + 1] = y; this.p3[i * 3 + 2] = z;
        const o = P.p(x, y, z); out[0] = o.x; out[1] = o.y; out[2] = clamp(P.f * 0.13 / o.z, 1.3, 16); out[3] = o.v ? 1 : 0;
      } else if (view === 'graph') {
        const g = this.graph, f = g.fit; out[0] = f.cx + g.x[i] * f.s; out[1] = f.cy + g.y[i] * f.s; out[2] = this.N > 1000 ? 2.6 : 3.4; out[3] = 1;
      } else { out[0] = this.bx[i]; out[1] = this.by[i]; out[2] = Math.max(1, this.bars.pitch * 0.36); out[3] = 1; }
    }
    _drawStage(now) {
      const ctx = this.ctx, W = this.w, H = this.h, N = this.N, P = makeProj(this.cam, W, H);
      if (this.running) {
        const k = clamp((now - this.runT0) / this.runDur, 0, 1), target = Math.floor(easeIO(k) * 0.15 * N + k * 0.85 * N);
        while (this.litCount < Math.min(N, target)) this.litAt[this.litCount++] = now;
        if (k >= 1) { while (this.litCount < N) this.litAt[this.litCount++] = now; this.running = false; this.runEnd = now; this._emitProgress(true); } else this._emitProgress();
      }
      if (this.view === 'graph' || this.prevView === 'graph') { if (this.graph) { this._tickGraph(this.graph, 0.03, 0.05); this._graphFit(this.graph); } }
      if (this.barsDirty && (this.view === 'bars' || this.prevView === 'bars')) this._layoutBars();
      const vk = clamp((now - this.viewT0) / this.viewDur, 0, 1), ve = easeIO(vk);
      const wOf = v => (this.view === v ? ve : 0) + (this.prevView === v && vk < 1 ? 1 - ve : 0);
      const tw = this.view === 'theater' && vk >= 1 ? 1 : wOf('theater');
      const occ = new Uint8Array(1500); for (let i = 0; i < N; i++) if (this.litAt[i] >= 0 && now >= this.moveT0[i] + 900) occ[this.slotOf[i]] = 1;
      drawHouse(ctx, W, H, P, this.rows, this.seats, tw, occ);
      const A = [0, 0, 0, 0], B = [0, 0, 0, 0];
      for (let i = 0; i < N; i++) {
        this._target(this.view, i, P, now, A);
        if (vk < 1 && this.prevView !== this.view) { this._target(this.prevView, i, P, now, B); this.sx[i] = lerp(B[0], A[0], ve); this.sy[i] = lerp(B[1], A[1], ve); this.ss[i] = lerp(B[2], A[2], ve); this.vis[i] = (A[3] || B[3]) ? 1 : 0; }
        else { this.sx[i] = A[0]; this.sy[i] = A[1]; this.ss[i] = A[2]; this.vis[i] = A[3]; }
      }
      const gw = wOf('graph');
      if (gw > 0.01 && this.graph) {
        ctx.save(); ctx.globalAlpha = gw; ctx.strokeStyle = 'rgba(255,255,255,0.13)'; ctx.lineWidth = 1; ctx.beginPath();
        const e = this.graph.edges; for (let k = 0; k < e.length; k += 2) { const i = e[k], j = e[k + 1]; if (this.litAt[i] < 0 || this.litAt[j] < 0) continue; ctx.moveTo(this.sx[i], this.sy[i]); ctx.lineTo(this.sx[j], this.sy[j]); }
        ctx.stroke(); ctx.restore();
      }
      // 좌석: 제품 × 밝기 4단계 × 선명도 3단계로 묶어 한 번에 칠한다
      const groups = {};
      for (let i = 0; i < N; i++) {
        if (!this.vis[i] || this.litAt[i] < 0) continue;
        const key = this.win[i] * 12 + Math.min(3, Math.floor(this.bright[i] * 4)) * 3 + this.clar[i];
        (groups[key] || (groups[key] = [])).push(i);
      }
      for (const key in groups) {
        const k = +key, p = Math.floor(k / 12), b = Math.floor((k % 12) / 3), cl = k % 3, shape = this.shapes[p], list = groups[key];
        ctx.globalAlpha = cl === 0 ? 0.5 : cl === 1 ? 0.8 : 1; ctx.fillStyle = this.colors[p][b]; ctx.beginPath();
        const sc = cl === 0 ? 1.2 : 1;
        for (let q = 0; q < list.length; q++) { const i = list[q]; shapePath(ctx, shape, this.sx[i], this.sy[i], this.ss[i] * 0.62 * sc); }
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (!this.reduced && this.litCount > 0) {
        ctx.lineWidth = 1.2;
        for (let i = Math.max(0, this.litCount - 140); i < this.litCount; i++) {
          const t = (now - this.litAt[i]) / 520; if (t < 0 || t > 1 || !this.vis[i]) continue;
          ctx.strokeStyle = `rgba(255,255,255,${(1 - t) * 0.8})`; ctx.beginPath(); ctx.arc(this.sx[i], this.sy[i], this.ss[i] * (0.6 + t * 1.8), 0, Math.PI * 2); ctx.stroke();
        }
      }
      this._drawCode(now);
      const bw = wOf('bars');
      if (bw > 0.01 && this.bars) this._drawBarLabels(bw);
      if (this.visited >= 0 && this.vis[this.visited]) {
        const i = this.visited, r = Math.max(8, this.ss[i] + 6);
        ctx.strokeStyle = '#E60012'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(this.sx[i], this.sy[i], r, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#ffffff'; ctx.font = `600 12px ${FONT}`; ctx.textBaseline = 'middle'; ctx.fillText(`방금 본 세상 #${i + 1}`, this.sx[i] + r + 6, this.sy[i]);
      }
      this._drawLens(now);
      this._emitHover();
    }
    _codeLine(i) {
      this.codeCache = this.codeCache || {}; const key = i + this.metric + this.pIds.join('');
      if (this.codeCache[key]) return this.codeCache[key];
      const w = this.D.worlds[i], c = w.cond, f = v => v.toFixed(2).slice(1), h = ((Math.imul(i + 1, 2654435761)) >>> 0).toString(16).padStart(8, '0').slice(0, 4);
      const vals = this.pIds.map(p => { const v = this.val(w, p); return p + ' ' + (this.metric === 'rev' ? String(Math.round(v)).padStart(3, ' ') : v.toFixed(3).slice(1)); }).join('  ');
      const out = { t: 'w' + String(i + 1).padStart(4, '0') + ' ' + h + '  ad' + f(c['광고 도달']) + ' fad' + f(c['유행 지속']) + ' sns' + f(c['SNS 전파력']) + ' px' + f(c['가격 민감도']) + '  ->  ' + vals + '  ', win: this.win[i] };
      return this.codeCache[key] = out;
    }
    // 세상이 계산되는 기록이 흘러가고, 각 줄이 그 좌석을 켠다
    _drawCode(now) {
      if (this.reduced || !this.litCount) return;
      const tail = this.runEnd ? now - this.runEnd : 0;
      if (!this.running && !(this.runEnd && tail < 1600)) return;
      const ctx = this.ctx, L = this.h < 560 ? 9 : 15, lh = 15, x0 = 24, y0 = this.h - 22 - (L - 1) * lh, n = this.litCount;
      ctx.save(); ctx.globalAlpha = this.running ? 1 : 1 - tail / 1600;
      const g = ctx.createLinearGradient(0, 0, 560, 0); g.addColorStop(0, 'rgba(10,10,10,0.88)'); g.addColorStop(1, 'rgba(10,10,10,0)');
      ctx.fillStyle = g; ctx.fillRect(0, y0 - 18, 560, L * lh + 26);
      ctx.font = '500 11.5px ui-monospace, "SF Mono", Menlo, Consolas, monospace'; ctx.textBaseline = 'alphabetic';
      for (let k = 0; k < L; k++) {
        const i = n - L + k; if (i < 0) continue;
        const y = y0 + k * lh, line = this._codeLine(i), newest = k === L - 1, a = newest ? 1 : 0.18 + 0.62 * (k / L);
        ctx.fillStyle = newest ? '#ffffff' : 'rgba(225,225,225,' + a + ')'; ctx.fillText(line.t, x0, y);
        const tw = ctx.measureText(line.t).width, p = this.D.products[line.win];
        ctx.fillStyle = this.colors[line.win][newest ? 3 : 2]; ctx.globalAlpha *= 1; ctx.fillText('> ' + p.id + ' 승', x0 + tw, y);
        if (k >= L - 5 && this.vis[i] && this.running) {
          const bx = x0 + tw + 46, by = y - 4, q = (k - (L - 5) + 1) / 5;
          ctx.strokeStyle = 'rgba(255,255,255,' + (0.28 * q) + ')'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, by);
          ctx.quadraticCurveTo((bx + this.sx[i]) / 2, Math.min(by, this.sy[i]) - 40, this.sx[i], this.sy[i]); ctx.stroke();
        }
      }
      ctx.restore();
    }
    _drawBarLabels(alpha) {
      const ctx = this.ctx, b = this.bars, N = this.N; ctx.save(); ctx.globalAlpha = alpha; ctx.textBaseline = 'alphabetic';
      b.ids.forEach(p => {
        const col = b.cols[p], prod = this.D.products[this.prodIdx[p]], rowsN = Math.ceil(col.n / 24), top = b.baseY - rowsN * b.pitch;
        ctx.fillStyle = '#ffffff'; ctx.font = `700 15px ${FONT}`; ctx.fillText(prod.name, col.x, b.baseY + 26);
        ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.fillRect(col.x, b.baseY + 2, b.cw, 2);
        ctx.fillStyle = '#ffffff'; ctx.font = `700 28px ${FONT}`; ctx.fillText(`${(col.n / N * 100).toFixed(1)}%`, col.x, top - 14);
        const tw = ctx.measureText(`${(col.n / N * 100).toFixed(1)}%`).width;
        ctx.font = `500 14px ${FONT}`; ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.fillText(`${col.n.toLocaleString()}개 세상`, col.x + tw + 10, top - 14);
      });
      ctx.restore();
    }
    _drawLens(now) {
      const i = this.hover >= 0 ? this.hover : this.focus, ctx = this.ctx;
      const want = i >= 0 && this.vis[i] && this.litAt[i] >= 0 && !this.drag?.moved && !this.fly;
      this.lensOpen = this.reduced ? (want ? 1 : 0) : clamp(this.lensOpen + (want ? 0.16 : -0.25), 0, 1);
      if (this.lensOpen <= 0 || i < 0) return;
      const R = 70 * easeOut(this.lensOpen), Z = 3.2, cx = this.sx[i], cy = this.sy[i];
      ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.clip();
      ctx.fillStyle = '#050505'; ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
      const lim = R / Z + 12;
      for (let j = 0; j < this.N; j++) {
        if (!this.vis[j] || this.litAt[j] < 0) continue; const dx = this.sx[j] - cx, dy = this.sy[j] - cy; if (Math.abs(dx) > lim || Math.abs(dy) > lim) continue;
        const p = this.win[j], b = Math.min(3, Math.floor(this.bright[j] * 4)), cl = this.clar[j];
        ctx.globalAlpha = cl === 0 ? 0.5 : cl === 1 ? 0.8 : 1; ctx.fillStyle = this.colors[p][b]; ctx.beginPath();
        shapePath(ctx, this.shapes[p], cx + dx * Z, cy + dy * Z, Math.min(18, this.ss[j] * 0.55 * Z)); ctx.fill();
      }
      ctx.globalAlpha = 1; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); shapePath(ctx, 'circle', cx, cy, Math.min(18, this.ss[i] * 0.55 * Z) + 5); ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R + 5, 0, Math.PI * 2); ctx.stroke();
    }
    _socMap(L, half) {
      const A = this.area(), two = !!this.soc2, w = two ? A.w / 2 : A.w, x0 = A.x + (two ? half * w : 0);
      let a = 1e9, b = -1e9, c = 1e9, d = -1e9; for (let i = 0; i < L.N; i++) { if (L.x[i] < a) a = L.x[i]; if (L.x[i] > b) b = L.x[i]; if (L.y[i] < c) c = L.y[i]; if (L.y[i] > d) d = L.y[i]; }
      if (!L.box || L.drag < 0) L.box = [a, b, c, d];
      const [bx0, bx1, by0, by1] = L.box, s = Math.min((w - 40) / (bx1 - bx0 || 1), (A.h - 40) / (by1 - by0 || 1));
      return { s, cx: x0 + w / 2 - (bx0 + bx1) / 2 * s, cy: A.y + A.h / 2 - (by0 + by1) / 2 * s };
    }
    _hitPerson(x, y) {
      if (!this.soc) return null; const L = this.socL;
      const halves = this.soc2 ? [[this.soc, 0], [this.soc2, 1]] : [[this.soc, 0]];
      let best = null, bd = 1e9;
      halves.forEach(([soc, h]) => { const m = this._socMap(L, h); soc.nodes.forEach((n, i) => { const px = m.cx + L.x[i] * m.s, py = m.cy + L.y[i] * m.s, d = Math.hypot(px - x, py - y), r = this._nodeR(n) + 4; if (d < r && d < bd) { bd = d; best = Object.assign({}, n, { product: soc.product, half: h }); } }); });
      return best;
    }
    _nodeR(n) { return clamp(1.6 + Math.sqrt(n.followers) * 0.32, 2, 11); }
    _drawSociety(now) {
      const ctx = this.ctx, L = this.socL;
      if (this.playing) {
        this.day = this.reduced ? Math.min(28.999, Math.floor(1 + (now - this.playT0) / 700) + 0.999) : 1 + (now - this.playT0) / 520;
        if (this.day >= 28.999) { this.day = 28.999; this.playing = false; this._emitDay(true); }
      }
      this._emitDay();
      if (L.alpha > 0.03) this._tickGraph(L, 0.04, 0.15);
      const t = this.day, appear = this.reduced ? 1 : clamp((now - this.socT0) / 900, 0, 1);
      const halves = this.soc2 ? [[this.soc, 0], [this.soc2, 1]] : [[this.soc, 0]];
      halves.forEach(([soc, h]) => {
        const m = this._socMap(L, h), N = soc.nodes.length, px = new Float32Array(N), py = new Float32Array(N);
        const ccx = m.cx, ccy = m.cy;
        for (let i = 0; i < N; i++) { const tx = m.cx + L.x[i] * m.s, ty = m.cy + L.y[i] * m.s, r = Math.hypot(L.x[i], L.y[i]), k = easeOut(clamp(appear * 1.6 - r * 0.6, 0, 1)); px[i] = lerp(ccx, tx, k); py[i] = lerp(ccy, ty, k); }
        ctx.strokeStyle = 'rgba(255,255,255,0.075)'; ctx.lineWidth = 1; ctx.beginPath();
        soc.links.forEach(([a, b]) => { ctx.moveTo(px[a], py[a]); ctx.lineTo(px[b], py[b]); }); ctx.stroke();
        const color = this.D.products[this.prodIdx[soc.product]].color;
        // 추천·불만이 선을 타고 번진다
        if (!this.reduced) soc.nodes.forEach(n => {
          if (!n.posted || t < n.posted || t >= n.posted + 1) return;
          const q = easeOut(t - n.posted), col = n.positive ? color : '#9a9a9a';
          ctx.strokeStyle = n.positive ? 'rgba(230,0,18,0.55)' : 'rgba(170,170,170,0.5)'; ctx.lineWidth = 1.4; ctx.beginPath();
          soc.adj[n.id].forEach(v => { ctx.moveTo(px[n.id], py[n.id]); ctx.lineTo(lerp(px[n.id], px[v], q), lerp(py[n.id], py[v], q)); }); ctx.stroke();
          ctx.fillStyle = col; ctx.beginPath(); soc.adj[n.id].forEach(v => { shapePath(ctx, 'circle', lerp(px[n.id], px[v], q), lerp(py[n.id], py[v], q), 2); }); ctx.fill();
        });
        const none = [], ex = [], bo = [], pos = [], neg = [];
        soc.nodes.forEach((n, i) => {
          if (n.posted && n.posted <= t) (n.positive ? pos : neg).push(i);
          else if (n.bought && n.bought <= t) bo.push(i);
          else if (n.exposed && n.exposed <= t) ex.push(i); else none.push(i);
        });
        ctx.fillStyle = '#3b3b3b'; ctx.beginPath(); none.forEach(i => shapePath(ctx, 'circle', px[i], py[i], this._nodeR(soc.nodes[i]) * 0.75)); ctx.fill();
        ctx.strokeStyle = '#d6d6d6'; ctx.lineWidth = 1.3; ctx.beginPath(); ex.forEach(i => shapePath(ctx, 'circle', px[i], py[i], this._nodeR(soc.nodes[i]))); ctx.stroke();
        ctx.fillStyle = color; ctx.beginPath(); bo.concat(pos).forEach(i => shapePath(ctx, 'circle', px[i], py[i], this._nodeR(soc.nodes[i]))); ctx.fill();
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); pos.forEach(i => shapePath(ctx, 'circle', px[i], py[i], this._nodeR(soc.nodes[i]) + 3.5)); ctx.stroke();
        ctx.fillStyle = '#9a9a9a'; ctx.beginPath(); neg.forEach(i => shapePath(ctx, 'square', px[i], py[i], this._nodeR(soc.nodes[i]) * 1.05)); ctx.fill();
        ctx.strokeStyle = '#0a0a0a'; ctx.lineWidth = 1.5; ctx.beginPath(); neg.forEach(i => { const r = this._nodeR(soc.nodes[i]) * 0.55; ctx.moveTo(px[i] - r, py[i] - r); ctx.lineTo(px[i] + r, py[i] + r); ctx.moveTo(px[i] + r, py[i] - r); ctx.lineTo(px[i] - r, py[i] + r); }); ctx.stroke();
        if (this.person) { const i = this.person.id, r = this._nodeR(soc.nodes[i]) + 7; ctx.strokeStyle = '#E60012'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px[i], py[i], r, 0, Math.PI * 2); ctx.stroke(); }
        if (this.soc2) { ctx.fillStyle = '#ffffff'; ctx.font = `700 15px ${FONT}`; ctx.textBaseline = 'top'; const A = this.area(); ctx.fillText(`${window.MM.eul(this.D.products[this.prodIdx[soc.product]].name)} 냈을 때`, A.x + h * A.w / 2 + 16, A.y + 8); }
      });
      if (this.soc2) { const A = this.area(); ctx.fillStyle = 'rgba(255,255,255,0.25)'; ctx.fillRect(A.x + A.w / 2 - 1, A.y, 2, A.h); }
    }

    // 구도 비교용 정지 그림
    static sketch(canvas, kind) {
      const r = canvas.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1), W = r.width, H = r.height;
      canvas.width = W * dpr; canvas.height = H * dpr; const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
      const b = buildSeats(kind, 1500), P = makeProj(CAMS[kind], W, H), rn = window.MM.mulberry(11);
      drawHouse(ctx, W, H, P, b.rows, b.seats, 1, b.seats.map(() => 1));
      const cols = [['#E60012', 'circle'], ['#EFEBE4', 'square'], ['#8D99A3', 'diamond']];
      b.seats.forEach(s => { const o = P.p(s.x, s.y, s.z); if (!o.v) return; const u = rn(), k = u < 0.9 ? 0 : u < 0.97 ? 1 : 2, br = 0.45 + rn() * 0.55; ctx.fillStyle = mix(cols[k][0], BG, br); ctx.beginPath(); shapePath(ctx, cols[k][1], o.x, o.y, clamp(P.f * 0.13 / o.z, 1.3, 16) * 0.62); ctx.fill(); });
    }
  }
  window.TheaterStage = TheaterStage;
})();
