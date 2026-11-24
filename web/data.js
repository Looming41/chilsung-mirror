// 실제 엔진(mirror/server.py) 결과를 window.DATA 에 넣고, 화면이 쓰는 계산 도우미를 window.MM 에 둔다.
(function () {
  function mulberry(seed) { let a = seed >>> 0; return function () { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const api = async (url, body) => {
    const r = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
    const j = await r.json().catch(() => ({ error: '서버 응답을 읽지 못했습니다' }));
    if (!r.ok || j.error && !('state' in j)) throw new Error(j.error || '서버 오류 ' + r.status);
    return j;
  };
  const wait = ms => new Promise(f => setTimeout(f, ms));
  async function job(id, tick) {
    for (;;) {
      const j = await api('/api/job/' + id);
      if (tick) tick(j);
      if (j.state === 'done') return j.result;
      if (j.state === 'error') throw new Error(j.error);
      await wait(400);
    }
  }
  // 연령×성별×권역 칸의 인구·응답자·가중치 제곱합 → 고른 집단의 유효표본과 오차
  function population(sel) {
    let p = 0, n = 0, q = 0;
    window.DATA.cross.forEach(c => { if (sel.age.includes(c[0]) && sel.sex.includes(c[1]) && sel.region.includes(c[2])) { p += c[3]; n += c[4]; q += c[5]; } });
    const e = q ? p * p / q : 0;
    return { people: p, respondents: n, effective: Math.round(e), margin: e ? Math.round(980 / Math.sqrt(e)) / 10 : 0 };
  }
  const price = id => window.DATA.products.find(p => p.id === id).price;
  const value = (w, pid, metric) => metric === 'rev' ? w.buy[pid] * price(pid) : w.buy[pid];
  function summarize(ws, pids, metric) {
    const n = ws.length, wins = {}, short = {}, D = window.DATA; pids.forEach(p => { wins[p] = 0; short[p] = []; });
    const winOf = ws.map(w => {
      let best = -1, bp = null; pids.forEach(p => { const v = value(w, p, metric); if (v > best) { best = v; bp = p; } });
      wins[bp]++; pids.forEach(p => short[p].push(best - value(w, p, metric))); return bp;
    });
    const q = (a, t) => a[Math.min(a.length - 1, Math.floor(t * a.length))];
    return pids.map(p => {
      const vals = ws.map(w => value(w, p, metric)).sort((a, b) => a - b);
      const worst = short[p].slice().sort((a, b) => b - a).slice(0, Math.max(1, Math.round(n * 0.1)));
      const losing = ws.filter((w, i) => winOf[i] !== p);
      let loseWhen = [];
      if (losing.length >= 3 && losing.length < n) loseWhen = D.conditions.map(c => {
        const all = ws.map(w => w.cond[c]), m = all.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(all.reduce((a, b) => a + (b - m) * (b - m), 0) / n) || 1;
        return { c, z: (losing.reduce((a, w) => a + w.cond[c], 0) / losing.length - m) / sd };
      }).sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, 3).filter(x => Math.abs(x.z) > 0.3).map(x => D.condLabels[x.c][x.z < 0 ? 0 : 1]);
      return { id: p, median: q(vals, .5), lo: q(vals, .05), hi: q(vals, .95), share: wins[p] / n, wins: wins[p], regret: worst.reduce((a, b) => a + b, 0) / worst.length, loseWhen };
    });
  }
  const cache = new Map(), key = (w, p) => window.DATA.run + '/' + w + '/' + p;
  async function loadSociety(w, p) {
    const k = key(w, p);
    if (!cache.has(k)) { if (cache.size > 12) cache.delete(cache.keys().next().value); cache.set(k, await api(`/api/society?run=${encodeURIComponent(window.DATA.run)}&world=${w}&product=${encodeURIComponent(p)}`)); }
    return cache.get(k);
  }
  const eul = w => { const c = w.charCodeAt(w.length - 1) - 0xAC00; return w + (c >= 0 && c <= 11171 && c % 28 ? '을' : '를'); };
  window.MM = { eul, mulberry, summarize, population, value, api, job, loadSociety, modifiers: () => ({}), society: (w, p) => cache.get(key(w, p)) };
  window.BOOT = { error: '' };
  (function load() {
    api('/api/data').then(d => { d.society = null; d.summary = null; d.all = d.worlds; window.DATA = d; }).catch(e => { window.BOOT.error = e.message; setTimeout(load, 2500); });
  })();
})();
