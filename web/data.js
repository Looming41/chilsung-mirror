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
  // 제품 하나의 성적: 과거 출시 5개 제품의 실제 구매경험률(최저·평균·최고)과 비교한 네 등급
  const value = w => w.buy[window.DATA.cur];
  const tier = w => { const v = value(w), b = window.DATA.bench; return v >= b[2] ? 0 : v >= b[1] ? 1 : v >= b[0] ? 2 : 3; };
  function tiers(b) {
    const p = v => Math.round(v * 100) + '%';
    return [{ id: `${p(b[2])} 이상`, name: `과거 최고(${p(b[2])}) 이상`, color: '#FF2B3D', shape: 'circle' }, { id: `${p(b[1])}~`, name: `과거 평균~최고 (${p(b[1])}~${p(b[2])})`, color: '#F5A3A8', shape: 'circle' },
      { id: `${p(b[0])}~`, name: `과거 최저~평균 (${p(b[0])}~${p(b[1])})`, color: '#EFEBE4', shape: 'square' }, { id: `${p(b[0])} 미만`, name: `과거 최저(${p(b[0])}) 미만`, color: '#7C8894', shape: 'diamond' }];
  }
  // 한 제품의 요약과 구매율을 좌우하는 조건
  function summarize(ws) {
    const D = window.DATA, n = ws.length, v = ws.map(value), s = v.slice().sort((a, b) => a - b), q = t => s[Math.min(n - 1, Math.floor(t * n))];
    // 구매경험률 순위와 조건 순위의 상관이 0.09 를 넘는 것만(작은 상관은 조건끼리 얽힌 착시일 수 있다)
    const rk = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]).reduce((r, x, k) => { r[x[1]] = (k + .5) / n - .5; return r; }, new Array(n));
    const top = D.conditions.map(c => ({ c, r: 12 * ws.reduce((a, w, i) => a + (w.cond[c] - .5) * rk[i], 0) / n })).filter(x => Math.abs(x.r) > 0.09).sort((a, b) => Math.abs(b.r) - Math.abs(a.r)).slice(0, 3);
    const good = top.map(x => D.condLabels[x.c][x.r > 0 ? 1 : 0]), bad = top.map(x => D.condLabels[x.c][x.r > 0 ? 0 : 1]);
    return { median: q(.5), lo: q(.05), hi: q(.95), over: v.filter(x => x >= D.bench[1]).length, under: v.filter(x => x < D.bench[0]).length, n, good, bad };
  }
  const cache = new Map(), key = (w, p) => window.DATA.run + '/' + w + '/' + p;
  async function loadSociety(w, p) {
    const k = key(w, p);
    if (!cache.has(k)) { if (cache.size > 12) cache.delete(cache.keys().next().value); cache.set(k, await api(`/api/society?run=${encodeURIComponent(window.DATA.run)}&world=${w}&product=${encodeURIComponent(p)}`)); }
    return cache.get(k);
  }
  const eul = w => { const c = w.charCodeAt(w.length - 1) - 0xAC00; return w + (c >= 0 && c <= 11171 && c % 28 ? '을' : '를'); };
  window.MM = { eul, mulberry, summarize, tier, population, value, api, job, loadSociety, modifiers: () => ({}), society: (w, p) => cache.get(key(w, p)) };
  window.BOOT = { error: '' };
  (function load() {
    api('/api/data').then(d => { d.society = null; d.summary = null; d.all = d.worlds; d.items = d.products; d.products = tiers(d.bench); d.cur = d.items[0].id; window.DATA = d; }).catch(e => { window.BOOT.error = e.message; setTimeout(load, 2500); });
  })();
})();
