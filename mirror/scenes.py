"""AI 가 설계하는 세상들.

1) 축: 제품의 성패를 가를 상황 축 8개와 상태·확률을 AI 가 설계한다
2) 세상: 축에서 뽑은 조합마다 AI 가 세상 하나를 설계한다(이야기, 조건 값 0~100, 1년 사건)
3) 사람: 세상마다 연령×성별 8집단에서 한 명씩, AI 페르소나가 그 세상에 들어가 살 가능성을 답한다
4) 엔진에 넘김: AI 가 매긴 조건·분위기는 '순서'로 쓰고, 퍼짐의 폭은 과거 5개 제품으로 걸러낸 값을 쓴다.
   어느 세상이 뜨거운지는 AI 가, 얼마나 뜨거울 수 있는지는 과거 실적이 정한다. 사건은 때(타이밍)만 바꾸고 1년 평균은 바꾸지 않는다(세기는 가정).
보내는 것: 컨셉 카드와 합성 인물뿐이고 대회가 준 주소로만 보낸다.
실행: python -m mirror.scenes check
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np

from . import agents as A
from . import consumer as C
from . import people as P
from . import worlds as W

DIR = os.path.join(P.ROOT, "data", "scenes")
BATCH, TRY, WORKERS = 5, 200, 120                       # 한 번에 설계하는 세상, 후보 조건, 동시 호출
COND = [c for c in W.COND if c != "시장 분위기"]           # 분위기는 AI 사람들의 답이 정한다
MAKER = "당신은 소비재 시장 시나리오 설계자입니다. 실제 기업·인물에 대한 사실을 지어내지 말고, 일어날 법한 상황만 설계하세요. JSON 하나로만 답하세요.\n"
AXES = ('신제품: {card}\n앞으로 1년 한국 음료 시장에서 이 제품의 성패를 가를, 서로 독립적인 상황 축 8개를 설계하세요. 축마다 서로 뚜렷이 다른 상태 4개와 현실적인 확률(합 1).\n'
        '{{"axes":[{{"name":"축 이름(8자 이내)","states":[{{"s":"상태 한 구절(20자 이내)","p":0.4}}]}}]}}')
WORLD = ('신제품: {card}\n상황마다 세상 하나를 설계하세요. 상황들이 서로 얽혀 무슨 일이 벌어지는지 구체적으로 상상하세요.\n{lines}\n'
         'cond 는 보통 세상을 50 으로 한 0~100 정수입니다(낮음/높음의 뜻): ' + ", ".join(f"{c}({W.LABEL[c][0]}/{W.LABEL[c][1]})" for c in COND) + '\n'
         'events 는 그 1년에 벌어지는 일 1~3개: m=출시 뒤 몇 달째(1~12), t=한 구절(20자 이내), e=이 제품 판매에 주는 영향(-2~2 정수).\n'
         '{{"worlds":[{{"no":1,"title":"12자 이내","story":"60자 이내","cond":{{' + ",".join(f'"{c}":50' for c in COND) + '}},"events":[{{"m":3,"t":"","e":1}}]}}]}}')
LIVE = ('당신이 사는 세상: {story}\n그동안 벌어진 일: {events}\n이 세상의 편의점에서 이 신제품을 봤습니다: {card}\n이 세상의 사정까지 생각해 살 가능성을 0(절대 안 삼)~100(바로 삼) 정수로 답하세요. JSON 하나로만: {{"p":정수,"say":"이유 한 문장(30자 이내)"}}')


@lru_cache(8)
def _sig(k):
    return hashlib.sha1("\n".join([MAKER, AXES, WORLD, LIVE, A.CARD[k], A._sig()]).encode()).hexdigest()[:12]


def _file(k, model):
    return os.path.join(DIR, f"{k}-{model}.json")


def load(k, model=None):
    fs = [_file(k, model)] if model else [os.path.join(DIR, f) for f in (os.listdir(DIR) if os.path.isdir(DIR) else []) if f.startswith(k + "-") and f.endswith(".json")]
    for f in sorted((f for f in fs if os.path.exists(f)), key=os.path.getmtime, reverse=True):
        with open(f) as h:
            d = json.load(h)
        if d["sig"] == _sig(k):
            return d
    return None


def _save(k, d):
    os.makedirs(DIR, exist_ok=True)
    f = _file(k, d["model"])
    with open(f + ".tmp", "w") as h:
        json.dump(d, h, ensure_ascii=False)
    os.replace(f + ".tmp", f)


def _json(t):
    return json.loads(re.search(r"\{.*\}", t, re.S).group(0))


def _ask(system, q, model, read):
    """못 읽는 답도 저장되므로 질문을 바꿔 한 번 더."""
    for more in ("", "\n다른 글 없이 올바른 JSON 하나만, 개수와 형식을 정확히 지켜 출력하세요."):
        try:
            return read(_json(C.chat(system, q + more, model)))
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            if more:
                raise ValueError("답을 읽지 못함") from e


def _s(x, n):
    return re.sub(r"\s+", " ", str(x)).strip()[:n]


def read_axes(m):
    out = []
    for a in m["axes"][:8]:
        st = [{"s": _s(x["s"], 30), "p": max(.02, float(x["p"]))} for x in a["states"][:4]]
        t = sum(x["p"] for x in st)
        if len(st) < 2 or not np.isfinite(t):
            raise ValueError
        out.append({"name": _s(a["name"], 14), "states": [{"s": x["s"], "p": x["p"] / t} for x in st]})
    if len(out) < 6:
        raise ValueError
    return out


def read_worlds(m, n):
    out = []
    for w in m["worlds"]:
        ev = [{"m": min(12, max(1, int(e["m"]))), "t": _s(e["t"], 30), "e": min(2, max(-2, int(e["e"])))} for e in w["events"][:3]]
        out.append({"title": _s(w["title"], 20), "story": _s(w["story"], 90), "cond": {c: min(100.0, max(0.0, float(w["cond"][c]))) for c in COND}, "events": ev})
    if len(out) != n:
        raise ValueError
    return out


def combo(axes, i):
    """세상 i 가 놓인 상태들(축마다 하나, AI 가 매긴 확률대로)."""
    g = np.random.default_rng([11, i])
    return [int(g.choice(len(a["states"]), p=[x["p"] for x in a["states"]])) for a in axes]


@lru_cache(1)
def _seg():
    c = A.cast()
    a = c.age.str.extract(r"(\d+)")[0].astype(int).values
    return np.clip(a // 10 - 2, 0, 3) * 2 + (c.sex.values == "남자")          # 19세가 든 첫 구간도 20대로


def crew(i):
    """세상 i 에 들어갈 여덟 명: 연령×성별 집단마다 한 명."""
    g = np.random.default_rng([13, i])
    return [int(g.choice(np.flatnonzero(_seg() == s))) for s in range(8)]


def _live(k, w, p, model):
    ev = ", ".join(f"{e['m']}달째 {e['t']}" for e in w["events"]) or "없음"
    r = _ask(A.SYS + "\n".join(P.memory(A.cast().iloc[p])), LIVE.format(story=w["story"], events=ev, card=A.CARD[k]), model,
             lambda m: {"p": min(100, max(0, int(m["p"]))), "say": _s(m["say"], 60)})
    return r | {"i": p}


def _all(jobs, tick, what, need=.9):
    """jobs: (이름, 함수). 잔액이 없으면 바로 멈추고, 90% 를 못 채우면 쓰지 않는다."""
    out, err = {}, ""
    with ThreadPoolExecutor(WORKERS) as ex:
        fs = {ex.submit(f): n for n, f in jobs}
        for f in as_completed(fs):
            try:
                out[fs[f]] = f.result()
            except Exception as e:
                err = str(e)[:80] if type(e) in (RuntimeError, ValueError) else type(e).__name__
                if "잔액" in err:
                    ex.shutdown(cancel_futures=True)
                    raise RuntimeError(err) from None
            if tick:
                tick(len(out))
    if len(out) < need * len(jobs):
        raise RuntimeError(f"{what} {len(jobs)}개 가운데 {len(out)}개만 됐습니다({err}). 다시 실행하면 된 것은 건너뜁니다")
    return out


def worlds(k, model, n, tick=None):
    """세상 n 개의 설계(축·이야기·조건·사건)만 채운다. 이미 설계한 세상은 다시 묻지 않는다. tick(단계, 끝난 수, 전체)."""
    tick = tick or (lambda *a: None)
    d = load(k, model) or {"sig": _sig(k), "model": model, "axes": None, "worlds": {}}
    if not d["axes"]:
        tick("axes", 0, 1)
        d["axes"] = _ask(MAKER, AXES.format(card=A.CARD[k]), model, read_axes)
        _save(k, d)
    ax, W_ = d["axes"], d["worlds"]
    blocks = [b for b in (list(range(s, s + BATCH)) for s in range(0, n, BATCH)) if any(str(i) not in W_ for i in b)]

    def make(b):
        cs = [combo(ax, i) for i in b]
        lines = "\n".join(f"{q + 1}. " + ", ".join(f"{a['name']}: {a['states'][x]['s']}" for a, x in zip(ax, c)) for q, c in enumerate(cs))
        ws = _ask(MAKER, WORLD.format(card=A.CARD[k], lines=lines), model, lambda m: read_worlds(m, len(b)))
        return [w | {"pick": c} for w, c in zip(ws, cs)]
    if blocks:
        for b, ws in _all([(tuple(b), lambda b=b: make(b)) for b in blocks], lambda v: tick("design", v * BATCH, len(blocks) * BATCH), "세상 설계").items():
            W_.update({str(i): w for i, w in zip(b, ws)})
        _save(k, d)
    return d


def design(k, model, n, tick=None):
    """세상 설계에 더해, 세상마다 여덟 명의 답까지 채운다."""
    tick = tick or (lambda *a: None)
    d = worlds(k, model, n, tick)
    W_ = d["worlds"]
    todo = [(i, p) for i in range(n) if str(i) in W_ and "people" not in W_[str(i)] for p in crew(i)]
    if todo:
        got = _all([(x, lambda x=x: _live(k, W_[str(x[0])], x[1], model)) for x in todo], lambda v: tick("live", v, len(todo)), "세상 속 사람의 답")
        for i in {x[0] for x in todo}:
            r = [got[(i, p)] for p in crew(i) if (i, p) in got]
            if len(r) >= 5:
                W_[str(i)]["people"] = r
        _save(k, d)
    full = [i for i in range(n) if "people" in W_.get(str(i), {})]
    if len(full) < .9 * n:
        raise RuntimeError(f"세상 {n}개 가운데 {len(full)}개만 완성됐습니다. 다시 실행하면 이어서 합니다")
    return d


# ---------- 엔진에 넘길 값 ----------
def _rank(v, seed):
    """0~1 순위(같은 값은 무작위)."""
    v = np.asarray(v, float)
    o = np.lexsort((np.random.default_rng(seed).random(len(v)), v))
    r = np.empty(len(v))
    r[o] = (np.arange(len(v)) + .5) / len(v)
    return r


def plan(k, d, base, n):
    """세상마다 {pick: 걸러낸 조건 번호, store, mood, pulse}. 설계가 빠진 세상(10% 미만)은 None(통계로 뽑음). base[i]=페르소나 i 의 보통 세상 답."""
    R = W.ready()
    pw = np.exp(R["lw"])
    pw /= pw.sum()
    ids = [i for i in range(n) if "people" in d["worlds"].get(str(i), {})]
    ws = [d["worlds"][str(i)] for i in ids]
    names = [c for c in COND if c != "입점"]
    val = np.array([[c[x] for x in names] for c in R["conds"]])
    u = np.empty_like(val)                                                     # 걸러낸 분포에서의 백분위
    for j in range(len(names)):
        o = np.argsort(val[:, j])
        u[o, j] = np.cumsum(pw[o]) - pw[o] / 2
    want = np.column_stack([_rank([w["cond"][x] for w in ws], j) for j, x in enumerate(names)])
    store = _rank([w["cond"]["입점"] for w in ws], 90)
    gap = [float(np.mean([p["p"] - base.get(p["i"], 0) for p in w["people"] if p["i"] in base] or [0])) for w in ws]      # 보통 세상보다 얼마나 더(덜) 사고 싶은가
    mood_v, mo = np.array([c["시장 분위기"] for c in R["conds"]]), None
    mo = np.argsort(mood_v)
    mood = mood_v[mo][np.minimum(np.searchsorted(np.cumsum(pw[mo]), _rank(gap, 91)), len(mo) - 1)]
    out, got = [None] * n, []
    for q, i in enumerate(ids):
        cand = np.random.default_rng([17, i]).choice(len(pw), TRY, p=pw)       # 걸러낸 조건 안에서만 AI 의 설계에 가장 가까운 것
        b = int(cand[((u[cand] - want[q]) ** 2).sum(1).argmin()])
        got.append(u[b])
        e = np.zeros(W.STEPS)
        for x in ws[q]["events"]:
            s = min(W.STEPS - 1, int(x["m"] * W.STEPS / 12))
            e[s:] += .2 * x["e"] * .6 ** np.arange(W.STEPS - s)
        pulse = np.exp(e)
        out[i] = {"pick": b, "store": float(store[q]), "mood": float(mood[q]), "pulse": pulse / pulse.mean(), "gap": gap[q]}
    fit = float(np.mean([np.corrcoef(want[:, j], np.array(got)[:, j])[0, 1] for j in range(len(names))])) if len(ids) > 2 else 1.0
    return out, {"worlds": len(ids), "fit": fit, "axes": [a["name"] for a in d["axes"]], "calls": len(ids) * 8}


def get(k, model, n, base, tick=None):
    """(세상마다의 값, 세상마다의 글, 요약). 연결이 없고 저장된 설계도 모자라면 None."""
    if model:
        d = design(k, model, n, tick)
    else:
        d = load(k)
        if not d or sum("people" in d["worlds"].get(str(i), {}) for i in range(n)) < .9 * n:
            return None
    sc, info = plan(k, d, base, n)
    text = [None if x is None else {q: d["worlds"][str(i)][q] for q in ("title", "story", "events", "people", "pick")} for i, x in enumerate(sc)]
    c = A.cast()
    for t in text:
        if t:
            t["talk"] = [{"who": f"{W.AGES[_seg()[p['i']] // 2]} {c.sex.values[p['i']]}", "p": p["p"], "say": p["say"]} for p in t.pop("people")]
    return sc, text, info | {"model": d["model"], "axis": d["axes"]}


def drivers(ws, text, axes, k):
    """AI 가 설계한 상태마다 그 세상들의 구매경험률이 전체보다 얼마나 높고 낮은가(15개 세상 이상)."""
    v, has = np.array([w["buy"][k] for w in ws]), np.array([t is not None for t in text])
    rows = []
    for a, x in enumerate(axes):
        for s, st in enumerate(x["states"]):
            m = np.array([t is not None and t["pick"][a] == s for t in text])
            if m.sum() >= 15 and (has & ~m).sum() >= 15:
                o = v[has & ~m]
                rows.append({"axis": x["name"], "state": st["s"], "n": int(m.sum()), "buy": float(v[m].mean()), "diff": float(v[m].mean() - o.mean()), "se": float(np.sqrt(v[m].var() / m.sum() + o.var() / len(o)))})
    sure = sorted((r for r in rows if abs(r["diff"]) > 2.6 * r["se"]), key=lambda r: -r["diff"])      # 축이 32개 상태라 우연을 줄이려고 높은 기준
    return {"up": [r for r in sure if r["diff"] > 0][:5], "down": [r for r in sure[::-1] if r["diff"] < 0][:5]}


def check():
    ax = read_axes({"axes": [{"name": f"축{j}", "states": [{"s": f"상태{q}", "p": .25} for q in range(4)]} for j in range(8)]})
    assert combo(ax, 3) == combo(ax, 3) and len({tuple(combo(ax, i)) for i in range(500)}) > 490
    assert [int(_seg()[p]) for p in crew(5)] == list(range(8))
    g, n = np.random.default_rng(0), 120
    d = {"sig": "", "model": "시험", "axes": ax, "worlds": {}}
    hot = g.random(n) < .5
    for i in range(n):
        c = {x: float(g.integers(0, 101)) for x in COND}
        d["worlds"][str(i)] = {"pick": [int(hot[i]), *combo(ax, i)[1:]], "title": "t", "story": "s", "cond": c, "events": [{"m": 6, "t": "e", "e": 2}], "people": [{"i": p, "p": 80 if hot[i] else 20, "say": "말"} for p in crew(i)]}
    sc, info = plan("화채", d, {i: 50 for i in range(A.N)}, n)
    assert info["worlds"] == n and info["fit"] > .4 and all(abs(x["pulse"].mean() - 1) < 1e-9 for x in sc)
    S, Z = W.run(n, workers=1, sc=sc), W.run(n, workers=1)
    b = np.array([w["buy"]["화채"] for w in S])
    assert b[hot].mean() > b[~hot].mean() * 1.1 and abs(b.mean() / np.mean([w["buy"]["화채"] for w in Z]) - 1) < .25, (b[hot].mean(), b[~hot].mean(), np.mean([w["buy"]["화채"] for w in Z]))
    assert W.one(4, sc=sc[4])["buy"] == S[4]["buy"]
    text = [{"pick": d["worlds"][str(i)]["pick"]} for i in range(n)]
    dr = drivers(S, text, ax, "화채")
    assert not dr["up"] or dr["up"][0]["axis"] == "축0" and dr["up"][0]["state"] == "상태1"
    for bad in ({"axes": []}, {"worlds": []}):
        try:
            read_axes(bad) if "axes" in bad else read_worlds(bad, 5)
        except ValueError:
            continue
        raise AssertionError(bad)
    print("scenes ok · 뜨거운 세상", round(b[hot].mean(), 4), "식은 세상", round(b[~hot].mean(), 4), "통계만", round(np.mean([w["buy"]["화채"] for w in Z]), 4), "설계 반영", round(info["fit"], 2), dr["up"][:1])


if __name__ == "__main__":
    check()
