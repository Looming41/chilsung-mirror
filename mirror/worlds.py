"""통합 엔진: 세상마다 사람 600명이 살고, 패널에서 배운 의향 · 살 수 있는 점포 · 입소문이 한 사회 안에서 함께 움직인다.

세상 하나
  사람   KREI 19~59세 실응답자를 베이지안 부트스트랩으로 다시 뽑음
  의향   패널 응답자를 재표집해 다시 배운 모델(200벌 중 한 벌) → 그 재표집의 보정 TOP2 에 수준을 맞춤(유도 문항은 범위로 깎음)
  시장   컨셉별로 살 수 있는 비율(채널 × 냉동·냉장 인프라 × 입점률)만큼의 사람만 살 수 있음
  사회   광고와 친구 추천이 28구간(한 구간 = 13일, 합쳐서 364일)에 걸쳐 퍼짐
걸러내기  조건 후보 여러 벌로 과거 5개 제품(의향·취급점포율)을 똑같이 돌려, 실제 구매경험률을 잘 재현하는 조건에 비중을 준다.
          과거 응답자 자료는 없어 의향의 개인차는 가정(HET)이다. 5개 제품뿐이고 모두 가상 데이터다.
실행: python -m mirror.worlds check | report
"""
import hashlib
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from . import given as G
from . import multiverse as U
from . import people as P
from . import viral as V

IDS = {"화채": "밀키스 화채", "프리즈": "레쓰비 프리즈", "쥬시": "쥬시모먼트"}
LOOK = {"화채": ("#E60012", "circle"), "프리즈": ("#EFEBE4", "square"), "쥬시": ("#8D99A3", "diamond")}
N, STEPS, BOOT, CAND, HET, TAU = 600, 28, 200, 6000, 1.0, .25
LABEL = {"가격 민감도": ("가격에 둔감함", "가격에 민감함"), "시장 분위기": ("시장이 가라앉음", "시장이 활발함"), "실구매 전환": ("구매로 덜 이어짐", "구매로 잘 이어짐"), "광고 도달": ("광고가 적음", "광고가 많음"),
         "SNS 전파력": ("SNS 전파가 약함", "SNS 전파가 강함"), "유행 지속": ("유행이 짧음", "유행이 김"), "게시 성향": ("글을 잘 안 올림", "글을 자주 올림"), "악평 영향": ("악평 영향이 큼", "악평 영향이 작음"),
         "동조": ("남을 덜 따라 함", "남을 잘 따라 함"), "피로": ("싫증이 늦음", "싫증이 빠름"), "입점": ("파는 곳이 적음", "파는 곳이 많음")}
COND = list(LABEL)
PREP, CACHE = (os.path.join(P.ROOT, "data", f) for f in ("worlds_prep.pkl", "worlds_run.pkl"))
AGES, SEXES, REGS = ["20대", "30대", "40대", "50대"], ["여자", "남자"], ["수도권", "중부", "영남", "호남·제주"]


def _sig():
    h = hashlib.sha1()
    for f in (__file__, G.__file__, V.__file__, U.__file__):
        h.update(open(f, "rb").read())
    return h.hexdigest()


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------- 준비(한 번 계산해 저장) ----------
def people():
    from . import youth as Y
    d, (r, _) = U.base(), G.panel()
    keep = d.age.between(19, 59).values
    q = lambda c, v: Y.qmap(d[c].values, d[c].values[keep], np.repeat(r[v].values, np.maximum(1, (r.w * 4).round().astype(int))))
    f = pd.DataFrame({"나이": d.age, "여성": (d.sex == "여자") * 1.0, "신제품": q("novelty", "신제품시도성향"), "건강": q("health", "건강관심도"), "가격민감": q("thrift", "가격민감도")})[keep]
    k = d[keep]
    return pd.DataFrame({"age10": k.age10, "sex": k.sex, "reg": k.reg, "w": k.w, "trend": k.trend, "novelty": k.novelty_q, "thrift": k.thrift_q, "health": k.health_q, "social": k.social_q}).reset_index(drop=True), G.design(f)


def panel_X():
    r, _ = G.panel()
    return G.design(pd.DataFrame({"나이": 2026 - r.출생연도, "여성": (r.성별 == "여성") * 1.0, "신제품": r.신제품시도성향, "건강": r.건강관심도, "가격민감": r.가격민감도}))


def boots(seed=0, extra=None):
    """패널 재표집마다: (컨셉별 로지스틱 계수·절편, 순서 보정 TOP2). extra={컨셉 번호: 패널 응답자마다의 값}이면 그 컨셉에 입력 하나를 더한다(AI 페르소나의 답)."""
    from sklearn.linear_model import LogisticRegression
    g, (r, _), extra = np.random.default_rng(seed), G.panel(), extra or {}
    X = panel_X()
    XX = [np.column_stack([X, extra[j]]) if j in extra else X for j in range(3)]
    T2, ps, out = G.wide("T2"), G.wide("제시순서"), []
    while len(out) < BOOT:
        ix = g.integers(0, len(r), len(r))
        if any(len(np.unique(T2[ix, j])) < 2 for j in range(3)):
            continue
        w = G.rake(r.iloc[ix].reset_index(drop=True))
        ms = [LogisticRegression(C=.5, max_iter=1000).fit(XX[j][ix], T2[ix, j], sample_weight=w) for j in range(3)]
        keep = []
        for j in extra:                                             # 뽑히지 않은 응답자로 잰 기울기: 1보다 작으면 성향의 차이가 부풀려진 것
            o = np.setdiff1d(np.arange(len(r)), ix)
            keep.append(LogisticRegression(C=1e6, max_iter=1000).fit((XX[j][o] @ ms[j].coef_[0] + ms[j].intercept_[0])[:, None], T2[o, j]).coef_[0, 0] if len(np.unique(T2[o, j])) > 1 else 1.0)
        out.append(([m.coef_[0] for m in ms] if extra else np.array([m.coef_[0] for m in ms]), np.array([m.intercept_[0] for m in ms]), G.adjusted(w, T2[ix], ps[ix])[0], *([keep] if extra else [])))
    return out


def draw(r):
    """조건 후보 한 벌. 실구매 전환은 '알게 된 사람이 한 구간에 살 확률'에 곱하는 값."""
    c = U.draw(r)
    c.pop("희소성")
    c["실구매 전환"] = float(np.exp(r.uniform(np.log(.03), np.log(.6))))
    return c


def shift(lp, w, goal):
    """가중 평균 확률이 goal 이 되도록 하는 절편(이분법)."""
    lo, hi = -8.0, 8.0
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if np.average(_sigmoid(lp + m), weights=w) < goal else (lo, m)
    return (lo + hi) / 2


def society(intent, access, trend, c, nb, dg, seed, ai=None):
    """ai 가 있으면 올릴 마음·만족·악평 반응·추천 반응이 사람마다 AI 페르소나의 답을 따른다."""
    post, sat, bad, rec = (ai["post"], ai["sat"], np.clip(c["악평 영향"], .01, 1) ** ai["bad"], ai["rec"]) if ai else (1.0, np.full(len(intent), .82), c["악평 영향"], 1.0)
    return V.simulate(intent * c["실구매 전환"] * access, np.clip((.04 + .45 * trend) * post * c["게시 성향"], 0, 1), sat, nb, dg, days=STEPS, ad_reach=min(.9, c["광고 도달"] * 4),
                      shortform=c["SNS 전파력"] * .85, decay=c["유행 지속"], neg_penalty=bad, conform=c["동조"], fatigue=c["피로"], trait=(.5 + trend) * rec, rng=np.random.default_rng(seed))


def past_fit(i, seed=1, n=N):
    """조건 후보 i 로 과거 5개 제품을 돌린 구매경험률 → (조건, 5개 값)."""
    r, t = np.random.default_rng([seed, i]), G.past()
    c = draw(r)
    nb, dg = V.build_network(n, seed=i % 64)
    u, z, tr, out = r.random(n), r.normal(0, HET, n), r.beta(2, 2, n), []
    for x in t.itertuples():
        p = _sigmoid(z + shift(z, np.ones(n), x.TOP2) + c["시장 분위기"])
        out.append(society(p, (u < x.점포율) * 1.0, tr, c, nb, dg, [seed, i, 0])["adopters"] / n)
    return c, out


def prep(tick=None, cand=CAND):
    s = _sig()
    if os.path.exists(PREP):
        with open(PREP, "rb") as f:
            d = pickle.load(f)                                   # 이 컴퓨터에서 이 코드가 만든 파일만 읽는다
        if d["sig"] == s and d["cand"] == cand:
            return d
    pe, X = people()
    pe["_seg"] = [AGES.index(a) * 2 + SEXES.index(b) for a, b in zip(pe.age10, pe.sex)]
    lev = {t: pe[t].map({"낮음": 0, "중하": 1, "중상": 2, "높음": 3}).values.astype(int) for t in TRAIT}
    res = []
    with ProcessPoolExecutor() as ex:
        for z in ex.map(past_fit, range(cand), chunksize=50):
            res.append(z)
            if tick and len(res) % 100 == 0:
                tick(len(res) / cand)
    act, sim = np.log(G.past().경험률.values), np.log(np.maximum([x[1] for x in res], .5 / N))
    lw = -.5 * (((sim - act) / TAU) ** 2).sum(1)
    d = {"sig": s, "cand": cand, "people": pe, "X": X, "levels": lev, "boots": boots(), "conds": [x[0] for x in res], "lw": lw - lw.max(), "sim": sim, "reach": {k: 1.0 if G.SPEC[v]["infra"] is None else G.reach(v) for k, v in IDS.items()}}
    with open(PREP, "wb") as f:
        pickle.dump(d, f)
    return d


@lru_cache(1)
def ready():
    return prep()


# ---------- 세상 ----------
def weights(flt):
    d = ready()["people"]
    w = d.w.values.copy()
    for c, v in (flt or {}).items():
        if v:
            w = w * d[c].isin(v).values
    return w


def one(i, flt=None, seed=7, log=None, ai=None):
    """세상 i → {id, 조건(원값), intent, buy, curve, seen}. log=컨셉이면 사람·관계·날짜 기록을 함께 준다. ai 는 agents.effects 의 값(그 컨셉에만 쓴다)."""
    d, r = ready(), np.random.default_rng([seed, i])
    pe, X, w0 = d["people"], d["X"], d["people"].w.values
    pw = np.exp(d["lw"])
    c = dict(d["conds"][r.choice(len(pw), p=pw / pw.sum())])
    for k in ("실구매 전환", "광고 도달", "게시 성향", "가격 민감도", "SNS 전파력"):              # 걸러낸 조건 주변을 조금씩 흔든다(똑같은 세상이 반복되지 않게)
        c[k] *= float(np.exp(r.normal(0, .06)))
    for k in ("유행 지속", "악평 영향", "동조", "피로", "시장 분위기"):
        c[k] += float(r.normal(0, .015))
    c["동조"], c["피로"], c["유행 지속"] = max(0, c["동조"]), max(0, c["피로"]), float(np.clip(c["유행 지속"], .2, .9))
    c["입점"] = float(r.uniform())                                  # 입점은 과거 제품이 아니라 이 컨셉들의 가정
    bt = int(r.integers(BOOT))
    B, a, top = d["boots"][bt]
    lead = r.uniform(*G.ASSUME["유도문항|쥬시모먼트"][:2])
    own = np.exp(r.normal(0, float(np.log(G.past().k).std(ddof=1)) * np.sqrt(1.2), 3))      # 제품마다 다른 전환(과거 5개 제품의 차이만큼)
    g = weights(flt) * r.exponential(size=len(w0))
    ix = r.choice(len(w0), N, p=g / g.sum())
    nb, dg = V.build_network(N, seed=int(r.integers(1 << 30)))
    ix = ix[np.argsort(-pe.trend.values[ix] + r.normal(0, .15, N))][np.argsort(np.argsort(-dg))]      # 친구 많은 자리에 유행 민감한 사람(잡음 섞음)
    tr, u = pe.trend.values[ix], r.random(N)
    out = {"id": i, "cond": c, "intent": {}, "buy": {}, "curve": {}, "seen": {}, "agg": {}}
    sg = d["people"]["_seg"].values[ix]
    for j, (k, con) in enumerate(IDS.items()):
        lo, hi = G.ASSUME["입점률|" + con][:2]
        acc = d["reach"][k] * (lo + (hi - lo) * c["입점"])
        on = ai if ai and ai["k"] == k else None
        b = on["B"][bt][:-1] if on else B[j]
        lp = X @ b + (on["s"] * on["B"][bt][-1] + on["a"][bt] if on else a[j])
        mine = None
        if on:                                                       # 사람마다 페르소나 한 명이 맡는다. 성향의 차이는 패널이 확인해 준 만큼(keep)만 남기고, 빠진 자리를 그 페르소나만의 차이로 채운다(전체 크기는 그대로)
            pick = np.random.default_rng([seed, i, 2]).integers(0, on["m"].shape[1], len(w0))
            who, row = on["m"][np.arange(len(w0)), pick], np.arange(len(w0))
            mid = np.average(lp, weights=w0)
            sd = np.sqrt(np.average((lp - mid) ** 2, weights=w0))
            lp = mid + on["keep"] * (lp - mid) + on["own"][row, pick] * sd * np.sqrt(1 - on["keep"] ** 2)
            who = who[ix]
            mine = {x: on[x][who] for x in ("post", "sat", "bad", "rec")}
        z = lp + shift(lp, w0, float(np.clip(top[j] * (lead if k == "쥬시" else 1), .01, .99)))
        # 가격 민감도: 가격이 1,800원보다 비쌀수록, 알뜰한 사람일수록 더 깎임(세기는 가정)
        price = -.5 * np.log(c["가격 민감도"]) * (G.SPEC[con]["srp"] / 1800 - 1) * (1 + .5 * X[:, 6])
        p = _sigmoid(z + price + c["시장 분위기"])[ix]
        o = society(p * own[j], (u < acc) * 1.0, tr, c, nb, dg, [seed, i, 1], mine)
        cum = np.cumsum(o["buy"]) / N
        out["intent"][k], out["buy"][k] = float(p.mean()), float(cum[-1])
        out["curve"][k] = [0.0, *np.round(cum, 4).tolist()]
        out["seen"][k] = float((o["log"]["seen"] >= 0).mean())
        code = why(o, X[ix], b, u < acc, p)
        got = o["first_day"] >= 0
        out["agg"][k] = {"seg": np.column_stack([np.bincount(sg, minlength=8), np.bincount(sg, got, 8)]).astype(int).tolist(), "code": np.bincount(sg * len(CODES) + code[0], minlength=8 * len(CODES)).reshape(8, -1).tolist(),
                         "trait": [np.column_stack([np.bincount(d["levels"][t][ix], minlength=4), np.bincount(d["levels"][t][ix], got, 4)]).astype(int).tolist() for t in TRAIT]}
        if on:                                                       # 산 사람이 말한 살 이유, 알고도 안 산 사람이 말한 망설일 이유
            no = ~got & (o["log"]["seen"] >= 0)
            out["agg"][k]["ai"] = [np.bincount(on["buy"][who][got], minlength=8).tolist(), np.bincount(on["hold"][who][no], minlength=8).tolist()]
        if log == k:
            out["log"] = detail(o, pe.iloc[ix].reset_index(drop=True), X[ix], b, u < acc, p, nb, dg)
            if on:
                for x, q in zip(out["log"]["nodes"], who):
                    x["say"] = on["text"][q][0 if x["bought"] else 1]
    return out


TRAIT = {"novelty": ("새로운 맛을 즐김", "익숙한 것을 고름"), "thrift": ("알뜰하게 비교함", "가격을 덜 따짐"), "health": ("건강을 챙김", "건강은 덜 따짐"), "social": ("SNS를 자주 씀", "SNS를 덜 씀")}
GOOD, POOR = ["나이·성별이 좋아하는 쪽", "새 제품을 시도하는 편", "건강을 챙기는 편", "가격 감각이 맞는 편"], ["나이·성별상 덜 끌림", "새 제품을 덜 시도하는 편", "건강 관심이 맞지 않음", "가격이 부담되는 편"]


CODES = ["buy_ad", "buy_friend", "unaware", "noplace", "bad", "poor", "cold"]


def why(o, X, b, ok, p):
    """한 사람씩의 까닭 → (CODES 번호, 성향 묶음 기여 4열)."""
    L, con = o["log"], X * b
    con = con - con.mean(0)
    grp = np.column_stack([con[:, :4].sum(1), con[:, 4], con[:, 5], con[:, 6]])
    code = np.where(grp.min(1) < -.1, 5, 6)
    code = np.where((L["neg_seen"] > 0) & (p > .3), 4, code)
    code = np.where(~ok, 3, code)
    code = np.where(L["seen"] < 0, 2, code)
    return np.where(o["first_day"] >= 0, np.where(L["friend"], 1, 0), code), grp


def detail(o, s, X, b, ok, p, nb, dg):
    """한 사람씩의 기록. 이유는 모델 안에서 가장 크게 작용한 항목이다(실제 사람의 말이 아님)."""
    L, nodes = o["log"], []
    code, grp = why(o, X, b, ok, p)
    q = lambda v: int(v) + 1 if v >= 0 else None
    for i, r in enumerate(s.itertuples()):
        lv = {k: {"낮음": -2, "중하": -1, "중상": 1, "높음": 2}[getattr(r, k)] for k in TRAIT}
        k = max(lv, key=lambda x: abs(lv[x]))
        buy, seen = o["first_day"][i], L["seen"][i]
        t = {0: "광고를 봄 · " + GOOD[int(grp[i].argmax())], 1: "친구 추천 글을 봄 · " + GOOD[int(grp[i].argmax())], 2: "광고도 추천 글도 보지 못함", 3: "근처에 파는 곳이 없음", 4: "불만 글을 봄", 5: POOR[int(grp[i].argmin())], 6: "사고 싶은 마음이 크지 않음"}[int(code[i])]
        nodes.append({"id": i, "age": r.age10, "sex": r.sex, "followers": int(dg[i]) * 12, "friends": int(dg[i]), "trait": TRAIT[k][0 if lv[k] > 0 else 1], "exposed": q(seen), "via": ("friend" if L["friend"][i] else "ad") if seen >= 0 else None,
                      "bought": q(buy), "posted": q(L["post"][i]), "positive": bool(L["pos"][i]) if L["post"][i] >= 0 else None, "access": bool(ok[i]), "why": t})
    return {"nodes": nodes, "links": [[i, int(j)] for i in range(len(nb)) for j in nb[i] if i < j]}


def _job(a):
    return one(*a)


_pool = None


def run(n=1500, flt=None, seed=7, tick=None, workers=None, ai=None):
    """세상 n 개. cond 는 0~1 백분위, raw 는 원값."""
    global _pool
    ready()
    ai = {k: v for k, v in ai.items() if k not in ("text", "info")} if ai else None      # 집계에 안 쓰는 글은 나눠 주지 않는다
    if workers == 1:
        it = map(_job, [(i, flt, seed, None, ai) for i in range(n)])
    else:
        _pool = _pool or ProcessPoolExecutor()
        it = _pool.map(_job, [(i, flt, seed, None, ai) for i in range(n)], chunksize=10)
    W = []
    for x in it:
        W.append(x)
        if tick:
            tick(len(W))
    rk = {c: (pd.Series([w["cond"][c] for w in W]).rank(method="first").values - .5) / n for c in COND}
    for k, w in enumerate(W):
        w["raw"], w["cond"] = w["cond"], {c: round(float(rk[c][k]), 4) for c in COND}
        w["winner"] = max(w["buy"], key=w["buy"].get)
    return W


def default(tick=None):
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        if d["sig"] == _sig():
            return d["W"]
    W = run(tick=tick)
    with open(CACHE, "wb") as f:
        pickle.dump({"sig": _sig(), "W": W}, f)
    return W


def cross():
    """연령×성별×권역 칸: 인구, 응답자, 가중치 제곱합(유효표본 계산용)."""
    t = ready()["people"].assign(w2=lambda x: x.w ** 2).groupby(["age10", "sex", "reg"]).agg(p=("w", "sum"), n=("w", "size"), q=("w2", "sum")).reset_index()
    return [[r.age10, r.sex, r.reg, round(r.p), int(r.n), round(float(r.q))] for r in t.itertuples()]


def check():
    d = ready()
    assert len(d["people"]) == len(d["X"]) > 3000 and len(d["boots"]) == BOOT
    ess = np.exp(d["lw"]).sum() ** 2 / np.exp(2 * d["lw"]).sum()
    assert ess > 50, ess
    a, b = one(3), one(3)
    assert a["buy"] == b["buy"] and set(a["buy"]) == set(IDS)                                  # 같은 번호면 같은 세상
    L = one(3, log="화채")
    nd = L["log"]["nodes"]
    assert abs(sum(x["bought"] is not None for x in nd) / N - L["buy"]["화채"]) < 1e-9 and L["buy"] == a["buy"]
    assert all(x["access"] for x in nd if x["bought"]) and all(x["exposed"] and x["exposed"] <= x["bought"] for x in nd if x["bought"])
    W = run(24, workers=1)
    assert all(0 <= v <= 1 for w in W for v in w["cond"].values()) and all(w["curve"]["화채"][-1] == round(w["buy"]["화채"], 4) for w in W)
    w = run(24, {"age10": ["20대"]}, workers=1)
    assert np.mean([x["intent"]["화채"] for x in w]) > np.mean([x["intent"]["화채"] for x in W])   # 20대는 화채 의향이 더 높다(패널)
    assert sum(x[3] for x in cross()) > 25e6
    print("worlds ok, 걸러낸 조건의 유효 수", round(ess))


def report():
    d = ready()
    pw = np.exp(d["lw"])
    pw /= pw.sum()
    t = G.past()[["제품", "경험률"]].copy()
    t["걸러내기 전(중앙값)"], t["걸러낸 뒤(가중 평균)"] = np.exp(np.median(d["sim"], 0)), np.exp((d["sim"] * pw[:, None]).sum(0))
    print(t.round(3).to_string(), "\n유효 조건 수", round(1 / (pw ** 2).sum()), "/", len(pw))
    W = default()
    B = pd.DataFrame([w["buy"] for w in W])
    print(B.quantile([.05, .5, .95]).round(4).to_string(), "\n1등", pd.Series([w["winner"] for w in W]).value_counts(normalize=True).round(3).to_dict(), "본 사람", pd.DataFrame([w["seen"] for w in W]).median().round(2).to_dict())
    pr = pd.Series({k: G.SPEC[v]["srp"] for k, v in IDS.items()})
    print("매출 1등", (B * pr).idxmax(axis=1).value_counts(normalize=True).round(3).to_dict())
    print("신제품 검증 계산과 비교\n", G.summary(G.forecast(1500))[["컨셉", "구매경험률", "하한", "상한"]].round(4).to_string())


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else check()
