"""현장 설문 → 즉시 보정: 실제 사람 100명쯤의 1분 설문으로, 사람이 정했던 '반응 가중치'를 바로잡는다.

1. 묻는 것: 나이·성별·제로 빈도·제로 거부·좋아하는 음료·성향 4문항(모두 KREI 와 같은 문항) + 제품 카드 최대 4장의 구매의향(1~5)
2. 매칭: 응답자마다 답이 비슷한 KREI 실응답자 30명을 찾아 효용 항목(마음의 재료)을 채운다
3. 학습: 반응 가중치 9개·가격 민감도·시장 분위기·제품 고유 매력을 순서형 로짓의 사후분포(사전 = multiverse.PRIOR, 라플라스 근사)로 갱신
   - 한 사람이 여러 장에 답하면 답끼리 닮으므로 가중을 깎는다(설계효과 ICC=.5)
   - 현장 사람은 들떠 있을 수 있어 '시장 분위기'에 지울 수 없는 흔들림(VENUE)을 남긴다
4. 시나리오 비교는 회차마다 이 사후분포에서 마음을 뽑는다
설문은 '말한 의향'이다. 실제 구매로 얼마나 이어지는지(실구매 전환)는 이 설문으로 배우지 못한다.
실행: python -m mirror.field check | bench
"""
import datetime as dt
import json
import os
import sqlite3
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB, BENCH = os.path.join(ROOT, "data", "field.db"), os.path.join(ROOT, "data", "field_bench.json")
ZERO = ["경험없음", "월1회미만", "월1회", "월2~3회", "주1회+"]
FAV = ["탄산", "제로탄산", "탄산수", "주스", "커피", "차", "이온", "기능성", "유제품", "두유", "전문점", "기타", "물만"]
IT = {"it_new": ("새로운 음식을 먹어보는 것을 좋아한다", 5), "it_thr": ("식품을 살 때 가격 대비 품질을 체크하는 편이다", 5), "it_hea": ("건강에 관심이 많다", 5), "it_sns": ("SNS를 얼마나 쓰나요 (1 전혀 · 2 가끔 · 3 자주)", 3)}
CSV = {"나이": "age", "성별": "sex", "제로빈도": "zero", "제로거부": "reject", "좋아하는음료": "fav", "새로움": "it_new", "가성비": "it_thr", "건강": "it_hea", "SNS": "it_sns"}
MAXCARD, VENUE, ICC = 4, .25, .5
CUTS = np.array([-np.inf, -2.0, -0.7, 0.6, 1.8, np.inf])         # consumer.likert_pmf 와 같은 경계


# ---------- 저장 ----------
def db(path=None):
    c = sqlite3.connect(path or DB, timeout=10)
    c.execute("create table if not exists study(id text primary key, ts text, concept text, scs text)")
    c.execute("create table if not exists resp(id text primary key, study text, ts text, src text, ans text, rate text)")
    return c


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def make_study(concept, scs, path=None):
    with db(path) as c:
        i = uuid.uuid4().hex[:8]
        c.execute("insert into study values(?,?,?,?)", (i, now(), concept, json.dumps(scs, ensure_ascii=False)))
    return i


def study(i=None, path=None):
    """지정한 설문(없으면 가장 최근) → dict 또는 None."""
    with db(path) as c:
        r = c.execute("select * from study " + ("where id=?" if i else "order by ts desc, rowid desc limit 1"), (i,) if i else ()).fetchone()
    return {"id": r[0], "ts": r[1], "concept": r[2], "scs": json.loads(r[3])} if r else None


def valid(a, rate, scs):
    """외부 입력 검사 → (정리된 답, 정리된 평점). 잘못되면 ValueError."""
    o = {"age": int(a["age"]), "sex": str(a["sex"]), "zero": str(a["zero"]), "reject": int(bool(a["reject"])), "fav": str(a["fav"]), **{k: int(a[k]) for k in IT}}
    if not 14 <= o["age"] <= 79 or o["sex"] not in ("남자", "여자") or o["zero"] not in ZERO or o["fav"] not in FAV or any(not 1 <= o[k] <= m for k, (_, m) in IT.items()):
        raise ValueError("답의 값이 범위를 벗어났습니다")
    q = {k: int(v) for k, v in rate.items() if k in scs}
    if not q or any(not 1 <= v <= 5 for v in q.values()):
        raise ValueError("구매의향은 1~5여야 합니다")
    return o, q


def save(st, a, rate, src="폰", path=None):
    o, q = valid(a, rate, st["scs"])
    with db(path) as c:
        c.execute("insert into resp values(?,?,?,?,?,?)", (uuid.uuid4().hex, st["id"], now(), src, json.dumps(o, ensure_ascii=False), json.dumps(q, ensure_ascii=False)))


def load(sid, path=None, src=None):
    with db(path) as c:
        rs = c.execute("select ans, rate, src from resp where study=?", (sid,)).fetchall()
    rs = [r for r in rs if src is None or r[2] in src]
    return pd.DataFrame([json.loads(r[0]) for r in rs]), [json.loads(r[1]) for r in rs]


def template(scs):
    return pd.DataFrame([{"나이": 23, "성별": "여자", "제로빈도": "월2~3회", "제로거부": "아니오", "좋아하는음료": "탄산", "새로움": 4, "가성비": 3, "건강": 3, "SNS": 3, **{k: 4 for k in scs}}])


def from_csv(t, st, path=None):
    """구글 폼 등에서 내려받은 표 → 저장. (성공 수, 오류 목록)"""
    miss = [c for c in CSV if c not in t]
    if miss:
        return 0, ["빠진 열: " + ", ".join(miss)]
    ok, bad = 0, []
    for i, r in t.iterrows():
        try:
            a = {v: r[k] for k, v in CSV.items()}
            a["reject"] = str(a["reject"]).strip() in ("예", "1", "True", "true", "네")
            save(st, a, {k: r[k] for k in st["scs"] if k in t and pd.notna(r[k])}, "파일", path)
            ok += 1
        except (ValueError, TypeError) as e:
            bad.append(f"{i + 2}행: {e}")
    return ok, bad


# ---------- 매칭 ----------
def match(A, d, k=30):
    """응답자 × base 행렬(행 합 1): 답이 비슷한 실응답자 k 명에 가우스 가중."""
    z = {v: i for i, v in enumerate(ZERO)}
    D = ((A.age.values[:, None] - d.age.values) / 8) ** 2 + 1.5 * (A.sex.to_numpy(object)[:, None] != d.sex.to_numpy(object))
    D = D + ((A.zero.map(z).values[:, None] - d.zero.map(z).values) / 1.5) ** 2 + (A.reject.values[:, None] != d.zero_reject.values) + 1.5 * (A.fav.to_numpy(object)[:, None] != d.fav_cat.to_numpy(object))
    for c, (_, m) in IT.items():
        D = D + .6 * ((A[c].values[:, None] - d[c].values) * (4 / (m - 1))) ** 2
    W, at = np.zeros(D.shape), np.argpartition(D, k, axis=1)[:, :k]
    v = np.exp(-.5 * (np.take_along_axis(D, at, 1) - D.min(1, keepdims=True)))
    np.put_along_axis(W, at, v / v.sum(1, keepdims=True), 1)
    return W


def terms(d, scs, concept):
    """시나리오별 효용 재료 {이름: (N, 9)}."""
    from . import choice as M
    from . import consumer as C
    return {k: np.column_stack(list(C.utility(d, s, M.concept_attrs(concept, s))[1].values())) for k, s in scs.items()}


# ---------- 사후분포 ----------
class Posterior:
    """θ = [로그 반응 가중치 9, 로그 가격 민감도, 시장 분위기, 제품 매력…] 의 정규 근사."""

    def __init__(self, mode, cov, prods, n):
        self.mode, self.cov, self.prods, self.n = mode, cov, prods, n
        self.L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mode)))

    def draw(self, seed):
        r = np.random.default_rng([int(seed), 77])
        t = self.mode + self.L @ r.standard_normal(len(self.mode))
        return {"coef": np.exp(t[:9]), "가격 민감도": float(np.exp(t[9])), "시장 분위기": float(t[10] + r.normal(0, VENUE)), "appeal": dict(zip(self.prods, t[11:]))}


def prior(P_, spread=1.0):
    from . import multiverse as U
    return np.r_[np.full(9, U.PRIOR["반응 가중치"][0]), U.PRIOR["가격 민감도"][0], U.PRIOR["시장 분위기"][0], np.full(P_, U.PRIOR["제품 고유 매력"][0])] * spread


def fit(A, rates, scs, concept, d=None, T=None):
    from scipy.optimize import minimize
    from . import multiverse as U
    d = U.base() if d is None else d
    T = terms(d, scs, concept) if T is None else T
    prods = sorted({U.pkey(s) for s in scs.values()})
    W = match(A, d)
    rows = [(i, k, y) for i, q in enumerate(rates) for k, y in q.items()]
    X, y = np.array([W[i] @ T[k] for i, k, _ in rows]), np.array([v for *_, v in rows])
    pi, sd = np.array([prods.index(U.pkey(scs[k])) for _, k, _ in rows]), prior(len(prods))
    m = np.array([len(rates[i]) for i, *_ in rows])
    tw = 1 / (1 + (m - 1) * ICC)

    def f(t):
        b = np.exp(t[:9])
        Xb = X * b
        Xb[:, 8] *= np.exp(t[9])
        u = Xb.sum(1) + t[10] + t[11:][pi]
        F = lambda c: np.where(np.isinf(c), (c > 0) * 1.0, 1 / (1 + np.exp(-np.clip(c - u, -40, 40))))
        a, e = F(CUTS[y]), F(CUTS[y - 1])
        p = np.maximum(a - e, 1e-12)
        du = -(a * (1 - a) - e * (1 - e)) / p * tw
        g = np.r_[(du[:, None] * Xb).sum(0), (du * Xb[:, 8]).sum(), du.sum(), np.bincount(pi, du, len(prods))]
        return -(tw * np.log(p)).sum() + .5 * ((t / sd) ** 2).sum(), -g + t / sd ** 2
    r = minimize(f, np.zeros(len(sd)), jac=True, method="L-BFGS-B")
    # 피셔 정보(가우스–뉴턴): 항상 사전보다 좁아진다
    b = np.exp(r.x[:9])
    Xb = X * b
    Xb[:, 8] *= np.exp(r.x[9])
    u = Xb.sum(1) + r.x[10] + r.x[11:][pi]
    F = np.where(np.isinf(CUTS), (CUTS > 0) * 1.0, 1 / (1 + np.exp(-np.clip(CUTS - u[:, None], -40, 40))))
    info = (np.diff(F * (1 - F), axis=1) ** 2 / np.maximum(np.diff(F, axis=1), 1e-9)).sum(1) * tw
    J = np.column_stack([Xb, Xb[:, 8], np.ones(len(u)), pi[:, None] == np.arange(len(prods))])
    cov = np.linalg.inv((J * info[:, None]).T @ J + np.diag(1 / sd ** 2))
    return Posterior(r.x, cov, prods, len(A))


def intents(T, scs, w, post=None, draws=200, seed=0):
    """모집단(가중 w) 구매의향(4~5점) 표본: 사전 또는 사후에서 마음을 뽑은 (draws, 시나리오)."""
    from . import consumer as C
    from . import multiverse as U
    w, out = np.asarray(w) / np.sum(w), []
    keep = np.flatnonzero(w > 0)
    for s in range(draws):
        r = np.random.default_rng([seed, s])
        if post is None:
            c, pr, mood, ap = r.lognormal(0, .25, 9), r.lognormal(0, .35), r.normal(0, .35), {}
        else:
            v = post.draw(seed * 1000 + s)
            c, pr, mood, ap = v["coef"], v["가격 민감도"], v["시장 분위기"], v["appeal"]
        c = c * np.r_[np.ones(8), pr]
        out.append([w[keep] @ C.likert_pmf(T[k][keep] @ c + mood + ap.get(U.pkey(sc), U.appeal(s, U.pkey(sc), .25)))[:, 3:].sum(1) for k, sc in scs.items()])
    return np.array(out)


def table(A, rates, scs, concept, w, post):
    """제품별: 설문 그대로 / 설문 전 예측 / 보정 후(대상 모집단)."""
    from . import multiverse as U
    T = terms(U.base(), scs, concept)
    a, b, rows = intents(T, scs, w), intents(T, scs, w, post), []
    for j, k in enumerate(scs):
        v = np.array([q[k] for q in rates if k in q])
        n, p = len(v), float((v >= 4).mean()) if len(v) else np.nan
        se = 1.645 * np.sqrt((p * (1 - p) + 1 / max(4 * n, 1)) / n) if n else np.nan
        lo, hi = np.quantile(b[:, j], [.05, .95])
        l0, h0 = np.quantile(a[:, j], [.05, .95])
        rows.append({"제품": k, "응답": n, "설문 그대로": p, "설문 오차(90%)": se, "설문 전 예측": float(np.median(a[:, j])), "전 하한": l0, "전 상한": h0, "보정 후": float(np.median(b[:, j])), "하한": lo, "상한": hi,
                     "의향 1등일 확률(전)": float((a.argmax(1) == j).mean()), "의향 1등일 확률(후)": float((b.argmax(1) == j).mean())})
    return pd.DataFrame(rows)


# ---------- 검증 ----------
DEMO_SC = {f"{p:,}원·{'제로' if z else '일반'}·{c}": dict(price=p, zero=z, flavor="새로운 맛", package="비주얼(숏폼형)", copy=c, influencer=False) for p, z, c in ((1500, 1, "재미"), (2200, 1, "재미"), (1500, 0, "재미"), (2200, 1, "건강"))}


def venue(d, n, rng):
    """현장 편의표본: 20대·수도권·SNS 쓰는 사람이 많이 잡힌다."""
    p = d.w.values * np.exp(-np.abs(d.age.values - 24) / 8) * np.where(d.reg == "수도권", 1.8, 1) * np.where(d.it_sns >= 3, 1.5, 1) * (d.age.values >= 14)
    return rng.choice(len(d), n, replace=False, p=p / p.sum())


def answers(d, ix):
    return pd.DataFrame({"age": d.age.values[ix], "sex": d.sex.values[ix], "zero": d.zero.values[ix], "reject": d.zero_reject.values[ix].astype(int), "fav": d.fav_cat.values[ix], **{c: d[c].values[ix].astype(int) for c in IT}})


def bench_sim(reps=60, ns=(50, 100, 300), seed=0):
    """가상의 '진짜 회차'를 하나 정하고 현장 설문을 흉내 내 되찾는지 본다. 진짜 회차에는 모델이 모르는 것도 넣는다(개인차, 여성의 제로 선호)."""
    from . import consumer as C
    from . import multiverse as U
    d, rows = U.base(), []
    T, w = terms(d, DEMO_SC, "자두맛 탄산음료"), d.w.values / d.w.sum()
    a = intents(T, DEMO_SC, w, draws=200)
    for r in range(reps):
        g = np.random.default_rng([seed, r])
        c, mood, ap = g.lognormal(0, .25, 9) * np.r_[np.ones(8), g.lognormal(0, .35)], g.normal(0, .35), {U.pkey(s): g.normal(0, .25) for s in DEMO_SC.values()}
        extra = g.normal(0, .4) * (d.sex.values == "여자")
        u = {k: T[k] @ c + mood + ap[U.pkey(s)] + extra * s["zero"] for k, s in DEMO_SC.items()}
        hid = np.random.default_rng([seed, r, 1]).normal(0, .7, len(d))                         # 개인 고유의 후함·짬
        true = np.array([w @ C.likert_pmf(v + hid)[:, 3:].sum(1) for v in u.values()])
        for n in ns:
            ix = venue(d, n, np.random.default_rng([seed, r, n]))
            y = [{k: int(1 + (np.random.default_rng([seed, r, n, i, j]).random() > C.likert_pmf(v[[i]] + hid[i]).cumsum(1)[0]).sum()) for j, (k, v) in enumerate(u.items())} for i in ix]
            # 자기 자신이 매칭되지 않게 설문 응답자를 base 에서 뺀다
            keep = np.setdiff1d(np.arange(len(d)), ix)
            po = fit(answers(d, ix), y, DEMO_SC, "자두맛 탄산음료", d.iloc[keep].reset_index(drop=True), {k: v[keep] for k, v in T.items()})
            b = intents(T, DEMO_SC, w, po, draws=200, seed=r)
            raw = np.array([np.mean([q[k] >= 4 for q in y]) for k in DEMO_SC])
            for nm, est, smp in (("설문 전 예측", np.median(a, 0), a), ("설문 그대로", raw, None), ("보정 후", np.median(b, 0), b)):
                lo, hi = (np.quantile(smp, [.05, .95], axis=0) if smp is not None else (raw - 1.645 * np.sqrt(raw * (1 - raw) / n), raw + 1.645 * np.sqrt(raw * (1 - raw) / n)))
                rows.append((n, nm, np.abs(est - true).mean(), ((lo <= true) & (true <= hi)).mean(), float(est.argmax() == true.argmax()), np.abs((est - est.mean()) - (true - true.mean())).mean()))
    t = pd.DataFrame(rows, columns=["응답자", "방법", "의향 오차", "90% 범위에 진실", "1등 맞힘", "제품 간 차이 오차"])
    return t.groupby(["응답자", "방법"], sort=False).mean().reset_index()


def bench_real(n=100, reps=10, seed=0):
    """실제 자료 시험: 현장처럼 치우친 KREI 응답자 n 명의 실제 답으로 전국 평균(1~5점)을 맞히나(설문에 없는 문항들)."""
    from . import multiverse as U
    from . import trust as TR
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    P, items, _ = TR.data()
    d, p = U.base(), P[P.year == 2025].reset_index(drop=True)
    d = d.iloc[:len(p)].reset_index(drop=True)
    assert (d.age.values == p.age.values).all()
    X = pd.concat([pd.get_dummies(d.fav_cat), pd.DataFrame({"a": (d.age - 45) / 15, "a2": ((d.age - 45) / 15) ** 2, "s": d.sex == "여자", "z": d.zero.map({v: i for i, v in enumerate(ZERO)}) - 2, "r": d.zero_reject}), d[list(IT)] - 3], axis=1).astype(float).values
    ask = [q for q in items if max(abs(np.corrcoef(p[q], d[c])[0, 1]) for c in IT) < .98]      # 설문에서 직접 묻는 문항은 뺀다
    w, band, rows = d.w.values, p.g.values, []
    for r in range(reps):
        ix = venue(d, n, np.random.default_rng([seed, r]))
        gw = pd.Series(w).groupby(band).sum() / w.sum()
        pw = (pd.Series(band[ix]).map(gw) / pd.Series(band[ix]).map(pd.Series(band[ix]).value_counts())).values
        for q in ask:
            y, t = p[q].values, np.average(p[q], weights=w)
            mk = lambda: RidgeCV(np.logspace(0, 3, 7))
            cf = np.empty(n)
            for tr, te in KFold(5, shuffle=True, random_state=r).split(ix):
                cf[te] = mk().fit(X[ix[tr]], y[ix[tr]]).predict(X[ix[te]])
            md = np.average(mk().fit(X[ix], y[ix]).predict(X), weights=w)
            rows.append((y[ix].mean() - t, np.average(y[ix], weights=pw) - t, md - t, md + np.average(y[ix] - cf, weights=pw) - t))
    e = pd.DataFrame(rows, columns=["현장 응답 그대로", "연령·성별 가중", "매칭 모델", "매칭 모델+잔차 보정"])
    return {"문항": len(ask), "n": n, "오차(1~5점, RMSE)": np.sqrt((e ** 2).mean()).round(4).to_dict()}


def check():
    import tempfile
    from . import multiverse as U
    p = os.path.join(tempfile.mkdtemp(), "t.db")
    st = study(make_study("자두맛 탄산음료", DEMO_SC, p), p)
    d = U.base()
    a, n = answers(d, venue(d, 60, np.random.default_rng(0))), from_csv(template(DEMO_SC), st, p)
    assert n == (1, []) and from_csv(template(DEMO_SC).assign(나이=9), st, p)[0] == 0 and from_csv(template(DEMO_SC).drop(columns="SNS"), st, p)[0] == 0
    for r in a.to_dict("records"):
        save(st, r, {k: 3 for k in DEMO_SC}, path=p)
    A, q = load(st["id"], p)
    assert len(A) == 61 and len(load(st["id"], p, ("파일",))[1]) == 1
    W = match(A, d)
    assert np.allclose(W.sum(1), 1) and ((W > 0).sum(1) <= 30).all()
    # 모두 5점이면 의향이 크게 오르고, 모두 1점이면 내려가며, 응답이 많을수록 사후가 좁아진다
    ks = list(DEMO_SC)
    T, w = terms(d, DEMO_SC, "자두맛 탄산음료"), d.w.values
    hi, lo = (fit(A, [{k: v for k in ks} for _ in range(len(A))], DEMO_SC, "자두맛 탄산음료") for v in (5, 1))
    m0, m1, m2 = (np.median(intents(T, DEMO_SC, w, x, 60), 0).mean() for x in (None, hi, lo))
    assert m2 < m0 < m1, (m2, m0, m1)
    g = np.random.default_rng(1)
    few, many = (fit(A.iloc[:m], [{k: int(g.integers(2, 5)) for k in ks} for _ in range(m)], DEMO_SC, "자두맛 탄산음료") for m in (10, 60))
    assert np.trace(many.cov) < np.trace(few.cov) < (prior(len(few.prods)) ** 2).sum()
    W1, _ = U.run({}, DEMO_SC, "자두맛 탄산음료", worlds=12, n=600, post=hi)
    W0, _ = U.run({}, DEMO_SC, "자두맛 탄산음료", worlds=12, n=600)
    assert W1["의향|" + ks[0]].mean() > W0["의향|" + ks[0]].mean()
    print("field ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["bench"]:
        t, r = bench_sim(), [{"응답자": n, **bench_real(n, 8)["오차(1~5점, RMSE)"]} for n in (50, 100, 300)]
        print(t.round(3).to_string(), "\n", r)
        json.dump({"sim": t.round(4).to_dict("records"), "real": r}, open(BENCH, "w"), ensure_ascii=False, indent=1)
    else:
        check()
