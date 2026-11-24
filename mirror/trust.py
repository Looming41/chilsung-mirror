"""신뢰 엔진: 합성 패널을 '사람 조사끼리의 차이'와 비교하고, 적은 실제 응답을 모델로 얼마나 늘릴 수 있는지 잰다.

① 재현 시험(Electric Twin 식): 2023~24 조사로 만든 합성 패널이 2025 조사를 얼마나 재현하나.
   기준은 사람 조사끼리의 차이(2024 ↔ 2025)와, 같은 조사를 반으로 나눴을 때의 우연 차이.
② 질문별 신뢰 등급(Simile 식): 평균이 좋다고 특정 답이 믿을 만한 건 아니므로 문항마다 등급을 매긴다.
③ 사람+모델 배합(Expected Parrot 식): 새 질문을 실제 m 명에게만 물었을 때 모델이 몇 배 표본의 효과를 내나.
   예측 기반 추론(PPI)처럼 실제 응답의 교차적합 잔차로 전체 수준을 바로잡는다.
실행: python -m mirror.trust check | bench
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import choice as M

FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trust.json")
BANDS, NAMES = [18, 29, 39, 49, 59, 69, 80], ["20대", "30대", "40대", "50대", "60대", "70대"]
SYN, HUM = "합성(2023~24→2025)", "사람(2024↔2025)"


def data():
    P = M.panel()
    P["band"] = pd.cut(P.age, BANDS, labels=NAMES).astype(str)
    P["g"] = P.band + P.sex
    items = [c for c in P if c.startswith("q") and c[1:].isdigit()]
    P[items] = P[items].fillna(P[items].median())
    return P, items, P.attrs["items"]


def dist(d, q, w=None):
    """집단(연령대×성별)별 1~5 응답 분포."""
    w = d.w if w is None else w
    t = pd.DataFrame({"g": d.g.values, "v": d[q].round().astype(int).values, "w": np.asarray(w)}).pivot_table(index="g", columns="v", values="w", aggfunc="sum", fill_value=0)
    t = t.reindex(columns=range(1, 6), fill_value=0)
    return t.div(t.sum(1), axis=0)


def tvd(a, b, size):
    ix = a.index.intersection(b.index)
    return float(np.average(.5 * (a.loc[ix] - b.loc[ix]).abs().sum(axis=1), weights=size.loc[ix]))


# ---------- ① ② 재현 시험 ----------
def synth(tr, q, n_per=800, seed=0, leaf=40):
    """2023~24 응답자로 '연령·성별·연도' 조건부 응답 분포를 학습해 2025 집단의 응답을 생성(비슷한 집단끼리 정보 공유)."""
    from sklearn.ensemble import RandomForestClassifier
    X = lambda age, sex, yr: np.column_stack([age, sex, yr])
    m = RandomForestClassifier(60, min_samples_leaf=leaf, random_state=seed, n_jobs=-1).fit(X(tr.age, tr.sex == "여자", tr.year), tr[q].round().astype(int), sample_weight=tr.w)
    rng, out = np.random.default_rng(seed), []
    for b, lo, hi in zip(NAMES, BANDS[:-1], BANDS[1:]):
        for s in ("남자", "여자"):
            p = m.predict_proba(X(rng.integers(lo + 1, hi + 1, n_per), np.full(n_per, s == "여자"), np.full(n_per, 2025)))
            out.append(pd.DataFrame({"g": b + s, q: m.classes_[(p.cumsum(1) > rng.random(n_per)[:, None]).argmax(1)]}))
    return pd.concat(out, ignore_index=True)


def replicate(P, items, names):
    rows, y = [], P[P.year == 2025].reset_index(drop=True)
    size = y.groupby("g").w.sum()
    for i, (q, nm) in enumerate(zip(items, names)):
        t, rng = dist(y, q), np.random.default_rng(i)
        h = np.mean([tvd(dist(y[m], q), dist(y[~m], q), size) for m in (rng.random(len(y)) < .5 for _ in range(6))])
        human = tvd(dist(P[P.year == 2024], q), t, size)
        s = synth(P[P.year < 2025], q, seed=i)
        syn = tvd(dist(s, q, np.ones(len(s))), t, size)
        al = (y.groupby(y[q].round().astype(int)).w.sum().reindex(range(1, 6), fill_value=0) / y.w.sum()).values
        flat = pd.DataFrame(np.tile(al, (len(t), 1)), index=t.index, columns=range(1, 6))
        rows.append({"문항": nm.split("_")[-1][:60], "우연(반분)": h, HUM: human, SYN: syn, "모두 평균": tvd(flat, t, size), "사람 대비": syn / human,
                     "등급": "A" if syn <= .10 else "B" if syn <= .15 else "C" if syn <= .22 else "D"})
    return pd.DataFrame(rows)


# ---------- ③ 사람 + 모델 ----------
def onehot(d):
    return np.column_stack([pd.get_dummies(d.band).values, (d.sex == "여자").values, (d.age - 45) / 15, d.sns - 2]).astype(float)


def blend(d, q, lab, X, seed=0):
    """lab 인 사람만 q 를 안다. 능형회귀로 전원을 예측하고 교차적합 잔차로 전체 수준을 보정한 집단별 평균."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    y, w, ix = d[q].values, d.w.values, np.flatnonzero(lab)
    mk = lambda: RidgeCV(alphas=np.logspace(0, 4, 9))
    pr = mk().fit(X[ix], y[ix], sample_weight=w[ix]).predict(X)
    cf = np.empty(len(ix))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(ix):
        cf[b] = mk().fit(X[ix[a]], y[ix[a]], sample_weight=w[ix[a]]).predict(X[ix[b]])
    pr = pr + np.average(y[ix] - cf, weights=w[ix])
    return pd.Series(pr * w).groupby(d.g.values).sum() / pd.Series(w).groupby(d.g.values).sum()


def mix(P, items, ms=(50, 100, 200, 400, 800), reps=6):
    """집단 12개 평균의 오차(1~5점, RMSE). 실제 m 명만 / 인구통계 모델 / 다른 답까지 쓰는 모델."""
    d = P[P.year == 2025].reset_index(drop=True)
    D, Z, size, rows = onehot(d), ((d[items] - d[items].mean()) / d[items].std()).values, d.groupby("g").w.sum(), []
    for j, q in enumerate(items):
        t, A = (d[q] * d.w).groupby(d.g).sum() / size, np.column_stack([D, np.delete(Z, j, axis=1)])
        for m in ms:
            for r in range(reps):
                lab = np.zeros(len(d), bool)
                lab[np.random.default_rng(1000 * j + 10 * m + r).choice(len(d), m, replace=False)] = True
                s = d[lab]
                raw = ((s[q] * s.w).groupby(s.g).sum() / s.groupby("g").w.sum()).reindex(t.index).fillna(np.average(s[q], weights=s.w))
                for nm, e in (("실제 응답만", raw), ("전체 평균만", raw * 0 + np.average(s[q], weights=s.w)), ("+인구통계 모델", blend(d, q, lab, D, r)), ("+기록 모델", blend(d, q, lab, A, r))):
                    rows.append((m, nm, q, float(np.sqrt(np.average((e.reindex(t.index) - t) ** 2, weights=size.loc[t.index])))))
    return pd.DataFrame(rows, columns=["실제 응답", "방법", "q", "오차"])


def equivalent(mx):
    """각 방법의 오차가 '실제 응답만'으로 몇 명을 조사해야 나오는 수준인가(로그-로그 직선)."""
    e = mx.groupby(["방법", "실제 응답"]).오차.mean().unstack(0)
    b, a = np.polyfit(np.log(e.index), np.log(e["실제 응답만"]), 1)
    return e, e.apply(lambda c: np.exp((np.log(c) - a) / b)).round(0)


def bench():
    P, items, names = data()
    r = replicate(P, items, names)
    mx = mix(P, items)
    e, eq = equivalent(mx)
    # 집단 차이가 큰 문항만 따로(차이가 거의 없는 문항은 '전체 평균만'이 당연히 유리)
    big = r.assign(q=items).nlargest(11, "모두 평균").q.tolist()
    e_big, eq_big = equivalent(mx[mx.q.isin(big)])
    s = {"문항 수": len(r), "합성이 사람 차이 이내": float((r[SYN] <= r[HUM]).mean()), "사람 평균": float(r[HUM].mean()), "합성 평균": float(r[SYN].mean()),
         "모두평균 평균": float(r["모두 평균"].mean()), "우연 평균": float(r["우연(반분)"].mean()), "등급": r.등급.value_counts().to_dict()}
    json.dump({"items": r.round(4).to_dict("records"), "mix": e.reset_index().round(4).to_dict("records"), "equiv": eq.reset_index().to_dict("records"), "mix_big": e_big.reset_index().round(4).to_dict("records"), "요약": s}, open(FILE, "w"), ensure_ascii=False, indent=1)
    return e, eq, e_big, eq_big, s


def check():
    rng, n = np.random.default_rng(0), 4000
    d = pd.DataFrame({"age": rng.integers(19, 80, n), "sex": rng.choice(["남자", "여자"], n), "w": 1.0, "sns": rng.integers(1, 4, n).astype(float), "year": 2025})
    d["band"] = pd.cut(d.age, BANDS, labels=NAMES).astype(str)
    d["g"] = d.band + d.sex
    f, it = rng.normal(0, 1, n) + (d.sex == "여자") * .8, [f"q{k:02d}" for k in range(6)]
    for q in it:
        d[q] = np.clip(np.round(3 + f + rng.normal(0, .5, n)), 1, 5)
    same = dist(d, "q00")
    assert tvd(same, same, d.groupby("g").w.sum()) == 0 and np.allclose(same.sum(1), 1)
    e, _ = equivalent(mix(d, it, ms=(60, 240), reps=3))
    assert e.loc[60, "+기록 모델"] < e.loc[60, "실제 응답만"] * .8, e                  # 답끼리 상관이 크면 이득이 커야 한다
    for q in it:
        d[q] = rng.integers(1, 6, n).astype(float)
    e2, _ = equivalent(mix(d, it, ms=(60, 240), reps=3))
    assert e.loc[240, "+기록 모델"] < e.loc[240, "전체 평균만"] * .6 and e2.loc[240, "+기록 모델"] > e2.loc[240, "전체 평균만"] * .9, (e, e2)   # 상관이 없으면 '전체 평균만'보다 나아질 수 없다
    print("trust ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        for x in bench():
            print(x.round(3).to_string() if hasattr(x, "to_string") else x)
    else:
        check()
