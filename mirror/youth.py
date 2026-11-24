"""10대(13~18세): KREI 2025 식품소비행태조사 청소년가구원 604명을 성인 응답자와 같은 열로 맞춘다.

- 성인과 같은 문항: 성·나이·지역·가구소득, 좋아하는 음료, 제로 빈도와 거부 이유, 건강 3문항, SNS 3문항
- 문항 수가 다른 척도(새로움: 성인 3 → 공통 2, 알뜰: 성인 3 → 공통 2)는 공통 문항의 백분위를 성인 척도 값으로 옮긴다
- 제로 수용성은 10대 응답으로 따로 학습한다
- 선택모델의 심리요인(청소년 조사에 없는 문항)은 공통 문항이 비슷한 19~29세 응답자에게서 빌린다
- 성인으로 배운 선택모델은 10대의 음료 1순위를 크게 틀린다(커피를 과대 예측). 10대 604명의 실제 1~3순위 응답으로 속성 선호를 통째로 옮긴다(shift)
12세 이하는 조사에 없다.
실행: python -m mirror.youth check | report
"""
import os
import sys

import numpy as np
import pandas as pd

from . import people as P
from . import records as R

SRC = os.path.join(P.ROOT, "data", "krei_youth.parquet")
XLSX = os.path.join(P.ROOT, "data", "raw", "krei", "(data_가중) 2025 식품소비행태조사_청소년가구원.xlsx")
# (성인 문항, 청소년 문항)
SHARED = {"novelty": [("H1_5", "H1_3"), ("H1_6", "H1_4")], "thrift": [("H1_7", "H1_5"), ("H1_8", "H1_6")],
          "health": [("BA5", "BA5"), ("BA2", "BA2"), ("BA5_1", "BA5_1")], "social": [("D4", "D2"), ("D4_1", "D2_1"), ("D5", "D3")]}
ONE = {"it_new": ("H1_6", "H1_4"), "it_thr": ("H1_8", "H1_6"), "it_hea": ("BA5", "BA5"), "it_sns": ("D4", "D2")}   # 현장 설문에서 묻는 한 문항씩


def raw():
    if not os.path.exists(SRC):
        pd.read_excel(XLSX, sheet_name="Numeric").to_parquet(SRC)
    return pd.read_parquet(SRC)


def items(adult=True):
    """공통 문항 원값 표(성인 또는 청소년). SNS 하위 문항 무응답은 1(전혀)."""
    d, j = (pd.read_parquet(R.SRC), 0) if adult else (raw(), 1)
    cs = {f"{k}{i}": d[pair[j]].astype(float).fillna(1) for k, v in SHARED.items() for i, pair in enumerate(v)}
    return pd.DataFrame(cs), pd.DataFrame({k: d[v[j]].astype(float).fillna(1) for k, v in ONE.items()})


def qmap(x, ref_x, ref_y, seed=0):
    """x 의 (ref_x 안) 백분위를 ref_y 의 값으로 옮긴다. 동점은 무작위로 푼다(계단 때문에 분포가 뭉치지 않게)."""
    r = np.random.default_rng(seed)
    a, b = np.sort(ref_x + r.uniform(-1e-3, 1e-3, len(ref_x))), np.sort(ref_y)
    return np.quantile(b, np.clip(np.searchsorted(a, x + r.uniform(-1e-3, 1e-3, len(x))) / len(a), 0, 1))


def load(adult, Z):
    """adult: 성인 base 표(records.load 순서), Z: 성인 선택모델 특성 → 청소년 표(같은 열)."""
    d, (ya, yo), (aa, _) = raw(), items(False), items(True)
    o = pd.DataFrame({"age": d.SQ2.astype(int), "sex": d.SQ1.map({1: "남자", 2: "여자"}), "province": d["지역"].map(R.PROV), "w": d.psfwt.astype(float)})
    o["income"] = np.nan                                         # 성인의 개인소득 자리(10대는 용돈이라 쓰지 않음)
    for k, v in SHARED.items():
        cols = [f"{k}{i}" for i in range(len(v))]
        if k in ("novelty", "thrift"):
            o[k] = qmap(ya[cols].mean(axis=1).values, aa[cols].mean(axis=1).values, adult[k].values)
        else:
            o[k] = ((ya[cols] - 1) / (2 if k == "social" else 4)).mean(axis=1).values
    fav = d[["A16_1", "A16_2", "A16_3"]]
    o["soda"] = sum(np.isin(fav[c], [7, 8, 12]) * w for c, w in zip(fav, (1, .6, .3))).clip(0, 1)
    o["fav"] = [" > ".join(dict.fromkeys(R.DRINK.get(int(v), "") for v in r if v == v)) for r in fav.values]
    o["zero_code"] = d.A19_1.astype(int)
    o["zero_freq"] = o.zero_code.map({1: 1.0, 2: .6, 3: .3, 4: .1, 9: 0.0})
    o["zero_why"], o["zero_no"] = d.A19_1_1.map(R.ZERO_WHY), d.A19_2_1.map(R.ZERO_NO)
    o["zero_reject"] = d.A19_2_1.isin([2, 3, 4, 5]).astype(float)
    z = Z[donors(o, ya, aa, adult)].copy()
    z[:, -3], z[:, -2], z[:, -1] = (o.age - 45) / 15, o.sex == "여자", yo.it_sns - 2      # choice.design 순서: …, 나이, 여성, SNS
    z = z + shift(z, d)[1]
    o = pd.concat([o, pd.DataFrame(z, columns=[f"z{i}" for i in range(Z.shape[1])])], axis=1)
    o["zero_aff"], auc = zero(o)
    o["trend"], o["drink"] = .5 * o.novelty + .5 * o.social, o.soda
    o["reg"], o["mar"], o["edu"], o["occ"] = o.province.map(P.REG), "미혼", "중졸이하", "비취업"
    o["single"] = d.J_SQ4.map({1: "1인", 2: "다인"}).fillna("다인")
    o["inc"] = pd.cut(d.J_BA2, [0, 2, 4, 6, 11], labels=["~200만", "~400만", "~600만", "600만+"]).astype(str)
    o["zero"] = o.zero_code.map({1: "주1회+", 2: "월2~3회", 3: "월1회", 4: "월1회미만", 9: "경험없음"})
    o["fav_cat"], o["age10"], o["band"] = d.A16_1.map(P.FAV).fillna("기타"), "10대", "13-18"
    for k in SHARED:                                              # 성인 기준 사분위
        o[k + "_q"] = np.array(["낮음", "중하", "중상", "높음"])[np.searchsorted(np.quantile(adult[k], [.25, .5, .75]), o[k])]
    o = pd.concat([o, yo], axis=1)
    o.attrs["auc"] = auc
    return o


def donors(o, ya, aa, adult, seed=0):
    """심리요인을 빌릴 19~29세 응답자(성별·공통 문항이 가장 비슷한 5명 중 1명)."""
    r, young = np.random.default_rng(seed), np.flatnonzero((adult.age <= 29).values)
    sd = aa.std().values
    A = np.column_stack([aa.values[young] / sd, 2.0 * (adult.sex.values[young] == "여자")])
    Y = np.column_stack([ya.values / sd, 2.0 * (o.sex.values == "여자")])
    D = ((Y[:, None] - A[None]) ** 2).sum(2)
    return young[np.argsort(D, axis=1)[np.arange(len(o)), r.integers(0, 5, len(o))]]


def ranks(d):
    from . import choice as M
    return pd.DataFrame({f"r{i}": d[f"A16_{i}"].map(M.CODE[2025]) for i in (1, 2, 3)} | {"w": d.psfwt / d.psfwt.mean()})


def shift(z, d, rows=None, lam=2.0):
    """10대의 속성 선호 이동 Δ(속성 수)를 순위 응답에서 추정(능형) → (Δ, 같은 효과를 내는 z 이동량: G·δz = Δ)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from . import choice as M
    m, R_ = M.model(), ranks(d)
    ix = np.arange(len(d)) if rows is None else rows
    pid, avail, ch = M._rows(R_.iloc[ix].reset_index(drop=True))
    w, V0 = R_.w.values[ix][pid], ((z[ix] @ m["G"].T) @ M.A.T + m["asc"])[pid]
    hot = np.zeros(avail.shape)
    hot[np.arange(len(ch)), ch] = 1

    def f(t):
        lp = np.where(avail, np.c_[V0 + M.A @ t, np.zeros(len(pid))], -np.inf)
        lp = lp - logsumexp(lp, axis=1, keepdims=True)
        return -(w * lp[np.arange(len(ch)), ch]).sum() + lam * (t ** 2).sum(), -(((hot - np.exp(lp)) * w[:, None])[:, :-1].sum(0) @ M.A) + 2 * lam * t
    t = minimize(f, np.zeros(M.A.shape[1]), jac=True, method="L-BFGS-B").x
    return t, np.linalg.lstsq(m["G"], t, rcond=None)[0]


def cv_shift(z, d, k=5, seed=0):
    """보지 않은 10대에서 1순위 분포 오차(TVD): 이동 전 / 이동 후, 그리고 1순위 로그우도(1인당)."""
    from . import choice as M
    m, r, t = M.model(), ranks(d), np.random.default_rng(seed).integers(0, k, len(d))
    alts = M.CATS + ["none"]
    P0, P1 = M.probs(z, m["G"], m["asc"]), np.zeros((len(d), len(alts)))
    for j in range(k):
        dz = shift(z, d, np.flatnonzero(t != j))[1]
        P1[t == j] = M.probs(z[t == j] + dz, m["G"], m["asc"])
    tr = pd.Series(r.w.values).groupby(r.r1.values).sum().reindex(alts, fill_value=0)
    tr, ok, ci = (tr / tr.sum()).values, r.r1.isin(alts).values, r.r1.map({a: i for i, a in enumerate(alts)})
    out = {}
    for nm, Q in (("성인 모델 그대로", P0), ("10대 응답으로 이동(교차검증)", P1)):
        out[nm] = (float(.5 * np.abs(np.average(Q, axis=0, weights=r.w) - tr).sum()), float(np.average(np.log(Q[ok, ci[ok].astype(int)]), weights=r.w[ok])))
    return out


def zero(o):
    """10대의 '제로 월 2회+' 로지스틱(교차검증 예측) → (확률, AUC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X, y = R._X(o)[:, :7], (o.zero_code <= 2).astype(int)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    p = cross_val_predict(LogisticRegression(C=.3), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p, float(roc_auc_score(y, p))


def report():
    """10대의 음료 1순위 분포: 실제 / 이동한 모델(같은 자료로 맞춘 값) / 20대 복사. 교차검증 값은 cv_shift."""
    from . import choice as M
    from . import multiverse as U
    b = U.base()
    y, d, m = b[b.age10 == "10대"], raw(), M.model()
    alts, code = M.CATS + ["none"], M.CODE[2025]
    sh = lambda v, w: (lambda t: t / t.sum())(pd.Series(w).groupby(np.asarray(v)).sum().reindex(alts, fill_value=0))
    a = pd.read_parquet(R.SRC)
    w20 = a.SQ2.between(19, 29)
    pr = pd.Series(np.average(M.probs(y.filter(regex=r"^z\d+$").values, m["G"], m["asc"]), axis=0, weights=y.w), index=alts)
    t = pd.DataFrame({"실제 10대": sh(d.A16_1.map(code).fillna("other"), d.psfwt.values), "모델(10대로 이동)": pr, "20대 복사": sh(a.A15_1.map(code).fillna("other")[w20], a.psfwt[w20].values)})
    adult = b[b.age10 != "10대"]
    z = adult.filter(regex=r"^z\d+$").values
    z0 = z[donors(y, items(False)[0], items(True)[0], adult)].copy()
    z0[:, -3], z0[:, -2], z0[:, -1] = (y.age - 45) / 15, y.sex == "여자", items(False)[1].it_sns - 2
    print(cv_shift(z0, d))
    return t, {c: float(.5 * (t[c] - t["실제 10대"]).abs().sum()) for c in t.columns[1:]}, b.attrs["youth_auc"]


def check():
    x = np.array([1.0, 2, 3, 4, 5])
    assert np.allclose(qmap(x, np.repeat(x, 200), np.repeat(x * 10, 200)), x * 10, atol=10) and len(raw()) == 604
    a, o = items(False)
    assert ((a >= 1) & (a <= 5)).all().all() and list(o) == list(ONE) and not a.isna().any().any()
    print("youth ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        for v in report():
            print(v.round(3).to_string() if hasattr(v, "to_string") else v)
    else:
        check()
