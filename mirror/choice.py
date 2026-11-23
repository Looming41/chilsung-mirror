"""실제 음료 선택 구조 학습: 3개년 KREI 패널 → 심리 요인분석 → 이질적 조건부 로짓(순위 확장형).

데이터: KREI 식품소비행태조사 성인가구원 2023·2024·2025 (각 약 5,800명, 가중치). '가장 좋아하는 음료 1~3순위'(A15).
1) 패널: 연도마다 다른 문항 번호를 '문항 문구'로 맞춘 공통 리커트 34문항 + SNS 사용 + 인구통계, 음료 코드 연도별 조화
2) 요인분석: 병렬분석(Horn)으로 요인 수 결정 → varimax 회전 → 사람별 요인 점수
3) 선택 모델: 음료를 '속성 묶음'(탄산·제로·단맛·과일·유제품·카페인·차·기능성·매장제조)으로 표현하고
   사람 i 의 속성 선호 β_i = Γ z_i (z = 요인점수·나이·성별·SNS) 로 이질성을 둔다.
   U_ij = a_j·β_i + ASC_j(능형 수축),  '물 외 안 마심' = 바깥 선택지(U=0).  1~3순위를 순위 확장(exploded) 로짓으로 모두 사용.
   → 데이터에 없던 속성 조합(예: 과일맛 제로 탄산)을 새 선택지로 넣어 개인별 선택확률·자기잠식을 계산할 수 있다.
4) 검증: 교차검증 예측력 / 범주 하나를 빼고 학습 후 '신제품처럼' 넣어 누가 고를지 맞히기 / 2023~24로 2025 예측
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

YEARS = (2023, 2024, 2025)
CATS = ["juice", "lowjuice", "milk", "flavmilk", "yogurt", "soymilk", "soda", "zerosoda", "ion", "tea",
        "functional", "sparkling", "coffee", "shop"]
KO = dict(juice="100% 주스", lowjuice="저과즙음료", milk="흰우유", flavmilk="가공우유", yogurt="발효유", soymilk="두유",
          soda="가당 탄산", zerosoda="제로 탄산", ion="이온음료", tea="차", functional="기능성 음료", sparkling="탄산수",
          coffee="커피", shop="음료전문점 음료", none="물만")
ATTR = ["carb", "zero", "sweet", "fruit", "dairy", "caffeine", "tea", "functional", "shop"]
ATTR_KO = dict(carb="탄산", zero="무설탕", sweet="단맛", fruit="과일", dairy="유제품", caffeine="카페인", tea="차",
               functional="기능성", shop="매장제조")
_A = {  # 범주별 속성(0~1)
    "juice": dict(fruit=1, sweet=.5), "lowjuice": dict(fruit=1, sweet=1), "milk": dict(dairy=1),
    "flavmilk": dict(dairy=1, sweet=1), "yogurt": dict(dairy=1, sweet=.5, functional=.5),
    "soymilk": dict(dairy=.5, functional=.5, sweet=.3), "soda": dict(carb=1, sweet=1),
    "zerosoda": dict(carb=1, zero=1, sweet=1), "ion": dict(functional=.5, sweet=.5), "tea": dict(tea=1, zero=1),
    "functional": dict(functional=1, sweet=.5, caffeine=.5), "sparkling": dict(carb=1, zero=1),
    "coffee": dict(caffeine=1, shop=.5), "shop": dict(shop=1, sweet=1, fruit=.5)}
A = np.array([[_A[c].get(a, 0.0) for a in ATTR] for c in CATS])
_BASE = {1: "juice", 2: "lowjuice", 3: "milk", 4: "flavmilk", 5: "yogurt", 6: "soymilk", 7: "soda", 8: "zerosoda",
         9: "ion", 17: "none"}
CODE = {2023: {**_BASE, 10: "coffee", 11: "tea", 12: "functional", 13: "sparkling", 14: "coffee", 15: "shop"},
        2024: {**_BASE, 10: "tea", 11: "functional", 12: "sparkling", 13: "coffee", 14: "coffee", 15: "shop"}}
CODE[2025] = CODE[2024]
CACHE = "data/choice_model.npz"
MODEL_VERSION = 2           # 모델 구조·규제·데이터가 바뀌면 올린다
# v1 요인(병렬분석 7개)을 적재량 상위 문항을 보고 붙인 이름. 버전이 바뀌면 다시 확인할 것
FACTOR_LABELS = ["로컬·친환경 식재료 관심", "간편식·간단 식사", "알뜰 비교 구매", "표시정보 신뢰·기능성 관심",
                 "식품 정책 관심", "음식물 낭비 회피", "안전·친환경 가치"]


# ---------- 1. 패널 ----------
def _files(y):
    s = "" if y == 2025 else f"_{y}"
    return f"data/krei_adult{s}.parquet", f"data/krei_adult_labels{s}.parquet", f"data/krei_adult_codebook{s}.parquet"


def _ensure(y):
    """없으면 KREI 에서 받아 parquet 로."""
    f, lf, cf = _files(y)
    if os.path.exists(f) and os.path.exists(lf) and os.path.exists(cf):
        return
    import io, urllib.request, zipfile
    ids = {2023: 164083, 2024: 503546, 2025: 505133}
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
        f"https://krei.re.kr/foodSurvey/board/atchDown.do?no={ids[y]}&cmd=a&nm={y}", timeout=300).read()))
    for i in z.infolist():
        name = i.filename if i.flag_bits & 0x800 else i.filename.encode("cp437").decode("cp949")
        if "성인" in name:
            x = io.BytesIO(z.read(i))
            pd.read_excel(x, sheet_name=0).to_parquet(f)
            pd.read_excel(x, sheet_name=2).to_parquet(lf)
            pd.read_excel(x, sheet_name=-1).ffill().to_parquet(cf)


def _likert(y):
    f, lf, cf = _files(y)
    d = pd.read_parquet(f)
    lab = pd.read_parquet(lf)
    lab.columns = ["v", "l"]
    cb = pd.read_parquet(cf)
    cb.columns = ["v", "val", "lab"]
    g = cb.groupby("v").agg(n=("val", "size"), first=("lab", "first"))
    lik = set(g[(g.n == 5) & g["first"].fillna("").astype(str).str.contains("전혀")].index)
    items, sns = {}, None
    for v, l in zip(lab.v, lab.l.fillna("").astype(str)):
        if "SNS 사용 여부" in l and v in d:
            sns = v
        if v in lik and v in d and d[v].notna().mean() > .95 and d[v].between(1, 5).mean() > .95:
            key = re.sub(r"\s|[.,·]", "", re.sub(r"^[^)]*\)\s*", "", l.split("_", 1)[-1]))
            items[key] = (v, re.sub(r"^[^.]*\.\s*", "", l))
    return d, items, sns


def panel():
    """3개년 공통 패널. 열: year, age, sex, w(연도별 평균 1), r1~r3(음료 범주), sns, q_*(공통 리커트)."""
    for y in YEARS:
        _ensure(y)
    raw = {y: _likert(y) for y in YEARS}
    common = sorted(set.intersection(*(set(raw[y][1]) for y in YEARS)))
    common = [k for k in common if raw[2025][1][k][0] != "J_BA6"]          # BA5(건강 관심) 중복 문항
    frames = []
    for y in YEARS:
        d, items, sns = raw[y]
        f = pd.DataFrame({"year": y, "age": d.SQ2.astype(int), "sex": d.SQ1.map({1: "남자", 2: "여자"}),
                          "w": d.psfwt / d.psfwt.mean(), "sns": d[sns].astype(float).fillna(1)})
        for r in (1, 2, 3):
            f[f"r{r}"] = d[f"A15_{r}"].map(CODE[y])
        for i, k in enumerate(common):
            f[f"q{i:02d}"] = d[items[k][0]].astype(float)
        frames.append(f)
    P = pd.concat(frames, ignore_index=True)
    P.attrs["items"] = [raw[2025][1][k][1] for k in common]
    return P


# ---------- 2. 요인분석 ----------
def factors(P, sims=20, seed=0):
    """병렬분석으로 k 결정 → varimax FA. 반환 (점수[n,k], 적재량 DataFrame, FA객체, 평균, 표준편차)."""
    from sklearn.decomposition import FactorAnalysis
    Q = P.filter(regex=r"^q\d+$")
    mu, sd = Q.mean(), Q.std()
    X = ((Q - mu) / sd).fillna(0).values
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    rng = np.random.default_rng(seed)
    rnd = np.array([np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(X.shape).T))[::-1] for _ in range(sims)])
    k = int((ev > np.percentile(rnd, 95, axis=0)).sum())
    fa = FactorAnalysis(k, rotation="varimax", random_state=seed).fit(X)
    S = fa.transform(X)
    S = (S - S.mean(0)) / S.std(0)
    L = pd.DataFrame(fa.components_.T, index=P.attrs["items"], columns=[f"F{i + 1}" for i in range(k)])
    return S, L, fa, mu.values, sd.values


def _short(item):
    t = item.split("_")[-1] if "_" in item else item
    t = re.sub(r"^\(\d+\)\s*", "", t)
    t = re.sub(r"(여부에 대한 관심 정도|에 대한 관심 정도|하는 편이다|편이다|여부)$", "", t).strip()
    return t[:22]


def factor_names(L, top=2):
    return [" · ".join(_short(i) for i in L[c].sort_values(key=abs, ascending=False).head(top).index) for c in L]


# ---------- 3. 선택 모델 ----------
def design(P, S, trend=False):
    """사람 특성 z: [1, 요인점수, 나이, 여성, SNS, (연도 추세)]."""
    cols = [np.ones(len(P)), *S.T, (P.age.values - 45) / 15, (P.sex.values == "여자").astype(float),
            P.sns.values - 2]
    if trend:
        cols.append(P.year.values - 2024.0)
    return np.column_stack(cols)


def _rows(P, drop=()):
    """순위 확장: (사람, 선택지 가용 마스크, 선택 인덱스). 'other'·결측 순위에서 멈춘다. drop 범주는 선택지에서 제거."""
    alts = CATS + ["none"]
    idx = {c: i for i, c in enumerate(alts)}
    pid, avail, ch = [], [], []
    base = np.array([c not in drop for c in alts])
    for i, (a, b, c) in enumerate(P[["r1", "r2", "r3"]].itertuples(index=False)):
        av = base.copy()
        for r in (a, b, c):
            if not isinstance(r, str) or r not in idx or r in drop:
                break
            if not av[idx[r]]:          # 같은 범주 중복 응답(커피 2종 합침·원자료 중복): 가능한 선택이 아니므로 건너뜀
                continue
            pid.append(i)
            avail.append(av.copy())
            ch.append(idx[r])
            av[idx[r]] = False
            if r == "none":
                break
    return np.array(pid), np.array(avail), np.array(ch)


def fit(Z, P, drop=(), lam_asc=0.1, lam_g=1.0, w=None, lam_main=20.0):   # LOCO 격자탐색으로 선택
    pid, avail, ch = _rows(P, drop)
    w = (P.w.values if w is None else w)[pid]
    Zr = Z[pid]
    nA, nZ, C = len(ATTR), Z.shape[1], len(CATS)
    onehot = np.zeros(avail.shape)
    onehot[np.arange(len(ch)), ch] = 1

    def f(th):
        G, asc = th[:nA * nZ].reshape(nA, nZ), th[nA * nZ:]
        V = np.c_[(Zr @ G.T) @ A.T + asc, np.zeros(len(Zr))]
        V = np.where(avail, V, -np.inf)
        lp = V - logsumexp(V, axis=1, keepdims=True)
        p = np.exp(lp)
        ll = (w * lp[np.arange(len(ch)), ch]).sum()
        dV = (onehot - p) * w[:, None]
        gG = (dV[:, :C] @ A).T @ Zr
        gasc = dV[:, :C].sum(0)
        obj = -ll + lam_asc * (asc ** 2).sum() + lam_g * (G[:, 1:] ** 2).sum() + lam_main * (G[:, 0] ** 2).sum()
        grad = -np.r_[gG.ravel(), gasc]
        grad[:nA * nZ] += 2 * lam_g * np.c_[np.zeros((nA, 1)), G[:, 1:]].ravel()
        grad[nA * nZ:] += 2 * lam_asc * asc
        grad[:nA * nZ] += 2 * lam_main * np.c_[G[:, :1], np.zeros((nA, nZ - 1))].ravel()
        return obj, grad

    r = minimize(f, np.zeros(nA * nZ + C), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return r.x[:nA * nZ].reshape(nA, nZ), r.x[nA * nZ:]


def probs(Z, G, asc, new=None, drop=()):
    """개인별 선택확률. 열 = 14범주 (+신제품) + 물만(마지막). new: 신제품 속성 dict(ASC 0 = 수축 사전평균)."""
    U = (Z @ G.T) @ A.T + asc
    for c in drop:
        U[:, CATS.index(c)] = -np.inf
    if new is not None:
        a = np.array([new.get(k, 0.0) for k in ATTR])
        U = np.c_[U, (Z @ G.T) @ a]
    U = np.c_[U, np.zeros(len(Z))]
    return np.exp(U - logsumexp(U, axis=1, keepdims=True))


# ---------- 4. 검증 ----------
def _seg(P):
    g = pd.cut(P.age, [18, 29, 39, 49, 59, 69, 200], labels=["20대", "30대", "40대", "50대", "60대", "70대+"])
    return g.astype(str) + " " + P.sex


def cv_eval(P, Z, Zd, k=5, seed=0):
    """1순위 예측(5겹, 사람 단위): 가중 로그손실·top1·top3 적중. 비교: 전체 점유율 / 인구통계만 / 요인+인구."""
    alts = CATS + ["none"]
    ok = P.r1.isin(alts).values
    y = P.r1.map({c: i for i, c in enumerate(alts)}).values
    fold = np.random.default_rng(seed).integers(0, k, len(P))
    out = {n: {"logloss": [], "top1": [], "top3": []} for n in ("전체 점유율", "인구통계만", "요인+인구통계")}
    for f in range(k):
        tr, te = fold != f, (fold == f) & ok
        share = np.bincount(y[tr & ok].astype(int), weights=P.w.values[tr & ok], minlength=len(alts))
        pr = {"전체 점유율": np.tile(share / share.sum(), (te.sum(), 1))}
        for name, ZZ in (("인구통계만", Zd), ("요인+인구통계", Z)):
            G, asc = fit(ZZ[tr], P[tr].reset_index(drop=True))
            pr[name] = probs(ZZ[te], G, asc)
        w, yy = P.w.values[te], y[te].astype(int)
        for name, p in pr.items():
            out[name]["logloss"].append(-np.average(np.log(p[np.arange(len(yy)), yy] + 1e-12), weights=w))
            out[name]["top1"].append(np.average(p.argmax(1) == yy, weights=w))
            out[name]["top3"].append(np.average((np.argsort(-p, 1)[:, :3] == yy[:, None]).any(1), weights=w))
    return pd.DataFrame({n: {m: np.mean(v) for m, v in d.items()} for n, d in out.items()}).T


def asc_prior(asc, c, keep):
    """뺀 범주의 ASC: 속성 거리(가우시안 커널)로 가중한 나머지 범주 ASC 평균."""
    i = CATS.index(c)
    d = ((A[keep] - A[i]) ** 2).sum(1)
    k = np.exp(-d / 0.5)
    return float((k * asc[keep]).sum() / k.sum())


def loco(P, Z, cats=None, **kw):
    """범주 하나를 통째로 빼고 학습 → 속성만으로 '신제품'처럼 다시 넣어 1순위 점유율을 예측.
    수준(전체 점유율)과 '누가 고르나'(연령×성별 12집단 상관)를 실제와 비교."""
    seg = _seg(P)
    rows = []
    for c in cats or CATS:
        G, asc = fit(Z, P, drop=(c,), **kw)
        asc_full = asc.copy()
        keep = [j for j in range(len(CATS)) if j != CATS.index(c)]
        asc_full[CATS.index(c)] = asc_prior(asc, c, keep)      # 뺀 범주는 ASC 미학습 → 비슷한 범주에서 빌림
        p = probs(Z, G, asc_full)[:, CATS.index(c)]
        real = (P.r1 == c).astype(float)
        df = pd.DataFrame({"seg": seg, "p": p * P.w, "y": real * P.w, "w": P.w}).groupby("seg")[["p", "y", "w"]].sum()
        ps, ys = df.p / df.w, df.y / df.w
        rows.append({"범주": KO[c], "실제 점유율": np.average(real, weights=P.w), "예측 점유율": np.average(p, weights=P.w),
                     "집단 상관 r": float(np.corrcoef(ps, ys)[0, 1]),
                     "집단 오차(%p)": float(100 * (ps - ys).abs().mean())})
    return pd.DataFrame(rows)


def temporal(P, S):
    """2023~24 로 학습 → 2025 집단(연령×성별)별 범주 점유율 예측. 비교: 2024 그대로 / 모델(추세 없음) / 모델(연도 추세)."""
    tr, te = (P.year < 2025).values, (P.year == 2025).values
    Pt, Pe = P[tr].reset_index(drop=True), P[te].reset_index(drop=True)

    def segshare(Px, pmat):
        w = Px.w.values[:, None]
        df = pd.DataFrame(pmat[:, :len(CATS)] * w, columns=CATS).assign(seg=_seg(Px).values, w=Px.w.values)
        g = df.groupby("seg").sum()
        return g[CATS].div(g.w, axis=0)

    onehot = lambda Px: (np.asarray(Px.r1, dtype=object)[:, None] == np.array(CATS, dtype=object)).astype(float)
    truth = segshare(Pe, onehot(Pe))
    P24 = P[P.year == 2024].reset_index(drop=True)
    out = {"2024 그대로": segshare(P24, onehot(P24))}
    for name, tr_ in (("모델(추세 없음)", False), ("모델+연도 추세", True)):
        Zt, Ze = design(Pt, S[tr], tr_), design(Pe, S[te], tr_)
        G, asc = fit(Zt, Pt)
        out[name] = segshare(Pe, probs(Ze, G, asc))
    rep = {n: {"평균 오차(%p)": 100 * (v - truth).abs().values.mean(),
               "제로 탄산 오차(%p)": 100 * (v.zerosoda - truth.zerosoda).abs().mean(),
               "커피 오차(%p)": 100 * (v.coffee - truth.coffee).abs().mean()} for n, v in out.items()}
    return pd.DataFrame(rep).T


# ---------- 시뮬레이터 연결 ----------
_MODEL = {}
def model():
    """3개년 전체로 학습한 모델 dict(fa, mu, sd, smu, ssd, G, asc, names, loadings). data/choice_model.pkl 캐시."""
    import pickle
    if "m" in _MODEL:
        return _MODEL["m"]
    path = CACHE.replace(".npz", ".pkl")
    if os.path.exists(path):
        m = pickle.load(open(path, "rb"))
        if m.get("version") == MODEL_VERSION:            # 학습 1회 ≈ 50분(L-BFGS) → 버전이 바뀔 때만 재학습
            m["names"] = FACTOR_LABELS if len(FACTOR_LABELS) == m["loadings"].shape[1] else factor_names(m["loadings"])
            if "lam" not in m:                           # 중첩 로짓 λ·묶음(데이터 추정)은 효용 고정 후 따로 → 재학습 없이 추가
                m["lam"], m["nests"], m["lam_ll"] = fit_lambda(m, k=4)
                pickle.dump(m, open(path, "wb"))
            _MODEL["m"] = m
            return m
    P = panel()
    S, L, fa, mu, sd = factors(P)
    raw = fa.transform(((P.filter(regex=r"^q\d+$") - mu) / sd).fillna(0).values)
    G, asc = fit(design(P, S), P)
    m = dict(fa=fa, mu=mu, sd=sd, smu=raw.mean(0), ssd=raw.std(0), G=G, asc=asc,
             names=factor_names(L), loadings=L, version=MODEL_VERSION)
    pickle.dump(m, open(path, "wb"))
    return m


def person_features(m=None):
    """2025 KREI 응답자 전원(records.load 와 같은 행 순서)의 선택모델 특성 z 와 요인점수."""
    m = m or model()
    P = panel()
    P = P[P.year == 2025].reset_index(drop=True)
    S = (m["fa"].transform(((P.filter(regex=r"^q\d+$") - m["mu"]) / m["sd"]).fillna(0).values) - m["smu"]) / m["ssd"]
    return design(P, S), S


KW = dict(carb=r"탄산|사이다|콜라|스파클링|소다", fruit=r"과일|자두|레몬|라임|복숭아|사과|포도|오렌지|유자|망고|딸기|수박|에이드|베리|파인",
          caffeine=r"커피|카페인|에너지|라떼", dairy=r"우유|라떼|요거|밀크", tea=r"녹차|홍차|보리차|아이스티|티백|허브티",
          functional=r"비타민|단백질|이온|기능성|전해질|프로바이오", shop=r"매장|수제")


def concept_attrs(concept, sc):
    """콘셉트 문장 + 시나리오 → 신제품 속성. 아무 키워드가 없으면 탄산음료로 본다."""
    a = {k: float(bool(re.search(p, concept or ""))) for k, p in KW.items()}
    if not any(a.values()):
        a["carb"] = 1.0
    a["zero"] = float(sc.get("zero", 0))
    a["sweet"] = 0.0 if re.search(r"무가당|플레인|탄산수", concept or "") else 1.0
    return a


_REF = {}


def ref_share():
    """기준점: 일반 가당 탄산(탄산·단맛)을 신제품으로 넣었을 때 KREI 2025 성인 전체 가중 평균 선택확률."""
    if "v" not in _REF:
        from . import records as R
        Z, _ = person_features()
        _REF["v"] = float(np.average(new_share(Z, dict(carb=1.0, sweet=1.0)), weights=R.load().w.values))
    return _REF["v"]


def new_share(Z, new, m=None):
    """개인별 '신제품을 1순위로 고를 확률' (중첩 로짓, 데이터 추정 묶음·λ)."""
    m = m or model()
    return nested_probs(Z, m["G"], m["asc"], m["lam"], m["nests"], new)[:, len(CATS)]


def cannibal(Z, new, w=None, m=None):
    """신제품 출시 전후 범주별 1순위 점유율. 열: 출시 전, 변화(%p), 잠식률(자기 점유율 중 잃는 %). 신제품 행 포함."""
    m = m or model()
    p0 = nested_probs(Z, m["G"], m["asc"], m["lam"], m["nests"])
    p1 = nested_probs(Z, m["G"], m["asc"], m["lam"], m["nests"], new)
    w = np.ones(len(Z)) if w is None else w
    before = np.average(np.c_[p0[:, :-1], np.zeros(len(Z)), p0[:, -1:]], axis=0, weights=w)
    after = np.average(p1, axis=0, weights=w)
    idx = [KO[c] for c in CATS] + ["신제품", "물만"]
    return pd.DataFrame({"출시 전": 100 * before, "변화(%p)": 100 * (after - before),
                         "잠식률(%)": 100 * np.where(before > 0, (after - before) / np.where(before > 0, before, 1), np.nan)},
                        index=idx)


# ---------- 중첩 로짓: IIA 완화(비슷한 음료끼리 더 잘 대체). 묶음은 데이터(2순위 상승도)로 찾는다 ----------
def nests_from_data(P, k=4):
    """1순위→2순위 상승도(lift) 대칭행렬로 평균연결 군집 → k 묶음. 실데이터상 '같이 고르는' 음료끼리 묶인다."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Q = P[P.r1.isin(CATS) & P.r2.isin(CATS) & (P.r1 != P.r2)]
    lift = pd.crosstab(Q.r1, Q.r2, normalize="index").div(Q.r2.value_counts(normalize=True), axis=1)
    lift = lift.reindex(index=CATS, columns=CATS).fillna(1.0)
    L = (lift + lift.T) / 2
    D = -np.log(np.clip(L.values, 1e-3, None))
    D = D - D.min()
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust")
    return {f"N{g}": [c for c, l in zip(CATS, lab) if l == g] for g in sorted(set(lab))}, lift


def _nest_of(new, nests):
    """신제품은 속성이 가장 가까운 범주가 속한 묶음으로."""
    a = np.array([new.get(k, 0.0) for k in ATTR])
    near = CATS[int(((A - a) ** 2).sum(1).argmin())]
    return next(k for k, v in nests.items() if near in v)


def nested_probs(Z, G, asc, lam, nests, new=None, avail=None):
    """2단 중첩 로짓(척도형: 묶음 안 변별력은 일반 로짓과 같고 묶음 간 대체만 λ로 조정). 열 = 14범주[+신제품]+물만."""
    U = (Z @ G.T) @ A.T + asc
    names = list(CATS)
    if new is not None:
        U = np.c_[U, (Z @ G.T) @ np.array([new.get(k, 0.0) for k in ATTR])]
        names.append("_new")
        nests = {k: v + (["_new"] if k == _nest_of(new, nests) else []) for k, v in nests.items()}
    if avail is not None:
        U = np.where(avail[:, :-1], U, -np.inf)
    out = np.zeros((len(Z), len(names) + 1))
    IV, idx = [], []
    for v in nests.values():
        ix = [names.index(c) for c in v]
        idx.append(ix)
        IV.append(logsumexp(U[:, ix], axis=1))                      # 묶음 포함가치(-inf = 다 빠진 묶음)
    IV = np.column_stack(IV)
    none = np.zeros(len(Z)) if avail is None else np.where(avail[:, -1], 0.0, -np.inf)
    top = np.c_[lam * IV, none]
    lt = top - logsumexp(top, axis=1, keepdims=True)
    for n, ix in enumerate(idx):
        inner = np.where(np.isfinite(IV[:, [n]]), U[:, ix] - np.where(np.isfinite(IV[:, [n]]), IV[:, [n]], 0), -np.inf)
        out[:, ix] = np.exp(lt[:, [n]] + inner)
    out[:, -1] = np.exp(lt[:, -1])
    return out


def fit_lambda(m=None, k=4, grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """효용 고정, 1~3순위 순위 확장 우도로 공통 λ 격자 추정. 반환 (λ, 묶음, {λ: 평균 로그우도})."""
    m = m or model()
    P = panel()
    S = (m["fa"].transform(((P.filter(regex=r"^q\d+$") - m["mu"]) / m["sd"]).fillna(0).values) - m["smu"]) / m["ssd"]
    Z = design(P, S)
    nests, _ = nests_from_data(P, k)
    pid, avail, ch = _rows(P)
    w = P.w.values[pid]
    ll = {}
    for lam in grid:
        p = nested_probs(Z[pid], m["G"], m["asc"], lam, nests, avail=avail)
        ll[lam] = float(np.average(np.log(p[np.arange(len(ch)), ch] + 1e-12), weights=w))
    return max(ll, key=ll.get), nests, ll

if __name__ == "__main__":
    import time
    t = time.time()
    P = panel()
    print("패널", P.shape, P.year.value_counts().to_dict(), "공통 문항", len(P.attrs["items"]))
    S, L, fa, *_ = factors(P)
    print("요인 수(병렬분석)", S.shape[1])
    for c, nm in zip(L, factor_names(L, 3)):
        print(" ", c, nm)
    Z, Zd = design(P, S), design(P, np.zeros((len(P), 0)))
    print(cv_eval(P, Z, Zd).round(3))
    lc = loco(P, Z)
    print(lc.round(3).to_string())
    print(temporal(P, S).round(2))
    print("choice %.0fs" % (time.time() - t))
