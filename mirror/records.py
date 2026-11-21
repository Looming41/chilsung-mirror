"""실제 한국인 응답 기록에 페르소나를 묶는다 ('기록 앵커 시민').

출처: 한국농촌경제연구원(KREI) 2025 식품소비행태조사 성인가구원 원자료 (n=5,874, 가중치 psfwt).
    https://krei.re.kr/foodSurvey/page/460

왜: LLM 에게 "20대 여성인 척" 시키면 집단 간 차이를 고정관념으로 채우고 뭉갠다(Pew 2026, Twin-2K-500 등).
    같은 사람의 실제 응답 기록이 새 질문의 답을 훨씬 잘 예측한다 → 페르소나 = 서사(Nemotron) + 실기록(KREI).

1. hot-deck 통계적 매칭: 페르소나마다 같은 성별·나이(±3세)·시도의 실제 응답자 1명을 가중치 비례로 뽑아 기록을 붙임
2. 심리 척도: 실제 리커트 문항 평균(0~1), Cronbach α 로 신뢰도 보고
3. ML: 실데이터로 '제로음료 월 2회 이상' 로지스틱 회귀 학습 → 페르소나별 제로 수용성(교차검증 AUC 보고)
"""
import os

import numpy as np
import pandas as pd

SRC = "data/krei_adult.parquet"
ZIP_URL = "https://krei.re.kr/foodSurvey/board/atchDown.do?no=505133&cmd=a&nm=2025"
XLSX = "data/raw/krei/(data_가중) 2025 식품소비행태조사_성인가구원.xlsx"

PROV = {1: "서울", 2: "부산", 3: "인천", 4: "대구", 5: "대전", 6: "광주", 7: "울산", 8: "경기", 9: "경상북",
        10: "경상남", 11: "전북", 12: "전라남", 13: "충청북", 14: "충청남", 15: "강원", 16: "제주", 17: "세종"}
DRINK = {1: "100% 과일주스", 2: "저과즙음료", 3: "흰우유", 4: "가공우유", 5: "발효유", 6: "두유", 7: "탄산음료",
         8: "제로 탄산음료", 9: "이온음료", 10: "녹차·곡물차", 11: "기능성 음료", 12: "탄산수", 13: "인스턴트·캔커피",
         14: "카페 커피", 15: "음료전문점 음료(버블티·에이드)", 16: "기타", 17: "물 외엔 안 마심"}
ZERO_FREQ = {1: "주 1회 이상", 2: "월 2~3회", 3: "월 1회", 4: "월 1회 미만", 9: "먹어본 적 없음"}
ZERO_NO = {1: "원래 안 먹는 종류라서", 2: "제로 특유의 맛이 싫어서", 3: "부작용 경험", 4: "인공감미료가 불안해서",
           5: "칼로리 0 같지 않아서", 6: "비싸다고 느껴서", 7: "본 적이 없어서", 8: "기타"}
ZERO_WHY = {1: "건강에 좋을 것 같아서", 2: "맛있어서", 3: "행사로 싸서", 4: "주변에서 먹으니까", 5: "신상·호기심",
            6: "다이어트", 7: "기타"}

# 척도: (문항, 역문항 여부). 1~5 리커트(SNS 류는 1~3)
SCALES = {
    # 실측 상관으로 재구성: '가격보다 맛'(H1_5)은 가격민감의 역문항이 아니라 새로움 추구와 같은 축(r=.45)
    "novelty": [("H1_4", 0), ("H1_5", 0), ("H1_6", 0)],          # 다양한 맛, 가격보다 맛, 새로운 음식 좋아함
    "thrift": [("H1_7", 0), ("H1_8", 0), ("H1_9", 0)],           # 가격 비교, 가성비 체크, 장보기 목록(계획형)
    "health": [("BA5", 0), ("BA2", 0), ("BA5_1", 0)],            # 건강 관심, 다이어트 관심, 건강 투자
    "social": [("D4", 0), ("D4_1", 0), ("D5", 0)],                # SNS 사용, 음식 SNS 시청, 온라인 추천 활용
}


def _fetch():
    """KREI 원자료 zip(약 60MB)을 받아 성인가구원 엑셀 → parquet. 한 번만."""
    import io
    import urllib.request
    import zipfile
    os.makedirs(os.path.dirname(XLSX), exist_ok=True)
    if not os.path.exists(XLSX):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP_URL, timeout=300).read()))
        for i in z.infolist():
            name = i.filename if i.flag_bits & 0x800 else i.filename.encode("cp437").decode("cp949")
            if "성인가구원" in name:
                open(XLSX, "wb").write(z.read(i))
    pd.read_excel(XLSX, sheet_name="Numeric").to_parquet(SRC)


def _scale(d, items):
    cols = []
    for v, rev in items:
        x = d[v].astype(float)
        top = 3 if v.startswith("D") else 5
        x = x.fillna(1) if v.startswith("D") else x            # SNS 안 쓰면 하위 문항 미응답 → 최저
        x = (x - 1) / (top - 1)
        cols.append(1 - x if rev else x)
    m = np.column_stack(cols)
    k = m.shape[1]
    alpha = k / (k - 1) * (1 - m.var(0, ddof=1).sum() / m.sum(1).var(ddof=1))
    return m.mean(1), alpha


def load(path=SRC):
    """KREI 성인 응답자 → 정리된 표(인구통계 + 성향 척도 + 음료 기록). attrs['alpha'] 에 척도 신뢰도."""
    if not os.path.exists(path):
        _fetch()
    d = pd.read_parquet(path)
    out = pd.DataFrame({"age": d.SQ2.astype(int), "sex": d.SQ1.map({1: "남자", 2: "여자"}),
                        "province": d["지역"].map(PROV), "w": d.psfwt.astype(float),
                        "income": d.BA7.astype(float)})
    alpha = {}
    for k, items in SCALES.items():
        out[k], alpha[k] = _scale(d, items)
    fav = d[["A15_1", "A15_2", "A15_3"]]
    out["soda"] = sum(np.isin(fav[c], [7, 8, 12]) * wgt for c, wgt in zip(fav, (1, .6, .3))).clip(0, 1)
    out["fav"] = [" > ".join(dict.fromkeys(DRINK.get(int(v), "") for v in r if v == v)) for r in fav.values]
    out["zero_code"] = d.A19_1.astype(int)
    out["zero_freq"] = out.zero_code.map({1: 1.0, 2: .6, 3: .3, 4: .1, 9: 0.0})
    out["zero_why"] = d.A19_1_1.map(ZERO_WHY)
    out["zero_no"] = d.A19_2_1.map(ZERO_NO)
    # 맛·감미료 불신으로 제로를 거부(신제품이 '제로'일 때 감점 근거)
    out["zero_reject"] = d.A19_2_1.isin([2, 3, 4, 5]).astype(float)
    out.attrs["alpha"] = alpha
    return out


# ---------- ML: 제로음료 수용성 ----------
FEATS = ["health", "novelty", "thrift", "social", "soda", "age10", "female", "inc"]


def _X(df):
    return np.column_stack([df.health, df.novelty, df.thrift, df.social, df.soda, df.age / 10,
                            (df.sex == "여자").astype(float), df.income.fillna(6) / 12])


def zero_model(k=None):
    """'제로음료 월 2회 이상' 로지스틱 회귀(가중). 반환 (모델, 리포트). 리포트: 5겹 교차검증 AUC(성향+인구 vs 인구만)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    k = load() if k is None else k
    X, y, w = _X(k), (k.zero_code <= 2).astype(int).values, k.w.values
    demo = [FEATS.index(c) for c in ("age10", "female", "inc")]
    auc = {"성향+인구통계": [], "인구통계만": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        for name, cols in (("성향+인구통계", slice(None)), ("인구통계만", demo)):
            m = LogisticRegression(max_iter=1000).fit(X[tr][:, cols], y[tr], sample_weight=w[tr])
            auc[name].append(roc_auc_score(y[te], m.predict_proba(X[te][:, cols])[:, 1], sample_weight=w[te]))
    m = LogisticRegression(max_iter=1000).fit(X, y, sample_weight=w)
    rep = {"auc": {n: float(np.mean(v)) for n, v in auc.items()},
           "coef": pd.Series(m.coef_[0], index=FEATS), "base_rate": float(np.average(y, weights=w))}
    return m, rep


# ---------- hot-deck 매칭 ----------
def attach(pool, k=None, seed=0):
    """페르소나마다 성별·나이(±3세, 80+ 는 75~79)·시도가 같은 실제 응답자 1명을 가중치 비례로 뽑아 기록을 붙인다.
    후보가 없으면 시도 조건을 풀고(전국), 그래도 없으면 나이 폭을 넓힌다."""
    k = load() if k is None else k
    rng = np.random.default_rng(seed)
    age_k, sex_k, prov_k, w_k = k.age.values, k.sex.values, k.province.values, k.w.values
    donor = np.empty(len(pool), int)
    level = np.empty(len(pool), int)
    for i, (a, s, pv) in enumerate(zip(np.minimum(pool.age.values, 79), pool.sex.values, pool.province.values)):
        for lv, (span, use_prov) in enumerate(((3, True), (3, False), (8, False), (99, False))):
            ok = (np.abs(age_k - a) <= span) & (sex_k == s)
            if use_prov:
                ok &= prov_k == pv
            idx = np.flatnonzero(ok)
            if len(idx):
                donor[i] = rng.choice(idx, p=w_k[idx] / w_k[idx].sum())
                level[i] = lv
                break
    rec = k.iloc[donor].drop(columns=["age", "sex", "province", "w"]).reset_index(drop=True)
    out = pd.concat([pool.reset_index(drop=True).drop(columns=[c for c in rec if c in pool]), rec], axis=1)
    out["donor"] = donor
    out["match_level"] = level                          # 0=시도까지 일치, 1=전국, 2·3=나이 폭 확대
    m, _ = zero_model(k)
    out["zero_aff"] = m.predict_proba(_X(out))[:, 1]    # 실데이터 학습 모델이 본 제로 수용 확률
    out["trend"] = 0.5 * out.novelty + 0.5 * out.social
    out["drink"] = out.soda
    return out


def record_text(row):
    """LLM 프롬프트에 넣을 실기록 요약(설문 응답 그대로)."""
    lv = lambda x: "매우 높음" if x > .8 else "높음" if x > .6 else "보통" if x > .4 else "낮음" if x > .2 else "매우 낮음"
    s = (f"[이 사람의 실제 설문 응답 기록 — 식품소비행태조사]\n"
         f"- 좋아하는 음료: {row.fav}\n- 제로 음료 섭취: {ZERO_FREQ.get(row.zero_code, '')}")
    if isinstance(row.zero_why, str):
        s += f" (먹는 이유: {row.zero_why})"
    if isinstance(row.zero_no, str):
        s += f" (안 먹는 이유: {row.zero_no})"
    s += (f"\n- 새로운 음식 시도 성향: {lv(row.novelty)}, 가격 민감도: {lv(row.thrift)}, "
          f"건강·다이어트 관심: {lv(row.health)}, SNS·온라인 추천 활용: {lv(row.social)}")
    return s


def segment_truth(k=None, ages=(19, 79)):
    """연령대×성별 '제로음료 월 2회 이상' 가중 비율(검증용 정답)."""
    k = load() if k is None else k
    k = k[k.age.between(*ages)]
    grp = pd.cut(k.age, [18, 29, 39, 49, 59, 69, 200], labels=["20대", "30대", "40대", "50대", "60대", "70대+"])
    y = (k.zero_code <= 2).astype(float)
    return (pd.DataFrame({"g": grp, "sex": k.sex, "yw": y * k.w, "w": k.w})
            .groupby(["g", "sex"], observed=True)[["yw", "w"]].sum().eval("yw / w").unstack())


def validate(model=None, seed=0):
    """보지 않은 실제 응답자(절반)로 채점: 집단(연령대×성별)별 '제로음료 월 2회 이상' 비율.
    ① 시뮬레이터: 나머지 절반 기록을 페르소나에 hot-deck 매칭 → 집단 비율
    ② LLM 단독: 인구통계만 주고 비율을 물음(1회 호출) ③ 기준선: 전체 평균을 모든 집단에 복사
    반환 (표, 지표). 지표: 평균 오차(%p), 집단 간 차이 재현(상관 r, 기울기; 1이면 차이 크기까지 정확)."""
    k = load()
    test = np.random.default_rng(seed).random(len(k)) < 0.5
    truth = segment_truth(k[test])
    from .consumer import load_pool, chat
    pool = pd.read_parquet("data/personas.parquet", columns=["age", "sex", "province"])
    pool = pool[pool.age.between(19, 79)].sample(8000, random_state=seed)
    sim = attach(pool, k[~test].reset_index(drop=True), seed=seed)
    sim_seg = segment_truth(sim.assign(w=1.0))
    tab = pd.DataFrame({"정답(실제 응답자)": truth.stack(), "시뮬레이터": sim_seg.stack()})
    tab["기준선(전체평균)"] = float(np.average(k[test].zero_code <= 2, weights=k[test].w))
    if model:
        import json, re
        segs = [f"{g} {s}" for g, s in tab.index]
        txt = chat("당신은 한국 소비자 조사 전문가입니다. JSON 만 출력하세요.",
                   "2025년 한국 성인 중 제로 음료(제로콜라·제로사이다 등)를 '월 2회 이상' 마시는 사람의 비율을 "
                   "아래 집단별로 추정하세요. 0~1 사이 소수. 형식: {\"20대 남자\": 0.xx, ...}\n집단: " + ", ".join(segs), model)
        guess = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        tab["LLM 단독"] = [float(guess.get(s, np.nan)) for s in segs]
    tab.index = [f"{g} {s}" for g, s in tab.index]
    t = tab["정답(실제 응답자)"]
    met = {}
    for c in tab.columns[1:]:
        e, dt = tab[c] - t, t - t.mean()
        dc = tab[c] - tab[c].mean()
        met[c] = {"평균 오차(%p)": 100 * e.abs().mean(),
                  "차이 재현 r": float(np.corrcoef(dt, dc)[0, 1]) if dc.std() > 0 else 0.0,
                  "차이 기울기": float((dt * dc).sum() / (dt ** 2).sum())}
    return tab, pd.DataFrame(met).T


if __name__ == "__main__":
    k = load()
    print("KREI", k.shape, "α", {a: round(v, 2) for a, v in k.attrs["alpha"].items()})
    assert all(v > 0.5 for v in k.attrs["alpha"].values())
    m, rep = zero_model(k)
    print("AUC", rep["auc"], "\n", rep["coef"].round(2).to_dict())
    assert rep["auc"]["성향+인구통계"] > rep["auc"]["인구통계만"]
    print(segment_truth(k).round(3))
    from mirror.consumer import load_pool
    pool = load_pool().sample(3000, random_state=0)
    a = attach(pool, k)
    assert (a.sex == pool.sex.values).all() and len(a) == 3000
    print("매칭 수준", np.bincount(a.match_level, minlength=4))
    print(a.zero_code.value_counts(normalize=True).sort_index().round(3).to_dict())
    print(record_text(a.iloc[0]))
    tab, met = validate()
    print(tab.round(3)); print(met.round(2))
    assert met.loc["시뮬레이터", "평균 오차(%p)"] < met.loc["기준선(전체평균)", "평균 오차(%p)"]
    print("records ok")
