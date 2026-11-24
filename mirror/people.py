"""페르소나 공장: 여러 방식으로 한국 성인 인구를 만들고, 보지 않은 실제 응답자로 같은 잣대로 채점한다.

방식 (참고한 도구)
- nemotron : 기성 페르소나 + 성·나이·시도 hot-deck (현재 시뮬레이터)
- record   : 실응답자 + 레이킹 가중 재표집 (PopulationSim·simPop 의 list balancing)
- cart     : 변수를 하나씩 나무로 조건부 합성 (R synthpop 방식). 실존 개인의 복사가 아니다
- bayes    : Chow–Liu 나무 구조 베이지안 네트워크 (Doppelganger 방식)
- llm      : LLM 이 짧은 설명만 보고 인물을 지어냄 (TinyTroupe 팩토리·Data Designer 방식). 실데이터를 쓰지 않는다
memory()   : 기록을 인터뷰 기억 문장으로 (genagents 의 기억 구조)
PersonaHub 는 CC BY-NC-SA(비상업)·영어/중국어라 쓰지 않는다.
실행: python -m mirror.people check | bench [seed수]
"""
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import records as R

ROOT = os.path.join(os.path.dirname(__file__), "..")
REG = {**dict.fromkeys(("서울", "경기", "인천"), "수도권"), **dict.fromkeys(("부산", "대구", "울산", "경상북", "경상남"), "영남"),
       **dict.fromkeys(("광주", "전북", "전라남", "제주"), "호남·제주"), **dict.fromkeys(("대전", "세종", "충청북", "충청남", "강원"), "중부")}
OCC = {1: "사무·전문", 2: "사무·전문", 3: "사무·전문", 4: "서비스·판매", 5: "서비스·판매", 6: "생산·단순", 7: "생산·단순", 8: "생산·단순", 9: "생산·단순",
       10: "비취업", 11: "사무·전문", 12: "비취업", 13: "비취업", 14: "비취업", 15: "비취업", 21: "사무·전문"}
EDU = {1: "중졸이하", 2: "중졸이하", 3: "고졸", 4: "대졸+", 5: "대졸+"}
FAV = {1: "주스", 2: "주스", 3: "유제품", 4: "유제품", 5: "유제품", 6: "두유", 7: "탄산", 8: "제로탄산", 9: "이온", 10: "차", 11: "기능성", 12: "탄산수",
       13: "커피", 14: "커피", 15: "전문점", 16: "기타", 17: "물만"}
BANDS, NAMES = [18, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 80], ["19-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-79"]
DEMO = ["age", "sex", "reg", "province", "edu", "mar", "single", "occ", "inc"]
BEH = ["zero", "fav", "novelty", "thrift", "health", "social"]
COLS = DEMO + BEH


def real():
    """KREI 2025 성인 → 범주형 표(COLS) + 가중치 w."""
    k, d = R.load(), pd.read_parquet(R.SRC)
    o = pd.DataFrame({"age": pd.cut(k.age, BANDS, labels=NAMES).astype(str), "sex": k.sex, "province": k.province, "reg": k.province.map(REG),
                      "edu": d.JJ_SQ3_4.map(EDU).fillna("고졸"), "mar": d.JJ_SQ3_7.map({1: "미혼", 2: "기혼"}).fillna("기혼"),
                      "single": d.J_SQ4.map({1: "1인", 2: "다인"}).fillna("다인"), "occ": d.JJ_SQ3_5.map(OCC).fillna("비취업"),
                      "inc": pd.cut(d.J_BA2, [0, 2, 4, 6, 11], labels=["~200만", "~400만", "~600만", "600만+"]).astype(str),
                      "zero": k.zero_code.map({1: "주1회+", 2: "월2~3회", 3: "월1회", 4: "월1회미만", 9: "경험없음"}), "fav": d.A15_1.map(FAV).fillna("기타")})
    for c in ("novelty", "thrift", "health", "social"):
        o[c] = pd.cut(k[c].rank(pct=True, method="average"), [0, .25, .5, .75, 1.01], labels=["낮음", "중하", "중상", "높음"]).astype(str)
    return o[COLS].assign(w=k.w.values)


# ---------- 생성기: gen(train, n, seed) → DataFrame[COLS] ----------
def rake(t, targets, iters=60):
    """IPF: targets = {열 또는 열 튜플: {값: 비율}}. 가중치가 극단으로 튀지 않게 평균의 0.1~10배로 자른다."""
    w = t.w.values / t.w.sum()
    for _ in range(iters):
        for c, tg in targets.items():
            key = t[c].astype(str) if isinstance(c, str) else t[list(c)].astype(str).agg("|".join, axis=1)
            cur = pd.Series(w).groupby(key.values).sum()
            w = w * key.map({k: tg[k] / cur[k] for k in cur.index if k in tg and cur[k] > 0}).fillna(1).values
            w = np.clip(w / w.mean(), .1, 10)
            w = w / w.sum()
    return w


def gen_record(t, n, seed, targets=None):
    w = rake(t, targets) if targets else (t.w / t.w.sum()).values
    return t.sample(n, replace=True, weights=w, random_state=seed)[COLS].reset_index(drop=True)


def gen_cart(t, n, seed, leaf=25):
    """첫 변수는 가중 분포에서, 다음부터는 앞 변수들로 만든 나무의 잎에서 실제 값을 가중 추출."""
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(seed)
    code = {c: {v: i for i, v in enumerate(sorted(t[c].unique()))} for c in COLS}
    X = pd.DataFrame({c: t[c].map(code[c]) for c in COLS}).values
    one = lambda M, k: np.column_stack([(M[:, j][:, None] == np.arange(len(code[COLS[j]]))) for j in range(k)]).astype(np.float32)
    w = (t.w / t.w.sum()).values
    S = np.zeros((n, len(COLS)), int)
    S[:, 0] = rng.choice(X[:, 0], n, p=w)
    for j in range(1, len(COLS)):
        m = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=seed).fit(one(X, j), X[:, j], sample_weight=w)
        lt, ls = m.apply(one(X, j)), m.apply(one(S, j))
        for leaf_id in np.unique(ls):
            ix, at = np.flatnonzero(lt == leaf_id), np.flatnonzero(ls == leaf_id)
            S[at, j] = rng.choice(X[ix, j], len(at), p=w[ix] / w[ix].sum())
    back = {c: {i: v for v, i in code[c].items()} for c in COLS}
    return pd.DataFrame({c: [back[c][v] for v in S[:, j]] for j, c in enumerate(COLS)})


def gen_bayes(t, n, seed, alpha=.5):
    """Chow–Liu: 상호정보량 최대 신장나무를 구조로 쓰고 조건부 확률표(평활)에서 조상부터 추출."""
    rng, w = np.random.default_rng(seed), (t.w / t.w.sum()).values
    val = {c: sorted(t[c].unique()) for c in COLS}
    cod = {c: t[c].map({v: i for i, v in enumerate(val[c])}).values for c in COLS}

    def joint(a, b):
        J = np.zeros((len(val[a]), len(val[b])))
        np.add.at(J, (cod[a], cod[b]), w)
        return J
    mi = {}
    for a, b in itertools.combinations(COLS, 2):
        J = joint(a, b)
        E = J.sum(1, keepdims=True) @ J.sum(0, keepdims=True)
        mi[a, b] = (J[J > 0] * np.log(J[J > 0] / E[J > 0])).sum()
    parent, left = {"age": None}, set(COLS) - {"age"}
    while left:
        a, b = max(((x, y) for x in parent for y in sorted(left)), key=lambda e: mi.get(e, mi.get(e[::-1])))
        parent[b] = a
        left.remove(b)
    S = {}
    for c, p in parent.items():                                  # 삽입 순서 = 조상 먼저
        if p is None:
            S[c] = rng.choice(len(val[c]), n, p=np.bincount(cod[c], w))
        else:
            J = joint(p, c) + alpha / len(t) / len(val[c])
            C = (J / J.sum(1, keepdims=True)).cumsum(1)
            S[c] = np.minimum((rng.random(n)[:, None] > C[S[p]]).sum(1), len(val[c]) - 1)
    return pd.DataFrame({c: np.array(val[c])[S[c]] for c in COLS})


def occ_name(s):
    if s == "무직":
        return "비취업"
    if re.search(r"청소|경비|하역|단순|배달|운전|조작|용접|정비|조립|기계|농|어업|임업|목공|배관|도장|건설|철근|미장|전기공|공$|제조|가공|검침|주차|포장|택배|수리|설치|인쇄|재단|재봉|작업|생산", s):
        return "생산·단순"
    if re.search(r"판매|영업|조리|미용|승무|서비스|주방|매장|상점|요양|간병|보육|캐디|바텐|계산|상품|방문|모델|음료|안내|예식|돌봄", s):
        return "서비스·판매"
    return "사무·전문"


def nemotron(n, seed):
    p = pd.read_parquet(os.path.join(ROOT, "data", "personas_rec.parquet"))
    p = p[p.age.between(19, 79)].sample(n, random_state=seed, replace=n > 25000).reset_index(drop=True)
    e = {"무학": "중졸이하", "초등학교": "중졸이하", "중학교": "중졸이하", "고등학교": "고졸"}
    return p, pd.DataFrame({"age": pd.cut(p.age, BANDS, labels=NAMES).astype(str), "sex": p.sex, "province": p.province, "reg": p.province.map(REG),
                            "edu": p.education_level.map(e).fillna("대졸+"), "mar": np.where(p.marital_status == "미혼", "미혼", "기혼"),
                            "single": np.where(p.family_type.str.startswith("혼자 거주"), "1인", "다인"), "occ": p.occupation.map(occ_name)})


def gen_nemotron(t, n, seed):
    """현재 방식: 인구통계는 Nemotron, 행동·소득은 성·나이·시도(없으면 성·나이)가 같은 실응답자에게서."""
    rng = np.random.default_rng(seed)
    _, q = nemotron(n, seed)
    don, w = np.empty(n, int), t.w.values
    for keys in (["sex", "age"], ["sex", "age", "province"]):     # 뒤(구체적) 조건이 앞을 덮어쓴다
        pool = {k: v.values for k, v in t.reset_index(drop=True).groupby(keys).groups.items()}
        for k, at in q.groupby(keys).groups.items():
            if k in pool:
                don[at] = rng.choice(pool[k], len(at), p=w[pool[k]] / w[pool[k]].sum())
    for c in ["inc"] + BEH:
        q[c] = t[c].values[don]
    return q[COLS]


LLM_VALUES = {"age": NAMES, "sex": ["남자", "여자"], "province": sorted(REG), "edu": ["중졸이하", "고졸", "대졸+"], "mar": ["미혼", "기혼"], "single": ["1인", "다인"],
              "occ": ["사무·전문", "서비스·판매", "생산·단순", "비취업"], "inc": ["~200만", "~400만", "~600만", "600만+"],
              "zero": ["주1회+", "월2~3회", "월1회", "월1회미만", "경험없음"], "fav": sorted(set(FAV.values())), **{c: ["낮음", "중하", "중상", "높음"] for c in ("novelty", "thrift", "health", "social")}}


def gen_llm(n, model, batch=25):
    """실데이터 없이 LLM 이 '2025년 한국 성인 인구를 대표하게' 지어낸 인물. 값이 목록 밖이면 버린다."""
    from . import consumer as C
    out = []
    for k in range(0, n, batch):
        txt = C._chat_raw("합성 인구 생성기. JSON 배열만 출력.", f"2025년 한국의 만 19~79세 인구를 대표하도록 서로 다른 인물 {batch}명을 만드세요(묶음 {k // batch + 1}). 실제 인구 비율을 따르고 값은 아래 목록에서만 고르세요.\n"
                          + "\n".join(f"{c}: {', '.join(v)}" for c, v in LLM_VALUES.items())
                          + "\nzero=제로음료 빈도, fav=가장 좋아하는 음료, novelty=새로움 추구, thrift=가격 비교, health=건강 관심, social=SNS 활용(인구 4등분 중 위치)\n"
                          '형식: [{"age":"25-29","sex":"여자",...}]', model)
        out += [{c: d.get(c) for c in LLM_VALUES} for d in json.loads(re.search(r"\[.*\]", txt, re.S).group(0)) if all(d.get(c) in v for c, v in LLM_VALUES.items())]
    d = pd.DataFrame(out)
    d["reg"] = d.province.map(REG)
    return d[COLS]


# ---------- 채점 ----------
def share(d, cols, w=None):
    return d.assign(_w=1.0 if w is None else w).groupby(cols)._w.sum() / (len(d) if w is None else np.sum(w))


def tvd(a, b):
    return .5 * a.subtract(b, fill_value=0).abs().sum()


AGE3 = {**dict.fromkeys(NAMES[:4], "19-39"), **dict.fromkeys(NAMES[4:8], "40-59"), **dict.fromkeys(NAMES[8:], "60-79")}


def gaps(d, w=None):
    """같은 연령대 안에서 집단(직업·가구·학력·권역)별 행동 비율의 차이."""
    y = pd.DataFrame({"제로월2+": d.zero.isin(["주1회+", "월2~3회"]), "탄산1순위": d.fav.isin(["탄산", "제로탄산", "탄산수"]), "커피1순위": d.fav.eq("커피"),
                      "새로움": d.novelty.isin(["중상", "높음"]), "가격": d.thrift.isin(["중상", "높음"]), "건강": d.health.isin(["중상", "높음"]), "SNS": d.social.isin(["중상", "높음"])}).astype(float)
    w = np.ones(len(d)) if w is None else np.asarray(w)
    out = {}
    for g in ("occ", "single", "edu", "reg"):
        key = [d.age.map(AGE3), d[g]]
        den = pd.Series(w, index=d.index).groupby(key).sum()
        n = d.groupby(key).size()
        m = y.mul(w, axis=0).groupby(key).sum().div(den, axis=0)[n >= 25]
        m = m - m.groupby(level=0).transform("mean")
        for (a, b), r in m.iterrows():
            for c, v in r.items():
                out[g, a, b, c] = v
    return pd.Series(out)


def score(s, test, train=None):
    """합성 s vs 보지 않은 실제 test(가중). 오차는 낮을수록, 집단차이 상관·기울기는 1에 가까울수록 좋다."""
    w = test.w.values
    r = {"단일 분포 오차": np.mean([tvd(share(s, c), share(test, c, w)) for c in COLS]),
         "두 변수 조합 오차": np.mean([tvd(share(s, list(p)), share(test, list(p), w)) for p in itertools.combinations(COLS, 2)]),
         "행동 분포 오차": np.mean([tvd(share(s, c), share(test, c, w)) for c in BEH])}
    a, b = gaps(test, w), gaps(s)
    ix = a.index.intersection(b.index)
    r["집단차이 상관"] = np.corrcoef(a[ix], b[ix])[0, 1] if len(ix) > 10 else np.nan
    r["집단차이 기울기"] = np.polyfit(a[ix], b[ix], 1)[0] if len(ix) > 10 else np.nan
    if train is not None:
        known = set(map(tuple, train[COLS].values))
        r["실존 응답자와 동일"] = np.mean([tuple(x) in known for x in s[COLS].values])
    r["서로 다른 인물"] = len(s[COLS].drop_duplicates()) / len(s)
    return r


def census_targets():
    """나이×성별·시도 목표: Nemotron(KOSIS 정합, 외부 감사에서 최대 1.8%p 차이)의 19~79세 분포."""
    _, q = nemotron(25000, 0)
    key = q[["age", "sex"]].agg("|".join, axis=1)
    return {("age", "sex"): key.value_counts(normalize=True).to_dict(), "province": q.province.value_counts(normalize=True).to_dict()}


def bench(seeds=6, n=5000, llm=None):
    k, tg, rows = real(), census_targets(), []
    for seed in range(seeds):
        te = np.random.default_rng(seed).random(len(k)) < .5
        tr, ts = k[~te].reset_index(drop=True), k[te].reset_index(drop=True)
        half = tr.sample(n, replace=True, weights="w", random_state=seed)[COLS]
        g = {"nemotron": gen_nemotron(tr, n, seed), "record": gen_record(tr, n, seed), "record+레이킹": gen_record(tr, n, seed, tg), "cart": gen_cart(tr, n, seed),
             "bayes": gen_bayes(tr, n, seed), "독립(하한)": pd.DataFrame({c: half[c].sample(frac=1, random_state=seed + i).values for i, c in enumerate(COLS)})}
        if llm is not None:
            g["llm"] = llm
        for nm, s in g.items():
            rows.append({"방식": nm, "seed": seed, **score(s, ts, tr)})
    return pd.DataFrame(rows).drop(columns="seed").groupby("방식").mean()


# ---------- 사람에게 입히기 ----------
def memory(r):
    """genagents 식 기억 문장. 기록에 있는 사실만 쓴다(꾸며 넣지 않음)."""
    z = {"주1회+": "제로 음료를 매주 마신다", "월2~3회": "제로 음료를 한 달에 두세 번 마신다", "월1회": "제로 음료를 한 달에 한 번쯤 마신다", "월1회미만": "제로 음료를 드물게 마신다", "경험없음": "제로 음료를 마셔 본 적이 없다"}
    lv = lambda v: {"낮음": "낮은 편", "중하": "조금 낮은 편", "중상": "조금 높은 편", "높음": "높은 편"}[v]
    home = ("혼자 산다" if r.single == "1인" else "가족과 함께 산다")
    return [f"나는 {r.age}세 {r.sex}이고 {r.province}에 산다.", f"학력은 {r.edu}, 일은 {r.occ}, 결혼 상태는 {r.mar}이고 {home}.", f"우리 집 월 소득은 {r.inc} 구간이다.",
            f"가장 좋아하는 음료 종류는 {r.fav}이고, {z[r.zero]}.", f"새로운 음식을 찾는 정도는 {lv(r.novelty)}, 가격을 비교하는 정도는 {lv(r.thrift)}이다.",
            f"건강과 다이어트 관심은 {lv(r.health)}, SNS와 온라인 추천을 쓰는 정도는 {lv(r.social)}이다."]


def users(d):
    """실제 사용군: 제로를 월 2회 이상 마시거나 탄산류를 1순위로 꼽은 사람."""
    return d[d.zero.isin(["주1회+", "월2~3회"]) | d.fav.isin(["탄산", "제로탄산", "탄산수"])].reset_index(drop=True)


def build(method="cart", n=3000, seed=0):
    """엔진에 넘길 인물 표. 기본값 cart: 실존 개인의 복사가 아닌 합성 인물."""
    k = real()
    return {"record": lambda: gen_record(k, n, seed, census_targets()), "cart": lambda: gen_cart(k, n, seed), "bayes": lambda: gen_bayes(k, n, seed),
            "nemotron": lambda: gen_nemotron(k, n, seed)}[method]()


def check():
    rng = np.random.default_rng(0)
    k = real()
    assert len(k) > 5000 and not k[COLS].isna().any().any()
    w = rake(k, {"sex": {"남자": .3, "여자": .7}})
    assert abs(w[k.sex.values == "여자"].sum() - .7) < .01
    tr = k.sample(2500, random_state=1).reset_index(drop=True)
    for f in (gen_cart, gen_bayes, gen_record, gen_nemotron):
        s = f(tr, 1500, 0)
        assert list(s.columns) == COLS and len(s) == 1500 and not s.isna().any().any(), f.__name__
    # 강한 의존을 심으면 cart·bayes 는 살려야 한다
    z = k.copy()
    z["zero"] = np.where(z.sex == "남자", "주1회+", "경험없음")
    for f in (gen_cart, gen_bayes):
        s = f(z, 2000, 0)
        assert ((s.sex == "남자") == (s.zero == "주1회+")).mean() > .98, f.__name__
    same = score(gen_record(k, 20000, 0), k)
    ind = score(pd.DataFrame({c: rng.choice(k[c], 20000, p=k.w / k.w.sum()) for c in COLS}), k)
    assert same["두 변수 조합 오차"] < ind["두 변수 조합 오차"] and same["집단차이 기울기"] > .9 > ind["집단차이 기울기"]
    assert len(memory(k.iloc[0])) == 6 and len(users(k)) < len(k)
    print("people ok")


if __name__ == "__main__":
    from . import consumer as _C                                # .env 로드
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        b = bench(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
        print(b.round(3).to_string())
        json.dump(b.reset_index().to_dict("records"), open(os.path.join(ROOT, "data", "people_bench.json"), "w"), ensure_ascii=False, indent=1)
    else:
        check()
