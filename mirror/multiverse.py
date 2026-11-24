"""시나리오 비교: 회차마다 사람·마음·시장·확산·운이 다르고, 모든 회차의 결과를 모아 결론을 낸다.

- 사람: 회차마다 실응답자를 베이지안 부트스트랩('다른 사람들이 조사됐다면')하고 그 회차의 구성원을 새로 뽑는다.
  고른 집단의 실응답자가 적으면 비슷한 사람에게서 빌려 온다(빌린 비율을 항상 표시). 1만 명을 복제하는 것과 달리 불확실성이 줄어든 척하지 않는다.
- 마음·시장·확산: 정답을 모르는 값들을 PRIOR 의 폭 안에서 회차마다 다르게 뽑는다. 폭은 사람이 정한 가정이다.
- 같은 회차 안에서는 모든 시나리오가 같은 구성원·같은 조건·같은 운을 쓴다.
- 심리: 동조(여러 지인의 추천), 희소성(한정 물량), 피로(반복 노출), 제품 고유 매력(모델이 모르는 호감)도 회차마다 다르다.
'1등 비율'은 조건을 바꿔 계산한 횟수 중의 비율이지 현실에서 이길 확률이 아니다. 실제 출시 실적이 생기면 reweight 로 회차의 비중을 바꾼다.
실행: python -m mirror.multiverse check
"""
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd

from . import choice as M
from . import consumer as C
from . import people as P
from . import records as R
from . import stats as S
from . import viral as V
from . import youth as Y

AGE10 = {"19-24": "20대", "25-29": "20대", "30-34": "30대", "35-39": "30대", "40-44": "40대", "45-49": "40대", "50-54": "50대", "55-59": "50대", "60-64": "60대", "65-69": "60대", "70-79": "70대"}
# 화면에 보이는 카테고리 → (열, 선택지). 같은 카테고리 안은 '또는', 카테고리끼리는 '그리고'
CATS = {"연령대": ("age10", ["10대", "20대", "30대", "40대", "50대", "60대", "70대"]), "성별": ("sex", ["남자", "여자"]), "권역": ("reg", ["수도권", "중부", "영남", "호남·제주"]),
        "시도": ("province", sorted(P.REG)), "결혼": ("mar", ["미혼", "기혼"]), "가구": ("single", ["1인", "다인"]), "학력": ("edu", ["중졸이하", "고졸", "대졸+"]),
        "직업군": ("occ", ["사무·전문", "서비스·판매", "생산·단순", "비취업"]), "가구소득(월)": ("inc", ["~200만", "~400만", "~600만", "600만+"]),
        "제로음료 빈도": ("zero", ["주1회+", "월2~3회", "월1회", "월1회미만", "경험없음"]), "가장 좋아하는 음료": ("fav", sorted(set(P.FAV.values()))),
        **{n: (c, ["낮음", "중하", "중상", "높음"]) for n, c in (("새로움 추구", "novelty_q"), ("가격 비교", "thrift_q"), ("건강 관심", "health_q"), ("SNS 활용", "social_q"))}}
STRICT = {"age10", "sex", "zero", "fav"}                    # 빌려 올 때 더 지키는 조건(행동과 가장 관련)
# 희소성 상한 1.0: 공급형 희소성 메타분석 δ=0.50(쾌락재 0.55, Barton 외 2022 J. Retailing)을 로그 오즈(×π/√3)로 바꾼 값. 실험실 의향이라 0~그 값 사이로 둔다.
# 동조·피로의 방향은 문헌에 있으나(Centola 2010 등) 크기는 확인하지 못해 폭을 사람이 정했다.
PRIOR = {"가격 민감도": (.35, "로그정규"), "반응 가중치": (.25, "로그정규"), "시장 분위기": (.35, "정규"), "제품 고유 매력": (.25, "정규"), "실구매 전환": ((.15, .50), "균등"), "광고 도달": ((.01, .5), "로그정규"),
         "SNS 전파력": ((1.0, 1.8), "균등"), "유행 지속": ((.40, .75), "균등"), "게시 성향": (.40, "로그정규"), "악평 영향": ((.3, .8), "균등"),
         "동조": ((0.0, .6), "균등"), "희소성": ((0.0, 1.0), "균등"), "피로": ((0.0, .3), "균등")}
WORLD = [k for k in PRIOR if k not in ("반응 가중치", "제품 고유 매력")]           # 회차 전체에 하나씩인 조건
TERMS = ["기본", "음료 선택구조", "제로 수용", "제로 거부", "새로움 추구", "고연령 거부감", "패키지×SNS", "문구", "가격 부담"]     # consumer.utility 의 항 순서


def pkey(sc):
    """설문·매력이 공유되는 '제품'(알리는 방법만 다른 시나리오는 같은 제품)."""
    return "|".join(str(sc[k]) for k in ("price", "zero", "flavor", "package", "copy")) + ("|한정" if sc.get("stock") else "")


@lru_cache(1)
def base():
    """실응답자 성인 5,874명 + 10대 604명: 효용 계산에 필요한 연속 성향 + 조건 선택용 범주."""
    k, q = R.load(), P.real()
    Z, _ = M.person_features()
    d = pd.concat([k.reset_index(drop=True), pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])], axis=1)
    m, _ = R.zero_model(k)
    d["zero_aff"], d["trend"], d["drink"] = m.predict_proba(R._X(k))[:, 1], .5 * k.novelty + .5 * k.social, k.soda
    for c in ("reg", "mar", "single", "edu", "occ", "inc", "zero"):
        d[c] = q[c].values
    d["fav_cat"], d["age10"], d["band"] = q.fav.values, q.age.map(AGE10).values, q.age.values
    for c in ("novelty", "thrift", "health", "social"):
        d[c + "_q"] = q[c].values
    d = pd.concat([d, Y.items(True)[1]], axis=1)
    y = Y.load(d, Z)
    out = pd.concat([d, y], ignore_index=True)
    out.attrs["youth_auc"] = y.attrs["auc"]
    return out


def synthetic(w, n, seed=0):
    """가중치 w 의 집단을 닮은 합성 인물(순차 나무, 실존 개인의 복사가 아님). LLM 에 보낼 때 쓴다."""
    d = base()
    t = pd.DataFrame({c: d[c] for c in ("sex", "reg", "province", "edu", "mar", "single", "occ", "inc", "zero")}).assign(
        age=d.band, fav=d.fav_cat, w=w, **{c: d[c + "_q"] for c in ("novelty", "thrift", "health", "social")})
    return P.gen_cart(t[t.w > 0].reset_index(drop=True), n, seed)


def col(c):
    return "fav_cat" if c == "fav" else c


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0


def select(filters, floor=200):
    """filters {열: [값]} → (가중치, 정보). 조건에 맞는 실응답자의 유효표본이 floor 보다 작으면, 조건을 어긴 정도에 따라 가중을 깎아 비슷한 사람을 빌린다."""
    d = base()
    mis, hit = np.zeros(len(d)), np.ones(len(d), bool)
    for c, v in filters.items():
        if v:
            bad = ~d[col(c)].isin(v).values
            hit &= ~bad
            mis += bad * (2.0 if c in STRICT else 1.0)
    w = d.w.values * hit
    info = {"인구": float(w.sum()), "실응답자": int(hit.sum()), "유효표본": ess(w), "빌린 비율": 0.0}
    if 0 < info["유효표본"] < floor or info["실응답자"] == 0:
        lo, hi = 0.0, 12.0
        for _ in range(40):
            lam = (lo + hi) / 2
            lo, hi = (lam, hi) if ess(d.w.values * np.exp(-lam * mis)) > floor else (lo, lam)
        w = d.w.values * np.exp(-lo * mis)
        info["빌린 비율"] = float(1 - w[hit].sum() / w.sum())
    info["오차"] = 1.96 * np.sqrt(.25 / info["유효표본"]) if info["유효표본"] > 0 else np.nan
    info["신호"] = "충분" if info["유효표본"] >= 385 else "보통" if info["유효표본"] >= 100 else "부족"
    return w, info


def relax(filters):
    """어느 조건을 풀면 근거가 가장 많이 늘어나나."""
    out = []
    for c, v in filters.items():
        if v:
            out.append((c, select({k: x for k, x in filters.items() if k != c})[1]["유효표본"]))
    return sorted(out, key=lambda x: -x[1])


def need(n_sc, margin=.03, diff=.05, p=.25, power=.8):
    """권장 인원: 비율 ±margin(95%) 과 '시나리오 간 diff 차이 탐지(모든 쌍 본페로니 보정)' 중 큰 값."""
    from scipy.stats import norm
    pairs = max(1, n_sc * (n_sc - 1) // 2)
    p2 = p - diff
    cmp = (norm.ppf(1 - .025 / pairs) + norm.ppf(power)) ** 2 * (p * (1 - p) + p2 * (1 - p2)) / diff ** 2
    return {"정밀도": S.sample_size(margin), "비교": int(np.ceil(cmp)) if n_sc > 1 else 0, "권장": int(max(S.sample_size(margin), np.ceil(cmp) if n_sc > 1 else 0))}


# ---------- 회차 ----------
def draw(r, spread=1.0):
    """회차의 조건 한 벌. spread 로 조건 변동 폭을 줄이거나 늘린다(0 이면 모두 기준값)."""
    u = lambda a, b: (a + b) / 2 + (r.uniform(a, b) - (a + b) / 2) * spread
    ln = lambda s: r.lognormal(0, s * spread)
    return {"가격 민감도": ln(.35), "시장 분위기": r.normal(0, .35 * spread), "실구매 전환": u(.15, .50), "광고 도달": .01 * ln(.5), "SNS 전파력": u(1.0, 1.8),
            "유행 지속": u(.40, .75), "게시 성향": ln(.40), "악평 영향": u(.3, .8), "동조": u(0, .6), "희소성": u(0, 1.0), "피로": u(0, .3)}


def appeal(seed, key, sd):
    """제품 고유 매력: 회차와 제품이 같으면 항상 같은 값."""
    return sd * np.random.default_rng([seed, zlib.crc32(key.encode())]).standard_normal() if sd else 0.0


def one(seed, w, scs, concept, n=500, days=28, spread=1.0, fix=None, post=None):
    """회차 하나: {시나리오: (의향, 구매율, 놓친 수요)}, 회차 조건. fix 로 일부 조건을 고정하면 '만약에', post(현장 설문 사후분포)가 있으면 마음을 거기서 뽑는다."""
    r, d = np.random.default_rng(seed), base()
    c = draw(r, spread) | (fix or {})
    coef, ap = r.lognormal(0, .25 * spread, len(TERMS)), {k: appeal(seed, k, .25 * spread) for k in {pkey(x) for x in scs.values()}}
    if post is not None:
        v = post.draw(seed)
        coef = v["coef"]
        ap |= {k: x for k, x in v["appeal"].items() if k in ap}
        c |= {k: v[k] for k in ("가격 민감도", "시장 분위기") if k not in (fix or {})}
    g = w * r.exponential(size=len(d))                        # 베이지안 부트스트랩
    s = d.iloc[r.choice(len(d), n, p=g / g.sum())].reset_index(drop=True)
    nb, dg = V.build_network(n, seed=seed)
    s = s.iloc[np.argsort(-s.trend.values + r.normal(0, .15, n))].reset_index(drop=True).iloc[np.argsort(np.argsort(-dg))].reset_index(drop=True)   # 팔로워 많은 자리에 유행 민감한 사람(잡음 섞음)
    res = {}
    for name, sc in scs.items():
        t = C.utility(s, sc, M.concept_attrs(concept, sc))[1]
        u = sum(k * x * (c["가격 민감도"] if n_ == "가격 부담" else 1.0) for k, (n_, x) in zip(coef, t.items())) + c["시장 분위기"] + ap[pkey(sc)]
        p = C.likert_pmf(u)[:, 3:].sum(1)
        o = V.simulate(p * c["실구매 전환"], np.clip(C.post_rate(s, sc) * c["게시 성향"], 0, 1), C.satisfaction(s, sc), nb, dg, days=days, ad_reach=c["광고 도달"],
                       influencer_seed=sc["influencer"], shortform=c["SNS 전파력"] * (1.0 if sc["package"].startswith("비주얼") else .7), decay=c["유행 지속"],
                       neg_penalty=c["악평 영향"], rng=np.random.default_rng(seed), conform=c["동조"], fatigue=c["피로"], scarce=c["희소성"], stock=sc.get("stock") or None,
                       trait=.5 + s.trend.values)
        res[name] = (float(p.mean()), o["adopters"] / n, o["missed"] / n)
    return res, c | {f"반응|{k}": v for k, v in zip(TERMS, coef)} | {f"매력|{k}": ap[pkey(x)] for k, x in scs.items()}


def run(filters, scs, concept, worlds=300, n=500, days=28, spread=1.0, fix=None, seed=0, tick=None, post=None):
    w, info = select(filters)
    rows = []
    for i in range(worlds):
        res, c = one(seed * 100003 + i, w, scs, concept, n, days, spread, fix, post)
        rows.append({"회차": i, **c, **{f"의향|{k}": v[0] for k, v in res.items()}, **{f"구매|{k}": v[1] for k, v in res.items()}, **{f"놓침|{k}": v[2] for k, v in res.items()}})
        if tick and i % 10 == 9:
            tick((i + 1) / worlds)
    return pd.DataFrame(rows), info


def summary(W, weight=None):
    """시나리오별: 1등 비율(동점 나눔), 구매율·의향의 5/50/95%, 나쁜 경우(95%)에서 놓치는 구매율, 기준(1번) 대비 차이."""
    B = W[[c for c in W if c.startswith("구매|")]]
    I = W[[c for c in W if c.startswith("의향|")]]
    nm = [c[3:] for c in B]
    wt = np.ones(len(W)) / len(W) if weight is None else np.asarray(weight) / np.sum(weight)

    def qt(x, q):
        o = np.argsort(x)
        return float(np.interp(q, np.cumsum(wt[o]) - wt[o] / 2, np.asarray(x)[o]))
    top = B.values == B.values.max(1, keepdims=True)
    share, reg = (top / top.sum(1, keepdims=True) * wt[:, None]).sum(0), B.values.max(1)[:, None] - B.values
    rows = []
    for j, k in enumerate(nm):
        df = B.values[:, j] - B.values[:, 0]
        rows.append({"시나리오": k, "1등 비율": share[j], "구매율": qt(B.values[:, j], .5), "구매 하한": qt(B.values[:, j], .05), "구매 상한": qt(B.values[:, j], .95),
                     "의향": qt(I.values[:, j], .5), "의향 하한": qt(I.values[:, j], .05), "의향 상한": qt(I.values[:, j], .95),
                     "나쁜 경우 손실": qt(reg[:, j], .95), "기준 대비": qt(df, .5), "기준 대비 하한": qt(df, .05), "기준 대비 상한": qt(df, .95)})
    e = np.sqrt(share * (1 - share) / ess(wt))                # 반복 수가 유한해서 생기는 오차
    t = pd.DataFrame(rows)
    t["1등 오차"] = 1.96 * e
    return t.sort_values("1등 비율", ascending=False).reset_index(drop=True)


def lead(W, name):
    B = W[[c for c in W if c.startswith("구매|")]]
    return B["구매|" + name] - B.drop(columns="구매|" + name).max(axis=1) if B.shape[1] > 1 else B.iloc[:, 0]


def conditions(W):
    """회차를 가르는 조건 열: 시장 조건 + 반응 가중치 + 제품 매력(변하는 것만)."""
    return [c for c in W if (c in WORLD or c.startswith(("반응|", "매력|"))) and W[c].std() > 1e-12]


def drivers(W, name):
    """어떤 조건이 이 시나리오의 승패를 가르나. 격차(1등과 차점의 구매율 차)와의 순위상관, 이긴 경우 수, 조건들이 격차를 설명하는 비율(교차검증 R², 나머지는 구성원 차이과 운)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    ld, ps = lead(W, name), conditions(W)
    win = ld >= 0
    rows = [{"조건": c, "이긴 경우 평균": W.loc[win, c].mean(), "진 경우 평균": W.loc[~win, c].mean(), "차이(표준편차)": (W.loc[~win, c].mean() - W.loc[win, c].mean()) / W[c].std(),
             "격차와의 상관": W[c].corr(ld, method="spearman")} for c in ps]
    r2 = np.nan
    if len(W) >= 40 and ld.std() > 0 and ps:
        X, y = ((W[ps] - W[ps].mean()) / W[ps].std()).values, ld.values
        r2 = max(0.0, float(1 - ((y - cross_val_predict(RidgeCV(np.logspace(-1, 4, 12)), X, y, cv=5)) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    return pd.DataFrame(rows).sort_values("격차와의 상관", key=abs, ascending=False).reset_index(drop=True), int(win.sum()), r2


def reweight(W, concept_sc, target, filters=None, days=40, sd=.5, n=400):
    """현실 갱신: 이미 출시된 제품을 각 회차에서 돌려 실제 시용률(target 하한·상한)에 가까운 회차의 비중을 키운다(로그정규 커널)."""
    w, _ = select(filters or {})
    goal = np.log(np.sqrt(target[0] * target[1]))
    lw = []
    for i, r in W.iterrows():
        res, _ = one(int(r.회차), w, {"x": concept_sc}, "", n, days, 1.0, {k: r[k] for k in WORLD})
        lw.append(-.5 * ((np.log(max(res["x"][1], .5 / n)) - goal) / sd) ** 2)
    lw = np.array(lw)
    return np.exp(lw - lw.max())


def representative(W, name):
    """(가장 전형적인 이긴 경우, 가장 전형적인 진 경우) 번호."""
    ld, X = lead(W, name).values, W[conditions(W)]
    X = ((X - X.mean()) / X.std().replace(0, 1)).values
    med = lambda m: int(W.회차[np.flatnonzero(m)[((X[m] - X[m].mean(0)) ** 2).sum(1).argmin()]]) if m.any() else None
    return med(ld >= 0), med(ld < 0)


def check():
    b = dict(zero=1, flavor="새로운 맛", package="비주얼(숏폼형)", copy="재미")
    sc = {"싼 것": {**b, "price": 1500, "influencer": False}, "비싼 것": {**b, "price": 2500, "influencer": False}}
    w, i = select({})
    assert i["실응답자"] == len(base()) and i["빌린 비율"] == 0 and 3300 < i["유효표본"] < 4500 and 44e6 < i["인구"] < 46e6
    assert select({"age10": ["10대"]})[1]["실응답자"] == 604
    w2, j = select({"age10": ["20대"], "sex": ["여자"], "reg": ["수도권"], "single": ["1인"]})
    assert j["신호"] == "부족" and j["빌린 비율"] > .3 and ess(w2) > 180
    assert relax({"age10": ["20대"], "single": ["1인"]})[0][1] > select({"age10": ["20대"], "single": ["1인"]})[1]["유효표본"]
    n = need(8)
    assert n["권장"] == n["비교"] > need(2)["비교"] > 1000 and need(1)["권장"] == S.sample_size(.03)
    W, _ = run({}, sc, "자두맛 탄산음료", worlds=40, n=600)
    t = summary(W)
    assert t.시나리오[0] == "싼 것" and t["1등 비율"][0] > .85 and abs(t["1등 비율"].sum() - 1) < 1e-9
    Z, _ = run({}, sc, "자두맛 탄산음료", worlds=40, n=600, spread=0.0)
    assert Z["구매|싼 것"].std() < W["구매|싼 것"].std() and Z["광고 도달"].std() < 1e-12         # 폭 0 이면 조건은 같고 사람·운만 다르다
    lim, _ = run({}, {"한정": {**b, "price": 1500, "influencer": True, "stock": .05}, "상시": {**b, "price": 1500, "influencer": True}}, "자두맛 탄산음료", worlds=15, n=600, fix={"광고 도달": .05})
    assert lim["구매|한정"].max() <= .05 + 1e-9 and (lim["놓침|상시"] == 0).all() and lim["놓침|한정"].max() > 0 and lim["의향|한정"].corr(lim["의향|상시"]) > .5
    again, _ = run({}, sc, "자두맛 탄산음료", worlds=5, n=600)
    assert np.allclose(again.iloc[:, 1:].values, W.iloc[:5, 1:].values)                        # 같은 씨앗이면 같은 결과
    s2 = summary(W, np.where(W["구매|비싼 것"] >= W["구매|싼 것"], 1.0, 1e-9) if (W["구매|비싼 것"] >= W["구매|싼 것"]).any() else None)
    assert abs(s2["1등 비율"].sum() - 1) < 1e-9
    d, k, r2 = drivers(W, "싼 것")
    assert len(d) == len(WORLD) + len(TERMS) + 2 and 0 < k <= 40 and (np.isnan(r2) or 0 <= r2 <= 1) and len(synthetic(select({"age10": ["10대"]})[0], 40)) == 40 and representative(W, "싼 것")[0] is not None
    print("multiverse ok")


if __name__ == "__main__":
    check()
