"""본선 과제(주어진 데이터): 정제 → 설문 편향 보정 → 과거 출시 실적으로 환산 → 반복 계산 예측.

자료는 모두 대회가 제공한 '가상 데이터'다(data/given, git 제외). 실제 롯데칠성 실적이 아니다.
핵심 식: 12개월 구매경험률 = 보정 구매의향 TOP2 × 살 수 있는 점포 비율 × 전환계수 k
  k 는 과거 5개 제품의 '구매경험률 ÷ (TOP2 × 취급점포율)'. 취급점포율로 나누면 제품마다 크게 다르던 환산계수가 비슷해진다(validate 로 확인).
과거 컨셉테스트가 어떤 정제·가중을 거친 TOP2 인지는 자료에 없다 → 원본 기준 결과도 함께 낸다.
실행: python -m mirror.given check | report
"""
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from . import people as P

DIR = os.path.join(P.ROOT, "data", "given")
A, B, C_ = (os.path.join(DIR, f) for f in ("3-1_내부데이터_POS_제품자산.xlsx", "3-2_소비자조사_패널데이터.xlsx", "3-3_구매여정_외부트렌드.xlsx"))
CON = ["밀키스 화채", "레쓰비 프리즈", "쥬시모먼트"]
LIKERT = ["컨셉이해도", "호감도", "차별성", "구매의향", "가격수용도"]
# 소매가·출고가(과제 설명서), 판매 채널, 냉동/냉장 조건
SPEC = {"밀키스 화채": dict(srp=1800, ship=1200, can=(1200, 780), chan=["편의점", "대형마트", "SSM", "온라인"], infra=None, cat=["MK01", "MK02", "MK03"], months="5~9월 한정"),
        "레쓰비 프리즈": dict(srp=1800, ship=1250, can=None, chan=["편의점", "대형마트", "SSM"], infra="냉동 음료 진열 가능 점포율", cat=["LB01", "LB02"], months="5월 출시"),
        "쥬시모먼트": dict(srp=3000, ship=2100, can=None, chan=["편의점", "SSM", "온라인(새벽배송)"], infra="냉장 쇼케이스 보유율", cat=["XD01"], months="3월 출시")}
# 사람이 정한 가정(화면에서 바꿀 수 있음). 근거를 함께 적는다
ASSUME = {"입점률|밀키스 화채": (.85, .95, "과거 밀키스 2종 취급점포율 0.88~0.92"),
          "입점률|레쓰비 프리즈": (.55, .95, "냉동 매대가 있어도 실제로 들여놓는 점포 비율. 근거 자료 없음. 아이스크림 냉동고 혼합진열 불가 다수"),
          "입점률|쥬시모먼트": (.35, .80, "아래 끝은 프레시원(냉장 14일) 실제 취급 35%. 소비기한이 45일로 길어 위로 열어 둠"),
          "유도문항|쥬시모먼트": (.65, 1.0, "쥬시모먼트만 '정성스럽게 만든 건강한…' 설명을 붙여 물었고, 정가 4,000원→3,000원으로 보여 줌. 얼마나 부풀었는지는 자료로 알 수 없음")}


def have():
    return all(os.path.exists(f) for f in (A, B, C_))


@lru_cache(16)
def sheet(f, s):
    return pd.read_excel(f, s)


# ---------- 정제 ----------
@lru_cache(1)
def pos():
    """정제한 POS(낱개 환산, 중복·테스트 제거). missing=POS 집계 누락 달, ret=반품(음수)."""
    p = sheet(A, "1_POS_판매").drop_duplicates()
    p = p[p.제품코드 != "TEST001"].copy()
    box = p.수량단위 != "EA"
    p["EA"] = np.where(box, p.판매수량 * 24, p.판매수량)
    p["ret"], p["missing"] = p.EA < 0, p.기준월 == "2025-11"
    return p.merge(sheet(A, "3_권역별기온").rename(columns={"평균기온(℃)": "temp"}), on=["기준월", "권역"])


@lru_cache(1)
def respondents():
    """모든 응답자 + 제외 사유."""
    r = sheet(B, "1_응답자정보").copy()
    age, lo, hi = 2026 - r.출생연도, r.연령대.str[:2].astype(int), r.연령대.str[3:5].astype(int)
    # 불성실 = 빠른 응답이면서 모든 문항이 같은 값
    same = sheet(B, "2_컨셉평가").groupby("응답자ID")[LIKERT].apply(lambda g: g.values.std() == 0)
    r["제외"] = ""
    r.loc[r.휴대폰해시.duplicated(), "제외"] = "중복 응답(같은 휴대폰)"
    r.loc[(r["응답시간(초)"] < 150) & r.응답자ID.map(same), "제외"] = "불성실(2분 30초 안에 전부 같은 값)"
    r.loc[~((age >= lo - 1) & (age <= hi + 1)) & (r.제외 == ""), "제외"] = "출생연도와 연령대 불일치"
    return r


def rake(r, w0=None):
    """모집단 구성비(연령×성별, 권역)에 맞춘 가중치(평균 1)."""
    t = sheet(B, "4_모집단구성비")
    tg = {"cell": t[t.구분 == "연령×성별"].set_index("항목").iloc[:, 1], "거주권역": t[t.구분 == "거주권역"].set_index("항목").iloc[:, 1]}
    k, w = {"cell": (r.연령대 + " " + r.성별).values, "거주권역": r.거주권역.values}, np.ones(len(r)) if w0 is None else np.asarray(w0, float)
    for _ in range(40):
        for c, v in tg.items():
            cur = pd.Series(w).groupby(k[c]).sum()
            f = v.reindex(cur.index) / v.reindex(cur.index).sum() * w.sum() / cur
            w = w * f.reindex(k[c]).values
    return w / w.mean()


@lru_cache(1)
def panel():
    """정제한 응답자 r(가중 w)와 컨셉평가 e(T2, 제시순서)."""
    r = respondents()
    r = r[r.제외 == ""].reset_index(drop=True)
    r["w"] = rake(r)
    e = sheet(B, "2_컨셉평가").merge(r[["응답자ID", "w"]], on="응답자ID")
    e["T2"] = (e.구매의향 >= 4) * 1.0
    return r, e


def wide(col):
    r, e = panel()
    return e.pivot(index="응답자ID", columns="컨셉", values=col).loc[r.응답자ID, CON].values


def adjusted(w, T2=None, pos_=None):
    """가중 TOP2 를 제시 순서 효과(공통 기울기)로 보정해 '세 자리 평균'에 맞춘 값(컨셉 3)."""
    T2, pos_ = wide("T2") if T2 is None else T2, wide("제시순서") if pos_ is None else pos_
    n = len(T2)
    X = np.column_stack([np.tile(np.eye(3), (n, 1)), (pos_ - 2).ravel()])
    W = np.repeat(np.asarray(w), 3)
    b = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ T2.ravel())
    return np.clip(b[:3], .01, .99), float(b[3])


def stages():
    """보정 단계별 TOP2: 원본 → 불성실 제거 → 모집단 가중 → 순서 보정, 그리고 '제시 가격을 낼 수 있다'까지."""
    e0 = sheet(B, "2_컨셉평가")
    r, e = panel()
    adj, slope = adjusted(r.w.values)
    g = lambda d, f: d.groupby("컨셉").apply(f, include_groups=False)
    k = e[e.최대지불의향가격.notna()]
    t = pd.DataFrame({"원본": g(e0, lambda d: (d.구매의향 >= 4).mean()), "불성실·중복 제거": g(e, lambda d: d.T2.mean()), "+ 모집단 가중": g(e, lambda d: np.average(d.T2, weights=d.w)),
                      "+ 순서 보정": pd.Series(adj, CON), "제시 가격을 낼 수 있음": g(k, lambda d: np.average(d.최대지불의향가격 >= d.제시가격, weights=d.w)),
                      "의향도 있고 가격도 냄": g(k, lambda d: np.average((d.최대지불의향가격 >= d.제시가격) & (d.T2 == 1), weights=d.w))}).loc[CON]
    return t, {"응답자": len(r), "유효표본": float(r.w.sum() ** 2 / (r.w ** 2).sum()), "순서 기울기": slope}


# ---------- 과거 실적 → 전환계수 ----------
def past():
    t = sheet(A, "6_과거컨셉테스트_실적").iloc[:, :6].copy()
    t.columns = ["제품", "연도", "카테고리", "TOP2", "점포율", "경험률"]
    t["의향만 환산"], t["k"] = t.경험률 / t.TOP2, t.경험률 / (t.TOP2 * t.점포율)
    return t


def validate():
    """하나씩 빼고 맞히기(5개 제품): 의향만 / 의향×점포율."""
    t, rows = past(), []
    for i in range(len(t)):
        o, x = t.drop(index=i), t.loc[i]
        gm = lambda s: float(np.exp(np.log(s).mean()))
        rows.append({"제품": x.제품, "실제": x.경험률, "의향만 환산": x.TOP2 * gm(o["의향만 환산"]), "의향 × 점포율 × k": x.TOP2 * x.점포율 * gm(o.k)})
    v = pd.DataFrame(rows)
    return v, {c: float((v[c] / v.실제 - 1).abs().mean()) for c in v.columns[2:]}


def miss_factor():
    """2025-11 POS 누락 보정: 출고 ÷ POS 의 중앙값(그 달)."""
    p, o = pos(), sheet(A, "2_출고_sell-in")
    q = (o[o.기준월 == "2025-11"].groupby("제품코드").출고수량.sum() / p[p.missing].groupby("제품코드").EA.sum()).dropna()
    return float(q.median())


@lru_cache(1)
def market():
    """POS 에서 배운 것: 구매자 1인당 수량(과거 2제품), 실판매가 비율, 채널 비중, 캔 비중."""
    from . import multiverse as U
    p, t, d = pos(), past().set_index("제품"), U.base()
    pop = float(d.w[d.age.between(19, 59)].sum())
    fix = p.assign(EA=np.where(p.missing, p.EA * miss_factor(), p.EA))
    units = {"밀키스 딸기(시즌한정)": fix[fix.제품코드 == "MK04"].EA.sum(), "밀키스 제로": fix[(fix.제품코드 == "MK03") & fix.기준월.between("2025-03", "2026-02")].EA.sum()}
    price = {c: float(p[p.제품코드 == c].판매금액.sum() / p[p.제품코드 == c].EA.sum() / s) for c, s in (("MK04", 1900), ("MK03", 1900), ("MK02", 1900), ("DM01", 2300))}
    share = {}
    for k, s in SPEC.items():
        x = p[p.제품코드.isin(s["cat"])].groupby("판매채널").EA.sum()
        share[k] = (x / x.sum()).to_dict()
    can = p[p.제품코드.isin(["MK01", "MK02"])].groupby("제품코드").EA.sum()
    return {"인구": pop, "1인당 수량": {k: float(v / (t.경험률[k] * pop)) for k, v in units.items()}, "실판매가 비율": price, "채널": share, "캔 비중": float(can["MK01"] / can.sum())}


def fe(d, y, X, groups):
    Z = d[[y, *X]].astype(float).copy()
    for _ in range(30):
        for g in groups:
            Z = Z - Z.groupby(d[g].values).transform("mean")
    X = [x for x in X if Z[x].std() > 1e-6]
    return dict(zip(X, np.linalg.lstsq(Z[X].values, Z[y].values, rcond=None)[0]))


@lru_cache(1)
def elasticity():
    """행사 배수(제품×월, 제품×권역×채널 고정효과)와 제품별 기온 1℃당 판매 변화율(비행사)."""
    q = pos()
    q = q[(q.EA > 0) & ~q.missing].copy()
    q["lg"], q["rc"], q["pm"] = np.log(q.EA), q.권역 + q.판매채널 + q.제품코드, q.제품코드 + q.기준월
    ev = ["1+1", "2+1", "할인10%", "할인20%"]
    for k in ev:
        q[k] = (q.행사구분 == k) * 1.0
    b = fe(q, "lg", ev, ["rc", "pm"])
    n = q[q.행사구분 == "없음"]
    return {k: float(np.exp(v)) for k, v in b.items()}, {c: float(np.exp(fe(g, "lg", ["temp"], ["rc"])["temp"]) - 1) for c, g in n.groupby("제품코드") if len(g) > 100 and c not in ("MK04", "LB03")}


# ---------- 예측(반복 계산) ----------
def reach(con):
    """채널 비중으로 가중한 '이론상 놓일 수 있는' 비율: 판매 채널 × 냉동/냉장 인프라."""
    s, m, inf = SPEC[con], market()["채널"][con], sheet(A, "5_유통채널_인프라").set_index("채널")
    out = 0.0
    for c, v in m.items():
        c2 = "온라인(새벽배송)" if c == "온라인" else c
        if c in s["chan"] or c2 in s["chan"]:
            out += v * (float(inf.loc[c2, s["infra"]]) if s["infra"] else 1.0)
    return out


def forecast(worlds=2000, seed=0, assume=None, raw=False, t2=None):
    """회차마다: 응답자 재표집(가중 다시) → 순서 보정 TOP2 → 입점률·전환계수·1인당 수량·실판매가를 뽑아 구매경험률과 매출.

    과거 취급점포율은 '전 채널' 기준이라 살 수 있는 비율 = reach × 입점률 로 맞춘다(밀키스 화채는 reach 를 1 로 본다: 입점률 가정이 이미 과거 취급점포율).
    raw=True 면 정제 전 TOP2(과거 조사도 원본이었다고 볼 때). t2={컨셉: 의향}이면 그 값으로 고정(만약에)."""
    g, as_ = np.random.default_rng(seed), {k: v[:2] for k, v in ASSUME.items()} | (assume or {})
    r, _ = panel()
    T2, ps, m, pa = wide("T2"), wide("제시순서"), market(), past()
    rawp, lk = stages()[0]["원본"].values, np.log(pa.k.values)
    per = np.log(list(m["1인당 수량"].values()))
    n, rows, base_reach = len(r), [], {c: 1.0 if SPEC[c]["infra"] is None else reach(c) for c in CON}
    for i in range(worlds):
        ix = g.integers(0, n, n)
        if raw:
            b = np.clip(g.normal(rawp, np.sqrt(rawp * (1 - rawp) / 304)), .01, .99)
        else:
            w = rake(r.iloc[ix].reset_index(drop=True)) if i % 4 == 0 else r.w.values[ix]       # ponytail: 네 번에 한 번만 가중을 다시 계산(느려서)
            b = adjusted(w, T2[ix], ps[ix])[0]
        k = np.exp(lk.mean() + lk.std(ddof=1) * np.sqrt(1 + 1 / len(lk)) * g.standard_t(len(lk) - 1, 3))
        u = np.exp(per.mean() + g.normal(0, .35))                                              # 1인당 수량: 과거 2제품 평균, 폭은 가정
        for j, c in enumerate(CON):
            s = SPEC[c]
            top = (t2 or {}).get(c, b[j])
            if c == "쥬시모먼트" and c not in (t2 or {}):
                top = top * g.uniform(*as_["유도문항|쥬시모먼트"])
            d = base_reach[c] * g.uniform(*as_["입점률|" + c])
            trial = float(np.clip(top * d * k[j], 0, 1))
            units = trial * m["인구"] * u
            disc = g.uniform(.72, .93)
            can = m["캔 비중"] if s["can"] else 0.0
            list_, ship = (s["srp"] * (1 - can) + (s["can"][0] * can if can else 0)), (s["ship"] * (1 - can) + (s["can"][1] * can if can else 0))
            rows.append((i, c, top, d, k[j], trial, units, units * list_ * disc / 1e8, units * ship / 1e8))
    return pd.DataFrame(rows, columns=["회차", "컨셉", "보정 의향", "살 수 있는 비율", "전환계수", "구매경험률", "수량", "소매 매출(억)", "출고 매출(억)"])


def summary(F):
    w1 = lambda v: F.pivot(index="회차", columns="컨셉", values=v).idxmax(axis=1).value_counts(normalize=True)
    top, tr, rows = w1("출고 매출(억)"), w1("구매경험률"), []
    for c in CON:
        q = lambda v, x=.5: float(F[F.컨셉 == c][v].quantile(x))
        rows.append({"컨셉": c, "보정 의향": q("보정 의향"), "살 수 있는 비율": q("살 수 있는 비율"), "구매경험률": q("구매경험률"), "하한": q("구매경험률", .05), "상한": q("구매경험률", .95),
                     "소매 매출(억)": q("소매 매출(억)"), "매출 하한": q("소매 매출(억)", .05), "매출 상한": q("소매 매출(억)", .95), "출고 매출(억)": q("출고 매출(억)"),
                     "경험률 1등": float(tr.get(c, 0)), "매출 1등": float(top.get(c, 0))})
    return pd.DataFrame(rows)


def whatif():
    """지렛대: 하나만 바꿨을 때 컨셉별 구매경험률(중앙값)이 어떻게 변하나."""
    med = lambda f, c: float(f[f.컨셉 == c].구매경험률.median())
    base, st = forecast(400), stages()[0]["의향도 있고 가격도 냄"]
    sc = [("프리즈: 입점률을 최대로(95%)", "레쓰비 프리즈", dict(assume={"입점률|레쓰비 프리즈": (.95, .95)})),
          ("프리즈: 입점률이 55%에 그침", "레쓰비 프리즈", dict(assume={"입점률|레쓰비 프리즈": (.55, .55)})),
          ("쥬시: 프레시원처럼 35%만 취급", "쥬시모먼트", dict(assume={"입점률|쥬시모먼트": (.35, .35)})),
          ("쥬시: 의향도 있고 3,000원도 낸다는 사람만", "쥬시모먼트", dict(t2={"쥬시모먼트": float(st["쥬시모먼트"])})),
          ("화채: 의향도 있고 1,800원도 낸다는 사람만", "밀키스 화채", dict(t2={"밀키스 화채": float(st["밀키스 화채"])})),
          ("프리즈: 의향도 있고 1,800원도 낸다는 사람만", "레쓰비 프리즈", dict(t2={"레쓰비 프리즈": float(st["레쓰비 프리즈"])}))]
    return pd.DataFrame([{"만약에": n, "컨셉": c, "지금": med(base, c), "바꾸면": med(forecast(400, **kw), c)} for n, c, kw in sc])


# ---------- 누가 사나: 패널 → 가상 소비자(KREI) ----------
@lru_cache(1)
def persona_model():
    """컨셉별 능형 로지스틱(나이 2차, 성별, 신제품 시도, 건강, 가격민감, 여성×나이). 5겹 교차검증 AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    r, _ = panel()
    X = design(pd.DataFrame({"나이": 2026 - r.출생연도, "여성": (r.성별 == "여성") * 1.0, "신제품": r.신제품시도성향, "건강": r.건강관심도, "가격민감": r.가격민감도}))
    Y, o = wide("T2"), {}
    for j, c in enumerate(CON):
        m = LogisticRegression(C=.5, max_iter=2000)
        p = cross_val_predict(m, X, Y[:, j], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        o[c] = (m.fit(X, Y[:, j], sample_weight=r.w), float(roc_auc_score(Y[:, j], p)))
    return o


def design(d):
    a, n, h, t = (d.나이 - 38) / 11, (d.신제품 - 3.4) / .95, (d.건강 - 3.3) / .95, (d.가격민감 - 3.5) / .95
    return np.column_stack([a, a ** 2, d.여성, d.여성 * a, n, h, t]).astype(float)


def personas(w=None):
    """KREI 응답자(19~59세)에 패널 성향 분포를 백분위로 옮겨 컨셉별 의향 확률을 붙인다."""
    from . import multiverse as U
    from . import youth as Y
    d, (r, _) = U.base(), panel()
    keep = d.age.between(19, 59).values
    q = lambda c, v: Y.qmap(d[c].values, d[c].values[keep], np.repeat(r[v].values, np.maximum(1, (r.w * 4).round().astype(int))))
    f = pd.DataFrame({"나이": d.age, "여성": (d.sex == "여자") * 1.0, "신제품": q("novelty", "신제품시도성향"), "건강": q("health", "건강관심도"), "가격민감": q("thrift", "가격민감도")})
    out = d.loc[keep].assign(w=(d.w.values if w is None else np.asarray(w))[keep])
    for c, (m, _) in persona_model().items():
        out[c] = m.predict_proba(design(f[keep]))[:, 1]
    return out[out.w > 0]


# ---------- 장부 ----------
def audit():
    """자료에서 찾은 문제와 처리. (자료, 문제, 처리, 확신)"""
    p0, p, n = sheet(A, "1_POS_판매"), pos(), respondents().제외.value_counts()
    log, si = sheet(C_, "2_구매여정_온라인로그"), sheet(A, "2_출고_sell-in")
    f15 = [3, 3, 5, 2, 5, 4, 5, 2, 3, 2, 4, 2, 3, 1, 4]                                          # FGI 표의 레쓰비 프리즈 점수
    mk = p[p.제품코드 == "MK04"].groupby("행사구분").EA.sum()
    ratio = si[si.제품코드 == "MK04"].출고수량.sum() / p[p.제품코드 == "MK04"].EA.sum()
    rows = [("패널조사", f"중복 {n.get('중복 응답(같은 휴대폰)', 0)}명, 불성실 {n.get('불성실(2분 30초 안에 전부 같은 값)', 0)}명(전부 같은 값, 구매의향 100%), 나이 모순 {n.get('출생연도와 연령대 불일치', 0)}명", "제외", "높음"),
            ("패널조사", "20대 여성·수도권에 치우침(20대 여성 81명, 50대 남성 13명)", "모집단 구성비로 가중. 유효표본은 273명이 아니라 158명", "높음"),
            ("패널조사", "제시 순서: 밀키스 화채가 61%에서 첫 번째. 원본에서는 첫 번째일수록 의향이 높아 보임", "보정하되, 불성실 제거·가중 뒤에는 순서 효과가 거의 사라짐(자리당 -0.7%p)", "중간"),
            ("패널조사", "쥬시모먼트만 좋게 설명하고 물음(Q4-B) + '정가 4,000원→3,000원' 표기", "의향을 65~100%로 깎는 가정(자료로는 크기를 알 수 없음)", "낮음"),
            ("패널조사", "밀키스 화채는 PET 1,800원만 물음. 캔 1,200원은 조사되지 않음", "캔 의향은 알 수 없음. 매출 계산에서만 캔 비중 반영", "높음"),
            ("POS", f"중복 {int(p0.duplicated().sum())}행, 테스트상품 {int((p0.제품코드 == 'TEST001').sum())}행", "삭제", "높음"),
            ("POS", f"온라인 캔 3종이 2026년부터 BOX(24) 단위 {int((p0.수량단위 != 'EA').sum())}행", "×24 로 낱개 환산", "높음"),
            ("POS", f"2025-11 판매가 모든 제품에서 약 {1 - 1 / miss_factor():.0%} 적음(출고는 정상)", "집계 누락으로 보고 출고 비율로 보정", "중간"),
            ("POS", f"온라인 음수 {int(p.ret.sum())}행", "반품으로 보고 합계에는 포함, 추정에서는 제외", "중간"),
            ("POS", f"밀키스 딸기 판매의 {mk.get('1+1', 0) / mk.sum():.0%}가 1+1 행사분", "1인당 수량·실판매가 비율에 그대로 반영", "높음"),
            ("출고", f"밀키스 딸기 출고가 실판매의 {ratio:.2f}배(초도 밀어내기)", "수요 추정에 출고를 쓰지 않음", "높음"),
            ("제품자산DB", "LB03 코드가 단종 제품과 리뉴얼안에 중복 사용, ID01 은 출시일이 중단일보다 늦음, LB03 은 단종일(6월) 뒤 10월까지 판매", "참고만", "높음"),
            ("제품자산DB", "비슷한 과거: 밀키스 수박(2021, 70억) · 얼려먹는 주스 파우치(개발중단, 사유 없음) · 프리미엄 착즙 2세대(손익분기 미달) · 델몬트 프리미엄 포도(3,000원, 가격 대비 수요 부족)", "예측 결과와 대조", "높음"),
            ("과거 컨셉테스트", "의향만 쓰면 환산계수가 0.07~0.23 으로 흔들림", "취급점포율로 나누면 0.19~0.25 로 모임 → 예측식에 사용(제품 5개뿐)", "중간"),
            ("유통 인프라", "편의점 냉동 음료 진열 가능 31%, 아이스크림 냉동고 혼합 진열 불가 다수", "프리즈의 상한으로 사용", "높음"),
            ("FGI", f"'12/15명이 프리즈에 높은 관심'이라고 적혔으나 표의 4점 이상은 {sum(x >= 4 for x in f15)}명", "표의 숫자를 따름", "높음"),
            ("FGI", "RTD 커피를 자주 사는 사람 우선 모집, 수도권만, 검수 전 초안, 쥬시 가격 '대부분 수용'은 패널 가격 응답과 반대", "정량 추정에 쓰지 않고 이유 문장만 참고", "높음"),
            ("오프라인 관찰", "2027-05 출시 예정인 밀키스 화채를 2026-08 에 구매한 기록 3건", "시점이 맞지 않아 구매 증거로 쓰지 않음", "높음"),
            ("온라인 로그", f"봇으로 보이는 쿠키(B…) {int(log.쿠키ID.str.startswith('B').sum())}건, 클릭이 노출보다 많음, 시간대 KST·UTC 섞임", "이 예측에는 쓰지 않음", "높음"),
            ("검색·SNS", "'수박화채'는 2025-07 한 번 튄 뒤 2026-07 은 39. 연관검색어 62%가 만드는 법·레시피(음료 아님)", "수요 근거로 쓰지 않음", "높음"),
            ("검색·SNS", "'얼려먹는 커피' 2026-06~08 SNS 의 70~79%가 광고·협찬", "자발적 관심으로 읽지 않음", "높음"),
            ("컨셉보드", "화채 '과일이 쏙쏙' 인데 과일농축액 0.1%, 프리즈는 고카페인 표기", "기대·실망 위험으로 표시(숫자에는 반영하지 않음)", "중간"),
            ("POS", "카니발: 밀키스 제로 수도권 선출시로 추정하려 했으나 통제 제품도 같이 움직여 판별 불가", "매출에서 빼지 않음. '순증이 아닌 총매출'로 읽어야 함", "중간")]
    return pd.DataFrame(rows, columns=["자료", "문제", "처리", "확신"])


def check():
    p = pos()
    assert (p.수량단위 != "EA").any() and "TEST001" not in set(p.제품코드) and p.duplicated().sum() == 0
    assert p[p.제품코드 == "LB01"].groupby("수량단위").apply(lambda g: g.판매금액.sum() / g.EA.sum(), include_groups=False).max() < 1300      # 박스를 낱개로 바꾸면 단가가 낱개 값으로 돌아온다
    r, e = panel()
    assert len(r) == 273 and len(e) == 273 * 3 and abs(np.average(r.성별 == "여성", weights=r.w) - .5) < .01 and 150 < r.w.sum() ** 2 / (r.w ** 2).sum() < 170
    a, b = adjusted(np.ones(len(r)), np.ones((len(r), 3)) * .5)
    assert np.allclose(a, .5) and abs(b) < 1e-9
    _, er = validate()
    assert er["의향 × 점포율 × k"] < er["의향만 환산"]
    assert reach("레쓰비 프리즈") < .5 < reach("밀키스 화채") and 1.2 < miss_factor() < 1.4
    s = summary(forecast(120)).set_index("컨셉")
    assert abs(s["매출 1등"].sum() - 1) < 1e-9 and (s.하한 < s.구매경험률).all() and (s.구매경험률 < s.상한).all()
    lo, hi = (summary(forecast(120, assume={"입점률|레쓰비 프리즈": (x, x)})).set_index("컨셉").구매경험률["레쓰비 프리즈"] for x in (.3, .9))
    assert 2 < hi / lo < 4
    assert personas()[CON].apply(lambda x: x.between(0, 1)).all().all() and len(audit()) > 15
    print("given ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 60)
        print(stages()[0].round(3).to_string(), stages()[1], "\n")
        print(past().round(3).to_string(), "\n", validate()[0].round(3).to_string(), validate()[1], "\n", market(), "\n", elasticity(), {c: round(reach(c), 3) for c in CON})
        print(summary(forecast(2000)).round(3).T.to_string(), "\n원본 의향 기준\n", summary(forecast(2000, raw=True)).round(3).T.to_string(), "\n", whatif().round(4).to_string(), {c: round(v[1], 3) for c, v in persona_model().items()})
        q_ = personas()
        print(q_.groupby(["age10", "sex"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in CON}), include_groups=False).round(2).to_string())
    else:
        check()
