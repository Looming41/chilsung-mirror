"""AI 페르소나: 합성 인물 600명이 컨셉을 보고 한 답을, 시뮬레이션 속 사람의 구매 판단과 입소문 반응으로 쓴다.

한 사람에게 한 번 묻고(처음 살 마음 · 친구가 추천했을 때 · 나쁜 후기를 봤을 때 · 올릴 마음 · 만족할 것 같은 정도 · 이유) 그 답을 모든 세상이 함께 쓴다.
- 세상 속 사람 하나를 성향이 가까운 페르소나 한 명이 맡는다
- 살 마음: 비슷한 사람들의 평균 답은 구매의향 모델의 입력 하나로 넣어 패널 응답으로 반영 크기를 배우고(worlds.boots), 성향의 차이는 뽑히지 않은 패널 응답자로 확인된 몫(약 70~80%)만 남기고 빠진 자리를 그 페르소나만의 차이로 채운다
  (사람들 사이 차이의 전체 크기는 바꾸지 않는다). 전체 수준은 패널 TOP2 에 맞춘다
- 나머지 반응: 사람마다의 차이만 바꾸고 평균은 과거 제품으로 걸러낸 값을 지킨다. 차이를 얼마나 크게 볼지는 가정이다
보내는 것: 합성 인물(실존 개인의 복사가 아님)과 컨셉 카드. 패널·KREI 개인 기록은 보내지 않고 이 컴퓨터에서 짝짓는다. 대회가 준 주소로만 보낸다.
실행: python -m mirror.agents check
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd

from . import consumer as C
from . import given as G
from . import multiverse as U
from . import people as P
from . import worlds as W

N, NEAR, WORKERS = 600, 5, 40
DIR = os.path.join(P.ROOT, "data", "agents")
CARD = {"화채": "밀키스 화채. 우유가 든 부드러운 탄산음료 밀키스의 수박화채맛. 여름 한정(5~9월). 1,800원. 편의점·마트·온라인에서 판다.",
        "프리즈": "레쓰비 프리즈. 얼려 먹는 커피, 냉동 파우치. 1,800원. 냉동 진열이 되는 편의점·마트에서 판다.",
        "쥬시": "쥬시모먼트. 비가열 프리미엄 착즙주스, 유리병, 냉장. 3,000원. 편의점·슈퍼·새벽배송에서 판다."}
BUY = ["맛 기대", "새로움·호기심", "가격이 괜찮음", "건강·성분", "브랜드 믿음", "계절·상황", "디자인·재미", "없음"]
HOLD = ["맛 걱정", "가격 부담", "건강·당 걱정", "필요 없음", "살 곳·휴대 불편", "낯섦", "차이를 모름", "없음"]
SYS = "당신은 아래 사람입니다. 마케터처럼 말하지 말고, 이 사람의 습관과 처지에서 솔직하게 답하세요. 관심 없으면 낮은 점수를 주세요.\n"
ASK = ("편의점에서 이 신제품을 봤습니다: {card}\nJSON 하나로만 답하세요. 점수는 1(전혀)~5(매우) 정수.\n"
       '{{"first":처음 봤을 때 살 마음,"friend":친구가 맛있다고 추천한 뒤 살 마음,"bad":맛없다는 후기를 본 뒤 살 마음,"post":마셔 보고 SNS에 올릴 마음,"like":마시면 만족할 것 같은 정도,'
       '"buy":살 이유 번호,"hold":망설일 이유 번호,"yes":"살 이유 한 문장(30자 이내)","no":"망설일 이유 한 문장(30자 이내)"}}\n살 이유: ' + " ".join(f"{i}={x}" for i, x in enumerate(BUY)) + "\n망설일 이유: " + " ".join(f"{i}={x}" for i, x in enumerate(HOLD)))
LV = {"낮음": 0, "중하": 1, "중상": 2, "높음": 3}


@lru_cache(1)
def cast():
    """19~59세를 닮은 합성 인물 600명(늘 같은 사람)."""
    d = U.base()
    return U.synthetic(d.w.values * d.age.between(19, 59).values, N, seed=0)


@lru_cache(1)
def _sig():
    return hashlib.sha1("\n".join([SYS, ASK, json.dumps(CARD, ensure_ascii=False, sort_keys=True)] + [" ".join(P.memory(r)) for r in cast().itertuples()]).encode()).hexdigest()[:12]


def parse(t):
    m = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    o = {k: int(m[k]) for k in ("first", "friend", "bad", "post", "like", "buy", "hold")}
    if not all(1 <= o[k] <= 5 for k in ("first", "friend", "bad", "post", "like")) or not (0 <= o["buy"] < len(BUY) and 0 <= o["hold"] < len(HOLD)):
        raise ValueError("범위 밖")
    return o | {k: re.sub(r"\s+", " ", str(m[k]))[:60] for k in ("yes", "no")}


def answer(i, k, model):
    q = ASK.format(card=CARD[k])
    for more in ("", "\n다른 글 없이 JSON 하나만 출력하세요."):                  # 못 읽는 답도 저장되므로 질문을 바꿔 한 번 더
        t = C.chat(SYS + "\n".join(P.memory(cast().iloc[i])), q + more, model)
        try:
            return parse(t) | {"i": i}
        except (AttributeError, KeyError, TypeError, ValueError):
            if more:
                raise ValueError("답을 읽지 못함") from None


def _file(k, model):
    return os.path.join(DIR, f"{k}-{model}.json")


def load(k, model=None):
    """저장된 답. 모델을 안 정하면 가장 최근 것. 인물이나 질문이 바뀌었으면 쓰지 않는다."""
    fs = [_file(k, model)] if model else [os.path.join(DIR, f) for f in (os.listdir(DIR) if os.path.isdir(DIR) else []) if f.startswith(k + "-") and f.endswith(".json")]
    for f in sorted((f for f in fs if os.path.exists(f)), key=os.path.getmtime, reverse=True):
        with open(f) as h:
            d = json.load(h)
        if d["sig"] == _sig():
            return d
    return None


def collect(k, model, tick=None):
    """600명에게 묻고 저장한다. 앞의 한 명으로 연결을 먼저 확인하고, 90% 를 못 채우면 쓰지 않는다."""
    d = load(k, model)
    if d:
        return d
    rows, err = [], ""
    try:
        rows.append(answer(0, k, model))
    except ValueError as e:
        err = str(e)
    with ThreadPoolExecutor(WORKERS) as ex:
        for f in as_completed([ex.submit(answer, i, k, model) for i in range(1, N)]):
            try:
                rows.append(f.result())
            except Exception as e:
                err = str(e)[:80] if type(e) in (RuntimeError, ValueError) else type(e).__name__
                if "잔액" in err:
                    ex.shutdown(cancel_futures=True)
                    raise RuntimeError(err) from None
            if tick:
                tick(len(rows))
    if len(rows) < .9 * N:
        raise RuntimeError(f"AI 페르소나 {N}명 가운데 {len(rows)}명만 답했습니다({err}). 다시 실행하면 답한 사람은 건너뜁니다")
    d = {"sig": _sig(), "model": model, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": sorted(rows, key=lambda r: r["i"])}
    os.makedirs(DIR, exist_ok=True)
    tmp = _file(k, model) + ".tmp"
    with open(tmp, "w") as h:
        json.dump(d, h, ensure_ascii=False)
    os.replace(tmp, _file(k, model))
    return d


# ---------- 시뮬레이션 속 사람과 짝짓기 ----------
def near(A, B, n=NEAR):
    """A 의 사람마다 B(합성 인물)에서 가장 가까운 n 명. 열: 나이, 여성, 신제품, 건강, 가격 비교[, SNS]."""
    s = np.array([8, .35, 1.2, 1.2, 1.2, 1.2])[:A.shape[1]]
    D = (((A[:, None, :] - B[None, :, :]) / s) ** 2).sum(2)
    return np.argsort(D, 1, kind="stable")[:, :n]


def _cast_x(ix):
    c = cast().iloc[ix]
    return np.column_stack([c.age.str.extract(r"(\d+)")[0].astype(float) + 2, (c.sex == "여자") * 1.0, *(c[x].map(LV).astype(float) for x in ("novelty", "health", "thrift", "social"))])


def _people_x():
    pe, X = W.ready()["people"], W.ready()["X"]
    return np.column_stack([X[:, 0] * 11 + 38, X[:, 2], *(pe[x].map(LV).astype(float) for x in ("novelty", "health", "thrift", "social"))])


def _panel_x():
    """패널에는 SNS 성향이 없어 다섯 열만. 1~5점을 패널 안 사분위로 바꾼다."""
    r, _ = G.panel()
    q = lambda c: np.minimum(3, (r[c].rank(pct=True) - 1e-9) * 4).astype(int).astype(float)
    return np.column_stack([2026 - r.출생연도, (r.성별 == "여성") * 1.0, q("신제품시도성향"), q("건강관심도"), q("가격민감도")])


def effects(k, d):
    """저장된 답 → worlds.one 에 넘길 값. 세상마다 사람 하나를 가까운 페르소나 다섯 가운데 한 명이 맡는다(m: 응답자 × 다섯 명)."""
    t = pd.DataFrame(d["rows"])
    B, w = _cast_x(t.i.values), W.ready()["people"].w.values
    m, mp = near(_people_x(), B), near(_panel_x(), B[:, :5])
    avg = lambda v: np.average(v[m].mean(1), weights=w)                          # 실제 인구에서의 평균
    first, like = t["first"].values.astype(float), t.like.values.astype(float)
    up, down = np.clip(t.friend.values - first, 0, None) + .25, np.clip(first - t.bad.values, 0, None) + .25
    rel = lambda v: np.clip(v / avg(v), .2, 3)                                   # 평균은 지키고 사람마다의 차이만
    mu, sd = avg(first), max(first.std(), .3)
    s, sp = (first[m].mean(1) - mu) / sd, (first[mp].mean(1) - mu) / sd
    own = first[m] - first[m].mean(1, keepdims=True)                             # 비슷한 사람들 사이에서 이 페르소나만의 차이
    j = list(W.IDS).index(k)
    coef = W.boots(extra={j: sp})
    keep = float(np.clip(np.median([x[3][0] for x in coef]), .3, 1))             # 성향의 차이 가운데 패널이 확인해 준 몫
    b = np.array([x[0][j] for x in coef])                                        # 마지막 값이 AI 답의 계수
    return {"k": k, "m": m, "s": s, "keep": keep, "own": own / max(np.sqrt(np.average((own ** 2).mean(1), weights=w)), .3), "B": b, "a": np.array([x[1][j] for x in coef]), "post": rel(t.post.values.astype(float)), "rec": rel(up), "bad": rel(down),
            "sat": 1 / (1 + np.exp(-(np.log(.82 / .18) + .6 * (like - avg(like))))), "buy": t.buy.values, "hold": t.hold.values, "text": t[["yes", "no"]].values.tolist(), "info": proof(j, sp, t, b, d) | {"keep": keep}}


_fx = {}


def get(k, model=None, tick=None):
    """시뮬레이션에 넘길 값. model(연결된 모델)이 있으면 그 모델의 답을 쓰고 없으면 묻는다. 연결이 없으면 저장된 답, 그것도 없으면 None."""
    d = collect(k, model, tick) if model else load(k)
    if not d:
        return None
    key = (k, d["model"], d["at"])
    if key not in _fx:
        _fx[key] = effects(k, d)
    return _fx[key]


def said(ws, ai):
    """세상들에서 산 사람·알고도 안 산 사람을 맡은 페르소나가 든 이유의 비율과 대표 문장."""
    t = np.array([w["agg"][ai["k"]]["ai"] for w in ws], float).sum(0)
    out = []
    for q, (names, col) in enumerate(((BUY, "buy"), (HOLD, "hold"))):
        c = t[q]
        pick = lambda i: next((ai["text"][n][q] for n in np.flatnonzero(ai[col] == i) if ai["text"][n][q]), "")
        out.append([{"why": names[i], "share": float(c[i] / max(c.sum(), 1)), "say": pick(i)} for i in np.argsort(-c) if c[i] > 0 and names[i] != "없음"][:6])
    return {"buy": out[0], "hold": out[1]}


def proof(j, sp, t, b, d):
    """AI 답이 패널의 실제 응답을 얼마나 맞히나(5겹 교차검증 판별력, 찍기 0.5)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    r, _ = G.panel()
    X, y = W.panel_X(), G.wide("T2")[:, j].astype(int)
    au = lambda Z: float(np.mean([roc_auc_score(y, cross_val_predict(LogisticRegression(C=.5, max_iter=1000), Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=q), method="predict_proba")[:, 1]) for q in range(10)]))
    return {"n": len(t), "model": d["model"], "at": d["at"], "top2": float((t["first"] >= 4).mean()), "panel": float(np.average(y, weights=r.w)), "alone": float(roc_auc_score(y, sp)), "base": au(X), "with": au(np.column_stack([X, sp])),
            "coef": [float(x) for x in np.quantile(b[:, -1], [.05, .5, .95])]}


def check():
    c = cast()
    assert len(c) == N
    ok = '앞말 {"first":3,"friend":4,"bad":2,"post":1,"like":4,"buy":0,"hold":1,"yes":"a","no":"b"}'
    assert parse(ok)["friend"] == 4
    for bad in ("싫어요", ok.replace('"first":3', '"first":9'), ok.replace('"hold":1', '"hold":8')):
        try:
            parse(bad)
        except (AttributeError, ValueError):
            continue
        raise AssertionError(bad)
    # 가짜 답: 절반은 사고 싶고 절반은 아니다(성향과 무관) → 전체 의향 수준은 그대로, 산 사람은 사고 싶다고 한 쪽에 몰린다
    hot = np.random.default_rng(0).random(N) < .5
    rows = [{"i": i, "first": 5 if y else 1, "friend": 5 if y else 1, "bad": 1, "post": 2, "like": 4, "buy": 1 if y else 7, "hold": 7 if y else 3, "yes": "신기해서", "no": "필요 없음"} for i, y in enumerate(hot)]
    ai = effects("화채", {"rows": rows, "model": "시험", "at": ""})
    A, Z = W.run(60, workers=1, ai=ai), W.run(60, workers=1)
    v = np.sum([x["agg"]["화채"]["ai"][0] for x in A], 0)
    m = lambda ws: np.mean([x["intent"]["화채"] for x in ws])
    assert v[1] / v.sum() > .55 and abs(m(A) - m(Z)) < .03, (v, m(A), m(Z))
    assert "ai" not in Z[0]["agg"]["화채"] and [x["buy"]["쥬시"] for x in A] == [x["buy"]["쥬시"] for x in Z] and A[3]["buy"] == W.one(3, ai=ai)["buy"]
    n = W.one(3, log="화채", ai=ai)["log"]["nodes"]
    assert all(x["say"] == ("신기해서" if x["bought"] else "필요 없음") for x in n)
    print("agents ok · 산 사람 중 사고 싶다던 페르소나", round(v[1] / v.sum(), 2), "· 의향", round(m(Z), 3), "→", round(m(A), 3), "· 구매", round(np.mean([x["buy"]["화채"] for x in Z]), 4), "→", round(np.mean([x["buy"]["화채"] for x in A]), 4))


if __name__ == "__main__":
    check()
