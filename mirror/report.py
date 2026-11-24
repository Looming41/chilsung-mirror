"""결과 보고: 누가 샀나 · 왜 샀나 · 왜 안 샀나 · 어떻게 사게 됐나 · 그래서 무엇을 할까.

두 가지를 섞지 않고 따로 낸다.
- 계산: 1,500개 세상에서 산 사람·안 산 사람의 집계(모델이 본 까닭)
- 사람의 말: 패널 응답자가 직접 쓴 자유응답(대회 제공 가상 데이터). 살 사람(4~5점)과 안 살 사람으로 나눠 가중 비율을 낸다
실행: python -m mirror.report check | show
"""
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from . import given as G
from . import worlds as W

# 자유응답 문장 → 이유. 서로 다른 문장이 34개뿐이라 사람이 분류했다
WHY = {"밀키스 특유의 부드러운 맛이 기대돼요": "맛", "어릴 때 먹던 화채 생각나서 좋아요": "추억", "여름에 시원하게 마시기 좋을 것 같아요": "계절·상황", "한정판이라 한 번쯤 마셔볼 것 같아요": "한정판", "사진 찍어 올리기 좋은 디자인이에요": "디자인", "수박화채맛이라 호기심이 생겨요": "호기심",
       "너무 달 것 같아요": "단맛 걱정", "수박맛이 인위적일까 걱정돼요": "인공 맛 걱정", "밀키스랑 크게 다르지 않아 보여요": "차이를 모름", "과일 함량이 낮다고 해서 기대가 줄었어요": "과일 함량", "여름 한정이면 굳이 찾아 마시진 않을 듯해요": "굳이 안 찾음",
       "여름에 슬러시처럼 먹으면 좋을 것 같아요": "계절·상황", "레쓰비라서 맛은 믿고 먹을 수 있을 듯해요": "브랜드 믿음", "얼려 먹는 커피라니 신선해요": "새로움", "카페 프라페 대체로 괜찮을 것 같아요": "카페 대신",
       "냉동이면 편의점에서 살 수 있을지 모르겠어요": "살 곳 걱정", "녹으면 어떻게 하나요? 휴대가 어려울 듯해요": "휴대 불편", "얼려 먹는다는 게 무슨 뜻인지 잘 모르겠어요": "이해 안 됨", "카페인이 높다는 게 부담이에요": "카페인", "파우치를 어떻게 먹는지 불편할 것 같아요": "먹는 법",
       "패키지가 예쁘고 건강해 보여요": "디자인·건강", "브랜드 컨셉이 깔끔하고 믿음이 가요": "브랜드 믿음", "과즙이 진할 것 같아요": "맛", "정성스럽게 만든 느낌이라 선물하기도 좋아요": "선물", "델몬트 같은 기존 주스랑 뭐가 다른지 모르겠어요": "차이를 모름",
       "처음 보는 브랜드라 아직 믿음이 안 가요": "낯선 브랜드", "이 가격이면 카페에서 사 마실래요": "가격", "가격이 너무 비싼 것 같아요": "가격", "유통기한이 짧을까 봐 걱정돼요": "유통기한"}
DO = {"단맛 걱정": "당 함량과 덜 단 맛을 앞면에 적는다", "인공 맛 걱정": "시음 행사로 맛을 먼저 확인시킨다", "차이를 모름": "기존 제품과 무엇이 다른지 한 문장으로 비교해 보여 준다", "과일 함량": "'과일이 쏙쏙' 문구를 실제 함량(농축액 0.1%)에 맞게 고친다", "굳이 안 찾음": "여름 한 철 진열을 눈에 띄는 자리에 몰아 준다",
      "살 곳 걱정": "냉동 매대가 있는 점포를 알려 주고 입점을 먼저 넓힌다", "휴대 불편": "사서 바로 먹는 상황(점심 뒤, 이동 중)을 광고에서 보여 준다", "이해 안 됨": "얼려 먹는 법을 3단계 그림으로 앞면에 넣는다", "카페인": "카페인을 줄인 안을 함께 시험한다", "먹는 법": "파우치 먹는 장면을 짧은 영상으로 보여 준다",
      "낯선 브랜드": "델몬트 같은 기존 브랜드를 보증으로 붙인다", "가격": "3,000원보다 낮은 가격이나 작은 용량을 시험한다", "유통기한": "45일이라는 기간과 비가열 공정을 앞면에 적는다"}
EMPTY = {"좋아요", "ㅇㅇ", "네", ".", "괜찮아요"}
SEG = [(a, s) for a in W.AGES for s in W.SEXES]
BAND = {"20대": "19~29세", "30대": "30~39세", "40대": "40~49세", "50대": "50~59세"}
ROUTE = {"unaware": "알지 못함", "noplace": "파는 곳이 없음", "bad": "불만 글을 봄", "cold": "알지만 끌리지 않음"}


def josa(w, a, b):
    """받침에 따라 조사를 고른다: josa('여자','이','가')."""
    c = ord(w[-1]) - 0xAC00
    return w + (a if 0 <= c <= 11171 and c % 28 else b)


@lru_cache(1)
def voice():
    """응답자 × 컨셉: 이유, 살 사람 여부, 가중치, 특성."""
    r, e = G.panel()
    d = e.merge(r[["응답자ID", "성별", "연령대", "직업", "주이용채널", "커피음료_월구매횟수", "탄산_월구매횟수", "주스_월구매횟수", "신제품시도성향", "건강관심도", "가격민감도"]], on="응답자ID")
    d["이유"] = d.자유응답.map(WHY)
    return d


def words(con, flt=None):
    """(살 사람이 든 이유, 안 살 사람이 든 이유, 응답자 수)."""
    d, f = voice(), flt or {}
    d = d[(d.컨셉 == con) & ~d.자유응답.isin(EMPTY)]
    if f.get("age10"):
        d = d[d.연령대.isin([BAND[a] for a in f["age10"]])]
    if f.get("sex"):
        d = d[d.성별.isin([s.replace("자", "성") for s in f["sex"]])]

    def top(v, worry):
        if not len(v):
            return []
        g = v.groupby(["이유", "자유응답"]).agg(w=("w", "sum"), n=("w", "size")).reset_index()
        out = [{"why": k, "share": float(x.w.sum() / g.w.sum()), "n": int(x.n.sum()), "say": x.loc[x.w.idxmax(), "자유응답"], "do": DO.get(k, "")} for k, x in g.groupby("이유") if (k in DO) == worry]
        return sorted(out, key=lambda x: -x["share"])[:6]
    return top(d[d.T2 == 1], False), top(d[d.T2 == 0], True), int(len(d))


def lift(con):
    """걱정 하나가 사라지면 의향이 얼마나 오르나(상한): 그 이유를 든 사람이 나머지 사람 평균만큼 사고 싶어진다고 볼 때."""
    d = voice()
    d = d[d.컨셉 == con]
    base, out = float(np.average(d.T2, weights=d.w)), []
    for k, g in d[d.이유.isin(DO)].groupby("이유"):
        o = d[d.이유 != k]
        sh = float(g.w.sum() / d.w.sum())
        out.append({"why": k, "say": g.groupby("자유응답").w.sum().idxmax(), "do": DO[k], "share": sh, "gain": float(sh * (np.average(o.T2, weights=o.w) - np.average(g.T2, weights=g.w)))})
    return base, sorted(out, key=lambda x: -x["gain"])


def cards(con, k=3):
    """대표 응답자(패널 행): 살 사람(안 살 사람)이 많은 집단부터, 그 집단이 가장 많이 든 이유를 말한 사람."""
    d = voice()
    d = d[(d.컨셉 == con) & ~d.자유응답.isin(EMPTY)]
    col, nm = {"밀키스 화채": ("탄산_월구매횟수", "탄산"), "레쓰비 프리즈": ("커피음료_월구매횟수", "커피음료"), "쥬시모먼트": ("주스_월구매횟수", "주스")}[con]

    def pick(v):
        out, used = [], set()
        for seg, _ in v.groupby(["연령대", "성별"]).w.sum().sort_values(ascending=False).items():      # 그런 사람이 많은 집단부터
            g = v[(v.연령대 == seg[0]) & (v.성별 == seg[1]) & ~v.이유.isin(used)]
            if not len(g):
                continue
            g = g[g.이유 == g.groupby("이유").w.sum().idxmax()].sort_values("w", ascending=False)       # 그 집단이 가장 많이 든 이유
            r = g.iloc[0]
            used.add(r.이유)
            tr = [t for t, x in (("새 제품을 잘 시도", r.신제품시도성향 >= 4), ("건강을 챙김", r.건강관심도 >= 4), ("가격에 민감", r.가격민감도 >= 4)) if x]
            out.append({"who": f"{r.연령대} {r.성별} · {r.직업}", "life": f"{r.주이용채널} 이용 · {nm} 월 {int(r[col])}회", "trait": ", ".join(tr) or "두드러진 성향 없음", "score": int(r.구매의향), "say": r.자유응답})
            if len(out) == k:
                break
        return out
    return pick(d[(d.T2 == 1) & ~d.이유.isin(DO)]), pick(d[(d.T2 == 0) & d.이유.isin(DO)])


def crowd(ws, k):
    """세상들의 집계: 집단별 구매경험률(범위), 산 사람 구성, 성향별, 경로, 안 산 까닭."""
    A = np.array([w["agg"][k]["seg"] for w in ws], float)                     # 세상 × 집단 8 × (사람, 산 사람)
    c = np.array([w["agg"][k]["code"] for w in ws], float).sum((0, 1))
    T = np.array([w["agg"][k]["trait"] for w in ws], float).sum(0)            # 성향 4 × 수준 4 × (사람, 산 사람)
    n, b = A[..., 0].sum(0), A[..., 1].sum(0)
    rate = np.where(A[..., 0] > 0, A[..., 1] / np.maximum(A[..., 0], 1), np.nan)
    seg = [{"who": f"{a} {s}", "rate": float(b[i] / n[i]), "lo": float(np.nanquantile(rate[:, i], .05)), "hi": float(np.nanquantile(rate[:, i], .95)), "people": float(n[i] / n.sum()), "buyers": float(b[i] / max(b.sum(), 1))} for i, (a, s) in enumerate(SEG) if n[i] > 0]
    tr = [{"trait": W.TRAIT[t][0], "low": W.TRAIT[t][1], "hi": float(T[i, 3, 1] / max(T[i, 3, 0], 1)), "lo": float(T[i, 0, 1] / max(T[i, 0, 0], 1))} for i, t in enumerate(W.TRAIT)]
    sh = dict(zip(W.CODES, c / c.sum()))
    sh["cold"] += sh.pop("poor")
    buy = float(sh["buy_ad"] + sh["buy_friend"])
    cur = np.array([w["curve"][k] for w in ws]).mean(0)
    return {"seg": sorted(seg, key=lambda x: -x["rate"]), "trait": sorted(tr, key=lambda x: -abs(np.log(max(x["hi"], 1e-6) / max(x["lo"], 1e-6)))), "friend": float(sh["buy_friend"] / max(buy, 1e-9)), "half": int(np.searchsorted(cur, cur[-1] / 2)) * 13,
            "curve": [round(float(x), 5) for x in cur], "not": sorted([{"why": v, "share": float(sh[x] / max(1 - buy, 1e-9))} for x, v in ROUTE.items()], key=lambda x: -x["share"]), "total": buy}


def build(run):
    ws, flt, out = run["worlds"], run["flt"], {}
    for k, con in W.IDS.items():
        if run.get("sc") and k != run["only"]:                      # AI 가 이 제품을 위해 설계한 세상이라 다른 제품에는 쓰지 않는다
            continue
        c = crowd(ws, k)
        yes, no, n = words(con, flt)
        base, up = lift(con)
        good, bad = cards(con)
        top, big, nb = c["seg"][0], max(c["seg"], key=lambda x: x["buyers"]), c["not"][0]
        head = [f"{josa(top['who'], '이', '가')} 가장 잘 샀습니다(구매경험률 {top['rate']:.1%}, 전체 평균 {c['total']:.1%}).",
                f"산 사람 중에는 {josa(big['who'], '이', '가')} {big['buyers']:.0%}로 가장 많습니다(인구 비중 {big['people']:.0%})."]
        if yes:
            head.append(f"살 사람이 가장 많이 든 이유는 ‘{yes[0]['say']}’({yes[0]['share']:.0%})입니다.")
        if no:
            head.append(f"안 살 사람은 ‘{no[0]['say']}’({no[0]['share']:.0%})를 가장 많이 들었습니다.")
        head.append(f"계산에서 안 산 가장 큰 까닭은 ‘{nb['why']}’({nb['share']:.0%})입니다.")
        out[k] = {"name": con, "head": head, **c, "yes": yes, "no": no, "n": n, "intent": base, "lift": up[:4], "good": good, "bad": bad}
    ai = run.get("ai")
    if ai:
        from . import agents as A
        from . import scenes as S
        out[ai["k"]]["ai"] = A.said(ws, ai) | ai["info"] | ({"scene": run["scene"] | S.drivers(ws, run["text"], run["scene"]["axis"], ai["k"])} if run.get("sc") else {})
    return out


def check():
    v = voice()
    assert set(v.자유응답) <= set(WHY) | EMPTY, set(v.자유응답) - set(WHY) - EMPTY
    assert josa("여자", "이", "가") == "여자가" and josa("한국", "이", "가") == "한국이"
    ws = W.run(20, workers=1)
    o = build({"worlds": ws, "flt": {}})
    for k, x in o.items():
        assert abs(x["total"] - np.mean([w["buy"][k] for w in ws])) < 1e-9 and abs(sum(s["buyers"] for s in x["seg"]) - 1) < 1e-9 and abs(sum(s["share"] for s in x["not"]) - 1) < 1e-9
        assert x["yes"] and x["no"] and x["good"] and x["bad"] and all(g["score"] >= 4 for g in x["good"]) and all(g["score"] < 4 for g in x["bad"]) and all(u["gain"] < 1 - x["intent"] for u in x["lift"])
    assert build({"worlds": W.run(10, {"age10": ["20대"]}, workers=1), "flt": {"age10": ["20대"]}})["화채"]["seg"][0]["who"].startswith("20대")
    s = lambda k: dict((x["why"], x["share"]) for x in o[k]["not"])["파는 곳이 없음"]
    assert s("프리즈") > s("화채")
    print("report ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["show"]:
        for k, x in build({"worlds": W.default(), "flt": {}}).items():
            print("##", k, " ".join(x["head"]))
            print([(s["who"], round(s["rate"], 3), round(s["buyers"], 2)) for s in x["seg"]])
            print([(t["trait"], round(t["hi"], 3), round(t["lo"], 3)) for t in x["trait"]], "친구", round(x["friend"], 2), "절반", x["half"])
            print([(s["why"], round(s["share"], 2)) for s in x["not"]])
            print([(y["why"], round(y["share"], 2), y["n"]) for y in x["yes"]], [(y["why"], round(y["share"], 2), y["n"]) for y in x["no"]])
            print("의향", round(x["intent"], 3), [(u["why"], round(u["gain"], 3)) for u in x["lift"]])
            print(x["good"][:1], x["bad"][:1])
    else:
        check()
