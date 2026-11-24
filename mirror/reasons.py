"""대표 세계에 LLM 을 넣어 이유 듣기: 이긴 세계와 진 세계에 같은 합성 인물을 보내 두 제품 중 고르게 하고 이유를 모은다.

- 인물은 순차 나무로 합성한 사람(실존 응답자가 아님)이고, 같은 인물이 두 세계를 모두 산다
- 시뮬레이션이 누구를 1등으로 봤는지는 알려 주지 않는다. 보기 순서는 인물의 절반씩 뒤집어 순서 편향을 잰다
- LLM 의 이유는 '그럴듯한 설명'이지 원인의 증거가 아니다. 시뮬레이션 승패와 얼마나 일치하는지, 조건이 바뀔 때 선택이 바뀌는지를 함께 보여 준다
실행: python -m mirror.reasons check
"""
import concurrent.futures as cf
import json
import re

import numpy as np
import pandas as pd

from . import consumer as C
from . import multiverse as U
from . import people as P

WHY = ["가격", "맛 기대", "건강·성분", "패키지·재미", "유행·추천", "한정·희소", "익숙함", "필요 없음"]
SAY = {"가격 민감도": ("물가가 올라 음료 한 병 값도 따져 보는 때다", "지갑 사정이 여유로워 음료 값은 크게 신경 쓰지 않는 때다"),
       "시장 분위기": ("요즘 신상 음료에 사람들이 들떠 있다", "요즘 신상 음료에 사람들이 시큰둥하다"),
       "광고 도달": ("이 제품 광고가 자주 보인다", "이 제품 광고를 거의 못 봤다"),
       "SNS 전파력": ("음료 이야기가 SNS에서 빨리 퍼진다", "음료 이야기가 SNS에서 잘 퍼지지 않는다"),
       "유행 지속": ("유행이 한 달 가까이 이어진다", "유행이 며칠이면 식는다"),
       "게시 성향": ("주변 사람들이 먹어 본 걸 SNS에 자주 올린다", "주변 사람들이 먹어 본 걸 거의 올리지 않는다"),
       "악평 영향": ("혹평을 봐도 크게 흔들리지 않는 분위기다", "혹평 하나에도 안 사는 분위기다"),
       "동조": ("여럿이 추천하면 따라 사는 분위기다", "남이 추천해도 각자 알아서 고르는 분위기다"),
       "피로": ("같은 광고를 여러 번 보면 질린다는 사람이 많다", "같은 광고를 여러 번 봐도 개의치 않는다")}
# '희소성' 은 말로 옮기지 않는다: 넣어 봤더니 LLM 이 그 문장을 그대로 따라 말해 답을 유도했다


def scene(W, i, top=4):
    """세계 i 의 조건 가운데 평균에서 가장 먼 것들을 말로 옮긴다(숫자·승패는 말하지 않는다)."""
    r = W[W.세계 == i].iloc[0]
    z = {k: (r[k] - W[k].mean()) / W[k].std() for k in SAY if k in W and W[k].std() > 0}
    ks = [k for k in sorted(z, key=lambda k: -abs(z[k])) if abs(z[k]) >= .3][:top]
    return [SAY[k][0 if z[k] > 0 else 1] for k in ks]


def card(sc, concept):
    return (f"{concept} {sc['price']:,}원, {'제로 슈거' if sc['zero'] else '설탕이 든 일반'}, {sc['flavor']}, {sc['package'].split('(')[0]} 패키지, '{sc['copy']}'을 강조"
            + (", 유명 인플루언서가 소개함" if sc["influencer"] else "") + (", 한정 수량 판매" if sc.get("stock") else ""))


def prompt(mem, world, a, b):
    s = "당신은 아래 사람입니다. 이 사람의 실제 처지와 성향대로, 솔직하게 답하세요. 마케터처럼 말하지 마세요. 안 사도 됩니다.\n" + "\n".join("- " + m for m in mem)
    u = ("[지금 세상]\n" + ("\n".join("- " + w for w in world) or "- 평소와 다름없다") + f"\n\n편의점에 두 신제품이 있습니다.\nA: {a}\nB: {b}\n\n"
         "무엇을 사겠습니까? JSON 하나로만 답하세요: {\"pick\":\"A|B|없음\",\"why\":\"" + "|".join(WHY) + " 중 하나\",\"say\":\"이유 한 문장(40자 이내)\"}")
    return s, u


def parse(t):
    m = re.search(r"\{.*?\}", t or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    pk, why = str(j.get("pick", "")).strip().upper(), str(j.get("why", "")).strip()
    if pk not in ("A", "B", "없음"):
        return None
    return pk, next((w for w in WHY if w in why or why in w and why), "기타"), str(j.get("say", ""))[:80]


def listen(W, scs, concept, w, first, second, n=10, model=None, ask=None, workers=3, tick=None, seed=0):
    """first(1등 시나리오)와 second(차점)를 이긴·진 대표 세계에서 묻는다 → (답 표, 요약)."""
    ask = ask or (lambda s, u: C.chat(s, u, model))
    yes, no = U.representative(W, first)
    who = U.synthetic(w, n, seed)
    worlds = [(k, i) for k, i in (("이긴 세계", yes), ("진 세계", no)) if i is not None]
    jobs = [(p, nm, i, p % 2) for nm, i in worlds for p in range(n)]          # 절반은 A 가 1등, 절반은 B 가 1등
    sc = {i: scene(W, i) for _, i in worlds}

    def go(j):
        p, nm, i, flip = j
        a, b = (second, first) if flip else (first, second)
        s, u = prompt(P.memory(who.iloc[p]), sc[i], card(scs[a], concept), card(scs[b], concept))
        try:
            g = parse(ask(s, u))
        except Exception as e:                                                # 한 사람의 실패가 전체를 멈추지 않게 하되 개수는 보고한다
            return {"인물": p, "세계": nm, "오류": f"{type(e).__name__}: {str(e)[:80]}"}
        if g is None:
            return {"인물": p, "세계": nm, "오류": "형식이 맞지 않는 답"}
        return {"인물": p, "누구": f"{who.age[p]} {who.sex[p]} · {who.fav[p]} · 제로 {who.zero[p]}", "세계": nm, "선택": {"A": a, "B": b}.get(g[0], "없음"), "자리": g[0], "이유": g[1], "말": g[2]}
    out = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(go, jobs)):
            out.append(r)
            if tick:
                tick((k + 1) / len(jobs))
    t = pd.DataFrame(out)
    return t, digest(t, first, second) | {"세계 설명": {nm: sc[i] for nm, i in worlds}}


def digest(t, first, second):
    ok = t[t["선택"].notna()] if "선택" in t else t.iloc[:0]
    o = {"성공": len(ok), "실패": len(t) - len(ok)}
    if not len(ok):
        return o
    o["선택"] = {w: {k: int(v) for k, v in g["선택"].value_counts().items()} for w, g in ok.groupby("세계")}
    o["이유"] = pd.crosstab(ok["이유"], ok["세계"]).to_dict()
    ab = ok[ok["자리"].isin(["A", "B"])]
    o["앞자리 선택률"] = float((ab["자리"] == "A").mean()) if len(ab) else np.nan
    p = ok.pivot_table(index="인물", columns="세계", values="선택", aggfunc="first").dropna()
    o["세계가 바뀌자 선택도 바뀜"] = float((p.iloc[:, 0] != p.iloc[:, 1]).mean()) if p.shape[1] == 2 and len(p) else np.nan
    g = {w: (v["선택"] == first).sum() - (v["선택"] == second).sum() for w, v in ok.groupby("세계")}
    o["시뮬레이션과 같은 방향"] = {w: bool(d > 0) if w == "이긴 세계" else bool(d < 0) for w, d in g.items()}
    return o


def check():
    b = dict(zero=1, flavor="새로운 맛", package="비주얼(숏폼형)", copy="재미", influencer=False)
    scs = {"싼": {**b, "price": 1500}, "비싼": {**b, "price": 1700}}
    w, _ = U.select({"age10": ["10대", "20대"]})
    W, _ = U.run({"age10": ["10대", "20대"]}, scs, "자두맛 탄산음료.", worlds=30, n=600)
    seen = []

    def fake(s, u):
        seen.append(u)
        return '답: {"pick":"A","why":"가격","say":"싸서"}' if len(seen) != 3 else "모르겠어요"
    t, o = listen(W, scs, "자두맛 탄산음료.", w, "싼", "비싼", n=6, ask=fake, workers=1)
    assert o["실패"] == 1 and o["성공"] + 1 == len(t) and o["앞자리 선택률"] == 1.0
    assert not any(x in u for u in seen for x in ("구매율", "이긴", "진 세계", "1등")) and not any("한정" in u.split("JSON")[0] for u in seen)
    assert {"싼", "비싼"} <= set(t["선택"].dropna())                         # 앞자리만 고르는 LLM 은 절반씩 나와야 한다
    assert parse('{"pick":"C"}') is None and parse("x") is None and parse('{"pick":"없음","why":"필요 없음","say":"."}')[1] == "필요 없음"
    print("reasons ok")


if __name__ == "__main__":
    check()
