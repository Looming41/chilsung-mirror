"""실험 실행기: 요인설계 시나리오 × 반복(R) → 설문형 구매의향 + 4주 바이럴 확산 → 통계 요약.

반복마다 (1) 페르소나 부트스트랩 재추출 (2) 응답 잡음 (3) 네트워크/확산 난수를 바꾼다.
같은 반복 안에서는 모든 시나리오가 같은 난수를 공유(공통 난수) → 시나리오 간 비교의 분산이 줄어든다.
"""
import itertools

import numpy as np
import pandas as pd

from . import consumer as C
from . import stats as S
from . import viral as V


def scenarios(levels):
    keys = list(levels)
    return [dict(zip(keys, combo)) for combo in itertools.product(*levels.values())]


def label(sc):
    return (f"{sc['price']:,}원·{'제로' if sc['zero'] else '일반'}·{sc['flavor']}·"
            f"{sc['package'][:2]}·{sc['copy']}·{'인플루언서' if sc['influencer'] else '광고만'}")


def run(pool, levels, *, n=1000, reps=20, ages=(19, 80), sexes=("남자", "여자"), say_do=0.35,
        ad_reach=0.01, days=28, mode="rule", concept="", model="claude-opus-5-5", seed=0, progress=None):
    base = C.stratified_sample(pool, n, ages=ages, sexes=sexes, seed=seed)
    scs = scenarios(levels)
    llm_cache = {}
    if mode == "llm":
        uniq = {tuple((k, v) for k, v in sc.items() if k != "influencer") for sc in scs}
        total, done = len(uniq) * n, [0]

        def tick():
            done[0] += 1
            if progress:
                progress(done[0] / total * 0.5, f"LLM 응답 {done[0]:,}/{total:,}")
        for sc in scs:
            key = tuple((k, v) for k, v in sc.items() if k != "influencer")
            if key not in llm_cache:
                llm_cache[key] = C.llm_respond(base, concept, sc, model, tick=tick)

    top2 = np.zeros((len(scs), reps))           # 반복별 설문형 구매의향률(4~5점)
    p_base = np.zeros((len(scs), n))            # 기준 표본 개인별 의향(반복 평균)
    curves = {k: np.zeros((len(scs), reps, days)) for k in ("buy", "pos", "neg")}
    adopters = np.zeros((len(scs), reps))
    nstar = np.zeros((len(scs), reps))
    spreaders = []
    texts = {}

    for r in range(reps):
        rng = np.random.default_rng(seed * 1000 + r)
        idx = rng.integers(0, n, n) if r else np.arange(n)   # r=0 은 원표본
        nbrs, deg = V.build_network(n, seed=seed * 1000 + r)
        # 팔로워 많은 노드에 트렌드 민감 페르소나가 가도록 배치(잡음 섞어 완전 정렬은 피함)
        by_trend = idx[np.argsort(-base.trend.values[idx] + rng.normal(0, 0.15, n))]
        idx = by_trend[np.argsort(np.argsort(-deg))]
        P = base.iloc[idx].reset_index(drop=True)
        noise = rng.normal(0, 0.35, n)                         # 응답 확률성(같은 사람도 매번 조금 다름)
        for j, sc in enumerate(scs):
            if mode == "llm":
                key = tuple((k, v) for k, v in sc.items() if k != "influencer")
                pmf_all, tx = llm_cache[key]
                pmf = pmf_all[idx]
                texts[label(sc)] = tx
                u = np.log(np.clip(pmf[:, 3:].sum(1), 1e-4, 1 - 1e-4) / np.clip(pmf[:, :3].sum(1), 1e-4, 1))
            else:
                u, contrib = C.utility(P, sc)
                u = u + noise
                pmf = C.likert_pmf(u)
            p_top2 = pmf[:, 3:].sum(1)
            top2[j, r] = (rng.random(n) < p_top2).mean()
            if r == 0:
                p_base[j] = p_top2
                if mode == "rule":
                    texts[label(sc)] = [C.reasons(contrib, i) for i in range(n)]
            out = V.simulate(p_top2 * say_do, C.post_rate(P, sc), C.satisfaction(P, sc), nbrs, deg,
                             days=days, ad_reach=ad_reach, influencer_seed=sc["influencer"],
                             shortform=1.4 if sc["package"].startswith("비주얼") else 1.0,
                             rng=np.random.default_rng(seed * 7919 + r))
            for k in curves:
                curves[k][j, r] = out[k]
            adopters[j, r] = out["adopters"]
            nstar[j, r] = out["n_star"]
            if r == 0:
                fd = out["first_day"]
                first = np.argsort(np.where(fd < 0, 10**6, fd))[:20]
                first = first[fd[first] >= 0]
                spreaders.append(pd.DataFrame({"시나리오": label(sc), "팔로워": deg[first],
                                               "연령대": P.age_group.values[first].astype(str),
                                               "성별": P.sex.values[first], "구매일": fd[first] + 1}))
        if progress:
            progress(0.5 * (mode == "llm") + (r + 1) / reps * (0.5 if mode == "llm" else 1.0), f"반복 {r + 1}/{reps}")

    return summarize(scs, base, top2, p_base, curves, adopters, nstar, n, reps, spreaders, texts)


def summarize(scs, base, top2, p_base, curves, adopters, nstar, n, reps, spreaders, texts):
    labels = [label(s) for s in scs]
    k = np.round(p_base.sum(1)).astype(int)      # 원표본 기대 응답자 수(독립 표본 n 기준 검정용)
    pbest = S.prob_best(k, np.full(len(k), n))
    # 승률: 반복마다(공통 난수) 4주 구매율 1위를 한 비율, 동점은 나눠 가짐
    top = adopters == adopters.max(0, keepdims=True)
    winrate = (top / top.sum(0, keepdims=True)).mean(1)
    rows = []
    for j, lb in enumerate(labels):
        m, lo, hi = S.bootstrap_ci(top2[j])
        am, alo, ahi = S.bootstrap_ci(adopters[j] / n)
        rows.append({"시나리오": lb, **scs[j], "승률": winrate[j], "구매의향": m, "의향_하한": lo, "의향_상한": hi,
                     "의향1등확률": pbest[j], "4주구매율": am, "구매_하한": alo, "구매_상한": ahi,
                     "바이럴확률": float((nstar[j] > 1).mean()), "분기비": float(nstar[j].mean()),
                     "부정비율": float(curves["neg"][j].sum() / max(1, curves["pos"][j].sum() + curves["neg"][j].sum()))})
    table = pd.DataFrame(rows).sort_values(["승률", "4주구매율"], ascending=False).reset_index(drop=True)

    best = int(np.argmax(k))                     # 설문 검정은 설문 1위 기준
    best_label = labels[best]
    tests = []
    for j, lb in enumerate(labels):
        if j == best:
            continue
        z, p = S.two_prop_ztest(k[best], n, k[j], n)
        tests.append({"비교 대상": lb, "1위 의향": k[best] / n, "대상 의향": k[j] / n,
                      "차이(%p)": 100 * (k[best] - k[j]) / n, "Cohen h": S.cohens_h(k[best] / n, k[j] / n), "p": p})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["Holm 보정 p"] = S.holm(tests.p)
        tests["유의(5%)"] = np.where(tests["Holm 보정 p"] < 0.05, "유의함", "차이 없음")
        tests = tests.sort_values("차이(%p)").reset_index(drop=True)

    seg = base[["age_group", "sex"]].astype(str).copy()
    segs = {lb: seg.assign(의향=p_base[j]).groupby(["age_group", "sex"]).의향.mean().unstack()
            for j, lb in enumerate(labels)}

    bass = {}
    for j, lb in enumerate(labels):
        fit = V.fit_bass(curves["buy"][j].mean(0))
        bass[lb] = fit
    return dict(table=table, tests=tests, best_label=best_label, curves=curves, labels=labels, segs=segs, bass=bass,
                spreaders=pd.concat(spreaders), texts=texts, base=base, scenarios=scs, n=n, reps=reps,
                p_base=p_base)


if __name__ == "__main__":
    import time
    pool = C.load_pool()
    levels = dict(price=[1500, 2000], zero=[1], flavor=["새로운 맛"], package=["비주얼(숏폼형)", "클래식"],
                  copy=["재미"], influencer=[True, False])
    t = time.time()
    res = run(pool, levels, n=600, reps=8)
    tb = res["table"]
    print(tb[["시나리오", "승률", "구매의향", "의향1등확률", "4주구매율", "바이럴확률", "분기비"]].round(3).to_string())
    assert abs(tb["의향1등확률"].sum() - 1) < 1e-6 and abs(tb["승률"].sum() - 1) < 1e-6
    assert tb.sort_values("구매의향").iloc[-1].price == 1500   # 싼 쪽이 의향 1위여야 함
    print(res["tests"].round(4).to_string())
    vis = tb[tb.package.str.startswith("비주얼") & tb.influencer]["4주구매율"].mean()
    cls = tb[tb.package.str.startswith("클래식") & ~tb.influencer]["4주구매율"].mean()
    assert vis > cls, (vis, cls)
    print("sim ok %.1fs" % (time.time() - t))
