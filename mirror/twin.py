"""개인 단위 디지털 트윈 시험 (Twin-2K-500 방식, 한국 실데이터).

실제 응답자 한 명의 '제로음료 섭취 빈도'(A19_1, 5단계)를 가리고, 그 사람의 다른 실제 답만으로 맞힌다.
① ML: 다른 응답자로 학습한 다항 로지스틱  ② LLM: 기록을 읽고 그 사람으로서 답  ③ 기준선: 같은 연령대×성별 최빈값
누설 방지: 제로 섭취 이유·미섭취 이유(A19_1_1, A19_2_1)는 입력에서 뺀다.
"""
import re

import numpy as np
import pandas as pd

from . import records as R

OPTS = [1, 2, 3, 4, 9]          # 주1회+, 월2~3, 월1, 월1미만, 안 먹어봄
ORD = {1: 4, 2: 3, 3: 2, 4: 1, 9: 0}   # 순서 거리용


def _features(k):
    raw = pd.read_parquet(R.SRC, columns=["A15_1", "A15_2", "A15_3"])
    fav = np.zeros((len(k), 18))
    for c, w in zip(raw, (1, .6, .3)):
        v = raw[c].fillna(0).astype(int).values
        fav[np.arange(len(k)), v] += w
    return np.column_stack([k.age / 10, (k.sex == "여자"), k.income.fillna(6) / 12, k.novelty, k.thrift, k.health,
                            k.social, fav[:, 1:]])


def _seg(k):
    return pd.cut(k.age, [18, 29, 39, 49, 59, 69, 200]).astype(str) + k.sex


def prompt(r):
    lv = lambda x: "매우 높음" if x > .8 else "높음" if x > .6 else "보통" if x > .4 else "낮음" if x > .2 else "매우 낮음"
    inc = int(r.income) if r.income == r.income else 0
    return (f"{r.age}세 {r.sex}, {r.province} 거주, 월소득 구간 {inc}/12.\n"
            f"좋아하는 음료(1~3순위): {r.fav}\n"
            f"새로운 음식 시도 성향 {lv(r.novelty)}, 가격 민감도 {lv(r.thrift)}, 건강·다이어트 관심 {lv(r.health)}, "
            f"SNS·온라인 추천 활용 {lv(r.social)}")


def run(n=60, model=None, seed=0):
    from sklearn.linear_model import LogisticRegression
    k = R.load()
    X, y = _features(k), k.zero_code.values
    rng = np.random.default_rng(seed)
    test = rng.choice(len(k), n, replace=False)
    train = np.setdiff1d(np.arange(len(k)), test)
    ml = LogisticRegression(max_iter=2000).fit(X[train], y[train], sample_weight=k.w.values[train])
    pred = {"ML(다른 응답자 학습)": ml.predict(X[test])}
    seg = _seg(k)
    mode = pd.Series(y[train]).groupby(seg.values[train]).agg(lambda s: s.value_counts().idxmax())
    pred["기준선(집단 최빈값)"] = mode.reindex(seg.values[test]).values
    if model:
        from .consumer import chat
        sys_ = ("당신은 아래 설문 기록의 실제 인물입니다. 이 사람으로서 질문에 답하세요. "
                "보기 번호 하나만 JSON 으로: {\"답\": 번호}")
        q = ("질문: 제로 음료(제로콜라·제로사이다 등)를 얼마나 자주 마십니까?\n"
             "1) 주 1회 이상 2) 월 2~3회 3) 월 1회 4) 월 1회 미만 9) 먹어본 적 없음")
        out = []
        for i in test:
            txt = chat(sys_, prompt(k.iloc[i]) + "\n\n" + q, model)
            m = re.search(r"[1-49]", txt.split("답")[-1] if "답" in txt else txt)
            out.append(int(m.group(0)) if m else 9)
        pred["LLM 트윈"] = np.array(out)
    truth = y[test]
    rep = {}
    for name, p in pred.items():
        d = np.abs(np.vectorize(ORD.get)(p) - np.vectorize(ORD.get)(truth))
        share_err = np.mean([abs((p == o).mean() - (truth == o).mean()) for o in OPTS])
        rep[name] = {"정확히 맞힘": (d == 0).mean(), "1단계 이내": (d <= 1).mean(), "분포 오차(%p)": 100 * share_err}
    dist = pd.DataFrame({name: [(p == o).mean() for o in OPTS] for name, p in {**pred, "정답": truth}.items()},
                        index=[R.ZERO_FREQ[o] for o in OPTS])
    return pd.DataFrame(rep).T, dist


if __name__ == "__main__":
    import sys
    rep, dist = run(model=sys.argv[1] if len(sys.argv) > 1 else None)
    print(rep.round(3))
    print(dist.round(3))
