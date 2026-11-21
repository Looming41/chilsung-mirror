"""페르소나 표본추출 + 반응 모델.

반응은 두 모드:
- rule : 페르소나 서술에서 뽑은 성향 점수로 만든 효용 모델(빠름, API 불필요, 이유 설명 가능)
- llm  : OpenAI로 자유 서술 응답 → SSR(의미 유사도)로 5점 척도 분포 변환
"""
import json
import os
import re

import numpy as np
import pandas as pd

AGE_BINS = [0, 19, 29, 39, 49, 59, 200]
AGE_LABELS = ["10대", "20대", "30대", "40대", "50대", "60대+"]

KW = {
    "health": r"건강|다이어트|운동|헬스|요가|필라테스|러닝|조깅|혈당|당 수치|칼로리|저당|제로|식단",
    "trend": r"인스타|SNS|유튜브|틱톡|핫플|신상|트렌드|팝업|맛집 탐방|카페 투어|인증|숏폼|릴스",
    "drink": r"탄산|콜라|사이다|음료|에이드|디저트|카페|커피|달콤|단 음식|간식|편의점",
    "thrift": r"알뜰|가성비|절약|할인|저렴|아끼|생활비|빠듯",
}
TEXT_COLS = ["culinary_persona", "sports_persona", "hobbies_and_interests", "persona", "career_goals_and_ambitions"]


def fetch_personas(path, n=30000):
    """Nemotron-Personas-Korea 첫 샤드(약 220MB, 11만 명)를 받아 n명 표본만 저장."""
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("nvidia/Nemotron-Personas-Korea", "data/train-00000-of-00009.parquet", repo_type="dataset")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.read_parquet(src).sample(n, random_state=0).to_parquet(path)


def load_pool(path="data/personas.parquet"):
    if not os.path.exists(path):
        fetch_personas(path)
    df = pd.read_parquet(path)
    text = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
    for k, pat in KW.items():
        cnt = text.str.count(pat)
        df[k] = 1 - np.exp(-cnt / 1.5)                 # 0~1 포화 점수
    young = np.clip((45 - df.age) / 30, 0, 1)
    df["trend"] = np.clip(0.6 * df.trend + 0.4 * young, 0, 1)
    student = df.occupation.fillna("").str.contains("학생|무직|구직")
    df["thrift"] = np.clip(df.thrift + 0.3 * student + 0.2 * (df.age >= 65), 0, 1)
    df["age_group"] = pd.cut(df.age, AGE_BINS, labels=AGE_LABELS)
    return df.reset_index(drop=True)


def stratified_sample(pool, n, ages=(15, 80), sexes=("남자", "여자"), seed=0):
    """연령대×성별 층에서 모집단(KOSIS 기반) 비율대로 비례 배분 추출."""
    sub = pool[pool.age.between(*ages) & pool.sex.isin(sexes)]
    strata = sub.groupby(["age_group", "sex"], observed=True)
    share = strata.size() / len(sub)
    alloc = np.floor(share * n).astype(int)
    alloc[alloc.index[np.argsort(-(share * n - alloc).values)[: n - alloc.sum()]]] += 1
    parts = [g.sample(alloc[key], random_state=seed) for key, g in strata if alloc[key] > 0]
    return pd.concat(parts).reset_index(drop=True)


# ---------- rule 모드 ----------
def utility(p, sc):
    """효용과 항목별 기여도. sc: 시나리오 dict."""
    novelty = 1.0 if sc["flavor"] == "새로운 맛" else 0.2
    contrib = {
        "기본": np.full(len(p), -1.3),
        "음료 선호": 1.2 * p.drink.values,
        "제로 슈거×건강관심": 1.1 * p.health.values * sc["zero"],
        "새로운 맛×트렌드": 0.8 * p.trend.values * novelty,
        "새로운 맛×고연령 거부감": -0.6 * (p.age.values >= 55) * novelty,
        "비주얼 패키지×트렌드": 0.7 * p.trend.values * (sc["package"] == "비주얼(숏폼형)"),
        "문구 적합": np.where(sc["copy"] == "건강", 0.5 * p.health.values, 0.5 * p.trend.values),
        "가격 부담": -0.9 * (0.3 + p.thrift.values) * (sc["price"] - 1500) / 500,
    }
    return sum(contrib.values()), contrib


def likert_pmf(u, cuts=(-2.0, -0.7, 0.6, 1.8)):
    """순서형 로짓: 효용 → 1~5점 확률."""
    cdf = 1 / (1 + np.exp(-(np.asarray(cuts)[None, :] - u[:, None])))
    cdf = np.c_[np.zeros(len(u)), cdf, np.ones(len(u))]
    return np.diff(cdf, axis=1)


def satisfaction(p, sc):
    """마셔본 뒤 만족 확률: 제로 슈거는 건강 무관심층에서 대체감미료 불만."""
    return np.clip(0.85 - 0.3 * sc["zero"] * (1 - p.health.values) - 0.1 * (sc["price"] > 2000), 0.2, 0.95)


def post_rate(p, sc):
    return np.clip(0.04 + 0.45 * p.trend.values, 0, 1)


def reasons(contrib, i, k=2):
    items = sorted(((v[i], name) for name, v in contrib.items() if name != "기본"))
    neg = [f"{n}({v:+.2f})" for v, n in items[:k] if v < 0]
    pos = [f"{n}({v:+.2f})" for v, n in items[::-1][:k] if v > 0]
    return "끌림: " + (", ".join(pos) or "없음") + " / 망설임: " + (", ".join(neg) or "없음")


# ---------- llm 모드 (SSR) ----------
ANCHORS = [
    "절대 안 살 것 같아요. 저랑은 안 맞아요.",
    "아마 안 살 것 같아요. 굳이 손이 안 가요.",
    "잘 모르겠어요. 보이면 한번 고민은 해볼 것 같아요.",
    "아마 한번 사 볼 것 같아요. 꽤 끌려요.",
    "무조건 사 볼 거예요. 당장 마셔보고 싶어요.",
]


def _client():
    from openai import OpenAI
    return OpenAI()


def llm_available():
    return bool(os.environ.get("OPENAI_API_KEY"))


def concept_text(concept, sc):
    return (f"{concept}\n가격 {sc['price']:,}원, {'제로 슈거' if sc['zero'] else '일반 당 함량'}, "
            f"{sc['flavor']}, 패키지: {sc['package']}, 광고 문구 톤: {sc['copy']}")


def persona_prompt(row):
    return (f"당신은 {row.age}세 {row.sex}, {row.province} {row.district} 거주, 직업 {row.occupation}입니다.\n"
            f"{row.persona}\n식생활: {row.culinary_persona}\n취미: {row.hobbies_and_interests}")


def llm_respond(personas, concept, sc, model):
    """각 페르소나 자유 서술 → SSR pmf. 반환 (pmf[n,5], texts)."""
    cl = _client()
    texts = []
    for row in personas.itertuples():
        r = cl.chat.completions.create(model=model, messages=[
            {"role": "system", "content": persona_prompt(row) +
             "\n이 사람으로서 솔직하게, 반말 섞인 실제 말투로 2~3문장만 답하세요. 마케팅 문구처럼 말하지 마세요."},
            {"role": "user", "content": f"편의점 냉장고 앞에서 이 신제품을 봤어요.\n{concept_text(concept, sc)}\n살 건가요? 이유는?"},
        ])
        texts.append(r.choices[0].message.content.strip())
    return ssr(texts), texts


def ssr(texts, temp=0.05):
    """Semantic Similarity Rating (Maier et al. 2025): 응답과 기준 문장 5개의 코사인 유사도 → 분포."""
    cl = _client()
    emb = cl.embeddings.create(model="text-embedding-3-small", input=ANCHORS + list(texts)).data
    v = np.array([e.embedding for e in emb])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sims = v[len(ANCHORS):] @ v[: len(ANCHORS)].T
    z = (sims - sims.max(1, keepdims=True)) / temp
    pmf = np.exp(z)
    return pmf / pmf.sum(1, keepdims=True)


def ask_why(row, concept, sc, question, model):
    cl = _client()
    r = cl.chat.completions.create(model=model, messages=[
        {"role": "system", "content": persona_prompt(row) + "\n이 사람으로서 솔직하게 짧게 답하세요."},
        {"role": "user", "content": f"제품: {concept_text(concept, sc)}\n{question}"},
    ])
    return r.choices[0].message.content.strip()


if __name__ == "__main__":
    pool = load_pool()
    s = stratified_sample(pool, 1000, seed=0)
    assert len(s) == 1000
    sub = pool[pool.age.between(15, 80)]
    share_pool = sub.groupby("age_group", observed=True).size() / len(sub)
    share_s = s.groupby("age_group", observed=True).size() / len(s)
    assert (share_pool - share_s).abs().max() < 0.01
    sc = dict(price=1800, zero=1, flavor="새로운 맛", package="비주얼(숏폼형)", copy="재미")
    u, c = utility(s, sc)
    pmf = likert_pmf(u)
    assert np.allclose(pmf.sum(1), 1)
    cheap, _ = utility(s, {**sc, "price": 1500})
    assert (cheap >= u).all()
    print("consumer ok", "top2=%.3f" % pmf[:, 3:].sum(1).mean(), reasons(c, 0))
