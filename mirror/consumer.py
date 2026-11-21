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


CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"   # 로컬 실행, 한국어 지원, 무료


GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")     # 대시보드 입력란에서 덮어씀(메모리에만 보관)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"   # Gemini 의 OpenAI 호환 엔드포인트


def gemini_models(key):
    """키로 쓸 수 있는 Gemini 텍스트 모델 목록. 키가 틀리면 예외."""
    from openai import OpenAI
    ids = [m.id.removeprefix("models/") for m in OpenAI(api_key=key, base_url=GEMINI_URL).models.list()]
    ids = [i for i in ids if i.startswith("gemini") and ("flash" in i or "pro" in i)
           and not any(x in i for x in ("image", "tts", "audio", "live", "embedding"))]
    def rank(i):   # 최신 버전 flash 정식판이 맨 위
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", i)
        return (-float(m.group(1)) if m else 0, "lite" in i, "pro" in i, "preview" in i or "exp" in i, i)
    return sorted(set(ids), key=rank)


def llm_available():
    """Claude(API 키 또는 `ant auth login`), Gemini 키, OpenAI 키 중 하나라도 있으면 True."""
    return bool(GEMINI_KEY or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.exists(os.path.expanduser("~/.config/anthropic/credentials"))
                or os.environ.get("OPENAI_API_KEY"))


_clients = {}
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/1M 입력·출력
EST_TOKENS = (700, 300)            # 호출당 입력·출력(사고 포함) 추정치, 실측으로 갱신
USAGE = {"calls": 0, "in": 0, "out": 0}

import threading
_lock = threading.Lock()


def cost(model, tin, tout):
    pi, po = PRICES.get(model, (0, 0))
    return (tin * pi + tout * po) / 1e6


def _track(tin, tout):
    with _lock:
        USAGE["calls"] += 1
        USAGE["in"] += tin
        USAGE["out"] += tout


def chat(system, user, model):
    """한 번의 페르소나 응답. 모델 이름으로 공급자를 고른다."""
    if model.startswith("claude"):
        import anthropic
        cl = _clients.setdefault("claude", anthropic.Anthropic())
        r = cl.beta.messages.create(
            model=model, max_tokens=4000, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},             # 짧은 소비자 반응이라 낮은 노력으로 충분
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        _track(r.usage.input_tokens, r.usage.output_tokens)
        if r.stop_reason == "refusal":
            return "(응답 거절)"
        return "".join(b.text for b in r.content if b.type == "text").strip()
    from openai import OpenAI
    if model.startswith("gemini"):
        # 무료 등급은 분당 한도가 낮아 429가 잦다 → 재시도 넉넉히
        cl = _clients.get(("gemini", GEMINI_KEY)) or _clients.setdefault(
            ("gemini", GEMINI_KEY), OpenAI(api_key=GEMINI_KEY, base_url=GEMINI_URL, max_retries=8))
    else:
        cl = _clients.setdefault("openai", OpenAI())
    r = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": user}])
    _track(r.usage.prompt_tokens, r.usage.completion_tokens)
    return r.choices[0].message.content.strip()


def _embedder():
    if "embed" not in _clients:
        from fastembed import TextEmbedding
        _clients["embed"] = TextEmbedding(EMBED_MODEL)
    return _clients["embed"]


def concept_text(concept, sc):
    return (f"{concept}\n가격 {sc['price']:,}원, {'제로 슈거' if sc['zero'] else '일반 당 함량'}, "
            f"{sc['flavor']}, 패키지: {sc['package']}, 광고 문구 톤: {sc['copy']}")


def persona_prompt(row):
    return (f"당신은 {row.age}세 {row.sex}, {row.province} {row.district} 거주, 직업 {row.occupation}입니다.\n"
            f"{row.persona}\n식생활: {row.culinary_persona}\n취미: {row.hobbies_and_interests}")


SYSTEM_TAIL = ("\n이 사람으로서 솔직하게, 실제 이 사람이 쓸 법한 말투로 2~3문장만 답하세요. "
               "광고 문구처럼 말하지 말고, 관심 없으면 관심 없다고 하세요. 대부분의 신제품은 그냥 지나칩니다.")


def llm_respond(personas, concept, sc, model, workers=8, tick=None):
    """각 페르소나 자유 서술 → SSR pmf. 반환 (pmf[n,5], texts). tick(): 호출 1건 끝날 때마다."""
    from concurrent.futures import ThreadPoolExecutor
    if model.startswith("gemini"):
        workers = 2                     # 무료 등급 분당 한도 보호
    q = f"편의점 냉장고 앞에서 이 신제품을 봤어요.\n{concept_text(concept, sc)}\n살 건가요? 이유는?"

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(chat, persona_prompt(row) + SYSTEM_TAIL, q, model): i
                for i, row in enumerate(personas.itertuples())}
        texts = [None] * len(futs)
        for f in as_completed(futs):       # 메인 스레드에서 진행 표시(Streamlit 은 다른 스레드에서 못 그림)
            texts[futs[f]] = f.result()
            if tick:
                tick()
    return ssr(texts), texts


def ping(model):
    """모델이 실제로 응답하는지 1회 시험. 성공 시 응답 문자열, 실패 시 예외."""
    return chat("짧게 답하세요.", "안녕? 한 단어로.", model)


def ssr(texts, temp=0.05):
    """Semantic Similarity Rating (Maier et al. 2025): 응답과 기준 문장 5개의 코사인 유사도 → 분포."""
    v = np.array(list(_embedder().embed(ANCHORS + list(texts))))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sims = v[len(ANCHORS):] @ v[: len(ANCHORS)].T
    z = (sims - sims.max(1, keepdims=True)) / temp
    pmf = np.exp(z)
    return pmf / pmf.sum(1, keepdims=True)


def ask_why(row, concept, sc, question, model):
    return chat(persona_prompt(row) + "\n이 사람으로서 솔직하게 짧게 답하세요.",
                f"제품: {concept_text(concept, sc)}\n{question}", model)


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
    pm = ssr(["오 이거 완전 내 스타일, 바로 집어야지", "단 거 안 먹어서 패스", "음 글쎄, 한번 보긴 할 듯"])
    assert pm[0].argmax() >= 3 and pm[1].argmax() <= 1, pm.round(2)
    print("ssr ok", pm.argmax(1) + 1)
