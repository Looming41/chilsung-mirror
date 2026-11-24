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

from .records import record_text

AGE_BINS = [0, 19, 29, 39, 49, 59, 200]
AGE_LABELS = ["10대", "20대", "30대", "40대", "50대", "60대+"]



def fetch_personas(path, n=30000):
    """Nemotron-Personas-Korea 첫 샤드(약 220MB, 11만 명)를 받아 n명 표본만 저장."""
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("nvidia/Nemotron-Personas-Korea", "data/train-00000-of-00009.parquet", repo_type="dataset")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.read_parquet(src).sample(n, random_state=0).to_parquet(path)


def load_pool(path="data/personas.parquet", cache="data/personas_rec.parquet"):
    """Nemotron 페르소나(서사) + KREI 실응답자 기록(성향·음료습관) hot-deck 매칭 + 선택모델 특성(z0..)·심리요인(f1..)."""
    from . import choice as M
    from . import records as R
    src_mtime = max(os.path.getmtime(R.__file__), os.path.getmtime(M.__file__))
    if not os.path.exists(cache) or os.path.getmtime(cache) < src_mtime:
        if not os.path.exists(path):
            fetch_personas(path)
        df = R.attach(pd.read_parquet(path))
        Z, S = M.person_features()
        df = pd.concat([df, pd.DataFrame(Z[df.donor.values], columns=[f"z{i}" for i in range(Z.shape[1])]),
                        pd.DataFrame(S[df.donor.values], columns=[f"f{i + 1}" for i in range(S.shape[1])])], axis=1)
        df.to_parquet(cache)
    df = pd.read_parquet(cache)
    df["age_group"] = pd.cut(df.age, AGE_BINS, labels=AGE_LABELS)
    return df


def zmat(p):
    return p.filter(regex=r"^z\d+$").values


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
def utility(p, sc, attr=None):
    """효용과 항목별 기여도. sc: 시나리오 dict, attr: 신제품 속성(choice.concept_attrs). 없으면 '탄산·단맛·sc.zero'."""
    from . import choice as M
    attr = attr or M.concept_attrs("", sc)
    fit_new = M.new_share(zmat(p), attr)          # 3개년 실선택 학습모델: 이 사람이 신제품을 1순위로 고를 확률
    # 성향은 모두 KREI 실응답 기록에서 온다(records.py). 제로 수용성은 실데이터로 학습한 로지스틱 모델 확률.
    novelty = 1.0 if sc["flavor"] == "새로운 맛" else 0.2
    contrib = {
        "기본": np.full(len(p), -1.0),
        "음료 선택구조 적합(3개년 학습)": 0.8 * np.log(fit_new / M.ref_share()),   # 일반 탄산 대비
        "제로 수용성(실데이터 학습)": 2.0 * (p.zero_aff.values - 0.2) * sc["zero"],
        "제로 거부: 맛·감미료 불신(실기록)": -0.8 * p.zero_reject.values * sc["zero"],
        "새로운 맛×새로움 추구": 0.9 * (p.novelty.values - 0.4) * novelty,
        "새로운 맛×고연령 거부감": -0.6 * (p.age.values >= 55) * novelty,
        "비주얼 패키지×SNS 활용": 0.8 * p.social.values * (sc["package"] == "비주얼(숏폼형)"),
        "문구 적합": np.where(sc["copy"] == "건강", 0.5 * p.health.values, 0.5 * p.trend.values),
        # 함의 가격탄력성 -1.3~-1.7(1500원 기준): 탄산 카테고리 -0.79(Andreyeva 2010)와 브랜드 평균 -2.62(Bijmolt 2005) 사이
        "가격 부담": -0.9 * (0.3 + p.thrift.values) * (sc["price"] - 1500) / 500,
    }
    return sum(contrib.values()), contrib


def likert_pmf(u, cuts=(-2.0, -0.7, 0.6, 1.8)):
    """순서형 로짓: 효용 → 1~5점 확률."""
    cdf = 1 / (1 + np.exp(-(np.asarray(cuts)[None, :] - u[:, None])))
    cdf = np.c_[np.zeros(len(u)), cdf, np.ones(len(u))]
    return np.diff(cdf, axis=1)


def satisfaction(p, sc):
    """마셔본 뒤 만족 확률: 제로 맛·감미료를 싫어한다고 실제로 답한 사람은 불만(부정 후기) 쪽."""
    return np.clip(0.85 - 0.35 * sc["zero"] * p.zero_reject.values - 0.15 * sc["zero"] * (1 - p.health.values)
                   - 0.1 * (sc["price"] > 2000), 0.2, 0.95)


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


def _dotenv(path=os.path.join(os.path.dirname(__file__), "..", ".env")):
    """프로젝트 루트 .env 의 KEY=VALUE 를 환경변수로(이미 있으면 유지). .env 는 git 제외."""
    if os.path.exists(path):
        for line in open(path):
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_dotenv()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")     # .env 또는 대시보드 입력란(메모리에만 보관)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"   # Gemini 의 OpenAI 호환 엔드포인트


def gemini_models(key):
    """키로 쓸 수 있는 Gemini 텍스트 모델 목록. 키가 틀리면 예외."""
    from openai import OpenAI
    ids = [m.id.removeprefix("models/") for m in OpenAI(api_key=key, base_url=GEMINI_URL).models.list()]
    ids = [i for i in ids if i.startswith("gemini") and ("flash" in i or "pro" in i)
           and not any(x in i for x in ("image", "tts", "audio", "live", "embedding"))]
    def rank(i):   # 최신 flash-lite 정식판이 맨 위: 무료 등급 일일 한도가 가장 넉넉함(flash 는 하루 20회)
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", i)
        return ("lite" not in i, "pro" in i, "preview" in i or "exp" in i, -float(m.group(1)) if m else 0, i)
    return sorted(set(ids), key=rank)


def llm_available():
    """Claude(API 키 또는 `ant auth login`), Gemini 키, OpenAI 키 중 하나라도 있으면 True."""
    return bool(GEMINI_KEY or os.environ.get("GATEWAY_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.exists(os.path.expanduser("~/.config/anthropic/credentials"))
                or os.environ.get("OPENAI_API_KEY"))


_clients = {}
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/1M 입력·출력
EST_TOKENS = (700, 300)            # 호출당 입력·출력(사고 포함) 추정치, 실측으로 갱신
USAGE = {"calls": 0, "in": 0, "out": 0}

import threading
import time
_lock = threading.Lock()


def cost(model, tin, tout):
    pi, po = PRICES.get(model, (0, 0))
    return (tin * pi + tout * po) / 1e6


GEMINI_RPM = 14                    # 무료 등급 flash-lite 분당 15회보다 살짝 낮게
_rl = {"last": 0.0}


def _throttle():
    """전역 호출 간격 유지(스레드 공유)."""
    with _lock:
        wait = _rl["last"] + 60 / GEMINI_RPM - time.time()
        if wait > 0:
            time.sleep(wait)
        _rl["last"] = time.time()


def _track(tin, tout):
    with _lock:
        USAGE["calls"] += 1
        USAGE["in"] += tin
        USAGE["out"] += tout


CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llm_cache.jsonl")
_cache = None


def chat(system, user, model):
    """디스크 캐시를 거친 페르소나 응답. 같은 (모델, 프롬프트)는 다시 호출하지 않아 무료 한도를 아낀다."""
    import hashlib
    global _cache
    key = hashlib.sha1(f"{model}\x00{system}\x00{user}".encode()).hexdigest()
    with _lock:
        if _cache is None:
            _cache = {}
            if os.path.exists(CACHE_PATH):
                for line in open(CACHE_PATH):
                    d = json.loads(line)
                    _cache[d["k"]] = d["v"]
        if key in _cache:
            return _cache[key]
    out = _chat_raw(system, user, model)
    with _lock:
        _cache[key] = out
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps({"k": key, "m": model, "v": out}, ensure_ascii=False) + "\n")
    return out


GATEWAY_URL = "https://gateway-us.pydantic.dev/proxy/lotte-ax-challenge"      # 대회가 준 OpenAI 주소(responses 방식)


def gateway(key):
    from openai import OpenAI
    return OpenAI(base_url=GATEWAY_URL, api_key=key, timeout=60, max_retries=0)


def ask_openai(cl, model, system, user, **kw):
    """→ (답, 입력 토큰, 출력 토큰). 생각 끄기를 받는 모델은 끈다."""
    r = cl.responses.create(model=model, instructions=system, input=user, store=False, **({"reasoning": {"effort": "none"}} if model in NO_THINK else {}), **kw)
    tin, tout = (r.usage.input_tokens, r.usage.output_tokens) if r.usage else (0, 0)
    with _lock:                                             # 사용량 장부: 때·모델·토큰만(질문·답·키는 적지 않는다)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "m": model, "in": tin, "out": tout}) + "\n")
    return (r.output_text or "").strip(), tin, tout


LEDGER = os.path.join(os.path.dirname(__file__), "..", "data", "openai_usage.jsonl")


def ledger():
    """모델마다 [호출, 입력 토큰, 출력 토큰]."""
    out = {}
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for d in map(json.loads, f):
                u = out.setdefault(d["m"], [0, 0, 0])
                u[0], u[1], u[2] = u[0] + 1, u[1] + d["in"], u[2] + d["out"]
    return out


NO_THINK = {"gpt-6-luna", "gpt-6.1-terra"}      # 생각 토큰도 출력 요금이라 끈다. gpt-6.2-sol 은 none 을 거부해서 기본값(공식 문서 2026-11-24)


def _chat_raw(system, user, model):
    """한 번의 실제 API 호출. 모델 이름으로 공급자를 고른다."""
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
    from openai import OpenAI, RateLimitError, InternalServerError, APITimeoutError, APIConnectionError
    if model.startswith("gemini"):
        cl = _clients.get(("gemini", GEMINI_KEY)) or _clients.setdefault(
            ("gemini", GEMINI_KEY), OpenAI(api_key=GEMINI_KEY, base_url=GEMINI_URL, timeout=60))
    else:
        cl = _clients.get("openai")                         # 화면에서 연결한 것이 있으면 그것
        if not cl:
            key = os.environ.get("GATEWAY_KEY")
            cl = _clients.setdefault("openai", gateway(key) if key else OpenAI())
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(6):
        if model.startswith("gemini"):
            _throttle()
        try:
            if not model.startswith("gemini"):
                text, tin, tout = ask_openai(cl.with_options(max_retries=0), model, system, user)
                _track(tin, tout)
                return text
            r = cl.with_options(max_retries=0).chat.completions.create(model=model, messages=msgs)
            break
        except RateLimitError as e:
            body = str(getattr(e, "body", e))
            if "insufficient_quota" in body:            # 잔액 없음은 기다려도 안 풀린다
                raise RuntimeError("OpenAI 계정에 잔액이 없습니다") from None
            if "per day" in body:
                raise RuntimeError(f"{model} 무료 등급 하루 한도를 다 썼습니다. 다른 모델(예: flash-lite)을 고르거나 내일 다시 하세요.") from None
            m = re.search(r"retry in ([\d.]+)s", body)      # 분당 한도: 구글이 알려준 만큼 기다렸다 재시도
            time.sleep(float(m.group(1)) + 1 if m else 20)
        except (InternalServerError, APITimeoutError, APIConnectionError):
            time.sleep(5 * (attempt + 1))                   # 503 과부하·끊김: 점점 길게 기다렸다 재시도
    else:
        raise RuntimeError(f"{model} 호출이 계속 실패합니다(한도 또는 서버 과부하). 잠시 뒤 다시 하세요.")
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
            f"{row.persona}\n식생활: {row.culinary_persona}\n취미: {row.hobbies_and_interests}\n"
            f"{record_text(row)}\n(서사와 설문 기록이 다르면 설문 기록 쪽이 실제 행동입니다.)")



SYSTEM_TAIL = ("\n이 사람으로서 솔직하게, 실제 이 사람이 쓸 법한 말투로 2~3문장만 답하세요. "
               "광고 문구처럼 말하지 말고, 끌리면 끌린다고, 관심 없으면 관심 없다고 하세요. "
               "설문 기록은 이 사람의 실제 습관이니 판단에 반영하되, 기록 문구('주 1회 이상' 등)를 그대로 읊지는 마세요.")
# 주의: "대부분 그냥 지나친다" 같은 기저율 암시는 응답을 전부 부정·획일화시킴(실측 평균 1.9점, 문장 반복) → 넣지 않는다


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
            try:
                texts[futs[f]] = f.result()
            except RuntimeError as e:
                if "하루 한도" in str(e):   # 하루 한도는 나머지도 다 실패하므로 즉시 중단
                    ex.shutdown(cancel_futures=True)
                    raise
                print(f"[llm] 페르소나 {futs[f]} 실패: {e}", flush=True)
            if tick:
                tick()
    ok = [i for i, t in enumerate(texts) if t]
    if len(ok) < 0.8 * len(texts):
        raise RuntimeError(f"응답 실패가 너무 많습니다({len(texts) - len(ok)}/{len(texts)}). 잠시 뒤 다시 하세요.")
    pmf = np.empty((len(texts), len(ANCHORS)))
    pmf[ok] = ssr([texts[i] for i in ok])
    pmf[[i for i in range(len(texts)) if i not in ok]] = pmf[ok].mean(0)   # 실패자는 응답자 평균으로 대체
    return pmf, [t or "(응답 실패 — 평균으로 대체)" for t in texts]


def ping(model):
    """모델이 실제로 응답하는지 1회 시험. 성공 시 응답 문자열, 실패 시 예외."""
    return _chat_raw("짧게 답하세요.", "안녕? 한 단어로.", model)   # 캐시 우회: 지금 실제로 되는지 확인


def embed(texts):
    """Gemini 키가 있으면 gemini-embedding-001(입장 구분이 됨), 없으면 로컬 모델.
    로컬 mpnet 은 '안 살래요'도 주제가 같으면 '사겠다'에 붙여서 SSR 용으로는 부정확함(실측)."""
    if GEMINI_KEY:
        from openai import OpenAI, RateLimitError
        cl = OpenAI(api_key=GEMINI_KEY, base_url=GEMINI_URL, max_retries=0, timeout=60)
        out, i = [], 0
        while i < len(texts):                               # 무료 등급: 문장 1개 = 요청 1회, 분당 100회
            try:
                out += [d.embedding for d in cl.embeddings.create(model="gemini-embedding-001", input=texts[i:i + 90]).data]
                i += 90
            except RateLimitError as e:
                body = str(getattr(e, "body", e))
                if "per day" in body:
                    raise RuntimeError("Gemini 임베딩 무료 하루 한도를 다 썼습니다. 내일 다시 하세요.") from None
                m = re.search(r"retry in ([\d.]+)s", body)
                time.sleep(float(m.group(1)) + 1 if m else 30)
        return np.array(out), 0.01                          # 유사도 폭이 좁아 온도를 낮춤
    return np.array(list(_embedder().embed(texts))), 0.05


def ssr(texts, temp=None):
    """Semantic Similarity Rating (Maier et al. 2025): 응답과 기준 문장 5개의 코사인 유사도 → 분포."""
    # 실측: '애매' 기준 세트를 추가하면 "눈은 가는데 안 살래요" 같은 거절이 0.01→0.41로 올라가 오히려 악화 → 단일 세트 유지
    v, t = embed(ANCHORS + list(texts))
    temp = temp or t
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
