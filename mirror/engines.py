"""에이전트 엔진 연결: 같은 합성 인물·같은 자극을 여러 엔진에 넣고 예측과 '기준 질문' 정확도를 비교한다.

엔진: ml(LLM 없는 통계 기준) · direct(프레임워크 없는 LLM) · concordia(DeepMind) · tinytroupe(MS) · oasis(CAMEL, MiroFish 의 내부 엔진)
- 엔진은 각자 격리 환경(data/engines/<이름>, 파이썬 3.11)에서 돌고, LLM 은 로컬 중계기(127.0.0.1)를 통해서만 부른다.
  중계기가 속도 제한·재시도·호출 수 집계를 맡고, 엔진에는 진짜 API 키를 주지 않는다.
- 기준 질문: 인물 기억에서 '제로음료 빈도'를 가리고 물어, 실제 KREI 분포와의 차이(TVD)와 인물별 순위상관을 잰다.
MiroFish 자체는 AGPL-3.0 이고 Zep 클라우드 계정이 필요해 넣지 않았다(엔진은 OASIS 와 같다).
실행: python -m mirror.engines check | run [인원]
"""
import datetime as dt
import http.server
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from . import people as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV, RUNS, SCRIPTS = (os.path.join(ROOT, *p) for p in (("data", "engines"), ("data", "engine_runs"), ("mirror", "engine_scripts")))
VENV = {"concordia": "concordia", "tinytroupe": "tiny", "oasis": "oasis"}
ZERO = ["경험없음", "월1회미만", "월1회", "월2~3회", "주1회+"]
ASK = ("편의점에서 이 신제품을 봤습니다.\n{concept}\n솔직한 반응을 2문장으로 말한 뒤, 마지막 두 줄을 정확히 이 형식으로 쓰세요.\n"
       "구매의향: 0~100 사이 숫자\n제로빈도: " + " / ".join(ZERO) + " 중 하나 (평소 제로 음료를 마시는 빈도)")


# ---------- LLM 중계기 ----------
class Relay:
    """OpenAI 호환 요청을 받아 실제 서비스로 넘긴다. 분당 호출 제한, 429/5xx 재시도, 호출·토큰 집계."""

    def __init__(self, url, key, model, rpm=13, limit=400):
        self.url, self.key, self.model, self.rpm, self.limit = url.rstrip("/"), key, model, rpm, limit
        self.token, self.calls, self.tokens, self.times, self.lock = secrets.token_hex(12), 0, [0, 0], [], threading.Lock()
        relay = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Authorization") != "Bearer " + relay.token:
                    return self.reply(401, b'{"error":{"message":"bad token"}}')
                if not self.path.endswith("/chat/completions"):
                    return self.reply(404, b'{"error":{"message":"only chat/completions"}}')
                self.reply(*relay.forward(body))

            def reply(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                self.times = [t for t in self.times if now - t < 60]
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
            time.sleep(1)

    def forward(self, body):
        j = json.loads(body)
        j["model"] = self.model
        j.pop("stream", None)
        for k in ("reasoning_effort", "seed", "frequency_penalty", "presence_penalty", "logprobs", "parallel_tool_calls", "store", "verbosity"):      # 호환 서비스가 거절하는 선택 항목
            j.pop(k, None)
        if "max_completion_tokens" in j:
            j["max_tokens"] = min(8000, j.pop("max_completion_tokens"))
        err = (503, b'{"error":{"message":"upstream failed"}}')
        for k in range(7):
            with self.lock:
                if self.calls >= self.limit:
                    return 429, b'{"error":{"message":"call limit reached"}}'
                self.calls += 1
            self.wait()
            r = urllib.request.Request(self.url + "/chat/completions", json.dumps(j).encode(), {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=90) as f:
                    b = f.read()
                u = json.loads(b).get("usage") or {}
                self.tokens[0] += u.get("prompt_tokens", 0)
                self.tokens[1] += u.get("completion_tokens", 0)
                return 200, b
            except urllib.error.HTTPError as e:
                err = (e.code, e.read())
                if e.code not in (408, 429, 500, 502, 503, 504) or b"per day" in err[1].lower():
                    return err
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = (503, json.dumps({"error": {"message": str(e)[:120]}}).encode())
            time.sleep(min(30, 3 * 2 ** k))
        return err


# ---------- 인물·자극 ----------
def hidden_memory(r):
    """기준 질문의 정답(제로 빈도)과, 정답을 흘리는 '제로탄산' 선호를 가린 기억."""
    m = P.memory(r)
    fav = "탄산" if r.fav == "제로탄산" else r.fav
    m[3] = f"가장 좋아하는 음료 종류는 {fav}이다."
    return m


def cast(n, seed=0, method="cart", only_users=False):
    d = P.build(method, 4000, seed)
    d = P.users(d) if only_users else d
    d = d.sample(n, random_state=seed).reset_index(drop=True)
    return d, [{"id": i, "name": f"P{i:03d}", "memory": hidden_memory(r)} for i, r in enumerate(d.itertuples())]


def launch(engine, relay, timeout=1500):
    """격리 환경에서 엔진 실행. 환경변수는 중계기 주소·임시 토큰만 넘긴다(진짜 키·다른 비밀은 넘기지 않음)."""
    exe = os.path.join(ENV, VENV[engine], "bin", "python")
    out = os.path.join(RUNS, engine + ".json")
    if not os.path.exists(exe):
        return {"engine": engine, "answers": [], "error": "설치되지 않음"}
    if os.path.exists(out):
        os.replace(out, out + ".prev")
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": os.environ["HOME"], "LANG": "ko_KR.UTF-8", "OPENAI_API_KEY": relay.token, "OPENAI_BASE_URL": relay.base,
           "OPENAI_API_BASE_URL": relay.base, "LLM_MODEL": relay.model, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CAMEL_LOGGING_DISABLED": "true"}
    cwd = os.path.join(RUNS, "work_" + engine)
    os.makedirs(cwd, exist_ok=True)
    try:
        p = subprocess.run([exe, os.path.join(SCRIPTS, engine + "_run.py"), os.path.join(RUNS, "in.json"), out], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        tail = (p.stderr or p.stdout)[-600:]
    except subprocess.TimeoutExpired:
        tail = f"{timeout}초 초과"
    if os.path.exists(out):
        return json.load(open(out))
    return {"engine": engine, "answers": [], "error": tail}


def direct(people, concept, relay):
    ans = []
    for p in people:
        body = {"messages": [{"role": "system", "content": "당신은 아래 인물입니다. 이 사람으로서 답하세요.\n" + "\n".join(p["memory"])}, {"role": "user", "content": ASK.format(concept=concept)}]}
        code, b = relay.forward(json.dumps(body).encode())
        if code == 200:
            ans.append({"id": p["id"], "text": json.loads(b)["choices"][0]["message"].get("content") or ""})
    return {"engine": "direct", "answers": ans}


def ml(d, seed=0):
    """LLM 없는 비교 기준: 실제 응답자(KREI)로 '가린 값'을 예측하는 다항 로지스틱. 인물의 가려지지 않은 정보만 쓰고, 확률대로 하나를 뽑는다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    k = P.real()
    X = [c for c in P.COLS if c != "zero"]
    vis = lambda t: t[X].assign(fav=t.fav.replace("제로탄산", "탄산"))
    m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=.3, max_iter=800)).fit(vis(k), k.zero, logisticregression__sample_weight=k.w.values / k.w.mean())
    pr, rng = m.predict_proba(vis(d)), np.random.default_rng(seed)
    return {"engine": "ml", "answers": [{"id": i, "text": "제로빈도: " + str(rng.choice(m.classes_, p=q))} for i, q in enumerate(pr)]}


# ---------- 채점 ----------
def parse(text):
    b = re.search(r"구매\s*의향\s*[:：]\s*\**\s*(\d{1,3})", text or "")
    z = re.search(r"제로\s*빈도\s*[:：]\s*\**\s*(" + "|".join(map(re.escape, ZERO)) + ")", text or "")
    buy = int(b.group(1)) if b else None
    return (buy if buy is not None and buy <= 100 else None), (z.group(1) if z else None)


def grade(res, d, real):
    """엔진 하나의 결과 → 구매의향 요약 + 기준 질문 정확도."""
    rows = {a["id"]: parse(a.get("text")) for a in res.get("answers", [])}
    buy = np.array([v[0] for v in rows.values() if v[0] is not None], float)
    z = pd.Series({i: v[1] for i, v in rows.items() if v[1] is not None}, dtype=object)
    out = {"엔진": res["engine"], "답한 인원": len(rows), "구매의향 평균": buy.mean() if len(buy) else np.nan, "구매의향 표준편차": buy.std(ddof=1) if len(buy) > 1 else np.nan,
           "60점 이상": (buy >= 60).mean() if len(buy) else np.nan, "오류": (res.get("error") or "")[-160:]}
    if len(z) >= 8:
        truth = d.zero.loc[z.index]
        o = {v: i for i, v in enumerate(ZERO)}
        a, b = z.map(o).astype(float), truth.map(o).astype(float)
        out |= {"기준질문 분포 오차": P.tvd(z.value_counts(normalize=True), real), "기준질문 순위상관": a.corr(b, method="spearman") if a.std() > 0 else 0.0,
                "월2회+ 예측": (a >= 3).mean(), "월2회+ 인물 실제": (b >= 3).mean(), "월2회+ 실제 인구": float(real[["월2~3회", "주1회+"]].sum())}
    return out


def run(n=12, concept="자두맛 제로 탄산음료. 500ml 페트, 1,800원.", engines=("ml", "direct", "concordia", "tinytroupe", "oasis"), seed=0, method="cart", only_users=False, model="gemini-3.5-flash-lite", rpm=13, tick=None):
    from . import consumer as C
    os.makedirs(RUNS, exist_ok=True)
    d, people = cast(n, seed, method, only_users)
    json.dump({"concept": concept, "people": people, "ask": ASK.format(concept=concept)}, open(os.path.join(RUNS, "in.json"), "w"), ensure_ascii=False)
    k = P.real()
    k = P.users(k) if only_users else k
    real = P.share(k, "zero", k.w.values)
    relay = Relay(C.GEMINI_URL, C.GEMINI_KEY, model, rpm, limit=max(150, 25 * n))
    rows, raw = [], {}
    for i, e in enumerate(engines):
        if tick:
            tick(i / len(engines), f"{e} 실행 중")
        t, c0, k0 = time.time(), relay.calls, list(relay.tokens)
        res = direct(people, concept, relay) if e == "direct" else ml(d, seed) if e == "ml" else launch(e, relay)
        raw[e] = res
        rows.append(grade(res, d, real) | {"호출": relay.calls - c0, "토큰": sum(relay.tokens) - sum(k0), "초": round(time.time() - t)})
        print(rows[-1], flush=True)
    relay.srv.shutdown()
    t = pd.DataFrame(rows)
    ok = t["구매의향 평균"].dropna()
    s = {"시각": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds"), "콘셉트": concept, "인원": n, "인물 생성": method, "모델": model,
         "엔진 간 구매의향 범위": [float(ok.min()), float(ok.max())] if len(ok) else None, "표": t.replace({np.nan: None}).to_dict("records"),
         "답변": {e: r.get("answers", []) for e, r in raw.items()}, "사건": {e: r.get("events", [])[:80] for e, r in raw.items()}, "인물": d.to_dict("records")}
    json.dump(s, open(os.path.join(RUNS, "summary.json"), "w"), ensure_ascii=False, indent=1)
    return s


def check():
    assert parse("좋네요.\n구매의향: 35\n제로빈도: 월2~3회") == (35, "월2~3회") and parse("**구매의향:** 70") == (70, None) and parse("구매의향: 300") == (None, None) and parse("") == (None, None)
    k = P.real()
    r = k[k.fav == "제로탄산"].iloc[0]
    m = " ".join(hidden_memory(r))
    assert "제로" not in m and "탄산" in m
    d = pd.DataFrame({"zero": ZERO * 4})
    real = pd.Series(.2, index=ZERO)
    good = grade({"engine": "x", "answers": [{"id": i, "text": f"구매의향: 50\n제로빈도: {z}"} for i, z in enumerate(d.zero)]}, d, real)
    assert good["기준질문 분포 오차"] < 1e-9 and good["기준질문 순위상관"] > .99
    flat = grade({"engine": "x", "answers": [{"id": i, "text": "제로빈도: 주1회+"} for i in range(20)]}, d, real)
    assert flat["기준질문 분포 오차"] > .7 and flat["기준질문 순위상관"] == 0
    # 중계기: 토큰이 틀리면 거절, 호출 한도를 지키고, 엔진에 준 모델명을 강제로 바꾼다
    seen = []

    class Up(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append((json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.headers["Authorization"]))
            b = b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
            self.send_response(200 if len(seen) > 1 else 503)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
    up = http.server.HTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    rl = Relay(f"http://127.0.0.1:{up.server_port}", "REALKEY", "real-model", rpm=100, limit=3)

    def post(tok):
        q = urllib.request.Request(rl.base + "/chat/completions", b'{"model":"gpt-x","seed":1,"messages":[]}', {"Authorization": "Bearer " + tok})
        try:
            return urllib.request.urlopen(q, timeout=20).status
        except urllib.error.HTTPError as e:
            return e.code
    assert post("wrong") == 401 and not seen
    assert post(rl.token) == 200 and seen[-1][0]["model"] == "real-model" and "seed" not in seen[-1][0] and seen[-1][1] == "Bearer REALKEY"      # 503 한 번 뒤 재시도 성공
    assert post(rl.token) == 200 and post(rl.token) == 429 and rl.tokens == [6, 4]
    print("engines ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        kw = {"concept": os.environ["LAB_CONCEPT"]} if os.environ.get("LAB_CONCEPT") else {}
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 12, engines=sys.argv[3].split(",") if len(sys.argv) > 3 else ("ml", "direct", "concordia", "tinytroupe", "oasis"), **kw)
    else:
        check()
