"""새 화면(web/) 서버. 이 컴퓨터(127.0.0.1)에서만 열린다.

실행: python -m mirror.server [포트=8517] · 자체검사: python -m mirror.server check
"""
import base64
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import given as G
from . import given_ui as GU
from . import link as K
from . import people as P
from . import report as RP
from . import worlds as W

WEB = os.path.join(P.ROOT, "web")
REPORTS = os.path.join(P.ROOT, "data", "reports")
PHOTOS = os.path.join(P.ROOT, "data", "photos")
HIST = os.environ.get("MM_HISTORY", os.path.join(P.ROOT, "data", "verify.db"))
IMG = {b"\x89PNG\r\n\x1a\n": ("png", "image/png"), b"\xff\xd8\xff": ("jpg", "image/jpeg"), b"RIFF": ("webp", "image/webp")}
JOBS, RUNS, LOCK, LOCK2, LOCK3 = {}, {}, threading.Lock(), threading.Lock(), threading.Lock()


# ---------- 화면에 넘길 자료 ----------
def products():
    return [{"id": k, "name": v, "concept": {"화채": "수박화채맛, 여름 한정 (5~9월)", "프리즈": "얼려 먹는 커피, 냉동 파우치", "쥬시": "비가열 프리미엄 착즙주스, 유리병"}[k], "color": W.LOOK[k][0], "shape": W.LOOK[k][1], "price": G.SPEC[v]["srp"]} for k, v in W.IDS.items()]


def lean(ws):
    return [{k: w[k] for k in ("id", "cond", "winner")} | {"buy": {k: round(v, 4) for k, v in w["buy"].items()}, "intent": {k: round(v, 3) for k, v in w["intent"].items()}, "curve": w["curve"], "seen": {k: round(v, 3) for k, v in w["seen"].items()}} for w in ws]


def segments():
    d = W.ready()["people"]
    f = lambda c, v: [{"id": x, "share": float(d.w[d[c] == x].sum() / d.w.sum()), "aff": {}} for x in v]
    return {"age": f("age10", W.AGES), "sex": f("sex", W.SEXES), "region": f("reg", W.REGS)}


def verify():
    st, _ = G.stages()
    s, k = G.summary(G.forecast(1500)).set_index("컨셉"), float(np.exp(np.log(G.past().k).mean()))
    fun = {}
    for i, c in W.IDS.items():
        lo, hi = G.ASSUME["유도문항|쥬시모먼트"][:2]
        lead = (lo + hi) / 2 if i == "쥬시" else 1.0
        fun[i] = {"raw": float(st["원본"][c]), "clean": float(st["불성실·중복 제거"][c]), "weighted": float(st["+ 순서 보정"][c]), "lead": lead, "store": float(s["살 수 있는 비율"][c]), "conv": k,
                  "pay": float(st["제시 가격을 낼 수 있음"][c]), "lo": float(s["하한"][c]), "hi": float(s["상한"][c]), "median": float(s["구매경험률"][c]), "why": G.ASSUME["입점률|" + c][2]}
    a, iss = G.audit(), []
    big = {"패널조사", "과거 컨셉테스트", "유통 인프라", "FGI"}
    for r in a.itertuples():
        st_ = "보류" if re.search(r"쓰지 않음|참고만|빼지 않음|알 수 없음\.|읽지 않음|반영하지 않음|대조", r.처리) else "검토 중" if r.확신 == "낮음" else "해결"
        iss.append([("높음" if r.자료 in big else "중간" if r.확신 != "높음" or r.자료 in ("POS", "출고") else "낮음"), r.자료, r.문제 + " → " + r.처리, st_])
    return {"funnel": fun, "issues": sorted(iss, key=lambda x: ["높음", "중간", "낮음"].index(x[0])), "excluded": int((G.respondents().제외 != "").sum()), "reviews": reviews()}


def reviews():
    h = GU.history()
    return [[r["시각"], r["메모"] or "메모 없음", r["구매경험률"]] for _, r in h.head(30).iterrows()] if len(h) else []


def evidence():
    d = W.ready()
    pw = np.exp(d["lw"])
    pw /= pw.sum()
    v, er = G.validate()
    sig = K.signal_test()
    from . import multiverse as U
    _, info = U.select({})
    return {"population": {"people": round(float(d["people"].w.sum())), "respondents": len(d["people"]), "effective": round(U.ess(d["people"].w)), "margin": round(98 / np.sqrt(U.ess(d["people"].w)), 1)},
            "all": {"people": round(info["인구"]), "respondents": info["실응답자"], "effective": round(info["유효표본"])},
            "panel": {"raw": 304, "clean": int(len(G.panel()[0])), "effective": round(G.stages()[1]["유효표본"]), "margin": round(98 / np.sqrt(G.stages()[1]["유효표본"]), 1)},
            "kept": round(float(1 / (pw ** 2).sum())), "tried": len(pw),
            "validation": [["과거 출시 5개 제품 구매경험률", f"오차 {er['의향 × 점포율 × k']:.0%}", f"의향만 쓰면 {er['의향만 환산']:.0%}", "하나씩 빼고 맞힘. 제품이 5개뿐이고 대회 제공 가상 데이터"],
                           ["사회 안에서 과거 제품 다시 팔아 보기", " · ".join(f"{np.exp((d['sim'][:, j] * pw).sum()):.1%}" for j in range(5)), "실제 " + " · ".join(f"{x:.1%}" for x in G.past().경험률), "이 5개로 세상의 조건을 걸러냈으므로 독립 검증이 아님"],
                           ["세 컨셉 구매의향 모델", " · ".join(f"{c[:2]} {m[1]:.2f}" for c, m in zip(W.IDS, G.persona_model().values())), "찍기 0.50", "패널 273명 교차검증 판별력"],
                           ["10대 음료 1순위 분포 오차", "0.143", "성인 모델 그대로 0.340", "학습에 쓰지 않은 10대 기준"],
                           ["관심(검색·SNS)이 판매를 설명하나", f"기온·추세를 빼면 p ≥ {sig.p.min():.2f}", f"겉보기 상관 최대 {sig['단순 상관'].max():.2f}", "그래서 반응량을 예측에 넣지 않음"],
                           ["두 계산의 차이 (구매경험률 중앙값)", "실험 " + " · ".join(f"{k} {np.median([w['buy'][k] for w in RUNS['default']['worlds']]):.1%}" for k in W.IDS), "신제품 검증 " + " · ".join(f"{x:.1%}" for x in G.summary(G.forecast(1500)).구매경험률), "실험은 사람과 입소문까지 넣은 값, 신제품 검증은 식으로 계산한 값"],
                           ["카니발(자기 제품 잠식)", "판별 불가", "—", "주어진 자료에서 통제 제품도 같이 움직임. 매출은 순증이 아닌 총매출"]]}


def react():
    from . import trends as T
    snap, rows = K.items()
    now = pd.Timestamp.now(tz="Asia/Seoul").normalize()
    t = pd.to_datetime(pd.Series([r.get("published") or r.get("first_seen") for r in rows], dtype=object), utc=True, errors="coerce", format="mixed").dt.tz_convert("Asia/Seoul").dt.normalize()
    age = (now - t).dt.days
    c = K.candidates(snap, 14)
    c = c[~c["이름"].str.contains("CU|GS25|세븐일레븐|이마트|편의점|올리브영|기타")].head(8) if len(c) else c
    out = {}
    for i, con in W.IDS.items():
        e = K.evidence(con, rows)
        near = [x for x in e["목록"] if x["직접 관련"] and not x["홍보·뉴스"]]
        out[i] = {"posts": e["직접 관련"], "recent": e["최근 30일"], "verdict": e["판정"], "worries": [f"{k} {v}건" for k, v in sorted(e["걱정"].items(), key=lambda x: -x[1])[:4]] or ["언급된 걱정 없음"], "quotes": [[x["채널"], x["글"].strip()[:70]] for x in near[:3]]}
    return {"daily": [int((age == d).sum()) for d in range(27, -1, -1)], "total": len(rows), "trending": [{"name": r["이름"], "kind": r["종류"], "status": r["상태"], "posts": int(r["최근 글"]), "authors": int(r["작성자"])} for _, r in c.iterrows()] if len(c) else [],
            "products": out, "channels": [[x["label"], {"success": "완료", "partial": "일부 실패", "error": "실패", "running": "수집 중"}.get(x["status"], "대기"), x["count"]] for x in snap["channels"]], "alive": snap["alive"], "at": pd.Timestamp.now(tz="Asia/Seoul").strftime("%H:%M")}


CH = {"SNS": "블루스카이·마스토돈", "social": "블루스카이·마스토돈", "뉴스": "뉴스", "news": "뉴스", "커뮤니티": "커뮤니티", "community": "커뮤니티", "instagram": "Instagram", "threads": "Threads", "tiktok": "TikTok", "youtube": "YouTube", "트렌드": "검색 트렌드"}
_posts = {"t": 0.0, "rows": []}


def posts():
    """모은 글 전체(식음료 낱말이 있는 것만) 한 벌. 1분 동안 재사용."""
    from . import food_scope as FS
    with LOCK2:
        if time.time() - _posts["t"] > 60:
            _, rows = K.items()
            num = lambda v: None if v is None or v != v else float(v)
            web = lambda u: u if isinstance(u, str) and re.match(r"https?://", u) else ""
            rx = {c: re.compile(v["core"], re.I) for c, v in K.TERMS.items()}
            out, seen = [], set()
            for r in rows:
                w = K.words(r).strip()
                key = re.sub(r"\W", "", w)[:80]
                if not key or key in seen or not (r.get("food_category") or FS.candidate(r)):
                    continue
                seen.add(key)
                t = pd.to_datetime(r.get("published") or r.get("first_seen"), utc=True, errors="coerce")
                out.append({"text": w[:400], "ch": CH.get(r.get("platform"), r.get("platform") or "기타"), "who": r.get("creator") or r.get("source") or "", "ts": 0 if pd.isna(t) else t.timestamp(), "at": "" if pd.isna(t) else t.tz_convert("Asia/Seoul").strftime("%Y.%m.%d %H:%M"),
                            "views": num(r.get("views")), "likes": num(r.get("likes")), "comments": num(r.get("comments")), "url": web(r.get("url")), "thumb": web(r.get("thumbnail")), "cat": r.get("food_category") or "", "topic": r.get("topic_label") or "",
                            "con": [i for i, c in W.IDS.items() if rx[c].search(w)], "ours": bool(FS.LOTTE.search(w))})
            _posts.update(t=time.time(), rows=out)
        return _posts["rows"]


def a_posts(req, j):
    q = req.query_params
    rows = posts()
    cnt = {}
    for r in rows:
        cnt[r["ch"]] = cnt.get(r["ch"], 0) + 1
    total, ch, con, word = len(rows), q.get("ch", ""), q.get("con", ""), q.get("q", "").strip().lower()
    rows = [r for r in rows if (not ch or r["ch"] == ch) and (not con or (r["ours"] if con == "롯데칠성" else con in r["con"])) and (not word or word in r["text"].lower())]
    key = {"recent": "ts", "likes": "likes", "views": "views", "comments": "comments"}[q.get("sort", "recent")]
    rows = sorted(rows, key=lambda r: r[key] or 0, reverse=True)
    n, size = max(0, int(q.get("page", 0))), 24
    return {"total": total, "count": len(rows), "page": n, "pages": max(1, -(-len(rows) // size)), "channels": sorted(cnt.items(), key=lambda x: -x[1]), "rows": rows[n * size:(n + 1) * size]}


async def observe(req):
    if not local(req):
        return PlainTextResponse("forbidden", 403)
    from . import trends as T
    h = await run_in_threadpool(lambda: T.html(T.snapshot(168)))
    return HTMLResponse(h.replace("</head>", "<style>.workspace,#insight-clear{display:none!important}</style></head>", 1))      # 콘텐츠 레이더 칸은 보이지 않게(화면 코드가 그 요소를 쓰므로 지우지는 않는다)


_base = {}


def base():
    with LOCK3:                                                  # 동시에 들어온 첫 요청들이 같은 준비를 여러 번 하지 않게
        if not _base:
            ws = W.default()
            RUNS["default"] = {"flt": {}, "worlds": ws}
            if os.path.exists(W.CACHE):                              # 미리 계산해 둔 기본 실험도 실제로 돌린 것이니 그 시각으로 남긴다
                log_experiment(ws, {}, "base-" + W._sig()[:8], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(W.CACHE))))
            ev = evidence()
            _base.update(products=products(), conditions=W.COND, condLabels={k: list(v) for k, v in W.LABEL.items()}, segments=segments(), cross=W.cross(), population=ev["population"], worlds=lean(ws), run="default",
                         verify=verify(), evidence=ev, concept={"steps": [x for x, _ in STEPS], "results": []}, step_days=13, bench=[float(x) for x in (G.past().경험률.min(), G.past().경험률.mean(), G.past().경험률.max())])
    return _base | {"verify": _base["verify"] | {"reviews": reviews()}}


# ---------- 작업(백그라운드) ----------
def start(fn):
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"state": "running", "n": 0, "result": None, "error": "", "t": time.time()}

    def go():
        try:
            JOBS[jid]["result"] = fn(JOBS[jid])
            JOBS[jid]["state"] = "done"
        except Exception as e:                                   # 화면에 실패로 알린다
            JOBS[jid].update(state="error", error=f"{type(e).__name__}: {str(e)[:160]}")
    threading.Thread(target=go, daemon=True).start()
    for k in [k for k, v in JOBS.items() if time.time() - v["t"] > 3600]:
        JOBS.pop(k, None)
    return jid


def clean_filter(j):
    f = {}
    for key, col, ok in (("age", "age10", W.AGES), ("sex", "sex", W.SEXES), ("region", "reg", W.REGS)):
        v = [x for x in j.get(key, []) if x in ok]
        if not v:
            raise ValueError("대상 소비자를 하나 이상 고르세요.")
        if len(v) < len(ok):
            f[col] = v
    return f


def run_job(flt, n, only):
    def fn(job):
        with LOCK:                                               # 한 번에 하나씩
            from . import agents as A
            job.update(phase="ask")
            ai = A.get(only, AI["model"] if AI["on"] else None, lambda k: job.update(n=k))
            job.update(phase="run", n=0)
            ws = W.default(lambda k: job.update(n=k)) if not ai and not flt and n == 1500 else W.run(n, flt, tick=lambda k: job.update(n=k), ai=ai)
        rid = uuid.uuid4().hex[:8]
        RUNS[rid] = {"flt": flt, "worlds": ws, "ai": ai}
        log_experiment(ws, flt, only=only)
        for k in list(RUNS)[1:-4]:
            RUNS.pop(k, None)
        return {"run": rid, "worlds": lean(ws), "ai": ai["info"] if ai else None}
    return fn


# ---------- OpenAI 연결 ----------
MODELS = {"gpt-6-luna": (.10, .50), "gpt-6.1-terra": (1.0, 5.0), "gpt-6.2-sol": (2.0, 10.0)}      # $/100만 토큰 (입력, 출력), developers.openai.com 2026-11-24 확인
AI = {"on": False, "model": "gpt-6-luna"}


def connect(key):
    """대회 주소로 모델마다 짧게 한 번 불러 확인하고, 키는 메모리에만 둔다(파일·기록에 남기지 않음). 실패하면 이유만 말한다."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

    from . import consumer as C
    cl = C.gateway(key)

    def probe(m):
        try:
            C.ask_openai(cl.with_options(timeout=25), m, "짧게", "안녕", max_output_tokens=16)
        except Exception as e:
            return e
    with ThreadPoolExecutor(len(MODELS)) as ex:
        res = dict(zip(MODELS, ex.map(probe, MODELS)))
    have = [m for m, e in res.items() if e is None]
    if not have:
        e = next(iter(res.values()))
        raise ValueError("키가 맞지 않습니다" if isinstance(e, AuthenticationError) else "사용 한도에 걸렸습니다. 잠시 뒤 다시 하세요" if isinstance(e, RateLimitError)
                         else "대회 서버에 연결하지 못했습니다. 인터넷을 확인하세요" if isinstance(e, APIConnectionError)
                         else f"연결하지 못했습니다(서버 응답 {e.status_code})" if isinstance(e, APIStatusError) else "연결하지 못했습니다: " + type(e).__name__)
    C._clients["openai"] = cl
    AI.update(on=True, have=have)
    if AI["model"] not in have:
        AI["model"] = have[0]


def auto_connect():
    """.env 나 환경변수에 GATEWAY_KEY 가 있으면 켤 때 연결한다."""
    if os.environ.get("GATEWAY_KEY"):
        try:
            connect(os.environ["GATEWAY_KEY"])
        except ValueError as e:
            print("OpenAI 자동 연결 실패:", e, flush=True)


def ai_state():
    from . import consumer as C
    use = [{"model": m, "calls": n, "in": a, "out": b, "usd": (a * MODELS[m][0] + b * MODELS[m][1]) / 1e6} for m, (n, a, b) in C.ledger().items() if m in MODELS]
    return {"use": use, "on": AI["on"], "model": AI["model"], "models": [{"id": m, "ok": m in AI.get("have", MODELS)} for m in MODELS]}


def a_openai(req, j):
    from . import consumer as C
    if req.method == "POST":
        if j.get("off"):
            C._clients.pop("openai", None)
            AI.update(on=False)
            AI.pop("have", None)
        if j.get("model"):
            if j["model"] not in AI.get("have", MODELS):
                raise ValueError("쓸 수 없는 모델입니다")
            AI["model"] = j["model"]
        key = str(j.get("key", "")).strip()
        if key:
            if not re.fullmatch(r"[!-~]{16,400}", key):
                raise ValueError("키 모양이 아닙니다. 받은 키를 그대로 넣어 주세요")
            connect(key)
    return ai_state()


# ---------- 검증 이력 ----------
def hdb():
    c = sqlite3.connect(HIST, timeout=10)
    c.execute("create table if not exists verify(id text primary key, ts text, kind text, name text, info text, photo text, state text, steps text, result text)")
    return c


def put(i, **kw):
    with hdb() as c:
        c.execute("insert or ignore into verify(id, ts, state) values(?,?,'진행 중')", (i, time.strftime("%Y-%m-%dT%H:%M:%S")))
        for k, v in kw.items():                                  # 열 이름은 코드에서만 정해진다
            c.execute(f"update verify set {k}=? where id=?", (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), i))


def past_runs():
    with hdb() as c:
        rs = c.execute("select id, ts, kind, name, info, photo, state, steps, result from verify order by ts desc, rowid desc limit 200").fetchall()
    live = {v.get("hid") for v in JOBS.values() if v["state"] == "running"}
    return [{"id": r[0], "ts": r[1][5:16].replace("T", " ").replace("-", "."), "kind": r[2], "name": r[3], "info": r[4] or "", "photo": bool(r[5]), "state": "중단됨" if r[6] == "진행 중" and r[0] not in live else r[6],
             "steps": json.loads(r[7]) if r[7] else [], "result": r[8] or ""} for r in rs]


BOARD = {"화채": "image5.jpg", "프리즈": "image4.jpg", "쥬시": "image3.jpg"}                  # 과제 설명서의 컨셉보드


def log_experiment(ws, flt, i=None, ts=None, only=None):
    """실험 한 번을 제품마다 따로 이력에 남긴다(서로 비교한 값은 남기지 않는다)."""
    i = i or uuid.uuid4().hex[:8]
    tgt = f"{', '.join(x for v in flt.values() for x in v) or '19~59세 전체'} · 세상 {len(ws):,}개"
    for p in [p for p in products() if only in (None, p["id"])]:
        put(f"{i}-{p['id']}", kind="신제품", name=p["name"], info=f"{p['concept']} · {p['price']:,}원", photo="board:" + BOARD[p["id"]], state="완료", steps=[["대상", tgt], ["컨셉", p["concept"]]], result="")
        if ts:
            with hdb() as c:
                c.execute("update verify set ts=? where id=?", (ts, f"{i}-{p['id']}"))


def save_photo(s):
    """data:image/...;base64 → 파일 이름. 그림이 아니거나 5MB 를 넘으면 거절한다."""
    if not s:
        return ""
    try:
        raw = base64.b64decode(str(s).split(",", 1)[1], validate=True)
    except (IndexError, ValueError):
        raise ValueError("사진을 읽지 못했습니다") from None
    ext = next((v[0] for k, v in IMG.items() if raw.startswith(k) and (k != b"RIFF" or raw[8:12] == b"WEBP")), None)
    if not ext or len(raw) > 5 * 1024 * 1024:
        raise ValueError("PNG, JPG, WEBP 사진만 5MB까지 넣을 수 있습니다")
    os.makedirs(PHOTOS, exist_ok=True)
    f = uuid.uuid4().hex + "." + ext
    with open(os.path.join(PHOTOS, f), "wb") as h:
        h.write(raw)
    return f


async def photo(req):
    if not local(req):
        return PlainTextResponse("forbidden", 403)
    with hdb() as c:
        r = c.execute("select photo from verify where id=?", (req.path_params["hid"],)).fetchone()
    f = r[0] if r and r[0] else ""
    f = os.path.join(G.DIR, "img", os.path.basename(f[6:])) if f.startswith("board:") else os.path.join(PHOTOS, os.path.basename(f)) if f else ""
    return FileResponse(f, media_type={v[0]: v[1] for v in IMG.values()}[f.rsplit(".", 1)[1]]) if f and os.path.exists(f) else PlainTextResponse("", 404)


# ---------- 새 컨셉 ----------
def past_like(c):
    """비슷한 과거 제품: 이름·설명에 겹치는 낱말 + 가격대."""
    d = G.sheet(G.A, "4_제품자산DB").fillna("")
    txt = " ".join((c["name"], c["desc"]))
    grams = lambda s: {s[i:i + 2] for s in [re.sub(r"[^가-힣A-Za-z]", "", s)] for i in range(len(s) - 1)}
    g = grams(txt)
    sc = [(len(g & grams(f"{r.제품명} {r.카테고리} {r.컨셉요약}")) + 2 * any(k in txt and k in f"{r.제품명}{r.컨셉요약}{r.카테고리}" for k in ("커피", "주스", "탄산", "우유", "제로", "냉동", "얼려", "수박", "프리미엄")), r) for r in d.itertuples()]
    return [r for s, r in sorted(sc, key=lambda x: -x[0])[:3] if s > 0]


def s_past(c, x):
    d = G.sheet(G.A, "4_제품자산DB").fillna("")
    top = past_like(c)
    col = d.columns[10]
    x["past"] = [{"제품": r.제품명, "상태": r.상태, "사유": str(d.loc[d.제품명 == r.제품명, col].iloc[0])} for r in top]
    return "; ".join(f"{p['제품']} ({p['상태']}" + (f", {p['사유']}" if p["사유"] else "") + ")" for p in x["past"]) or "비슷한 과거 제품을 찾지 못했습니다"


def s_posts(c, x):
    kw = [w for w in re.findall(r"[가-힣]{2,}", c["desc"]) if w not in ("편의점", "판매", "제품", "음료")][:8]
    _, rows = K.items()
    hit = [r for r in rows if sum(w in K.words(r) for w in kw) >= 2]
    x["posts"] = [K.words(r)[:100] for r in hit[:5]]
    x["kw"] = kw
    return f"모아 둔 글 {len(rows):,}건 가운데 낱말 2개 이상 겹치는 글 {len(hit)}건 (찾은 낱말: {', '.join(kw) or '없음'}). 새로 검색하지는 않았습니다"


def s_talk(c, x):
    """합성 인물에게 LLM 으로 묻는다. 대회 컨셉은 외부로 보내지 않는다."""
    from . import consumer as C
    from . import multiverse as U
    if any(re.sub(r"\s", "", v) in re.sub(r"\s", "", c["name"] + c["desc"]) for v in W.IDS.values()):
        raise PermissionError("대회 제공 컨셉은 외부 AI로 보내도 되는지 확인되지 않아 인터뷰하지 않았습니다")
    model = AI["model"] if AI["on"] else "gemini-3.5-flash-lite" if C.GEMINI_KEY else ""
    if not model:
        raise PermissionError("AI가 연결되지 않아 인터뷰를 건너뛰었습니다")
    w, _ = U.select({})
    who = U.synthetic(w, 6, seed=0)      # 늘 같은 6명: 제품끼리 비교되고, 같은 컨셉은 저장된 답을 써서 비용이 없다
    q = (f"편의점에서 이 신제품을 봤습니다: {c['name']}. {c['desc']} 가격 {c['price']}.\n솔직하게 답하세요. JSON 하나로만: "
         '{"score":1~5 정수(5=꼭 산다),"why":"이유 한 문장(40자 이내)"}')
    out, why = [], "답을 읽지 못함"
    for i in range(len(who)):
        try:
            for more in ("", "\n다른 글 없이 JSON 하나만 출력하세요."):      # 못 읽는 답도 저장되므로 질문을 바꿔 한 번 더
                t = C.chat("당신은 아래 사람입니다. 마케터처럼 말하지 말고 이 사람의 처지대로 답하세요.\n" + "\n".join(P.memory(who.iloc[i])), q + more, model)
                try:
                    m = json.loads(re.search(r"\{.*?\}", t, re.S).group(0))
                    m = {"score": int(m["score"]), "why": str(m["why"])}
                    break
                except (AttributeError, KeyError, TypeError, ValueError):
                    if more:
                        raise
            out.append({"누구": f"{who.age[i]} {who.sex[i]}", "점수": m["score"], "이유": m["why"][:60]})
        except Exception as e:
            why = str(e) if type(e) is RuntimeError else type(e).__name__
            if "잔액" in why:
                break
    if not out:
        raise RuntimeError(f"AI가 응답하지 않았습니다({why})")
    x["talk"] = out
    hi = sum(o["점수"] >= 4 for o in out)
    return f"가상 소비자 {len(out)}명 가운데 {hi}명이 4~5점. " + " / ".join(f"{o['누구']}: {o['이유']}" for o in out[:3]) + " (AI의 답이라 실제 조사와 맞는지 검증되지 않음)"


def s_forecast(c, x):
    """조사 전 예측: 의향은 세 컨셉과 과거 5개의 범위에서, 취급점포율은 과거 5개의 범위에서 뽑는다. 이 컨셉만의 정보는 아직 없다."""
    g, p = np.random.default_rng(0), G.past()
    st = G.stages()[0]["+ 순서 보정"].values
    top, lk = g.uniform(st.min(), st.max(), 4000), np.log(p.k.values)
    cold = bool(re.search(r"냉동|얼려|얼린", c["desc"] + c["name"]))
    chill = bool(re.search(r"냉장|착즙|비가열", c["desc"] + c["name"]))
    lo, hi = (.18, .33) if cold else (.28, .56) if chill else (.80, .92)
    d = g.uniform(lo, hi, 4000)
    t = top * d * np.exp(lk.mean() + lk.std(ddof=1) * np.sqrt(1.2) * g.standard_t(4, 4000))
    a, b, e = np.quantile(t, [.05, .5, .95])
    x["forecast"] = {"lo": float(a), "median": float(b), "hi": float(e)}
    return f"12개월 구매경험률 {b:.1%} (90% 범위 {a:.1%}~{e:.1%}). 조사 전이라 의향은 기존 컨셉들의 범위({st.min():.0%}~{st.max():.0%})를, 살 수 있는 점포는 {'냉동' if cold else '냉장' if chill else '상온'} 제품 범위({lo:.0%}~{hi:.0%})를 썼습니다"


def s_report(c, x):
    os.makedirs(REPORTS, exist_ok=True)
    f = os.path.join(REPORTS, f"{time.strftime('%Y%m%d-%H%M%S')}.md")
    L = [f"# {c['name']}", "", f"- 설명: {c['desc']}", f"- 가격: {c['price']} · 주 대상: {c['target']}", "", *[f"## {n}\n\n{r or '건너뜀'}\n" for (n, _), r in zip(STEPS, x["results"])], "## 주의\n\n조사하지 않은 컨셉이라 범위가 넓습니다. 실제 설문 50~100명을 받으면 좁힐 수 있습니다."]
    with open(f, "w") as h:
        h.write("\n".join(L))
    GU.save(f"신제품 검증: {c['name']}", {}, pd.DataFrame([{"컨셉": c["name"][:20], "구매경험률": x["forecast"]["median"], "소매 매출(억)": 0}]))
    return f"보고서를 저장했습니다: {os.path.relpath(f, P.ROOT)}"


STEPS = [("비슷한 과거 제품 찾기", s_past), ("소비자 글 찾기", s_posts), ("가상 소비자 인터뷰", s_talk), ("예측", s_forecast), ("보고서", s_report)]


def concept_job(c, hid):
    def fn(job):
        job["hid"] = hid
        x = {"results": []}
        job.update(steps=[{"state": "wait", "text": ""} for _ in STEPS])
        for k, (_, f) in enumerate(STEPS):
            job["steps"][k]["state"] = "run"
            try:
                t = f(c, x)
                job["steps"][k] = {"state": "done", "text": t}
            except PermissionError as e:                         # 하지 않기로 한 단계는 건너뛰고 계속
                t = str(e)
                job["steps"][k] = {"state": "skip", "text": t}
            except Exception as e:
                t = f"{type(e).__name__}: {str(e)[:120]}"
                job["steps"][k] = {"state": "fail", "text": t}
            x["results"].append(t)
            put(hid, steps=[[n, r] for (n, _), r in zip(STEPS, x["results"])])
        f = x.get("forecast")
        bad = any(y["state"] == "fail" for y in job["steps"])
        put(hid, state="일부 실패" if bad else "완료", result=f"12개월 구매경험률 {f['median']:.1%} ({f['lo']:.1%}~{f['hi']:.1%})" if f else "예측 실패")
        return {"forecast": f}
    return fn


# ---------- 주소 ----------
def local(req):
    """다른 사이트가 이 서버를 부르지 못하게: 호스트와 출처가 이 컴퓨터여야 한다."""
    ok = {"127.0.0.1", "localhost", "testserver"}
    host = (req.headers.get("host") or "").split(":")[0]
    org = req.headers.get("origin")
    return host in ok and (not org or re.sub(r"^https?://", "", org).split(":")[0] in ok)


def api(fn):
    async def h(req):
        if not local(req):
            return PlainTextResponse("forbidden", 403)
        try:
            body = await req.json() if req.method == "POST" else {}
            return JSONResponse(await run_in_threadpool(fn, req, body))
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse({"error": str(e)[:200]}, 400)
    return h


def a_run(req, j):
    n = int(j.get("n", 1500))
    if n not in (200, 500, 1000, 1500):
        raise ValueError("세상 개수가 맞지 않습니다")
    if j.get("product") not in W.IDS:
        raise ValueError("제품을 하나 고르세요")
    return {"job": start(run_job(clean_filter(j), n, j["product"]))}


def a_job(req, j):
    x = JOBS.get(req.path_params["jid"])
    if not x:
        raise ValueError("없는 작업입니다")
    return {k: x.get(k) for k in ("state", "n", "result", "error", "steps", "phase")}


def a_society(req, j):
    q, r = req.query_params, RUNS.get(req.query_params["run"])
    i, p = int(q["world"]), q["product"]
    if not r or p not in W.IDS or not 0 <= i < len(r["worlds"]):
        raise ValueError("다시 실행해 주세요")
    L = W.one(i, r["flt"], log=p, ai=r.get("ai"))["log"]
    days = [{"d": d, "exposed": [x["id"] for x in L["nodes"] if x["exposed"] == d], "bought": [x["id"] for x in L["nodes"] if x["bought"] == d], "positive": [x["id"] for x in L["nodes"] if x["posted"] == d and x["positive"]], "negative": [x["id"] for x in L["nodes"] if x["posted"] == d and not x["positive"]]} for d in range(1, W.STEPS + 1)]
    return {"worldId": i, "product": p, "days": days, **L}


def a_report(req, j):
    base()
    r = RUNS.get(req.query_params.get("run", "default"))
    if not r:
        raise ValueError("다시 실행해 주세요")
    if "report" not in r:
        r["report"] = RP.build(r)
    return {"products": r["report"], "worlds": len(r["worlds"]), "target": ", ".join(x for v in r["flt"].values() for x in v) or "19~59세 전체", "panel": int(len(G.panel()[0]))}


def a_concept(req, j):
    c = {k: str(j.get(k, "")).strip()[:m] for k, m in (("name", 60), ("desc", 300), ("price", 20), ("target", 40))}
    if not c["name"] or not c["desc"]:
        raise ValueError("제품 이름과 한 줄 설명을 적어 주세요")
    f, hid = save_photo(j.get("photo")), uuid.uuid4().hex[:10]
    put(hid, kind="신제품", name=c["name"], info=" · ".join(x for x in (c["desc"], c["price"], c["target"]) if x), photo=f)
    return {"job": start(concept_job(c, hid)), "id": hid}


def a_review(req, j):
    GU.save(str(j.get("memo", ""))[:300], {k: v[:2] for k, v in G.ASSUME.items()}, G.summary(G.forecast(1500)))
    return {"reviews": reviews()}


class Fresh(StaticFiles):
    """화면 파일은 쓸 때마다 바뀌었는지 확인하게 한다(예전 파일과 섞이지 않게)."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app = Starlette(routes=[Route("/api/data", api(lambda r, j: base())), Route("/api/reactions", api(lambda r, j: react())), Route("/api/posts", api(a_posts)), Route("/observe", observe), Route("/api/run", api(a_run), methods=["POST"]), Route("/api/job/{jid}", api(a_job)), Route("/api/society", api(a_society)),
                        Route("/api/report", api(a_report)), Route("/api/openai", api(a_openai), methods=["GET", "POST"]), Route("/api/history", api(lambda r, j: {"rows": past_runs()})), Route("/api/photo/{hid}", photo), Route("/api/concept", api(a_concept), methods=["POST"]), Route("/api/review", api(a_review), methods=["POST"]), Mount("/", Fresh(directory=WEB, html=True, check_dir=False))])


def check():
    import tempfile
    from starlette.testclient import TestClient
    GU.DB = os.path.join(tempfile.mkdtemp(), "h.db")
    global REPORTS, HIST, PHOTOS
    REPORTS, PHOTOS, HIST = tempfile.mkdtemp(), tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "v.db")
    c = TestClient(app)
    d = c.get("/api/data").json()
    assert len(d["worlds"]) == 1500 and len(d["verify"]["issues"]) > 15 and d["population"]["people"] > 25e6
    assert c.get("/data.js").headers["cache-control"] == "no-cache"
    assert c.get("/api/data", headers={"host": "evil.com"}).status_code == 403 and c.post("/api/run", json={}, headers={"origin": "https://evil.com"}).status_code == 403
    s = c.get("/api/society", params={"run": "default", "world": 4, "product": "화채"}).json()
    assert len(s["nodes"]) == W.N and abs(sum(x["bought"] is not None for x in s["nodes"]) / W.N - d["worlds"][4]["buy"]["화채"]) < 1e-3
    assert c.get("/api/society", params={"run": "default", "world": 99999, "product": "화채"}).status_code == 400 and c.post("/api/run", json={"product": "화채", "age": [], "sex": [], "region": [], "n": 200}).status_code == 400
    z = c.get("/api/report").json()
    assert z["worlds"] == 1500 and set(z["products"]) == set(W.IDS) and c.get("/api/report", params={"run": "x"}).status_code == 400
    p = c.get("/api/posts", params={"sort": "likes"}).json()
    assert p["total"] > 500 and len(p["rows"]) == 24 and all(not r["url"] or r["url"].startswith("http") for r in p["rows"]) and c.get("/api/posts", params={"q": "없는낱말없는낱말"}).json()["count"] == 0
    assert "trend-data" in c.get("/observe").text and ".workspace,#insight-clear{display:none" in c.get("/observe").text and c.get("/observe", headers={"host": "evil.com"}).status_code == 403
    bad = c.post("/api/openai", json={"key": "x" * 5})
    assert bad.status_code == 400 and "xxxxx" not in bad.text and c.get("/api/openai").json()["on"] is False
    assert c.post("/api/openai", json={"model": "gpt-6.1-terra"}).json()["model"] == "gpt-6.1-terra" and c.post("/api/openai", json={"model": "아무거나"}).status_code == 400
    assert c.post("/api/openai", json={"off": True}, headers={"origin": "https://evil.com"}).status_code == 403
    old = STEPS[2]
    STEPS[2] = (old[0], lambda c_, x: (_ for _ in ()).throw(PermissionError("시험")))
    try:
        j = c.post("/api/concept", json={"name": "청귤 제로", "desc": "청귤 향 탄산음료, 당 0g", "price": "1,800원", "target": "20대", "photo": "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 40).decode()}).json()
        hid, j = j["id"], j["job"]
        for _ in range(100):
            z = c.get("/api/job/" + j).json()
            if z["state"] != "running":
                break
            time.sleep(.2)
    finally:
        STEPS[2] = old
    assert z["state"] == "done" and [x["state"] for x in z["steps"]] == ["done", "done", "skip", "done", "done"], z
    h = c.get("/api/history").json()["rows"]
    assert h[0]["id"] == hid and h[0]["state"] == "완료" and len(h[0]["steps"]) == 5 and h[0]["photo"] and sum(x["name"] in W.IDS.values() and x["photo"] and not x["result"] for x in h) == 3
    assert c.get("/api/photo/" + hid).headers["content-type"] == "image/png" and c.get("/api/photo/x").status_code == 404 and c.get("/api/photo/" + hid, headers={"host": "evil.com"}).status_code == 403
    assert c.post("/api/concept", json={"name": "a", "desc": "b", "photo": "data:image/png;base64," + base64.b64encode(b"<script>").decode()}).status_code == 400
    try:
        s_talk({"name": "밀키스 화채", "desc": "", "price": ""}, {})
        raise AssertionError("대회 컨셉이 외부로 나갈 수 있음")
    except PermissionError:
        pass
    assert c.post("/api/review", json={"memo": "시험"}).json()["reviews"][0][1] == "시험"
    print("server ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        import uvicorn
        from . import consumer      # .env 읽기
        threading.Thread(target=auto_connect, daemon=True).start()
        threading.Thread(target=base, daemon=True).start()       # 첫 화면을 빨리 열기 위해 미리 계산
        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8517, log_level="warning")
