"""지속 관측: 기존 수집기 → Decode → 로컬 Laya → BERTopic. 네트워크는 워커에서만 실행."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import datetime as dt
import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time

import pandas as pd

from . import decode as D, formats as F, media as M, food_scope as FS, trend_social as Social

CAMPAIGN = "internet-live"
ROOT = Path(M.ROOT).resolve()


def init():
    with D.db() as c:
        c.executescript("""
        create table if not exists trend_control(id integer primary key check(id=1),enabled integer default 0,interval integer default 600,requested integer default 0);
        insert or ignore into trend_control(id) values(1);
        create table if not exists trend_worker(id integer primary key check(id=1),pid integer,heartbeat real,stage text,status text,next_run real,last_cycle real,error text);
        insert or ignore into trend_worker(id,status,stage) values(1,'stopped','대기');
        create table if not exists trend_sources(id text primary key,label text,status text,started real,finished real,last_success real,next_due real,n integer default 0,new integer default 0,error text);
        create table if not exists trend_events(id integer primary key,ts real,kind text,source text,message text,n integer);
        """)
        cfg = dict(brand="", sources=list(D.source_options()), keywords=[], formats=F.formats(), scope="food")
        c.execute("insert or ignore into campaigns values(?,?,?)", (CAMPAIGN, "식음료 트렌드", json.dumps(cfg, ensure_ascii=False)))
    return cfg


def config():
    init()
    with D.db() as c:
        return json.loads(c.execute("select config from campaigns where id=?", (CAMPAIGN,)).fetchone()[0])


def event(kind, source, message, n=0):
    with D.db() as c:
        c.execute("insert into trend_events(ts,kind,source,message,n) values(?,?,?,?,?)", (time.time(), kind, source, message, n))
        c.execute("delete from trend_events where id < (select coalesce(max(id),0)-2000 from trend_events)")


def stage(message, status="running"):
    with D.db() as c:
        c.execute("update trend_worker set stage=?,status=?,heartbeat=? where id=1", (message, status, time.time()))


def sync(sources=None):
    cfg = config()
    # 소스 설정이 늘어났어도 전체 관측 캠페인은 새 소스를 자동으로 포함한다.
    cfg["sources"] = list(D.source_options())
    cfg["brand"] = ""
    cfg["scope"] = "food"
    with D.db() as c:
        c.execute("update campaigns set config=?,name=? where id=?", (json.dumps(cfg, ensure_ascii=False), "식음료 트렌드", CAMPAIGN))
    result = D.collect(CAMPAIGN, cfg if sources is None else cfg | {"sources": sources})
    with D.db() as c:
        candidates = [r["id"] for r in (json.loads(v[0]) for v in c.execute("select payload from videos")) if FS.candidate(r)]
        c.execute("delete from members where campaign=?", (CAMPAIGN,))
        c.executemany("insert or ignore into members values(?,?)", ((CAMPAIGN, vid) for vid in candidates))
    return result


@lru_cache(maxsize=1)
def classifier():
    import laya
    import torch
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return laya.load(M.MODEL, device=device, vegaml=False)


def analyze(stopped=lambda: False):
    cfg = config()
    if any(r.get("video_url") and not r.get("words_done") for r in D.records(CAMPAIGN)):
        stage("Decode · 대본·화면 자막 추출")
        done, errors = D.words(CAMPAIGN, stage, available_only=True, stopped=stopped)
        event("decoded", "Decode", f"대본·자막 {done}건 · 미완료 {len(errors)}건", done)
    if stopped():
        return
    stage("Laya 준비 · 로컬 모델")
    n = D.classify(CAMPAIGN, cfg, stage, agent=classifier(), batch_size=16, stopped=stopped)
    event("classified", "Laya", f"콘텐츠 {n:,}건 분류 완료", n)
    if stopped():
        return
    from . import trend_topics
    stage("BERTopic · 주제 탐색")
    result = trend_topics.update(CAMPAIGN, tick=stage)
    event("topics", "BERTopic", result)
    if not stopped():
        from . import trend_insights
        result = trend_insights.update(CAMPAIGN, tick=stage, stopped=stopped)
        event("interpreted", "AI 트렌드 해석", result)


def source_jobs():
    jobs = {"format:" + n: (n, True, "", 600) for n in F.PAGES}
    jobs |= {"media:" + n: (n, on, why, 21600 if "유튜브" in n else 600)
             for n, (_, _, on, why) in M.sources().items() if n not in F.PAGES and "유튜브" not in n and n != "Hacker News"}
    for platform, label in Social.PLATFORMS.items():
        on, _, reason = Social.connection(platform)
        jobs["social:" + platform] = (label, on, reason, 3600)
    return jobs


def fetch_one(key, label, stopped=lambda: False):
    if stopped():
        return 0, 0
    started = time.time()
    with D.db() as c:
        c.execute("update trend_sources set status='running',started=?,error=null where id=?", (started, key))
    event("fetching", label, "새 글과 지표를 확인하고 있습니다")
    n, new, error = 0, 0, ""
    try:
        if key.startswith("social:"):
            platform = key.split(":", 1)[1]
            rows, issues = Social.collect(platform, tick=lambda message: event("discovering", label, message))
            before = {r["id"] for r in D.records(CAMPAIGN)}
            n, invalid = D.save_rows(CAMPAIGN, rows, platform)
            new = sum(r["id"] not in before and r["platform"] == platform for r in D.records(CAMPAIGN))
            error = " · ".join([*issues, *invalid])[:180]
        elif key.startswith("format:"):
            c = F.db()
            before = c.execute("select count(*) from content where grp=?", (label,)).fetchone()[0]
            c.close()
            report = F.community(pages=3, names=[label], predicate=FS.candidate)[label]
            n = int(re.match(r"\d+", report)[0])
            c = F.db()
            new = c.execute("select count(*) from content where grp=?", (label,)).fetchone()[0] - before
            c.close()
            if "중단:" in report:
                error = report.split("중단:", 1)[1].strip()
        else:
            kind, fn, _, _ = M.sources()[label]
            rows = [r for r in fn() if FS.candidate(r)]
            c = M.db()
            try:
                n, new = M.save(c, label, kind, rows)
                c.execute("insert into runs values(?,?,?,?,?,?)", (M.now(), label, 1, n, new, ""))
                c.commit()
            finally:
                c.close()
        if n and not key.startswith("social:"):
            sync([key])
    except Exception as exc:
        # URL/토큰이 들어갈 수 있는 예외 전문은 상태 API에 노출하지 않는다.
        error = str(exc)[:140] if isinstance(exc, PermissionError) else type(exc).__name__
    status = "partial" if n and error else "error" if error else "success" if n else "empty"
    delay = source_jobs()[key][3]
    with D.db() as c:
        c.execute("update trend_sources set status=?,finished=?,last_success=case when ? then ? else last_success end,next_due=?,n=?,new=?,error=? where id=?",
                  (status, time.time(), bool(n) or not error, time.time(), time.time() + delay, n, new, error, key))
    event(status, label, f"{n:,}건 관측 · 새 콘텐츠 {new:,}건" + (" · " + error if error else ""), n)
    return n, new


def cycle(stopped=lambda: False):
    init()
    stage("기존 콘텐츠 연결")
    sync()
    now = time.time()
    jobs = source_jobs()
    with D.db() as c:
        c.execute("delete from trend_sources where id not in (" + ",".join("?" for _ in jobs) + ")", tuple(jobs))
        for key, (label, enabled, why, _) in jobs.items():
            c.execute("insert or ignore into trend_sources(id,label,status,error) values(?,?,?,?)",
                      (key, label, "waiting" if enabled else "disabled", "" if enabled else why))
            if not enabled:
                c.execute("update trend_sources set status='disabled',error=? where id=?", (why, key))
        due = [key for key, (_, enabled, _, _) in jobs.items() if enabled and
               (c.execute("select coalesce(next_due,0) from trend_sources where id=?", (key,)).fetchone()[0] <= now)]
    event("cycle", "수집 워커", f"이번 관측 · {len(due)}개 소스")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, key, jobs[key][0], stopped): key for key in due if not stopped()}
        done = 0
        for f in as_completed(futures):
            f.result()
            done += 1
            stage(f"수집 · {done}/{len(due)}개 소스 완료")
    if not stopped():
        sync()
        analyze(stopped)
    with D.db() as c:
        c.execute("update trend_worker set last_cycle=? where id=1", (time.time(),))
    event("paused" if stopped() else "cycle_done", "수집 워커", "진행 중인 배치 저장 후 중지" if stopped() else "관측·분류·주제 탐색 주기 완료")


def control():
    init()
    with D.db() as c:
        return dict(c.execute("select * from trend_control where id=1").fetchone())


def start():
    init()
    with D.db() as c:
        c.execute("update trend_control set enabled=1 where id=1")
        worker = dict(c.execute("select * from trend_worker where id=1").fetchone())
    if time.time() - (worker["heartbeat"] or 0) < 30 and worker["status"] != "stopped":
        return
    log = D.DB.parent / "trend-worker.log"
    with log.open("ab") as out:
        subprocess.Popen([sys.executable, "-u", "-m", "mirror.trends", "watch"], cwd=ROOT, stdout=out, stderr=out, start_new_session=True)


def stop():
    init()
    with D.db() as c:
        c.execute("update trend_control set enabled=0 where id=1")
    event("stop", "수집 워커", "중지 요청 · 진행 중인 저장을 마친 뒤 멈춥니다")


def request_cycle():
    init()
    with D.db() as c:
        c.execute("update trend_control set requested=1 where id=1")
    start()


def watch(once=False, collect_sources=True):
    init()
    lock = (D.DB.parent / "trend-worker.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return
    quit_event = threading.Event()
    def heartbeat():
        while not quit_event.wait(5):
            with D.db() as c:
                c.execute("update trend_worker set heartbeat=? where id=1", (time.time(),))
    with D.db() as c:
        c.execute("update trend_worker set pid=?,heartbeat=?,status='running',error=null where id=1", (os.getpid(), time.time()))
        c.execute("update trend_sources set status='error',error='이전 작업이 중단되었습니다' where status='running'")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        while once or control()["enabled"]:
            try:
                with D.db() as c:
                    c.execute("update trend_worker set error=null where id=1")
                if collect_sources:
                    cycle(stopped=lambda: not once and not control()["enabled"])
                else:
                    sync()
                    analyze()
            except Exception as exc:
                message = f"{type(exc).__name__}: {str(exc)[:180]}" if isinstance(exc, (ImportError, ValueError, RuntimeError)) else type(exc).__name__
                event("error", "분석 파이프라인", message)
                with D.db() as c:
                    c.execute("update trend_worker set error=? where id=1", (message,))
            if once:
                break
            next_run = time.time() + control()["interval"]
            with D.db() as c:
                c.execute("update trend_worker set next_run=?,stage='다음 관측 대기',status='waiting' where id=1", (next_run,))
            while control()["enabled"] and time.time() < next_run and not control()["requested"]:
                time.sleep(2)
            with D.db() as c:
                c.execute("update trend_control set requested=0 where id=1")
    finally:
        quit_event.set()
        thread.join(timeout=6)
        stage("관측 중지", "stopped")
        lock.close()


def snapshot(hours=24):
    from . import trend_discovery, trend_insights
    cfg = config()
    d = D.load(CAMPAIGN, cfg)
    raw = d
    pending = int(d.food_category.isna().sum()) if not d.empty else 0
    accepted = d.apply(FS.accepted, axis=1) if not d.empty else None
    excluded = int((d.food_category.notna() & ~accepted).sum()) if not d.empty else 0
    if not d.empty:
        d = d[accepted].copy()
        d["brand_group"] = [FS.brand_group(r) for r in d.to_dict("records")]
    with D.db() as c:
        worker = dict(c.execute("select * from trend_worker where id=1").fetchone())
        ctl = dict(c.execute("select * from trend_control where id=1").fetchone())
        sources = [dict(r) for r in c.execute("select * from trend_sources order by label")]
        events = [dict(r) for r in c.execute("select * from trend_events order by id desc limit 30")]
        try:
            topics = [dict(r) for r in c.execute("select * from trend_topics")]
            members = {r["video"]: dict(r) for r in c.execute("select * from trend_topic_members")}
            topic_run = dict(c.execute("select * from trend_topic_runs order by ts desc limit 1").fetchone() or {})
        except sqlite3.OperationalError:
            topics, members, topic_run = [], {}, {}
        observations = c.execute("select count(*) from snapshots s join members m on s.video=m.video where m.campaign=?", (CAMPAIGN,)).fetchone()[0]
    now = time.time()
    alive = now - (worker["heartbeat"] or 0) < 30 and worker["status"] != "stopped"
    result = dict(now=now, worker=worker, enabled=bool(ctl["enabled"]), alive=alive, sources=sources, events=events,
                  topics=[], topic_run=topic_run, items=[], total=len(d), observations=observations, hours=hours,
                  pending=pending, excluded=excluded, scope="food")
    result["discovery"] = trend_discovery.snapshot()
    result["insights"] = dict(groups=[], tagged=0, total=0, run={})
    result["channels"] = []
    for platform, label in Social.PLATFORMS.items():
        on, mode, reason = Social.connection(platform)
        status = next((r for r in sources if r["id"] == "social:" + platform), {})
        group = d[(d.platform == platform) & (pd.to_datetime(d.observed, utc=True, format="mixed", errors="coerce") >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours))] if not d.empty else d
        result["channels"].append(dict(platform=platform, label=label, mode=mode, count=len(group),
            measured=int(group.views.notna().sum()) if not group.empty else 0,
            reactions=int(group.likes.notna().sum()) if not group.empty else 0,
            pending=int(((raw.platform == platform) & raw.food_category.isna()).sum()) if not raw.empty else 0,
            creators=int(group.creator.replace("", None).dropna().nunique()) if not group.empty else 0,
            status=status.get("status", "waiting" if on else "disabled"), error=status.get("error") or reason,
            last_success=status.get("last_success"), docs=Social.DOCS[platform]))
    if d.empty:
        return result
    for col in ["first_seen", "observed"]:
        if col not in d:
            d[col] = None
    timestamps = pd.to_datetime(d.first_seen, utc=True, format="mixed", errors="coerce")
    observed = pd.to_datetime(d.observed, utc=True, format="mixed", errors="coerce")
    result["new_hour"] = int((timestamps >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).sum())
    d = d[observed >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)].copy()
    # 신규 발견 시각과 게시 시각은 구분하며, 재관측된 오래된 글도 화면에서 찾을 수 있다.
    d["age_hours"] = ((pd.Timestamp.now(tz="UTC") - pd.to_datetime(d.published, utc=True, format="mixed", errors="coerce")).dt.total_seconds() / 3600).clip(lower=0)
    members = {r["id"]: members[r["id"]] for r in d.to_dict("records") if r["id"] in members and members[r["id"]].get("input_hash") == D.digest(D.state(r))}
    d["topic"] = d.id.map(lambda x: members.get(x, {}).get("topic"))
    d["topic_label"] = d.id.map(lambda x: members.get(x, {}).get("label", "주제 탐색 대기"))
    d["trend_metric"] = d.view_metric.where(d.views.notna(), "likes")
    d["trend_rate"] = d.rate.where(d.views.notna(), d.reaction_rate)
    d["growth_rank"] = d.groupby(["platform", "source", "trend_metric"]).trend_rate.rank(pct=True)
    # 비교군이 5개 미만이면 서로 다른 게시판의 원시 증가량을 억지로 순위화하지 않는다.
    enough = d.groupby(["platform", "source", "trend_metric"]).trend_rate.transform("count") >= 5
    d.loc[~enough, "growth_rank"] = None
    result["rising"] = int((d.trend_rate.gt(0) & d.growth_rank.ge(.8)).sum())
    result["classified"] = int(d.fmt.notna().sum())
    result["visible"] = len(d)
    records = json.loads(d.sort_values(["growth_rank", "observed"], ascending=False, na_position="last").to_json(orient="records", force_ascii=False))
    result["items"] = [{k: v for k, v in row.items() if k != "video_url"} for row in records]
    result["insights"] = trend_insights.snapshot(result["items"], now)
    for topic in topics:
        group = d[d.topic == topic["id"]]
        if group.empty:
            continue
        seen = pd.to_datetime(group.first_seen, utc=True, format="mixed", errors="coerce")
        edge = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)
        recent = int((seen >= edge).sum())
        prior = int(((seen < edge) & (seen >= edge - pd.Timedelta(hours=1))).sum())
        topic.pop("centroid", None)
        topic.update(count=len(group), sources=int(group.source.nunique()), recent=recent, prior=prior,
                     change=recent - prior, timeline=[dict(source=str(source), first=str(g.first_seen.min()), count=len(g)) for source, g in group.groupby("source")])
        result["topics"].append(topic)
    result["topics"].sort(key=lambda t: (t["recent"], t["sources"], t["count"]), reverse=True)
    return result


def html(payload):
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return Path(__file__).with_name("trend_component").joinpath("index.html").read_text().replace("__TREND_DATA__", raw)


if __name__ == "__main__":
    from . import consumer as _C  # 기존 .env 로딩. 새 외부 분석 API는 사용하지 않는다.
    command = sys.argv[1] if len(sys.argv) > 1 else "once"
    if command == "stop":
        stop()
    else:
        if command == "watch":
            init()
            with D.db() as c:
                c.execute("update trend_control set enabled=1 where id=1")
        watch(once=command != "watch")
