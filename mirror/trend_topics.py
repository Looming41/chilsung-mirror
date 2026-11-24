"""BERTopic + 로컬 다국어 임베딩. 모델 API·사용자 브라우저에 의존하지 않는다."""
from functools import lru_cache
import json
import re
import time
import uuid

import numpy as np
import pandas as pd

from . import decode as D, food_scope as FS

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VERSION = D.digest([MODEL, "bertopic-0.17.4", "umap5-hdbscan-leaf", 3])
MAX_DOCUMENTS = 3000
STOP = ["그리고", "하지만", "하는", "있는", "없는", "이번", "오늘", "진짜", "그냥", "이런", "저런", "합니다", "있다", "했다", "the", "and", "for", "with", "this", "that", "from", "are", "you", "https", "www", "com", "jpg", "jpeg", "png", "gif", "mp4", "html", "기자", "뉴스", "종합", "통해", "위해", "관련", "대한", "밝혔다"]


def init():
    with D.db() as c:
        c.executescript("""
        create table if not exists trend_embeddings(video text primary key,input_hash text,model text,vector blob);
        create table if not exists trend_topics(id text primary key,label text,centroid text,n integer,updated real);
        create table if not exists trend_topic_members(video text primary key,topic text,label text,input_hash text);
        create table if not exists trend_topic_runs(ts real primary key,digest text,model text,n integer,topics integer,noise integer,status text);
        """)


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL, cache_dir=str(D.DB.parent / "models" / "topics"), threads=4)


def documents(cid):
    cfg = next(c["config"] for c in D.campaigns() if c["id"] == cid)
    rows = D.records(cid)
    if cfg.get("scope") == "food":
        frame = D.load(cid, cfg)
        rows = [r for r in frame.to_dict("records") if FS.accepted(r)]
    edge = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    rows = [r for r in rows if re.search(r"[^\W\d_]{2,}", " ".join(D.state(r).values())) and pd.to_datetime(r.get("observed"), utc=True, errors="coerce") >= edge]
    return sorted(rows, key=lambda r: r.get("observed") or "", reverse=True)[:MAX_DOCUMENTS]


def keyword_text(value):
    return re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", re.sub(r"https?://\S+", " ", value.lower()))


def cluster(texts, vectors):
    from bertopic import BERTopic
    from umap import UMAP
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    model = BERTopic(language="multilingual", embedding_model=None, umap_model=UMAP(n_neighbors=min(15, len(texts)-1), n_components=min(5, len(texts)-2), min_dist=0, metric="cosine", random_state=42, n_jobs=1, compatibility_layout=True),
        hdbscan_model=HDBSCAN(min_cluster_size=max(5, min(15, len(texts) // 50)), min_samples=3, cluster_selection_method="leaf", copy=True),
        vectorizer_model=CountVectorizer(preprocessor=keyword_text, token_pattern=r"(?u)\b[^\W\d_]{2,}\b", stop_words=STOP, max_features=12000),
        calculate_probabilities=False, verbose=False)
    labels, _ = model.fit_transform(texts, vectors)
    names = {k: " · ".join(word for word, score in model.get_topic(k)[:3] if score > 0) for k in set(labels) if k != -1}
    return np.asarray(labels), names


def stable_ids(centroids, previous):
    pairs = sorted(((float(np.dot(v, np.asarray(json.loads(p["centroid"])))), key, p["id"])
                    for key, v in centroids.items() for p in previous), reverse=True)
    assigned, used = {}, set()
    for similarity, key, old in pairs:
        if similarity >= .82 and key not in assigned and old not in used:
            assigned[key] = old
            used.add(old)
    return {key: assigned.get(key, uuid.uuid4().hex[:12]) for key in centroids}


def update(cid, tick=lambda s: None):
    init()
    rows = documents(cid)
    signature = D.digest([VERSION, sorted((r["id"], D.digest(D.state(r))) for r in rows)])
    with D.db() as c:
        last = c.execute("select digest from trend_topic_runs order by ts desc limit 1").fetchone()
        if last and last[0] == signature:
            return "콘텐츠 변경 없음 · 기존 주제 유지"
    if len(rows) < 10:
        return f"주제 탐색 대기 · 콘텐츠 {len(rows)}/10건"
    # shortcut: 모델은 앞 128토큰을 읽는다. 긴 영상의 세부 주제가 필요해지면 구간 임베딩으로 확장한다.
    texts = [" ".join(str(r.get(k) or "") for k in ("caption", "transcript", "on_screen_text"))[:6000] for r in rows]
    with D.db() as c:
        cached = {r["video"]: dict(r) for r in c.execute("select * from trend_embeddings where model=?", (MODEL,))}
    vectors = {}
    missing = []
    for i, r in enumerate(rows):
        old = cached.get(r["id"])
        if old and old["input_hash"] == D.digest(D.state(r)):
            vectors[i] = np.frombuffer(old["vector"], dtype=np.float32)
        else:
            missing.append(i)
    if missing:
        tick(f"BERTopic · 다국어 모델 준비 · 새 콘텐츠 {len(missing):,}건")
        model = embedder()
        for start in range(0, len(missing), 64):
            part = missing[start:start + 64]
            tick(f"BERTopic · 의미 분석 {start}/{len(missing)}")
            encoded = list(model.embed([texts[i] for i in part], batch_size=64))
            with D.db() as c:
                for i, vector in zip(part, encoded):
                    vector = np.asarray(vector, dtype=np.float32)
                    vector /= max(float(np.linalg.norm(vector)), 1e-12)
                    vectors[i] = vector
                    c.execute("insert or replace into trend_embeddings values(?,?,?,?)", (rows[i]["id"], D.digest(D.state(rows[i])), MODEL, vector.tobytes()))
    matrix = np.stack([vectors[i] for i in range(len(rows))])
    if not np.isfinite(matrix).all():
        raise ValueError("주제 임베딩에 유효하지 않은 값이 있습니다.")
    tick(f"BERTopic · {len(rows):,}건 주제 묶기")
    labels, names = cluster(texts, matrix)
    centroids = {int(k): matrix[labels == k].mean(axis=0) for k in names}
    centroids = {k: v / max(float(np.linalg.norm(v)), 1e-12) for k, v in centroids.items()}
    with D.db() as c:
        previous = [dict(r) for r in c.execute("select * from trend_topics")]
    ids = stable_ids(centroids, previous)
    # 성공한 결과만 한 트랜잭션으로 교체한다. 실패하면 마지막으로 읽을 수 있던 주제가 남는다.
    with D.db() as c:
        c.execute("delete from trend_topics")
        c.execute("delete from trend_topic_members")
        for key, vector in centroids.items():
            c.execute("insert into trend_topics values(?,?,?,?,?)", (ids[key], names[key], json.dumps(vector.tolist()), int((labels == key).sum()), time.time()))
        for row, key in zip(rows, labels):
            c.execute("insert into trend_topic_members values(?,?,?,?)", (row["id"], ids.get(int(key)), names.get(int(key), "기타·단독 주제"), D.digest(D.state(row))))
        c.execute("insert into trend_topic_runs values(?,?,?,?,?,?,?)", (time.time(), signature, MODEL, len(rows), len(names), int((labels == -1).sum()), "success"))
        c.execute("delete from trend_topic_runs where ts not in (select ts from trend_topic_runs order by ts desc limit 100)")
    return f"{len(names)}개 주제 · {len(rows):,}건 분석 · 단독 콘텐츠 {int((labels == -1).sum()):,}건"
