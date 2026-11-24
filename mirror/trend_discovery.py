"""키워드에서 새 작성자를 발견하고, 다음 탐색어를 누적하는 공개 검색 큐."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import json
import re
import time
import urllib.parse

from . import decode as D, food_scope as FS, trend_public as P

SEEDS = ["한정선", "디저트", "말차", "편의점 신상", "하이볼", "제로 음료", "과일 모찌", "떡볶이",
         "#먹스타그램", "#디저트", "#홈카페", "#하이볼", "신상 과자", "찹쌀떡", "베이커리", "아이스크림",
         "라면", "치킨", "김밥", "가성비 맛집", "두바이 초콜릿", "요거트", "단백질 간식", "밀키트",
         "커피", "카페 신메뉴", "탄산음료", "주스", "소주", "맥주", "위스키", "와인"]
PLATFORMS = ("instagram", "tiktok", "threads")
BATCH = 8


def init():
    with D.db() as c:
        c.executescript("""
        create table if not exists trend_queries(platform text,query text,origin text,added real,last_run real,status text,n integer default 0,verified integer default 0,error text,primary key(platform,query));
        create table if not exists trend_discoveries(platform text,external_id text,query text,url text,creator text,first_seen real,last_seen real,primary key(platform,external_id,query));
        """)
        for platform in PLATFORMS:
            c.executemany("insert or ignore into trend_queries(platform,query,origin,added,status) values(?,?,'초기 식음료 탐색어',?,'waiting')",
                          ((platform, query, i) for i, query in enumerate(SEEDS)))


def learn(terms, origin, platforms=PLATFORMS):
    init()
    terms = list(dict.fromkeys(" ".join(str(t).split()) for t in terms if isinstance(t, str)))
    terms = [t for t in terms if re.fullmatch(r"#?[\w가-힣][\w가-힣 .·-]{1,34}", t) and not t.isdigit()][:12]
    with D.db() as c:
        for platform in platforms:
            room = max(0, 200 - c.execute("select count(*) from trend_queries where platform=?", (platform,)).fetchone()[0])
            c.executemany("insert or ignore into trend_queries(platform,query,origin,added,status) values(?,?,?,?, 'waiting')",
                          ((platform, t, origin, time.time()) for t in terms[:room]))


def due(platform):
    init()
    with D.db() as c:
        rows = [dict(r) for r in c.execute("select * from trend_queries where platform=? and coalesce(last_run,0)<? order by coalesce(last_run,0),added", (platform, time.time() - 6 * 3600))]
    learned = [r for r in rows if r["origin"] != "초기 식음료 탐색어"][:2]
    seed = [r for r in rows if r["origin"] == "초기 식음료 탐색어"]
    selected = seed[:BATCH-len(learned)] + learned
    return (selected + [r for r in rows if r not in selected])[:BATCH]


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and dict(attrs).get("href"):
            self.urls.append(dict(attrs)["href"])


def tiktok_links(html):
    from .trend_social import canonical
    parser = Links(); parser.feed(html)
    found = {}
    for url in parser.urls:
        try:
            platform, url = canonical(url)
        except ValueError:
            continue
        if platform == "tiktok":
            key = url.rstrip("/").rsplit("/", 1)[-1]
            found[key] = dict(id=key, url=url, creator=urllib.parse.urlsplit(url).path.split("/")[1][1:])
    return list(found.values())


def search(platform, query):
    if platform == "instagram":
        slug = urllib.parse.quote(query.lstrip("#").replace(" ", "-"), safe="")
        url = "https://www.instagram.com/popular/" + slug + "/"
        html = P.get(url)
        related = [edge.get("node", {}).get("query_text") for obj in P.objects(P.scripts(html))
                   if isinstance(obj.get("popular_search_related_keywords_connection"), dict)
                   for edge in obj["popular_search_related_keywords_connection"].get("edges", [])]
        learn([q for q in related if isinstance(q, str) and FS.candidate({"caption": q})], "공개 인기 검색 연관어")
        return P.instagram(html, search=True), url
    if platform == "threads":
        url = "https://www.threads.com/search?" + urllib.parse.urlencode({"q": query})
        return P.threads(P.get(url), search=True)[:20], url
    url = "https://search.naver.com/search.naver?" + urllib.parse.urlencode({"query": "site:tiktok.com " + query})
    return tiktok_links(P.get(url)), url


def collect(platform, tick=lambda message: None):
    from .trend_social import public_post, PLATFORMS as LABELS
    rows, errors = {}, []
    queries = due(platform)
    for index, job in enumerate(queries):
        query, issues = job["query"], []
        tick(f"키워드 탐색 {index+1}/{len(queries)} · {query}")
        with D.db() as c:
            c.execute("update trend_queries set status='running',error=null where platform=? and query=?", (platform, query))
        candidates, verified = [], {}
        try:
            candidates, discovery_url = search(platform, query)
            if platform == "threads":
                verified = {r["id"]: r for r in candidates}
            else:
                # 새 작성자를 우선해 같은 계정의 반복 노출이 표본을 독점하지 않게 한다.
                selected, authors = [], {}
                ordered = sorted(candidates, key=lambda r: int(r["id"]), reverse=True) if platform == "tiktok" else candidates
                for row in ordered:
                    if authors.get(row.get("creator"), 0) >= 2 or len(selected) >= 12:
                        continue
                    authors[row.get("creator")] = authors.get(row.get("creator"), 0) + 1
                    if row["id"] in rows:
                        verified[row["id"]] = rows[row["id"]]
                    else:
                        selected.append(row)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pending = {pool.submit(public_post, row["url"]): row for row in selected}
                    for future in as_completed(pending):
                        original = pending[future]
                        try:
                            full = future.result()
                            if platform == "instagram" and original.get("views") is not None:
                                full.update(views=original["views"], view_metric=original["view_metric"])
                            verified[full["id"]] = full
                        except Exception as exc:
                            issues.append(str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__)
                            # 인기 검색 본문의 실제 재생수는 살리고, 미확인 상세 지표는 비워 둔다.
                            if platform == "instagram" and original.get("caption"):
                                verified[original["id"]] = original | {"collection_method": "공개 인기 키워드 미리보기"}
            now = time.time()
            with D.db() as c:
                for row in candidates:
                    c.execute("insert into trend_discoveries values(?,?,?,?,?,?,?) on conflict(platform,external_id,query) do update set last_seen=excluded.last_seen,creator=excluded.creator",
                              (platform, row["id"], query, row["url"], row.get("creator"), now, now))
            for key, row in verified.items():
                row = row | dict(source=LABELS[platform], discovery_query=query, discovery_food=True,
                                 collection_method=(row.get("collection_method") if row.get("collection_method") == "공개 인기 키워드 미리보기" else "공개 인기 키워드 + 원문") if platform == "instagram" else "공개 검색 → TikTok 원문" if platform == "tiktok" else "공개 키워드·해시태그 검색",
                                 evidence_url=discovery_url)
                if FS.candidate(row):
                    rows[key] = row
        except Exception as exc:
            issues.append(str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__)
        error = " · ".join(dict.fromkeys(issues))[:180]
        status = "partial" if verified and error else "error" if error else "success" if verified else "empty"
        with D.db() as c:
            c.execute("update trend_queries set last_run=?,status=?,n=?,verified=?,error=? where platform=? and query=?",
                      (time.time(), status, len(candidates), len(verified), error, platform, query))
        if error:
            errors.append(query + ": " + error)
        tick(f"‘{query}’ · 발견 {len(candidates)} / 공개 본문 {len(verified)} · 누적 {len(rows)}건")
    return list(rows.values()), errors


def snapshot():
    init()
    with D.db() as c:
        jobs = [dict(r) for r in c.execute("select * from trend_queries order by coalesce(last_run,0) desc,added")]
        totals = [dict(r) for r in c.execute("select platform,count(distinct external_id) posts,count(distinct creator) creators,count(distinct query) queries from trend_discoveries group by platform")]
    return dict(queries=jobs, totals=totals, batch=BATCH, refresh_hours=6)
