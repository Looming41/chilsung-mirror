"""필수 4채널의 익명 공개 수집과 선택적 공식 API, 지정 링크 관측."""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import decode as D, food_scope as FS, trend_public as Public

PLATFORMS = {"youtube": "YouTube", "instagram": "Instagram", "tiktok": "TikTok", "threads": "Threads"}
DOCS = {
    "youtube": "https://developers.google.com/youtube/v3/getting-started",
    "instagram": "https://developers.facebook.com/documentation/instagram-platform/instagram-api-with-facebook-login/hashtag-search",
    "threads": "https://developers.facebook.com/documentation/threads/keyword-search",
    "tiktok": "https://developers.tiktok.com/docs/en/display-api-overview",
}


def init():
    with D.db() as c:
        c.execute("create table if not exists trend_social_urls(platform text,url text primary key,added real,checked real,status text,error text)")


def canonical(url):
    if not D.public_url(url):
        raise ValueError("공개 HTTPS 게시물 주소를 입력하세요.")
    u = urllib.parse.urlsplit(url)
    host, path = u.hostname.lower().removeprefix("www."), u.path
    if host in {"youtube.com", "m.youtube.com", "youtu.be"}:
        key = path.strip("/") if host == "youtu.be" else urllib.parse.parse_qs(u.query).get("v", [""])[0] if path == "/watch" else path.rsplit("/", 1)[-1] if path.startswith("/shorts/") else ""
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", key):
            return "youtube", "https://www.youtube.com/watch?v=" + key
    rules = {"instagram.com": ("instagram", r"/(?:[A-Za-z0-9_.]+/)?(?:p|reel)/[A-Za-z0-9_-]+/?"),
             "tiktok.com": ("tiktok", r"/@[A-Za-z0-9_.-]+/video/\d+/?"),
             "threads.com": ("threads", r"/@[A-Za-z0-9_.-]*/post/[A-Za-z0-9_-]+/?"),
             "threads.net": ("threads", r"/@[A-Za-z0-9_.-]*/post/[A-Za-z0-9_-]+/?")}
    if host in rules and re.fullmatch(rules[host][1], path):
        platform = rules[host][0]
        if platform == "instagram":
            path = "/" + "/".join(path.strip("/").split("/")[-2:])
        return platform, "https://www." + ("threads.com" if platform == "threads" else host) + path.rstrip("/") + "/"
    raise ValueError("YouTube 영상, Instagram 게시물/릴스, TikTok 영상, Threads 게시물의 전체 주소가 필요합니다.")


def add_urls(lines):
    urls = list(dict.fromkeys(x.strip() for x in lines if x.strip()))
    if len(urls) > 20:
        raise ValueError("한 번에 최대 20개 주소를 등록할 수 있습니다.")
    checked = [canonical(u) for u in urls]
    init()
    with D.db() as c:
        for platform, url in checked:
            c.execute("insert or ignore into trend_social_urls(platform,url,added,status) values(?,?,?,'waiting')", (platform, url, time.time()))
    return len(checked)


def links(platform):
    init()
    with D.db() as c:
        return [dict(r) for r in c.execute("select * from trend_social_urls where platform=? order by coalesce(checked,0),added limit 5", (platform,))]


def connection(platform):
    from .consumer import _dotenv
    if platform == "youtube":
        return True, "공식 API 검색" if os.environ.get("YOUTUBE_API_KEY") else "공개 검색 · 관련도순", ""
    ready = bool(os.environ.get({"instagram":"INSTAGRAM_ACCESS_TOKEN", "threads":"THREADS_ACCESS_TOKEN", "tiktok":"TIKTOK_ACCESS_TOKEN"}[platform]))
    if platform == "instagram":
        ready = ready and bool(os.environ.get("INSTAGRAM_USER_ID"))
    mode = {"instagram": "공개 인기 키워드 탐색", "tiktok": "키워드 검색 → 공개 영상", "threads": "키워드·해시태그 탐색"}[platform]
    return True, mode + (" + 공식 API" if ready else " · 계정 불필요"), ""


def api(url, token=None, payload=None):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    headers = {"User-Agent": D.M.UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=20) as r:
            result = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"공식 API HTTP {e.code} · 토큰·승인 권한을 확인하세요") from None
    if result.get("error") and (not isinstance(result["error"], dict) or result["error"].get("code") != "ok"):
        raise RuntimeError("공식 API가 요청을 거절했습니다. 토큰·권한·사용량을 확인하세요.")
    return result


def graph(path, **query):
    version = os.environ.get("META_GRAPH_VERSION", "")
    if version and not re.fullmatch(r"v\d+\.\d+", version):
        raise ValueError("META_GRAPH_VERSION 형식을 확인하세요.")
    return api("https://graph.facebook.com/" + (version + "/" if version else "") + path + "?" + urllib.parse.urlencode(query), os.environ["INSTAGRAM_ACCESS_TOKEN"])


def instagram():
    user = os.environ["INSTAGRAM_USER_ID"]
    fields = "id,caption,permalink,timestamp,media_type,like_count,comments_count,media_url"
    for tag in FS.HASHTAGS:  # 고정 8개 태그: 7일 30개 제한 안에서 반복 관측한다.
        found = graph("ig_hashtag_search", user_id=user, q=tag).get("data", [])
        for hashtag in found[:1]:
            for edge in ("top_media", "recent_media"):
                for r in graph(str(hashtag["id"]) + "/" + edge, user_id=user, fields=fields, limit=25).get("data", []):
                    yield dict(id=r["id"], caption=r.get("caption"), url=r.get("permalink"), published=r.get("timestamp"),
                               likes=r.get("like_count"), comments=r.get("comments_count"),
                               thumbnail=r.get("media_url") if r.get("media_type") == "IMAGE" else "", source="Instagram")


def threads():
    fields = "id,text,permalink,timestamp,username,media_type,thumbnail_url"
    for query in ["제로음료", "편의점 신상", "맛집", "하이볼"]:
        params = urllib.parse.urlencode(dict(q=query, search_type="RECENT", fields=fields, limit=25))
        for r in api("https://graph.threads.com/v1.0/keyword_search?" + params, os.environ["THREADS_ACCESS_TOKEN"]).get("data", []):
            yield dict(id=r["id"], caption=r.get("text"), url=r.get("permalink"), published=r.get("timestamp"),
                       creator=r.get("username"), thumbnail=r.get("thumbnail_url"), source="Threads")


def tiktok():
    fields = "id,title,video_description,share_url,create_time,view_count,like_count,comment_count,share_count,cover_image_url,duration"
    result = api("https://open.tiktokapis.com/v2/video/list/?fields=" + fields, os.environ["TIKTOK_ACCESS_TOKEN"], {"max_count":20})
    for r in result.get("data", {}).get("videos", []):
        yield dict(id=r["id"], caption=r.get("video_description") or r.get("title"), url=r.get("share_url"), published=r.get("create_time"),
                   views=r.get("view_count"), likes=r.get("like_count"), comments=r.get("comment_count"), shares=r.get("share_count"),
                   thumbnail=r.get("cover_image_url"), duration=r.get("duration"), source="TikTok")


def ytdlp(target, flat=False):
    import yt_dlp
    class Quiet:
        def debug(self, message): pass
        def warning(self, message): pass
        def error(self, message): pass
    options = dict(quiet=True, no_warnings=True, logger=Quiet(), skip_download=True, extract_flat=flat,
                   cachedir=False, usenetrc=False, cookiefile=None, cookiesfrombrowser=None,
                   socket_timeout=12, retries=0, extractor_retries=0, noplaylist=True)
    try:
        with yt_dlp.YoutubeDL(options) as y:
            return y.extract_info(target, download=False)
    except Exception:
        raise RuntimeError("공개 페이지에서 읽지 못했습니다 · 로그인·접근 제한 또는 일시 오류") from None


def video_row(r, platform):
    url = r.get("webpage_url") or r.get("url")
    if platform == "youtube":
        url = "https://www.youtube.com/watch?v=" + r["id"]
    return dict(id=r["id"], caption=r.get("title") or r.get("description"), url=url,
                published=r.get("timestamp"), creator=r.get("channel") or r.get("uploader"),
                views=r.get("view_count"), likes=r.get("like_count"), comments=r.get("comment_count"), shares=r.get("share_count"), metrics={"reposts":r.get("repost_count")},
                duration=r.get("duration"), thumbnail=r.get("thumbnail") or next((x.get("url") for x in r.get("thumbnails", []) if x.get("url")), ""),
                source=PLATFORMS[platform])


def youtube():
    for query in ["편의점 신상 음식 쇼츠", "제로 음료 리뷰 쇼츠", "먹방 맛집", "하이볼 맥주 소주"]:
        if os.environ.get("YOUTUBE_API_KEY"):
            headers = {"X-Goog-Api-Key": os.environ["YOUTUBE_API_KEY"]}
            params = urllib.parse.urlencode(dict(part="snippet", q=query, type="video", maxResults=8, regionCode="KR", relevanceLanguage="ko", order="date"))
            search = json.loads(D.M._raw("https://www.googleapis.com/youtube/v3/search?" + params, 20, headers))
            ids = ",".join(x["id"]["videoId"] for x in search.get("items", []))
            if not ids:
                continue
            params = urllib.parse.urlencode(dict(part="snippet,statistics,contentDetails", id=ids))
            for r in json.loads(D.M._raw("https://www.googleapis.com/youtube/v3/videos?" + params,20,headers)).get("items", []):
                sn, stats = r["snippet"], r.get("statistics", {})
                yield dict(id=r["id"],caption=sn["title"],url="https://www.youtube.com/watch?v="+r["id"],source="YouTube",creator=sn.get("channelTitle"),
                           published=sn.get("publishedAt"),views=stats.get("viewCount"),likes=stats.get("likeCount"),comments=stats.get("commentCount"),
                           thumbnail=sn.get("thumbnails",{}).get("medium",{}).get("url"))
        else:
            entries = (ytdlp("ytsearch8:" + query, flat=True) or {}).get("entries", [])
            for index, r in enumerate(entries):
                if not r or not r.get("id"):
                    continue
                if index < 2:
                    try:
                        r = ytdlp("https://www.youtube.com/watch?v=" + r["id"])
                    except RuntimeError:
                        pass  # 검색에서 확인한 조회수는 보존한다. 숨겨진 지표는 채우지 않는다.
                yield video_row(r, "youtube")


def public_post(url):
    platform, url = canonical(url)
    if platform == "youtube":
        r = video_row(ytdlp(url), platform)
        r["url"] = url
        return r
    code = url.rstrip("/").rsplit("/", 1)[-1]
    row = getattr(Public, platform)(Public.get(url), code)[0]
    return row | dict(source=PLATFORMS[platform], collection_method="공개 원문 JSON", evidence_url=url)


def public_collect(platform, tick=lambda message: None):
    from . import trend_discovery as Discovery
    discovered, errors = Discovery.collect(platform, tick)
    rows = {r["id"]: r for r in discovered}
    # 최근 발견한 자료 중 오래 관측한 원문을 반복 확인해 실제 증가량을 만든다.
    with D.db() as c:
        prior = [json.loads(r[0]) for r in c.execute("select payload from videos where julianday(updated)>=julianday('now','-7 days') and json_extract(payload,'$.platform')=? and json_extract(payload,'$.collection_method') is not null order by updated", (platform,))]
    refresh = [r for r in prior if r["external_id"] not in rows and r.get("url")][:20]
    for old in refresh:
        try:
            row = public_post(old["url"]) | {"discovery_food":old.get("discovery_food",False),"discovery_query":old.get("discovery_query","")}
            if FS.candidate(row):
                rows[row["id"]] = row | {"collection_method":"공개 원문 재관측"}
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    if refresh:
        tick(f"기존 원문 {len(refresh)}건 재관측 · 전체 {len(rows)}건")
    return list(rows.values()), list(dict.fromkeys(errors))


def identity(row, platform):
    if platform in {"instagram", "threads"} and row.get("url"):
        found, url = canonical(row["url"])
        if found != platform:
            raise ValueError("게시물의 플랫폼이 일치하지 않습니다.")
        row = row | {"id": url.rstrip("/").rsplit("/", 1)[-1], "url": url}
    return row


def collect(platform, tick=lambda message: None):
    rows, errors = {}, []
    enabled, mode, reason = connection(platform)
    if not enabled:
        return [], [reason]
    token_name = {"instagram":"INSTAGRAM_ACCESS_TOKEN", "threads":"THREADS_ACCESS_TOKEN", "tiktok":"TIKTOK_ACCESS_TOKEN"}.get(platform)
    if platform == "youtube" or (token_name and os.environ.get(token_name)):
        try:
            for row in globals()[platform]():
                if FS.candidate(row):
                    row = identity(row, platform)
                    rows[str(row["id"])] = row
        except Exception as exc:
            errors.append(str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__)
    if platform != "youtube":
        public_rows, issues = public_collect(platform, tick)
        errors.extend(issues)
        for row in public_rows:
            row = identity(row, platform)
            rows[str(row["id"])] = row
    for link in links(platform):
        error = ""
        try:
            row = public_post(link["url"])
            if FS.candidate(row):
                row = identity(row, platform)
                key = str(row["id"])
                # 원문에 없는 재생수는 같은 주기에 읽은 공개 릴스의 값만 보존한다.
                previous = rows.get(key, {})
                rows[key] = previous | {k: v for k, v in row.items() if v is not None}
                if row.get("views") is None and previous.get("views") is not None:
                    for field in ("views", "view_metric", "collection_method", "evidence_url"):
                        rows[key][field] = previous.get(field)
            else:
                error = "식음료 범위 밖"
        except Exception as exc:
            error = str(exc) if isinstance(exc, (RuntimeError, ValueError, PermissionError)) else type(exc).__name__
            errors.append(error)
        with D.db() as c:
            c.execute("update trend_social_urls set checked=?,status=?,error=? where url=?", (time.time(), "error" if error else "success", error, link["url"]))
    return list(rows.values()), errors
