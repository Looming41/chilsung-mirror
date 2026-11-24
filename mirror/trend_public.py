"""로그인 없는 공개 페이지의 JSON만 읽는다. 쿠키·비공개 API·접근 제한 우회 없음."""
import json
import gzip
import io
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip",
    "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1",
}
HOSTS = {"instagram.com", "www.instagram.com", "tiktok.com", "www.tiktok.com", "threads.com", "www.threads.com", "threads.net", "www.threads.net", "search.naver.com"}


def get(url):
    def check(target):
        u = urllib.parse.urlsplit(target)
        if u.scheme != "https" or u.hostname not in HOSTS or u.username or u.password or u.port not in (None, 443):
            raise ValueError("지원하는 공개 소셜 페이지 주소가 아닙니다.")
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)
    check(url)
    try:
        with urllib.request.build_opener(Redirect()).open(urllib.request.Request(url, headers=HEADERS), timeout=35) as response:
            if "error=invalid_post" in response.url or "/accounts/login" in response.url:
                raise RuntimeError("공개 게시물 접근 불가 · 로그인 또는 삭제된 게시물")
            body = response.read(8_000_001)
            encoding = response.headers.get("Content-Encoding", "").lower()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"공개 페이지 HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("공개 페이지 응답 지연 또는 네트워크 오류") from None
    if len(body) > 8_000_000:
        raise RuntimeError("공개 페이지가 8MB 제한을 초과했습니다.")
    if encoding == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
            body = compressed.read(8_000_001)
        if len(body) > 8_000_000:
            raise RuntimeError("압축 해제한 공개 페이지가 8MB 제한을 초과했습니다.")
    return body.decode("utf-8", errors="replace")


class Scripts(HTMLParser):
    def __init__(self, script_id=None):
        super().__init__()
        self.script_id, self.active, self.blocks, self.parts = script_id, False, [], []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attrs = dict(attrs)
            self.active = attrs.get("id") == self.script_id if self.script_id else attrs.get("type") == "application/json"
            self.parts = []

    def handle_data(self, value):
        if self.active:
            self.parts.append(value)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            try:
                self.blocks.append(json.loads("".join(self.parts)))
            except ValueError:
                pass
            self.active = False


def scripts(html, script_id=None):
    parser = Scripts(script_id)
    parser.feed(html)
    return parser.blocks


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def instagram(html, code=None, search=False):
    payloads, owner = scripts(html), None
    if search:
        payloads = [edge.get("node", {}) for obj in objects(payloads)
                    if isinstance(obj.get("xig_logged_out_popular_search_media_info"), dict)
                    for edge in obj["xig_logged_out_popular_search_media_info"].get("edges", [])]
    elif code is None:
        profiles = [o["xig_user_by_username"] for o in objects(payloads) if isinstance(o.get("xig_user_by_username"), dict)]
        if any(p.get("is_private") is True for p in profiles):
            raise RuntimeError("비공개 Instagram 프로필")
        owner = next((p.get("username") for p in profiles if p.get("username")), None)
        payloads = [p.get(k) or {} for p in profiles for k in ("polaris_ordered_timeline_connection", "polaris_clips_connection")]
    media = {}
    for obj in objects(payloads):
        key = obj.get("code")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) or (code and code != key):
            continue
        if obj.get("media_type") not in (1, 2, 8) and obj.get("__typename") not in ("XIGPolarisImageMedia", "XIGPolarisVideoMedia", "XIGPolarisCarouselMedia"):
            continue
        if key not in media or len(obj) > len(media[key]):
            media[key] = obj
    rows = []
    for key, obj in media.items():
        hidden = obj.get("like_and_view_counts_disabled") is True
        plays = obj.get("play_count")
        images = (obj.get("image_versions2") or {}).get("candidates") or []
        rows.append(dict(id=key, url="https://www.instagram.com/" + ("reel" if obj.get("product_type") == "clips" or (search and obj.get("__typename") == "XIGPolarisVideoMedia") else "p") + "/" + key + "/",
            creator=(obj.get("user") or {}).get("username") or owner, caption=(obj.get("caption") or {}).get("text"),
            published=obj.get("taken_at"), thumbnail=obj.get("display_uri") or next((i.get("url") for i in images if i.get("url")), None),
            likes=None if hidden else obj.get("like_count"), comments=obj.get("comment_count"),
            views=None if hidden else plays if plays is not None else obj.get("view_count"), view_metric="plays" if plays is not None else "views"))
    if not rows and not search:
        raise RuntimeError("Instagram 공개 JSON 없음 · 접근 제한 또는 페이지 변경")
    return rows


def tiktok(html, video_id=None, handle=None):
    script_id = "__UNIVERSAL_DATA_FOR_REHYDRATION__" if video_id else "__FRONTITY_CONNECT_STATE__"
    blocks = scripts(html, script_id)
    if not blocks:
        raise RuntimeError("TikTok 공개 JSON 없음 · 접근 제한 또는 페이지 변경")
    state = blocks[0]
    if video_id:
        detail = state.get("__DEFAULT_SCOPE__", {}).get("webapp.video-detail", {})
        item = detail.get("itemInfo", {}).get("itemStruct", {})
        if detail.get("statusCode") != 0 or str(item.get("id")) != video_id:
            raise RuntimeError("요청한 TikTok 영상의 공개 본문을 확인하지 못했습니다.")
        stats = dict(item.get("stats") or {})
        stats.update({k: v for k, v in (item.get("statsV2") or {}).items() if v is not None})
        creator, video = (item.get("author") or {}).get("uniqueId"), item.get("video") or {}
        if not isinstance(creator, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", creator):
            raise RuntimeError("TikTok 공개 작성자 정보 없음")
        return [dict(id=video_id, url=f"https://www.tiktok.com/@{creator}/video/{video_id}/", creator=creator,
            caption=item.get("desc"), published=int(item["createTime"]) if str(item.get("createTime", "")).isdigit() else None,
            views=stats.get("playCount"), likes=stats.get("diggCount"), comments=stats.get("commentCount"), shares=stats.get("shareCount"),
            thumbnail=video.get("cover") or video.get("originCover"), duration=video.get("duration"))]
    embedded = state.get("source", {}).get("data", {}).get("/embed/@" + handle, {})
    if not isinstance(embedded.get("videoList"), list):
        raise RuntimeError("TikTok 공개 프로필 영상 목록 없음")
    # shortcut: 공개 임베드는 최근·고정 영상 일부만 제공, 전체 이력이 필요하면 별도 승인된 소스를 추가한다.
    rows = [dict(id=str(r["id"]), url=f"https://www.tiktok.com/@{handle}/video/{r['id']}/", creator=handle, caption=r.get("desc"))
            for r in embedded["videoList"] if str(r.get("id", "")).isdigit() and not r.get("privateItem")]
    return sorted(rows, key=lambda r: int(r["id"]), reverse=True)


def threads(html, code=None, search=False):
    payloads = scripts(html)
    if search:
        payloads = [item.get("post", {})
                    for obj in objects(payloads) if isinstance(obj.get("searchResults"), dict)
                    for edge in obj["searchResults"].get("edges", [])
                    for item in edge.get("node", {}).get("thread", {}).get("thread_items", [])]
    found = {}
    for obj in payloads if search else objects(payloads):
        key, user = obj.get("code"), obj.get("user") or {}
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) or (code and key != code):
            continue
        username = user.get("username")
        if "caption" not in obj or not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
            continue
        if user.get("is_private") is True:
            continue
        if key not in found or len(obj) > len(found[key]):
            found[key] = obj
    rows = []
    for key, obj in found.items():
        app, user = obj.get("text_post_app_info") or {}, obj["user"]
        media = obj if obj.get("image_versions2") else next(iter(obj.get("carousel_media") or []), {})
        images = (media.get("image_versions2") or {}).get("candidates") or []
        rows.append(dict(id=key, url=f"https://www.threads.com/@{user['username']}/post/{key}/", creator=user["username"],
            caption=(obj.get("caption") or {}).get("text"), published=obj.get("taken_at"),
            thumbnail=next((i.get("url") for i in images if i.get("url")), None), views=obj.get("view_count"),
            likes=obj.get("like_count"), comments=app.get("direct_reply_count"), shares=app.get("reshare_count"),
            metrics={"reposts": app.get("repost_count")}))
    if not rows:
        raise RuntimeError("Threads 공개 본문 없음 · 접근 제한 또는 페이지 변경")
    return rows
