"""소비자 반응 모니터링(관측) ↔ 신제품 검증(예측) 연결.

- 관측 → 검증: 검증 중인 컨셉과 맞는 실제 글을 찾아 근거·걱정·표본 크기를 보여 준다(evidence)
- 검증 → 관측: 컨셉의 일반 검색어로 허용된 소스(빙 뉴스 RSS·블루스카이 공개 API)를 찾아 같은 저장소에 넣는다(seek). 출시 전 제품명은 외부로 보내지 않는다
- 관측 → 시뮬레이션: 뜨는 음식·테마를 신제품 후보 문장으로 넘긴다(candidates)
- 반응량은 매출 예측에 넣지 않는다: 주어진 자료에서 검색·SNS 는 기온을 빼면 판매를 설명하지 못했다(signal_test)
실행: python -m mirror.link check | report
"""
import json
import re
import sys
import time
import urllib.parse
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

# queries 는 외부 검색에 쓰이므로 일반어·이미 팔리는 브랜드만 넣는다
TERMS = {"밀키스 화채": {"queries": ["수박화채", "화채 음료", "밀키스", "과일 탄산음료"], "core": r"화채.{0,8}(?:음료|맛|탄산|에이드|소다)|수박.{0,8}(?:음료|에이드|주스|탄산|맛)|밀키스(?!정식)", "rx": r"화채|수박.{0,12}(?:음료|에이드|주스|탄산|맛)|밀키스(?!정식)"},
         "레쓰비 프리즈": {"queries": ["얼려먹는 커피", "컵얼음 커피", "커피 슬러시", "편의점 커피"], "core": r"얼려\s*먹.{0,8}커피|컵\s*얼음|얼음\s*컵|커피\s*슬러시|슬러시\s*커피|프라페|프라푸치노|레쓰비", "rx": r"얼려\s*먹|컵\s*얼음|얼음\s*컵|프라페|슬러시|레쓰비|(?:편의점|캔|파우치)\s*커피|프라푸치노"},
         "쥬시모먼트": {"queries": ["착즙주스", "프리미엄 주스", "과일주스", "생과일 주스"], "core": r"착즙|(?:생과일|프리미엄)\s*주스|콜드프레스|델몬트", "rx": r"착즙|(?:과일|생과일|프리미엄|오렌지|사과|자몽)\s*주스|콜드프레스|델몬트"}}
WORRY = {"가격": r"비싸|비쌈|가격|가성비|돈값|원값", "단맛": r"너무\s*달|달아서|느끼|설탕|당류", "구매처": r"어디서\s*(?:사|파|팔)|안\s*팔|품절|못\s*구|재고", "맛·품질": r"맛없|별로|인위적|밍밍|실망|이상한\s*맛", "건강": r"칼로리|카페인|살찌|당뇨|건강"}
AD = re.compile(r"광고|협찬|제공받|체험단|(?<![A-Za-z])AD(?![A-Za-z])|이벤트|당첨자")
OFFICIAL = re.compile(r"chilsung|lotte|milkis|pepsi|official|공식", re.I)
GAP = 600                                              # 같은 컨셉 재검색 최소 간격(초)


def words(r):
    return " ".join(str(r.get(k) or "") for k in ("caption", "transcript", "on_screen_text"))


def items(hours=720):
    """관측 화면과 같은 자료(trends) + 같은 저장소의 나머지 글(media.db), 주소 기준 중복 제거."""
    from . import media as M
    from . import trends as T
    s = T.snapshot(hours)
    seen = {r.get("url") for r in s["items"]}
    c = M.db()
    rows = c.execute("select text, url, coalesce(published, ''), kind, metrics from items").fetchall()
    c.close()
    extra = []
    for t, u, p, k, m in rows:
        if u not in seen:
            m = json.loads(m) if m else {}
            extra.append(dict(caption=t, url=u, published=p, platform=k, creator="", likes=m.get("likes"), comments=m.get("comments"), views=m.get("views")))
    return s, s["items"] + extra


def evidence(con, rows, now=None):
    """컨셉과 맞는 글 요약. 홍보를 뺀 직접 관련 글이 최근 30일에 5건 이상이어야 '참고 가능'."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    rx, core = re.compile(TERMS[con]["rx"], re.I), re.compile(TERMS[con]["core"], re.I)
    uniq = {re.sub(r"\W", "", words(r))[:80]: r for r in rows if rx.search(words(r))}          # 같은 글이 다른 경로로 두 번 들어온 것 제거
    hit = list(uniq.values())
    t = pd.to_datetime(pd.Series([r.get("published") for r in hit], dtype=object), utc=True, errors="coerce", format="mixed")
    out = []
    for r, q in zip(hit, (t >= now - pd.Timedelta(days=30)).tolist()):
        w = words(r)
        out.append({"글": w[:140], "직접 관련": bool(core.search(w)), "채널": {"SNS": "소셜", "social": "소셜", "뉴스": "뉴스", "news": "뉴스"}.get(r.get("platform"), r.get("platform")), "작성자": r.get("creator") or "", "게시": str(r.get("published") or "")[:10], "최근 30일": q, "조회": r.get("views"), "좋아요": r.get("likes"), "댓글": r.get("comments"),
                    "홍보·뉴스": bool(r.get("platform") in ("news", "뉴스") or AD.search(w) or OFFICIAL.search(str(r.get("creator") or ""))), "걱정": ", ".join(k for k, v in WORRY.items() if re.search(v, w)), "링크": r.get("url")})
    org = [x for x in out if not x["홍보·뉴스"]]
    near = [x for x in org if x["직접 관련"]]
    n30 = sum(x["최근 30일"] for x in near)
    num = lambda v: v if isinstance(v, (int, float)) and v == v else 0
    return {"글": len(out), "소비자 글": len(org), "직접 관련": len(near), "최근 30일": n30, "채널": dict(Counter(x["채널"] for x in near)), "걱정": dict(Counter(k for x in near for k in x["걱정"].split(", ") if k)),
            "판정": "참고 가능" if n30 >= 5 else "표본 부족", "목록": sorted(org, key=lambda x: (x["직접 관련"], x["최근 30일"], num(x["좋아요"]), num(x["조회"])), reverse=True) + [x for x in out if x["홍보·뉴스"]]}


def seek(con, tick=None, get=None, rss=None):
    """컨셉 일반 검색어로 허용된 소스를 찾아 저장 → (찾은 수, 새 글 수, 실패 목록). 같은 컨셉은 10분에 한 번."""
    from . import media as M
    get, rss = get or M.get, rss or M.rss
    c, src, found, new, bad = M.db(), "컨셉 검색: " + " · ".join(TERMS[con]["queries"][:2]), 0, 0, []
    try:
        last = c.execute("select max(ts) from runs where source=?", (src,)).fetchone()[0]
        if last and (pd.Timestamp.now(tz="Asia/Seoul") - pd.Timestamp(last).tz_convert("Asia/Seoul")).total_seconds() < GAP:
            raise ValueError("10분 안에 이미 찾았습니다.")
        for i, q in enumerate(TERMS[con]["queries"]):
            u = urllib.parse.quote(q)
            for kind, fn in (("뉴스", lambda: rss(f"https://www.bing.com/news/search?q={u}&format=rss&setlang=ko")),
                             ("SNS", lambda: [dict(text=p["record"].get("text"), url="https://bsky.app/profile/" + p["author"]["did"] + "/post/" + p["uri"].rsplit("/", 1)[-1], published=M.when(p["record"].get("createdAt")),
                                                   metrics={"likes": p.get("likeCount"), "comments": p.get("replyCount")}) for p in json.loads(get(f"https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={u}&limit=50&lang=ko")).get("posts", [])])):
                try:
                    rows = fn()
                    a, b = M.save(c, src, kind, rows)
                    found, new = found + a, new + b
                except Exception as e:                 # 한 소스 실패가 나머지를 막지 않게 하되 보고한다
                    bad.append(f"{q} ({kind}): {type(e).__name__}")
                time.sleep(0 if fn is None else .5)
            if tick:
                tick((i + 1) / len(TERMS[con]["queries"]))
        c.execute("insert into runs values(?,?,?,?,?,?)", (M.now(), src, int(not bad), found, new, "; ".join(bad)[:200]))
        c.commit()
    finally:
        c.close()
    return found, new, bad


def candidates(snap, n=30):
    """관측에서 나온 신제품 후보: 표본이 부족하지 않은 음식·장르·테마."""
    g = [x for x in snap["insights"]["groups"] if x["status"] in ("증가 확인", "최근 인기 후보", "최근 관측")]
    return pd.DataFrame([{"이름": x["label"], "종류": {"food": "음식·브랜드", "genre": "장르", "theme": "소비 테마"}[x["kind"]], "상태": x["status"], "최근 글": x["current"], "작성자": x["creators"], "채널": ", ".join(x["platforms"]), "반응 점수": x["score"]} for x in g[:n]])


def concept_text(label):
    return f"{label} 콘셉트의 음료. 500ml 페트."


@lru_cache(1)
def signal_test():
    """주어진 자료로 '관심이 판매를 설명하나'를 본다: 단순 상관과, 기온·기온²·추세를 뺀 뒤의 상관(월별, 행사 없는 판매)."""
    from scipy.stats import t as T
    from . import given as G
    p = G.pos()
    q = p[(p.EA > 0) & ~p.missing & (p.행사구분 == "없음")]
    s = G.sheet(G.C_, "3_검색트렌드지수").set_index("기준월").clip(lower=0)
    n = G.sheet(G.C_, "5_SNS언급량")
    org = n.assign(v=n["SNS언급량(건)"].clip(lower=0) * (1 - n["광고·협찬 비중"])).pivot(index="기준월", columns="키워드", values="v")
    nm, rows = p.drop_duplicates("제품코드").set_index("제품코드").제품명, []
    for code, kw in (("MK03", "제로 탄산"), ("MK02", "수박화채"), ("XE01", "얼려먹는 커피"), ("XE01", "슬러시 커피"), ("XD01", "착즙주스"), ("XD01", "프리미엄 주스")):
        g = q[q.제품코드 == code].groupby("기준월")
        y, tm = np.log(g.EA.mean()), g.temp.mean()
        for lab, S in (("검색", s), ("SNS(광고 제외)", org)):
            d = pd.DataFrame({"y": y, "t": tm, "x": np.log1p(S[kw].reindex(y.index)), "k": np.arange(len(y))}).dropna()
            X = np.c_[np.ones(len(d)), d.t, d.t ** 2, d.k]
            r = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
            pr, df = np.corrcoef(r(d.y.values), r(d.x.values))[0, 1], len(d) - 5
            rows.append({"제품": nm[code], "키워드": kw, "신호": lab, "개월": len(d), "단순 상관": np.corrcoef(d.y, d.x)[0, 1], "기온·추세 제외": pr, "p": float(2 * T.sf(abs(pr) * np.sqrt(df / (1 - pr ** 2)), df))})
    return pd.DataFrame(rows)


def check():
    import os
    import tempfile
    from . import media as M
    now = pd.Timestamp("2026-11-24", tz="UTC")
    mk = lambda i, cap, d="2026-11-20", who=None, **k: {"caption": cap, "platform": "threads", "creator": who or f"u{i}", "published": d + "T00:00:00+09:00", "likes": 1, "url": f"https://x/{i}", **k}
    rows = [mk(1, "수박화채 맛 음료 너무 달아서 별로"), mk(2, "밀키스 가격 비싸"), mk(3, "화채 만들었어", d="2026-01-01"), mk(4, "밀키스 신상 #광고"), mk(5, "밀키스 나왔어요", who="lottechilsung"), mk(6, "삼겹살"), mk(7, "수박에이드 음료"),
            mk(8, "수박 농사"), mk(9, "밀키스 출시", platform="news"), mk(10, "밀키스정식님뜯어먹기")]
    e = evidence("밀키스 화채", rows, now)
    assert (e["글"], e["소비자 글"], e["직접 관련"], e["최근 30일"]) == (7, 4, 3, 3) and e["판정"] == "표본 부족" and e["걱정"] == {"단맛": 1, "맛·품질": 1, "가격": 1}, e
    assert evidence("밀키스 화채", rows + rows, now)["글"] == 7                              # 같은 글은 한 번만 센다
    assert evidence("밀키스 화채", rows + [mk(i, f"밀키스 좋아 {i}") for i in (20, 21)], now)["판정"] == "참고 가능" and evidence("쥬시모먼트", [], now)["글"] == 0
    for c, v in TERMS.items():                                   # 출시 전 제품명은 검색어에 없어야 한다
        assert not any(c.replace(" ", "") in q.replace(" ", "") or q.replace(" ", "") in ("쥬시모먼트", "레쓰비프리즈") for q in v["queries"])
    old, M.DB = M.DB, os.path.join(tempfile.mkdtemp(), "m.db")
    try:
        post = {"record": {"text": "착즙주스 마심", "createdAt": "2026-11-24T01:00:00Z"}, "author": {"did": "d1"}, "uri": "at://d1/p/1"}
        bsky = lambda u: json.dumps({"posts": [post]})
        boom = lambda u: (_ for _ in ()).throw(OSError("x"))
        f, n, bad = seek("쥬시모먼트", get=bsky, rss=boom)
        assert f == 4 and n == 1 and len(bad) == 4
        try:
            seek("쥬시모먼트", get=bsky, rss=boom)
            raise AssertionError("10분 간격이 지켜지지 않음")
        except ValueError:
            pass
    finally:
        M.DB = old
    g = lambda **k: dict(label="말차", kind="food", current=3, creators=3, platforms=["threads"], score=None) | k
    assert len(candidates({"insights": {"groups": [g(status="최근 관측"), g(status="표본 부족")]}})) == 1
    print("link ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        s_, all_ = items()
        for c_ in TERMS:
            print(c_, {k: v for k, v in evidence(c_, all_).items() if k != "목록"})
        print(signal_test().round(2).to_string(), "\n", candidates(s_).to_string())
    else:
        check()
