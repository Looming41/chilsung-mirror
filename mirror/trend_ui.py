"""실제 워커와 연결된 콘텐츠 관측 화면. 읽기는 5초마다, 네트워크 수집은 워커에서."""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from . import trends as T, trend_social as Social

_graph = components.declare_component("trend_observatory", path=str(Path(__file__).with_name("trend_component")))


def render():
    T.init()
    st.markdown("""<style>
    .stMainBlockContainer{padding-top:4.4rem;padding-bottom:1rem;max-width:1800px}
    [data-testid="stAppViewContainer"]{background:#f7f8f5}
    [data-testid="stSidebar"]{background:#f0f2ec}
    [data-stale="true"]{opacity:1}
    @media(max-width:700px){
      .stMainBlockContainer{padding-left:12px;padding-right:12px}
      .st-key-trend_controls [data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:8px}
      .st-key-trend_controls [data-testid="stColumn"]{min-width:calc(50% - 4px);flex:1 1 calc(50% - 4px)}
      .st-key-trend_controls button{min-height:44px;font-size:12px}
    }
    </style>""", unsafe_allow_html=True)
    with st.container(key="trend_controls"):
        a, b, c, d = st.columns([1.4, 1, 1, 1.2])
    if a.button("관측 시작 / 재연결", type="primary", width="stretch", key="trend_start"):
        T.start()
    if b.button("관측 중지", width="stretch", key="trend_stop"):
        T.stop()
    if c.button("분석 재시도", width="stretch", key="trend_retry", help="수집 간격은 지키고 미완료 분석을 다시 실행합니다."):
        T.request_cycle()
    hours = d.selectbox("관측 기간", [24, 72, 168], format_func=lambda n: f"최근 {n // 24}일", label_visibility="collapsed", key="trend_hours")

    @st.fragment(run_every=5)
    def live():
        _graph(payload=T.snapshot(hours), key="trend_live", default=None)
    live()
    with st.expander("공개 수집 범위·링크 추가 · Instagram / TikTok / Threads"):
        st.markdown("**세 채널 모두 계정 없이 자동 수집합니다.** 공개 목록에서 새 콘텐츠를 발견하고 1시간 간격으로 지표를 다시 확인합니다.")
        st.caption("Instagram: 공개 인기 키워드 → 게시물 원문 / TikTok: 네이버 공개 검색 → 영상 원문 / Threads: 공개 키워드·해시태그 검색. 공개 검색 첫 페이지 범위이며 최신순·전체 순위를 보장하지 않습니다.")
        st.caption("32개 식음료 탐색어에서 시작해 연관 검색어와 AI가 확인한 음식명으로 확장합니다. 채널당 8개씩 순환하며 같은 검색어는 최소 6시간 후 다시 검색합니다.")
        st.caption("아래에서 관측할 게시물을 추가할 수 있습니다. 개인 쿠키를 사용하지 않으며 접근 제한은 실패로 기록합니다.")
        urls = st.text_area("관측할 게시물 주소", placeholder="YouTube · Instagram 릴스/게시물 · TikTok 영상 · Threads 게시물의 전체 주소를 한 줄에 하나씩", key="social_urls")
        if st.button("공개 링크 등록 · 관측 요청", key="social_add"):
            try:
                n = Social.add_urls(urls.splitlines())
                if n:
                    T.request_cycle()
                    st.success(f"{n}개 주소를 등록했습니다. 채널 수집 주기에 따라 접근과 식음료 관련성을 확인합니다.")
                else:
                    st.info("등록할 게시물 주소를 입력하세요.")
            except ValueError as exc:
                st.error(str(exc))
        with T.D.db() as conn:
            Social.init()
            watched = [dict(r) for r in conn.execute("select platform,url,status,error from trend_social_urls order by added desc limit 30")]
        if watched:
            st.dataframe(watched, hide_index=True, width="stretch")
        st.markdown("""**선택 사항: 공식 연결 추가** — 기본 공개 수집에는 필요하지 않습니다. 프로젝트 `.env`에 설정 후 워커를 중지하고 다시 시작하세요. 토큰을 채팅이나 공개 링크 칸에 입력하지 마세요.

- **YouTube**: 공개 검색이 기본이며 `YOUTUBE_API_KEY`가 있으면 공식 검색 API를 사용합니다.
- **Instagram**: `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_USER_ID`. 프로페셔널 계정과 공개 콘텐츠 접근 승인이 필요합니다. 고정 식음료 해시태그 8개를 검색합니다.
- **Threads**: `THREADS_ACCESS_TOKEN`. `threads_basic`, 승인된 `threads_keyword_search` 권한으로 식음료 키워드를 검색합니다.
- **TikTok**: `TIKTOK_ACCESS_TOKEN`. Display API는 연결한 계정의 영상만 가져옵니다. 전체 공개 키워드 검색은 지원하지 않습니다.

공개되지 않은 조회수·반응은 `—`로 남깁니다. 등록한 외부 공개 링크의 접근 가능 여부는 게시물마다 다릅니다.
""")
        st.markdown(" · ".join(f"[{name} 공식 안내]({Social.DOCS[key]})" for key, name in Social.PLATFORMS.items()))
    with st.expander("캠페인별 Decode · 파일 가져오기 · 대본 편집"):
        from .decode_ui import render_campaign
        render_campaign()
