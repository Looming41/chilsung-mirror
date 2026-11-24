"""Decode 캠페인 화면. 실제 데이터만 표시하고 외부 실행은 버튼으로 시작한다."""
import io
import json

import pandas as pd
import streamlit as st

from . import decode as D


def run(label, fn):
    with st.status(label, expanded=True) as status:
        def tick(message):
            status.update(label=message)
        try:
            result = fn(tick)
            if isinstance(result, tuple):
                n, errors = result
                st.write(f"처리 {n}건")
                for e in errors[:20]:
                    st.warning(e)
            else:
                st.write(f"처리 {result}건")
            status.update(label="완료", state="complete")
        except Exception as e:
            message = str(e) if isinstance(e, (ValueError, RuntimeError)) else f"{type(e).__name__}: 연결·파일·응답 형식을 확인하세요."
            st.error(message)
            status.update(label="중단 · 저장된 결과는 유지됩니다", state="error")


def render():
    st.subheader("Decode · Laya 콘텐츠 분석")
    st.caption("① 기존 수집 데이터 연결 → ② 대본·화면 자막 → ③ 포맷 분류 → ④ 중앙 조회수 비교 → ⑤ 콘텐츠별로 확인")
    all_campaigns = D.campaigns()
    if "_decode_next" in st.session_state:
        st.session_state.decode_campaign = st.session_state.pop("_decode_next")
    opts = {c["id"]: c["name"] for c in all_campaigns}
    st.session_state.setdefault("decode_campaign", next(iter(opts), "new"))
    current = st.selectbox("캠페인", ["new", *opts],
                           format_func=lambda x: "＋ 새 캠페인" if x == "new" else opts[x], key="decode_campaign")
    old = next((c for c in all_campaigns if c["id"] == current), None)
    sources = D.source_options()
    cfg = old["config"] if old else dict(brand="칠성사이다", sources=["format:" + n for n in D.F.PAGES], keywords=[], formats=D.F.formats())
    with st.expander("캠페인·기존 수집 소스·포맷 설정", expanded=old is None):
        with st.form("decode_setup_" + current):
            a, b = st.columns(2)
            name = a.text_input("캠페인 이름", value=old["name"] if old else "칠성 콘텐츠 탐색")
            brand = b.text_input("브랜드", value=cfg["brand"])
            selected = st.multiselect("기존 수집 소스", list(sources), default=[x for x in cfg["sources"] if x in sources], format_func=sources.get)
            keywords = st.text_area("가져올 본문·제목 검색어 (한 줄에 하나, 하나라도 포함하면 선택)", value="\n".join(cfg["keywords"]),
                                    placeholder="비워 두면 선택한 소스 전체를 가져옵니다.")
            st.caption("콘텐츠에서 실제 반복되는 구체적인 포맷을 적으세요. '해당 없음'은 자동 추가합니다.")
            edited = st.data_editor(pd.DataFrame(list(cfg["formats"].items()), columns=["포맷", "한 줄 설명"]),
                                    num_rows="dynamic", hide_index=True, width="stretch", key="decode_formats_" + current)
            if st.form_submit_button("캠페인 저장", type="primary"):
                try:
                    rows = edited.dropna()
                    if rows["포맷"].str.strip().duplicated().any():
                        raise ValueError("포맷 이름이 중복되었습니다.")
                    st.session_state._decode_next = D.configure(name, brand, selected, keywords.splitlines(), dict(rows.values), old["id"] if old else None)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    if old is None:
        st.info("캠페인을 저장한 다음 기존 수집 데이터를 연결하거나 JSON/CSV를 가져오세요.")
        return
    cid = old["id"]
    st.caption("기존 media.db와 뉴스·커뮤니티·SNS 수집기를 재사용합니다. 포맷 분류는 Laya 로컬 모델만 사용합니다.")
    with st.expander("JSON/CSV 가져오기"):
        upload = st.file_uploader("콘텐츠 JSON 또는 CSV", type=["json", "csv"], key="decode_upload")
        platform = st.selectbox("가져올 데이터의 플랫폼", ["instagram", "tiktok", "youtube", "community", "news", "social"], key="decode_import_platform")
        st.caption("기본 열: id, url, creator, caption, views, likes, comments, shares, thumbnail, video_url, transcript, on_screen_text, published. 일반적인 SNS 내보내기 필드도 인식합니다.")
        if st.button("파일 가져오기", disabled=upload is None):
            try:
                raw = upload.getvalue()
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError("파일은 20MB 이하로 올려 주세요.")
                rows = json.loads(raw) if upload.name.endswith(".json") else pd.read_csv(io.BytesIO(raw), dtype=str).where(lambda x: x.notna(), None).to_dict("records")
                n, errors = D.save_rows(cid, rows, platform, batch="import:" + D.digest([platform, raw.decode("utf-8-sig")]))
                st.success(f"{n}건 저장. 같은 파일을 다시 넣어도 지표 이력은 중복되지 않습니다.")
                for e in errors[:10]:
                    st.warning(e)
            except (ValueError, UnicodeError, pd.errors.ParserError) as e:
                st.error(str(e))
    st.caption("기존 제목 분류 평가에서 Laya 정확도는 44.5%였습니다. 이 캠페인의 정확도는 별도 검증이 필요합니다.")
    st.caption("영상 파일 URL이 있는 자료는 Whisper 대본·한국어 OCR을 추가할 수 있습니다. OCR은 첫 6초의 네 프레임을 읽습니다. 릴스·틱톡은 현재 JSON/CSV 가져오기로 연결합니다.")
    refresh = st.checkbox("가져오기 전에 기존 수집기로 선택 소스 갱신", False, key="decode_refresh",
                           help="기존 요청 간격·robots 규칙을 따릅니다. 유튜브 공식 API는 기존 YOUTUBE_API_KEY가 필요합니다.")
    a, b, c = st.columns(3)
    if a.button("① 기존 수집 데이터 가져오기", disabled=not cfg["sources"], width="stretch"):
        run("기존 데이터 연결", lambda tick: D.collect(cid, cfg, tick, refresh=refresh))
    if b.button("② 영상 대본·화면 자막 추출", width="stretch"):
        run("대본 추출", lambda tick: D.words(cid, tick))
    if c.button("③ Laya 미분류·변경분 분류", type="primary", width="stretch"):
        run("Laya 포맷 분류", lambda tick: D.classify(cid, cfg, tick))
    full = st.checkbox("영상 대본·화면 자막 추출 포함", False, key="decode_full")
    if st.button("전체 Decode 실행"):
        def pipeline(tick):
            if cfg["sources"]:
                n, errors = D.collect(cid, cfg, tick, refresh=refresh)
                st.write(f"기존 데이터 {n}건 연결")
                for error in errors[:20]:
                    st.warning(error)
            if full:
                _, errors = D.words(cid, tick)
                for error in errors[:20]:
                    st.warning(error)
            return D.classify(cid, cfg, tick)
        run("전체 Decode", pipeline)
    d = D.load(cid, cfg)
    if d.empty:
        st.info("아직 콘텐츠가 없습니다. 기존 수집 데이터 또는 파일 가져오기를 실행하세요. 예시 수치는 실제 결과에 섞지 않습니다.")
    else:
        a, b, c = st.columns([2, 1, 1])
        d["source"] = d.get("source", pd.Series("", index=d.index)).fillna("")
        groups = d.platform + " · " + d.source + " · " + d.view_metric
        metrics = sorted(set(groups))
        metric = a.selectbox("비교할 플랫폼·소스·조회 지표", metrics, key="decode_metric")
        brand_only = b.checkbox("브랜드 언급 확인만", key="decode_brand_only")
        sure = c.slider("최소 분류 확신", 0., 1., 0., .05, key="decode_sure")
        d = d[groups == metric]
        if brand_only:
            d = d[d.brand == "확인"]
        if sure:
            d = d[d.confidence.ge(sure)]
        if d.empty:
            st.info("현재 필터에 해당하는 콘텐츠가 없습니다.")
        else:
            k = st.columns(4)
            k[0].metric("콘텐츠", f"{len(d):,}")
            k[1].metric("크리에이터(제공분)", d.loc[d.creator.ne(""), "creator"].nunique())
            k[2].metric("캠페인 중앙 조회수", f"{d.views.median():,.0f}" if d.views.notna().any() else "미제공")
            k[3].metric("분류 완료", f"{d.fmt.notna().sum()} / {len(d)}")
            if not d.views.notna().any():
                st.info("이 소스는 조회수를 제공하지 않습니다. 포맷 분류와 원문은 확인할 수 있으며, 조회수 순위는 계산하지 않습니다.")
            st.iframe(D.chart_html(d), height="content", alt="콘텐츠 산점도, 포맷 선택과 원문 상세")
            r = D.ranking(d)
            if len(r):
                st.dataframe(r.style.format({"중앙 조회수": "{:,.0f}", "캠페인 대비": "{:.2f}×"}, na_rep="—"), width="stretch", hide_index=True)
            st.caption("배수 = 포맷 중앙 조회수 ÷ 현재 필터 전체 콘텐츠의 중앙 조회수. 5건 미만은 판단하지 않습니다. 게시 시점·계정 규모·광고비를 보정한 인과 효과나 미래 바이럴 확률은 아닙니다. 브랜드 판정은 텍스트 근거이며 로고 인식은 포함하지 않습니다.")
            cols = ["source", "creator", "caption", "fmt", "views", "likes", "shares", "rate", "brand", "url"]
            st.dataframe(d[cols], hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn("원문")})
            st.download_button("분석 결과 CSV", D.csv_bytes(d), "decode.csv", "text/csv")
            with st.expander("대본·화면 자막 확인 및 수정"):
                vid = st.selectbox("콘텐츠", d.id.tolist(), format_func=lambda x: next((str(v.caption or v.external_id)[:70] for v in d.itertuples() if v.id == x), x))
                v = d[d.id == vid].iloc[0]
                if v.get("word_error"):
                    st.warning(v.word_error)
                with st.form("decode_words_" + vid):
                    transcript = st.text_area("대본", v.transcript or "", height=160)
                    overlay = st.text_area("화면 자막", v.on_screen_text or "")
                    if st.form_submit_button("텍스트 저장 · 다음 분류에 반영"):
                        D.edit_words(vid, transcript, overlay)
                        st.rerun()
