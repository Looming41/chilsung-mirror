"""미디어 대응실 화면(Streamlit). 수집·분류는 별도 프로세스로 돌려 끝나면 모델 메모리를 돌려준다."""
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from . import media as M

LOG = os.path.join(M.ROOT, "data", "media.log")
SURE = .8                                       # 이 확신 미만의 laya 라벨은 '확인 필요'


def _running():
    p = st.session_state.get("media_proc")
    return p is not None and p.poll() is None


def _start():
    if not _running():
        with open(LOG, "w") as f:
            st.session_state.media_proc = subprocess.Popen([sys.executable, "-u", "-m", "mirror.media", "collect"], cwd=M.ROOT,
                                                           stdout=f, stderr=subprocess.STDOUT)


@st.fragment(run_every=5)
def _status():
    if _running():
        tail = [l for l in open(LOG).read().splitlines() if l.startswith("20")][-1:]
        st.info("수집·분류 중… " + (tail[0] if tail else "소스에 접속하는 중"))
    elif st.session_state.pop("media_proc", None) is not None:
        st.rerun(scope="app")


def _to_sim(text):
    st.session_state.concept_in = text          # 콜백: 다음 실행에서 시뮬레이터 콘셉트 입력란에 채워짐
    st.session_state.view = "시뮬레이터"


def render(model):
    st.title("📡 미디어 대응실")
    c = M.db()
    src = M.sources()
    tot, rel, cls = c.execute("select count(*), sum(relevant), sum(topic is not null) from items").fetchone()
    last = c.execute("select max(ts) from runs where source not like '위키%'").fetchone()[0]
    al = M.alerts()
    k = st.columns(5)
    k[0].metric("켜진 소스", f"{sum(v[2] for v in src.values())} / {len(src)}")
    k[1].metric("모은 글", f"{tot or 0:,}")
    k[2].metric("식음료 관련", f"{rel or 0:,}")
    k[3].metric("확인할 위험 글", len(al))
    k[4].metric("마지막 수집", last[5:16].replace("T", " ") if last else "없음")
    st.button("▶ 지금 수집·분류", on_click=_start, disabled=_running(), type="primary", help="소스마다 10분에 한 번만 요청합니다.")
    _status()
    t1, t2, t3, t4, t5, t6 = st.tabs(["🚨 위험", "📈 추세", "📰 글", "✍️ 지정 분석", "📝 대응 초안", "🔌 소스"])

    with t1:
        st.caption("안전 키워드(이물·배탈·회수 등)가 있는 식음료 글입니다. 오경보가 섞이니 사람이 확인해야 합니다. "
                   "laya의 안전 판정은 실제 글에서 쓸 수 없을 만큼 틀려서 사용하지 않습니다.")
        if al.empty:
            st.success("지금 확인할 글이 없습니다.")
        else:
            st.dataframe(al[["t", "source", "text", "url"]].rename(columns={"t": "시각", "source": "소스", "text": "내용", "url": "링크"}),
                         width="stretch", hide_index=True, column_config={"링크": st.column_config.LinkColumn(display_text="열기")})
            pick = st.selectbox("초안을 만들 글", al.index, format_func=lambda i: al.text[i][:70])
            if st.button("대응 초안 만들기"):
                try:
                    with st.spinner("작성 중…"):
                        st.session_state.last_draft = M.draft(al.text[pick][:60], [al.text[pick]], model)
                    st.success("'대응 초안' 탭에 승인 대기로 저장했습니다.")
                except Exception as e:
                    st.error(f"초안 실패: {type(e).__name__}: {str(e)[:200]}")

    with t2:
        st.subheader("급증 단어 (최근 48시간)")
        b = M.bursts()
        hot = b[(b["q(BH)"] < .05) & (b["이야기 수"] >= 2)] if len(b) else b
        if hot.empty:
            st.info("서로 다른 이야기 2건 이상에서 함께 늘어난 단어가 아직 없습니다. 수집이 쌓일수록 정확해집니다.")
        else:
            st.dataframe(hot.style.format({"점유율 배수": "{:.1f}", "p": "{:.4f}", "q(BH)": "{:.4f}"}), width="stretch", hide_index=True)
            w = st.selectbox("시뮬레이터로 보낼 단어", hot["단어"])
            st.button("이 추세를 신제품 콘셉트로 시뮬레이션", on_click=_to_sim, args=(f"{w} 콘셉트의 탄산음료. 500ml 페트.",))
        st.caption("보도자료를 여러 매체가 실은 것은 한 이야기로 셉니다. Fisher 정확검정 + BH 보정이며, 수집 초기에는 소스 구성 차이로 흔들립니다.")
        st.subheader("위키백과 조회수 추세")
        wt = M.wiki_table()
        if len(wt):
            st.dataframe(wt.style.format({"최근 7일 평균": "{:.0f}", "직전 28일 대비": "{:+.0%}", "2년 중 백분위": "{:.0%}"}), width="stretch", hide_index=True,
                         column_config={"추이": st.column_config.LineChartColumn("최근 60일")})
            bt = st.cache_data(M.wiki_backtest, ttl=3600)()
            if bt:
                f = lambda x: f"{x[0]:.0%} ({x[1]:.0%}~{x[2]:.0%})"
                st.markdown(f"**추세를 따라가도 되나 (과거 검증, 키워드 {bt['키워드 수']}개·급등 {bt['급등 수']}회)**  \n"
                            f"- 급등한 다음 주가 평소보다 높을 확률 {f(bt['급등 후 평소(28일)보다 높을 확률'])} · 급등이 아닐 때 {f(bt['비급등 시 평소보다 높을 확률'])}  \n"
                            f"- 급등한 다음 주에 **더 오를** 확률 {f(bt['급등 후 더 오를 확률(직전 7일 대비)'])} · 급등이 아닐 때 {f(bt['비급등 시 더 오를 확률'])}")
                st.caption("관심은 평소보다 높게 남지만 대개 꺾입니다. 하루 조회가 한 자릿수인 문서가 많아 잡음이 크고, 위키 조회는 구매가 아닌 관심의 대리 지표입니다.")
        else:
            st.info("`python -m mirror.media wiki` 로 한 번 받아야 합니다.")

    with t3:
        ev = os.path.join(M.ROOT, "data", "media_eval.json")
        if os.path.exists(ev):
            e = json.load(open(ev))
            st.caption(f"laya 정확도(수집 글 {e['n']}건을 예측을 보지 않고 1인이 라벨): 유형 {e['topic']:.0%} (한 가지로만 찍으면 {e['topic_base']:.0%}), "
                       f"감정 {e['sent']:.0%} (한 가지로만 찍으면 {e['sent_base']:.0%}). 확신 {SURE:.0%} 미만은 '확인 필요'로 표시합니다.")
        d = pd.read_sql("select coalesce(published, first_seen) 시각, kind 종류, source 소스, text 내용, url 링크, topic, topic_p, sent, sent_p from items "
                        "where relevant=1 order by 1 desc limit 1500", c)
        d["유형"] = d.topic.where(d.topic_p >= SURE, "확인 필요")
        d["감정"] = d.sent.where(d.sent_p >= SURE, "확인 필요")
        f1, f2 = st.columns(2)
        kinds = f1.multiselect("종류", sorted(d.종류.unique()))
        tops = f2.multiselect("유형", sorted(d.유형.unique()))
        q = st.text_input("단어로 찾기")
        v = d[(d.종류.isin(kinds) | (not kinds)) & (d.유형.isin(tops) | (not tops)) & d.내용.str.contains(q, case=False, regex=False)] if (kinds or tops or q) else d
        st.dataframe(v[["시각", "종류", "소스", "유형", "감정", "내용", "링크"]], width="stretch", hide_index=True, height=460,
                     column_config={"링크": st.column_config.LinkColumn(display_text="열기")})

    with t4:
        st.caption("글이나 링크를 한 줄에 하나씩 넣으세요. 링크는 사이트가 수집을 허용할 때만 제목과 요약을 읽습니다.")
        txt = st.text_area("분석할 내용", height=140)
        up = st.file_uploader("또는 CSV (리뷰·댓글 등)", type="csv")
        col, df = None, None
        if up is not None:
            df = pd.read_csv(up, nrows=3000)
            col = st.selectbox("글이 들어 있는 열", df.columns)
        if st.button("넣고 분류하기", disabled=_running()):
            (n, new), notes = M.add_text(txt.splitlines()) if txt.strip() else ((0, 0), [])
            if df is not None:
                n2, new2 = M.add_text(df[col].dropna().astype(str).tolist(), source="업로드: " + up.name[:40], links=False)[0]
                n, new = n + n2, new + new2
            for x in notes:
                st.warning("읽지 못함: " + x)
            st.success(f"{n}건 중 새 글 {new}건을 넣었습니다. 분류가 끝나면 '글' 탭에 나옵니다.")
            _start()

    with t5:
        dr = pd.read_sql("select * from drafts order by id desc limit 30", c)
        if dr.empty:
            st.info("아직 초안이 없습니다.")
        for r in dr.itertuples():
            with st.expander(f"[{r.status}] {r.title} · {r.ts[5:16]}", expanded=r.status == "승인 대기"):
                st.markdown(r.body)
                b1, b2 = st.columns(2)
                for b, s in ((b1, "승인"), (b2, "반려")):
                    if b.button(s, key=f"{s}{r.id}", disabled=r.status != "승인 대기"):
                        c.execute("update drafts set status=? where id=?", (s, r.id))
                        c.commit()
                        st.rerun()
        st.caption("승인해도 자동으로 게시되지 않습니다. 게시는 담당자가 직접 합니다.")

    with t6:
        runs = pd.read_sql("select source, ts, ok, n, new, note from runs where rowid in (select max(rowid) from runs group by source)", c).set_index("source")
        rows = []
        for n, (kind, _, on, why) in src.items():
            r = runs.loc[n] if n in runs.index else None
            rows.append({"소스": n, "종류": kind, "상태": "꺼짐" if not on else "대기" if r is None else "정상" if r.ok else "실패",
                         "최근 건수": None if r is None else int(r.n), "새 글": None if r is None else int(r.new), "비고": why or ("" if r is None else r.note)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=480)
        st.caption("robots.txt가 금지해 쓰지 않는 곳: 구글 뉴스 RSS, 유튜브 채널 RSS, 네이버 블로그 RSS, 네이트판, MLB파크, 한국소비자원, 애플 앱 리뷰. "
                   "인스티즈는 AI 수집 차단을 명시해 제외. 네이버·쿠팡은 약관 위험 때문에 `.env`에서 직접 켜야만 동작합니다. 작성자 아이디는 저장하지 않습니다.")
