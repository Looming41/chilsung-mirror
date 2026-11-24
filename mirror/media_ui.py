"""미디어 대응실 화면(Streamlit). 수집·분류는 별도 프로세스로 돌려 끝나면 모델 메모리를 돌려준다."""
import json
import os
import re
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import formats as F
from . import media as M

LOG = os.path.join(M.ROOT, "data", "media.log")
SURE = .8                                       # 이 확신 미만의 laya 라벨은 '확인 필요'


def _running():
    p = st.session_state.get("media_proc")
    return p is not None and p.poll() is None


def _start(mod="mirror.media"):
    if not _running():
        with open(LOG, "w") as f:
            st.session_state.media_proc = subprocess.Popen([sys.executable, "-u", "-m", mod, "collect"], cwd=M.ROOT,
                                                           stdout=f, stderr=subprocess.STDOUT)


@st.fragment(run_every=5)
def _status():
    if _running():
        tail = [l for l in open(LOG).read().splitlines() if re.match(r"[가-힣0-9]", l)][-1:]
        st.info("수집·분류 중… " + (tail[0] if tail else "소스에 접속하는 중"))
    elif st.session_state.pop("media_proc", None) is not None:
        st.rerun(scope="app")


def _to_sim(text):
    st.session_state.concept_in = text          # 콜백: 다음 실행에서 시뮬레이터 콘셉트 입력란에 채워짐
    st.session_state.view = "시뮬레이터"


@st.cache_data(show_spinner="포맷 분석 중…")
def _fmt(engine, food, lanes, selected, stamp):
    return F.load(engine, food, list(lanes), selected)


@st.cache_data(show_spinner="한 곳씩 빼며 확인 중…")
def _rank(_d, key):
    return F.robust(_d, F.rank(_d))


def _formats(model):
    """어떤 포맷이 터지나: 많이 쓰이는 포맷과 성과 좋은 포맷을 가른다."""
    c = F.db()
    n, nl, ng, last = c.execute("select count(*), sum(fmt_laya is not null), sum(fmt_llm is not null), max(last_seen) from content").fetchone()
    b = st.columns([1.3, 1.6, 1.6, 1.5])
    b[0].button("▶ 콘텐츠 새로 모으기", on_click=_start, args=("mirror.formats",), disabled=_running(), help="게시판 8곳의 앞 5쪽, 키가 있으면 유튜브까지. 2~4분")
    engine = b[1].radio("분류기", ["llm", "laya"], format_func=lambda x: {"llm": f"LLM ({ng or 0:,}건)", "laya": f"laya ({nl or 0:,}건)"}[x], horizontal=True)
    food = b[2].toggle("식음료 콘텐츠만")
    sel = b[2].toggle("인기글 모음 게시판 포함", True, help="베스트 게시판은 이미 터진 글만 모여 있어 포맷 차이가 작게 보입니다.")
    if b[3].button("남은 글 LLM으로 분류", disabled=not n or ng == n, help="제목이 선택한 LLM 서비스로 전송됩니다."):
        bar = st.progress(0.0)
        try:
            F.classify("llm", model, limit=2000, tick=bar.progress)
        except Exception as e:
            st.warning(f"중간에 멈춤(한 것까지는 저장됨): {str(e)[:150]}")
    lanes = [r[0] for r in c.execute("select distinct lane from content")]
    lanes = st.multiselect("어디 콘텐츠", lanes, lanes, label_visibility="collapsed") if len(lanes) > 1 else lanes
    d = _fmt(engine, food, tuple(lanes), sel, (last, nl, ng, F.version())) if n else pd.DataFrame()
    if len(d) < 20:
        st.info("분석할 콘텐츠가 20건보다 적습니다. 새로 모으거나, 식음료 필터를 끄거나, 아래에서 CSV를 올려 주세요.")
    else:
        r = _rank(d, (engine, food, tuple(lanes), sel, last, nl, ng))
        real = r[(r.포맷 != F.NONE) & (r.개수 >= 5)]
        k = st.columns(4)
        k[0].metric("분석한 콘텐츠", f"{len(d):,}", f"{d.grp.nunique()}개 채널·게시판", delta_color="off")
        k[1].metric("중앙 조회수", f"{d.views.median():,.0f}")
        if len(real):
            win = real[real.판정 == "터지는 포맷"]
            m, h = real.loc[real.개수.idxmax()], (win if len(win) else real).iloc[0]
            k[2].metric("가장 많이 쓰는 포맷", m.포맷, f"{m.개수:.0f}건 · {m.배수:.2f}배", delta_color="off")
            k[3].metric("실제로 터지는 포맷" if len(win) else "배수가 가장 높은 포맷", h.포맷, f"{h.배수:.2f}배 · {h.판정}", delta_color="off")
        f = st.pills("포맷을 누르면 산점도에서 강조됩니다", real.포맷.tolist(), default=real.포맷.iloc[0] if len(real) else None)
        x = d[d.likes.fillna(0) > 0]
        left, right = st.columns([1.15, 1])
        with left:
            if len(x) >= 10:
                fig = go.Figure()
                for nm, g, col, op in (("그 외", x[x.fmt != f], "#b9b9b9", .45), (f or "", x[x.fmt == f], "#E60012", .9)):
                    fig.add_scatter(x=g.views, y=g.likes, mode="markers", name=nm, text=g.title.str[:60], hovertemplate="%{text}<br>조회 %{x:,} · 추천 %{y:,}<extra></extra>",
                                    marker=dict(color=col, opacity=op, size=8))
                fig.update_xaxes(type="log", title="조회수").update_yaxes(type="log", title="추천·좋아요")
                fig.update_layout(height=430, margin=dict(t=10, b=0), legend=dict(orientation="h"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("추천 수가 있는 콘텐츠가 적어 산점도를 그리지 않습니다.")
        with right:
            q = real.sort_values("배수")
            col = q.판정.map({"터지는 포맷": "#E60012", "안 터지는 포맷": "#4a6fa5"}).fillna("#999999")
            fig = go.Figure(go.Scatter(x=q.배수, y=q.포맷, mode="markers", marker=dict(size=11, color=col),
                                       error_x=dict(type="data", symmetric=False, array=q.상한 - q.배수, arrayminus=q.배수 - q.하한, color="#bbb")))
            fig.add_vline(x=1, line_dash="dot", line_color="#888")
            fig.update_layout(height=430, margin=dict(t=10, b=0), xaxis_title="배수 (비슷한 때 같은 곳에 올라온 글 대비, 1 = 평소)")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(r[["포맷", "판정", "의존한 곳", "개수", "사용 비중", "중앙 조회수", "배수", "하한", "상한", "2배 이상 터진 비율", "q"]].style.format(
            {"사용 비중": "{:.0%}", "중앙 조회수": "{:,.0f}", "배수": "{:.2f}", "하한": "{:.2f}", "상한": "{:.2f}", "2배 이상 터진 비율": "{:.0%}", "q": "{:.3f}"}, na_rep="—"), width="stretch", hide_index=True)
        ev = os.path.join(M.ROOT, "data", "format_eval.json")
        e = json.load(open(ev)) if os.path.exists(ev) else None
        st.caption("배수: 같은 게시판·채널에서 비슷한 때 올라온 글들의 조회수 중앙값 대비. 빨강/파랑은 나머지와의 차이가 우연이 아니고(BH 보정 q<0.05) 게시판·채널을 하나씩 빼도 유지될 때만. 5개 미만은 판단하지 않습니다. "
                   + (f"분류 정확도(1인 라벨 {e['n']}건): LLM {e['llm']:.0%}, laya {e['laya']:.0%}. " if e else "") + "관찰 자료라 '이 포맷으로 하면 터진다'는 인과는 아닙니다.")
        if f:
            g = d[d.fmt == f].sort_values("mult", ascending=False)
            st.markdown(f"**{f}** — 배수가 높은 콘텐츠")
            st.dataframe(g[["mult", "title", "grp", "views", "likes", "comments", "url"]].head(30).rename(columns={"mult": "배수", "title": "제목", "grp": "게시판·채널", "views": "조회", "likes": "추천", "comments": "댓글", "url": "링크"})
                         .style.format({"배수": "{:.1f}", "조회": "{:,.0f}", "추천": "{:,.0f}", "댓글": "{:,.0f}"}, na_rep="—"), width="stretch", hide_index=True, height=260,
                         column_config={"링크": st.column_config.LinkColumn(display_text="열기")})
            p1, p2 = st.columns([3, 1])
            prod = p1.text_input("이 포맷으로 만들 제품", st.session_state.get("concept_in", "")[:80])
            if p2.button("훅 아이디어 5개"):
                try:
                    with st.spinner("작성 중…"):
                        st.session_state.fmt_ideas = F.ideas(f, g.title.head(8).tolist(), prod, model)
                except Exception as e2:
                    st.error(f"실패: {str(e2)[:150]}")
            if st.session_state.get("fmt_ideas"):
                st.markdown(st.session_state.fmt_ideas)
        hk = F.hooks(d)
        if len(hk):
            with st.expander("제목 표면 특징별 차이 (분류기를 쓰지 않은 교차 확인)"):
                st.dataframe(hk.style.format({"있을 때 배수": "{:.2f}", "없을 때 배수": "{:.2f}", "p": "{:.3f}", "q": "{:.3f}"}), width="stretch", hide_index=True)
    with st.expander("포맷 목록 고치기"):
        ed = st.data_editor(pd.DataFrame(list(F.formats().items()), columns=["포맷", "한 줄 설명"]), num_rows="dynamic", width="stretch", hide_index=True)
        if st.button("저장 (다음 분류부터 적용, 기존 분류는 다시 함)"):
            new = {str(a).strip(): str(b).strip() for a, b in ed.dropna().values if str(a).strip()}
            if 2 <= len(new) <= 30 and F.NONE not in new:
                json.dump(new, open(F.FILE, "w"), ensure_ascii=False, indent=1)
                st.rerun()
            st.error(f"포맷은 2~30개여야 하고 '{F.NONE}'은 자동으로 들어갑니다.")
    with st.expander("릴스·틱톡 등 CSV 올리기"):
        up = st.file_uploader("제목(또는 캡션·대본)과 조회수가 있는 CSV", type="csv", key="fmt_csv")
        if up is not None:
            df = pd.read_csv(up, nrows=5000)
            cs, no = list(df.columns), ["(없음)", *df.columns]
            u = st.columns(5)
            a = [u[0].selectbox("제목", cs), u[1].selectbox("조회수", cs), u[2].selectbox("좋아요", no), u[3].selectbox("계정", no), u[4].selectbox("게시일", no)]
            if st.button("넣기"):
                nn = [None if z == "(없음)" else z for z in a[2:]]
                st.success("%d건 중 새 콘텐츠 %d건" % F.from_csv(df, a[0], a[1], *nn, name="업로드: " + up.name[:30]))


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
    t0, t1, t2, t3, t4, t5, t6 = st.tabs(["🏆 포맷", "🚨 위험", "📈 추세", "📰 글", "✍️ 지정 분석", "📝 대응 초안", "🔌 소스"])

    with t0:
        _formats(model)

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
            f = e.get("feel", {})
            st.caption(f"정확도(예측을 보지 않고 1인이 라벨) — 감정은 SNS 글 {f.get('n')}건: Level {f.get('level', 0):.0%}, laya {f.get('laya', 0):.0%}, 전부 '중립'이라고 하면 {f.get('base', 0):.0%}. "
                       f"부정으로 판정한 글의 {f.get('neg_prec', 0):.0%}만 실제 부정이니 묶음으로 보고 한 건씩은 확인하세요. 유형(laya)은 {e['n']}건에서 {e['topic']:.0%}. 확신 {SURE:.0%} 미만 유형은 '확인 필요'. "
                       "Level은 제품 평가가 아니라 '글쓴이의 감정'을 읽습니다.")
        d = pd.read_sql("select coalesce(published, first_seen) 시각, kind 종류, source 소스, text 내용, url 링크, topic, topic_p, feel, emo from items "
                        "where relevant=1 order by 1 desc limit 1500", c)
        d["유형"] = d.topic.where(d.topic_p >= SURE, "확인 필요")
        d["감정"] = d.feel.fillna("미판정")
        d["세부 감정"] = [", ".join(l for l, _ in json.loads(x) if l != "neutral") if isinstance(x, str) else "" for x in d.emo]
        f1, f2 = st.columns(2)
        kinds = f1.multiselect("종류", sorted(d.종류.unique()))
        tops = f2.multiselect("유형", sorted(d.유형.unique()))
        q = st.text_input("단어로 찾기")
        v = d[(d.종류.isin(kinds) | (not kinds)) & (d.유형.isin(tops) | (not tops)) & d.내용.str.contains(q, case=False, regex=False)] if (kinds or tops or q) else d
        from collections import Counter
        cnt = Counter(l for x in v["세부 감정"] for l in x.split(", ") if l)
        if len(v) and cnt:
            m = st.columns(3)
            for box, k in zip(m, ("긍정", "중립", "부정")):
                box.metric(k, f"{(v.감정 == k).mean():.0%}", f"{(v.감정 == k).sum()}건", delta_color="off")
            st.bar_chart(pd.Series(dict(cnt.most_common(12)), name="글 수").sort_values(), horizontal=True, height=300)
        st.dataframe(v[["시각", "종류", "소스", "유형", "감정", "세부 감정", "내용", "링크"]], width="stretch", hide_index=True, height=460,
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
