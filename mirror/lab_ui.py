"""디지털 트윈 실험실 화면: 페르소나 생성 방식 비교 + 에이전트 엔진 예측 비교."""
import itertools
import json
import os
import re
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import engines as E
from . import multiverse as U
from . import people as P
from . import trust as T

BENCH = os.path.join(P.ROOT, "data", "people_bench.json")
SUM, LOG = os.path.join(E.RUNS, "summary.json"), os.path.join(P.ROOT, "data", "engines.log")
DESC = {"nemotron": "기존 방식 (Nemotron + 실기록 붙이기)", "record": "실응답자 재표집 (PopulationSim 방식)", "record+레이킹": "실응답자 + 인구 보정", "cart": "순차 나무 합성 (R synthpop 방식)",
        "bayes": "베이지안 네트워크 (Doppelganger 방식)", "llm(161명)": "LLM이 지어낸 인물 (TinyTroupe 팩토리 방식)", "cart(161명)": "위와 같은 인원의 cart (공정 비교용)", "독립(하한)": "변수를 따로따로 뽑음 (최악 기준)"}


def _busy():
    p = st.session_state.get("lab_proc")
    return p is not None and p.poll() is None


def _run(n, engines):
    os.makedirs(E.RUNS, exist_ok=True)
    with open(LOG, "w") as f:
        st.session_state.lab_proc = subprocess.Popen([sys.executable, "-u", "-m", "mirror.engines", "run", str(n), ",".join(engines)], cwd=P.ROOT, stdout=f, stderr=subprocess.STDOUT,
                                                     env={**os.environ, "LAB_CONCEPT": st.session_state.get("concept_in", "")})


@st.fragment(run_every=5)
def _status():
    if _busy():
        done = re.findall(r"'엔진': '(\w+)'", open(LOG).read())
        st.info("실험 중… 끝난 엔진: " + (", ".join(done) or "아직 없음"))
    elif st.session_state.pop("lab_proc", None) is not None:
        st.rerun(scope="app")


def _multiverse():
    st.markdown("##### 1. 누구를 대상으로 할까요  (같은 칸 안은 '또는', 칸끼리는 '그리고')")
    f, cs = {}, st.columns(4)
    for i, (n, (c, v)) in enumerate(U.CATS.items()):
        x = cs[i % 4].multiselect(n, v, key="mv_" + c, placeholder="전체")
        if x:
            f[c] = x
    _, info = U.select(f)
    st.markdown("##### 2. 무엇을 비교할까요")
    z = st.columns(6)
    pr = z[0].multiselect("가격(원)", [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000], [1500, 2200])
    ze = z[1].multiselect("당", ["제로", "일반"], ["제로"])
    fl = z[2].multiselect("맛", ["새로운 맛", "익숙한 맛"], ["새로운 맛"])
    pk = z[3].multiselect("패키지", ["비주얼(숏폼형)", "클래식"], ["비주얼(숏폼형)"])
    cp = z[4].multiselect("문구", ["재미", "건강"], ["재미"])
    nf = z[5].multiselect("인플루언서", ["협업", "없음"], ["협업", "없음"])
    concept = st.text_input("제품 콘셉트", st.session_state.get("concept_in", "자두맛 탄산음료. 500ml 페트."), key="mv_concept")
    scs = {f"{a:,}원·{b}·{c[:2]}·{d[:2]}·{e}·{g}": dict(price=a, zero=int(b == "제로"), flavor=c, package=d, copy=e, influencer=g == "협업") for a, b, c, d, e, g in itertools.product(pr, ze, fl, pk, cp, nf)}
    nd = U.need(max(1, len(scs)))
    m = st.columns(5)
    m[0].metric("대상 인구", f"{info['인구'] / 1e4:,.0f}만 명")
    m[1].metric("실제 응답자", f"{info['실응답자']:,}명", f"유효표본 {info['유효표본']:,.0f}", delta_color="off")
    m[2].metric(f"실데이터 근거 {info['신호']}", f"±{100 * info['오차']:.1f}%p" if info["유효표본"] > 0 else "없음")
    m[3].metric("비슷한 사람에게서 빌림", f"{info['빌린 비율']:.0%}")
    m[4].metric("권장 시뮬레이션 인원", f"{nd['권장']:,}명", f"시나리오 {len(scs)}개 비교 기준", delta_color="off")
    if info["신호"] != "🟢" and f:
        nm = {c: n for n, (c, _) in U.CATS.items()}
        st.warning("고른 집단의 실제 응답자가 적어 범위가 넓게 나옵니다. " + " · ".join(f"'{nm[c]}' 조건을 빼면 유효표본 {e:,.0f}" for c, e in U.relax(f)[:2]))
    with st.expander("만약에 · 세계 설정"):
        q = st.columns(4)
        worlds = q[0].select_slider("세계 수", [50, 100, 200, 300, 500], 200)
        n = q[1].select_slider("세계당 사람 수", [600, 1000, 1500, 2500], 600, help="권장 인원은 모든 세계를 합친 값으로 이미 넘습니다. 세계당 인원은 입소문 사회의 크기입니다.")
        spread = q[2].slider("상상의 폭", 0.0, 1.5, 1.0, .1, help="1은 기본 가정. 0이면 조건은 같고 사람과 운만 달라집니다.")
        days = q[3].select_slider("기간(일)", [14, 28, 56, 84], 28)
        w = st.columns(3)
        fix = {}
        for box, (k, lo, hi, dv) in zip(w, (("광고 도달", .002, .05, .01), ("유행 지속", .3, .85, .6), ("SNS 전파력", .8, 2.2, 1.4))):
            if box.checkbox(f"'{k}' 고정"):
                fix[k] = box.slider(k, lo, hi, dv)
        st.caption("세계마다 달라지는 조건의 폭은 사람이 정한 가정입니다: " + ", ".join(U.PRIOR))
    if not 1 <= len(scs) <= 24:
        st.error("시나리오는 1~24개로 골라 주세요.")
        return
    if st.button(f"🌌 {worlds}개 세계 열기 (연 {worlds * n:,}명)", type="primary", disabled=info["유효표본"] == 0 and not f):
        bar = st.progress(0.0)
        W, i2 = U.run(f, scs, concept, worlds, n, days, spread, fix, tick=bar.progress)
        st.session_state.mv = dict(W=W, info=i2, scs=scs, f=f, concept=concept, n=n)
        bar.empty()
    r = st.session_state.get("mv")
    if not r:
        return
    W, t = r["W"], U.summary(r["W"])
    top = t.iloc[0]
    st.markdown(f"##### 3. 결론 — {len(W)}개 세계 중 **{top['1등 세계']:.0%}** (±{100 * top['1등 오차']:.0f}%p)에서 **{top.시나리오}** 가 1등")
    a, b = st.columns([1, 1.2])
    fig = go.Figure(go.Bar(x=t["1등 세계"], y=t.시나리오, orientation="h", marker_color=["#E60012"] + ["#c9c9c9"] * (len(t) - 1), error_x=dict(type="data", array=t["1등 오차"], color="#777")))
    fig.update_layout(height=90 + 34 * len(t), margin=dict(t=10, b=0), xaxis_tickformat=".0%", xaxis_title="1등을 한 세계의 비율")
    fig.update_yaxes(autorange="reversed")
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=t.구매율, y=t.시나리오, mode="markers", marker=dict(size=11, color="#E60012"), error_x=dict(type="data", symmetric=False, array=t["구매 상한"] - t.구매율, arrayminus=t.구매율 - t["구매 하한"], color="#bbb")))
    fig.update_layout(height=90 + 34 * len(t), margin=dict(t=10, b=0), xaxis_tickformat=".0%", xaxis_title="구매율 (점=가운데 세계, 막대=90% 세계)")
    fig.update_yaxes(autorange="reversed")
    b.plotly_chart(fig, width="stretch")
    pc = ["1등 세계", "구매율", "구매 하한", "구매 상한", "의향", "의향 하한", "의향 상한", "나쁜 세계 손실", "기준 대비", "기준 대비 하한", "기준 대비 상한"]
    st.dataframe(t[["시나리오", *pc]].style.format({c: "{:.0%}" if c == "1등 세계" else "{:.1%}" for c in pc}), width="stretch", hide_index=True)
    st.caption(f"기준은 첫 시나리오({list(r['scs'])[0]}). '1등 세계'는 우리가 상상한 세계들 가운데의 비율이지 현실에서 이길 확률이 아닙니다. 구매율·의향의 절대값은 검증되지 않았으니 범위와 순위로 보세요. "
               f"실응답자 {r['info']['실응답자']:,}명이 근거이고 {r['info']['빌린 비율']:.0%}는 비슷한 사람에게서 빌렸습니다.")
    st.markdown("##### 4. 우주 지도 — 점 하나가 세계 하나")
    B = W[[c for c in W if c.startswith("구매|")]]
    win = B.idxmax(axis=1).str[3:]
    ps = [c for c in U.PRIOR if c in W]
    g = st.columns(2)
    xa, ya = g[0].selectbox("가로", ps, ps.index("광고 도달")), g[1].selectbox("세로", ps, ps.index("유행 지속"))
    fig = go.Figure()
    pal = ["#E60012", "#2f6db5", "#3a9d5d", "#e0a100", "#7a4fb0", "#888"]
    for i, k in enumerate(t.시나리오):
        q = win == k
        if q.any():
            fig.add_scatter(x=W[xa][q], y=W[ya][q], mode="markers", name=f"{k} ({q.sum()})", marker=dict(size=9, color=pal[min(i, 5)], opacity=.75), text=[f"세계 {v}" for v in W.세계[q]], hovertemplate="%{text}<extra></extra>")
    fig.update_layout(height=430, xaxis_title=xa, yaxis_title=ya, legend=dict(orientation="h", y=-.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    if len(r["scs"]) > 1:
        st.markdown("##### 5. 무엇이 승패를 가르나")
        dr, k = U.drivers(W, top.시나리오)
        st.dataframe(dr.style.format({c: "{:.2f}" for c in dr.columns[1:]} | {"이긴 세계 평균": "{:.3f}", "진 세계 평균": "{:.3f}"}), width="stretch", hide_index=True)
        lost = dr[dr["차이(표준편차)"].abs() > .3].head(3)
        if 0 < k < len(W) and len(lost):
            st.info(f"1등 시나리오가 **지는 {len(W) - k}개 세계**의 특징: " + ", ".join(f"{x['조건']}이(가) {'높음' if x['차이(표준편차)'] > 0 else '낮음'}" for _, x in lost.iterrows()))
        yes, no = U.representative(W, top.시나리오)
        st.caption(f"대표 세계: 이긴 세계 {yes}번" + (f", 진 세계 {no}번" if no is not None else "") + ". 이 세계들의 이유는 '엔진 예측 비교' 탭에서 LLM에게 들을 수 있습니다.")


def _trust():
    if not os.path.exists(T.FILE):
        st.info("`python -m mirror.trust bench` 를 한 번 실행하세요(약 1분).")
        return
    j = json.load(open(T.FILE))
    s = j["요약"]
    m = st.columns(4)
    m[0].metric("합성 패널의 오차", f"{s['합성 평균']:.3f}", "0이면 완전 일치", delta_color="off")
    m[1].metric("사람 조사끼리의 차이 (2024↔2025)", f"{s['사람 평균']:.3f}")
    m[2].metric("같은 조사를 반으로 나눈 우연 차이", f"{s['우연 평균']:.3f}")
    m[3].metric("신뢰 등급", " · ".join(f"{k} {v}" for k, v in sorted(s["등급"].items())))
    st.markdown("**① 재현 시험.** 2023~24년 조사로 만든 합성 패널이 2025년 조사의 12개 집단별 응답을 얼마나 재현하나 (33문항).")
    st.warning(f"합성 패널은 사람 조사끼리의 차이와 거의 같은 수준이지만, **쉬운 시험**입니다. 2025년 전체 평균을 안다면 모든 집단에 그 값을 복사해도 오차가 {s['모두평균 평균']:.3f}입니다(집단 차이가 작은 문항이 많음). 이미 물어본 질문의 재현이지 새 질문 예측이 아닙니다.")
    r = pd.DataFrame(j["items"])
    st.dataframe(r.style.format({c: "{:.3f}" for c in r.columns if c not in ("문항", "등급")}), width="stretch", hide_index=True, height=280)
    st.markdown("**② 사람 + 모델.** 새 질문을 실제 몇 명에게만 묻고 모델을 더했을 때, 12개 집단 평균의 오차(1~5점 척도, 낮을수록 좋음).")
    e = pd.DataFrame(j["mix"]).set_index("실제 응답")[["실제 응답만", "전체 평균만", "+인구통계 모델", "+기록 모델"]]
    big = pd.DataFrame(j["mix_big"]).set_index("실제 응답")[["실제 응답만", "전체 평균만", "+인구통계 모델", "+기록 모델"]]
    a, b = st.columns(2)
    a.markdown("전체 33문항")
    a.dataframe(e.style.format("{:.3f}"), width="stretch")
    b.markdown("집단 차이가 큰 11문항")
    b.dataframe(big.style.format("{:.3f}"), width="stretch")
    st.caption("'기록 모델'은 사람들이 다른 질문에 한 답으로 새 질문을 예측하고 실제 응답으로 수준을 바로잡습니다. 가장 단순한 '전체 평균만'보다 나아야 의미가 있습니다.")


def render():
    st.title("🧬 디지털 트윈 실험실")
    t0, tv, t1, t2, t3 = st.tabs(["🌌 다중우주", "✅ 신뢰 검증", "페르소나 생성 비교", "엔진 예측 비교", "인물 살펴보기"])
    with t0:
        _multiverse()
    with tv:
        _trust()
    with t1:
        st.markdown("같은 실제 응답자(KREI 2025, 성인 5,874명) **절반으로 사람을 만들고, 보지 않은 나머지 절반으로 채점**합니다. 오차는 낮을수록, 상관·기울기는 1에 가까울수록 좋습니다.")
        if os.path.exists(BENCH):
            b = pd.DataFrame(json.load(open(BENCH)))
            b.insert(1, "설명", b.방식.map(DESC))
            b = b.sort_values("두 변수 조합 오차")
            st.dataframe(b.style.format({c: "{:.3f}" for c in b.columns if c not in ("방식", "설명", "실존 응답자와 동일", "서로 다른 인물")} | {"실존 응답자와 동일": "{:.1%}", "서로 다른 인물": "{:.0%}"}, na_rep="—"),
                         width="stretch", hide_index=True)
            f = go.Figure(go.Bar(x=b["두 변수 조합 오차"], y=b.방식, orientation="h", marker_color=["#E60012" if m == "cart" else "#bbb" for m in b.방식]))
            f.update_layout(height=300, margin=dict(t=10, b=0), xaxis_title="두 변수 조합 오차 (낮을수록 실제 인구와 가까움)")
            f.update_yaxes(autorange="reversed")
            st.plotly_chart(f, width="stretch")
            st.caption("'실존 응답자와 동일'은 합성 인물이 실제 응답자 한 명과 모든 값이 같은 비율입니다. record 계열은 실존 응답을 그대로 쓰므로 100%이고 외부 LLM에 보내기 부적절합니다. "
                       "같은 조사 안에서 나눈 검증이라 다른 조사로 확인한 것은 아니며, 인구가 닮은 것과 신제품 반응을 맞히는 것은 다릅니다.")
        else:
            st.info("`python -m mirror.people bench` 를 한 번 실행하세요.")
    with t2:
        have = [e for e in ("ml", "direct", "concordia", "tinytroupe", "oasis") if e in ("ml", "direct") or os.path.exists(os.path.join(E.ENV, E.VENV[e], "bin", "python"))]
        c = st.columns([1, 2, 1])
        n = c[0].slider("인원", 4, 40, 12, 4)
        pick = c[1].multiselect("엔진", have, have)
        if c[2].button("▶ 실험", type="primary", disabled=_busy() or not pick):
            _run(n, pick)
        _status()
        st.caption("콘셉트는 왼쪽 '제품 콘셉트'를 씁니다. 합성 인물의 성향과 콘셉트가 선택된 LLM 서비스로 전송됩니다. 무료 등급은 분당 13회라 12명·4엔진에 10분 안팎이 걸립니다.")
        if os.path.exists(SUM):
            s = json.load(open(SUM))
            t = pd.DataFrame(s["표"])
            st.markdown(f"**{s['콘셉트']}** · 합성 인물 {s['인원']}명 ({s['인물 생성']}) · {s['모델']} · {s['시각'][5:16].replace('T', ' ')}")
            fmt = {k: f for k, f in {"구매의향 평균": "{:.0f}", "구매의향 표준편차": "{:.0f}", "60점 이상": "{:.0%}", "기준질문 분포 오차": "{:.2f}", "기준질문 순위상관": "{:.2f}", "월2회+ 예측": "{:.0%}", "월2회+ 인물 실제": "{:.0%}", "월2회+ 실제 인구": "{:.0%}"}.items() if k in t}
            st.dataframe(t.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
            if s["엔진 간 구매의향 범위"]:
                lo, hi = s["엔진 간 구매의향 범위"]
                st.metric("엔진 간 구매의향 범위 (불확실성)", f"{lo:.0f} ~ {hi:.0f}점")
            st.caption("기준질문: 인물 기억에서 제로음료 빈도를 가리고 물어 실제 한국 성인 분포와 비교한 값입니다(분포 오차 0이 완벽, 순위상관 1이 완벽). "
                       "ml은 LLM 없이 실제 응답자로 학습한 통계 모델(비교 기준)입니다. 구매의향 숫자 자체는 검증된 적이 없습니다. 엔진끼리 크게 다르면 그만큼 믿을 수 없다는 뜻입니다.")
            e = st.selectbox("답변 보기", list(s["답변"]))
            who = {i: r for i, r in enumerate(s["인물"])}
            rows = [{"인물": f"{who[a['id']]['age']} {who[a['id']]['sex']} {who[a['id']]['occ']}", "실제(가린 값)": who[a["id"]]["zero"], "답변": a["text"]} for a in s["답변"][e] if a["id"] in who]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
            if s["사건"].get(e):
                with st.expander("엔진 안에서 일어난 일"):
                    st.dataframe(pd.DataFrame(s["사건"][e]), width="stretch", hide_index=True)
    with t3:
        m = st.radio("생성 방식", ["cart", "bayes", "nemotron"], horizontal=True, format_func=lambda x: DESC[x])
        u = st.toggle("실제 사용군만 (제로 월 2회 이상 또는 탄산 1순위)")
        d = st.cache_data(P.build, show_spinner="인물 만드는 중…")(m, 2000)
        d = P.users(d) if u else d
        st.caption(f"{len(d):,}명. 기억 문장은 기록에 있는 사실만 씁니다.")
        i = st.number_input("번호", 0, len(d) - 1, 0)
        st.markdown("\n".join("- " + x for x in P.memory(d.iloc[int(i)])))
        st.dataframe(d.head(200), width="stretch", hide_index=True, height=260)
