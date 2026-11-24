"""디지털 트윈 실험실 화면: 페르소나 생성 방식 비교 + 에이전트 엔진 예측 비교."""
import json
import os
import re
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import engines as E
from . import people as P

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


def render():
    st.title("🧬 디지털 트윈 실험실")
    t1, t2, t3 = st.tabs(["① 페르소나 생성 비교", "② 엔진 예측 비교", "③ 인물 살펴보기"])
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
