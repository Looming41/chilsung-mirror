"""신제품 검증 화면: 주어진 3개 컨셉의 결론 · 보정 과정 · 만약에 · 대상별 반응 · 데이터 장부 · 검토 이력."""
import datetime as dt
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import given as G
from . import multiverse as U
from . import people as P

DB = os.path.join(P.ROOT, "data", "history.db")
COL = dict(zip(G.CON, ("#E60012", "#2f6db5", "#e0a100")))


def db():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("create table if not exists review(id integer primary key, ts text, memo text, assume text, result text)")
    return c


def save(memo, assume, t):
    with db() as c:
        c.execute("insert into review(ts, memo, assume, result) values(?,?,?,?)", (dt.datetime.now().isoformat(timespec="seconds"), memo[:500], json.dumps(assume, ensure_ascii=False), t.to_json(force_ascii=False, orient="records")))


def history(word=""):
    with db() as c:
        rs = c.execute("select ts, memo, assume, result from review order by id desc").fetchall()
    out = []
    for ts, memo, a, r in rs:
        r = json.loads(r)
        out.append({"시각": ts[5:16].replace("T", " "), "메모": memo, "가정": ", ".join(f"{k.split('|')[-1]} {k.split('|')[0]} {v[0]:.0%}~{v[1]:.0%}" for k, v in json.loads(a).items()),
                    "구매경험률": " · ".join(f"{x['컨셉']} {x['구매경험률']:.1%}" for x in r), "매출(억)": " · ".join(f"{x['컨셉']} {x['소매 매출(억)']:.0f}" for x in r)})
    t = pd.DataFrame(out)
    return t[t.apply(lambda x: word in " ".join(map(str, x)), axis=1)] if word and len(t) else t


@st.cache_data(show_spinner="계산 중…")
def _run(assume, raw, worlds=1500):
    return G.forecast(worlds, assume=dict(assume), raw=raw)


@st.cache_data(show_spinner="계산 중…")
def _whatif():
    return G.whatif()


def render():
    st.title("신제품 검증")
    if not G.have():
        st.info("data/given 폴더에 주어진 파일(3-1~3-3)을 넣어 주세요.")
        return
    st.caption("대회 제공 가상 데이터 기준 · 구매경험률 = 보정 구매의향 × 살 수 있는 점포 비율 × 전환계수")
    with st.expander("가정 바꾸기"):
        assume = {}
        for k, (lo, hi, why) in G.ASSUME.items():
            assume[k] = st.slider(k.replace("|", " · "), 0.1, 1.0, (lo, hi), .05, help=why, key="gv_" + k)
        raw = st.toggle("과거 컨셉테스트도 원본 의향이었다고 보고 계산", False, help="과거 조사가 어떤 정제를 거쳤는지 자료에 없습니다. 켜면 정제 전 의향을 씁니다.")
    F = _run(tuple(assume.items()), raw)
    t = G.summary(F)
    top = t.sort_values("경험률 1등", ascending=False).iloc[0]
    st.markdown(f"#### {top.컨셉}의 구매경험률이 가장 높습니다 — {len(F) // 3:,}번 계산 중 **{top['경험률 1등']:.0%}** 에서 1등")
    a, b = st.columns(2)
    for box, (m, lo, hi, ttl, f) in zip((a, b), (("구매경험률", "하한", "상한", "출시 12개월 구매경험률 (세로선=90% 범위)", ".1%"), ("소매 매출(억)", "매출 하한", "매출 상한", "소매 매출, 억 원 (세로선=90% 범위)", ".0f"))):
        fig = go.Figure(go.Bar(x=t.컨셉, y=t[m], marker_color=[COL[c] for c in t.컨셉], error_y=dict(type="data", symmetric=False, array=t[hi] - t[m], arrayminus=t[m] - t[lo], color="#555")))
        fig.update_layout(height=300, margin=dict(t=30, b=0), title=dict(text=ttl, font_size=13), yaxis_tickformat=f)
        box.plotly_chart(fig, width="stretch")
    pc = ["보정 의향", "살 수 있는 비율", "구매경험률", "하한", "상한", "경험률 1등", "매출 1등"]
    st.dataframe(t.style.format({c: "{:.1%}" for c in pc} | {c: "{:.0f}" for c in ("소매 매출(억)", "매출 하한", "매출 상한", "출고 매출(억)")}), width="stretch", hide_index=True)
    st.caption("'1등'은 조건을 바꿔 계산한 횟수 중의 비율이지 성공 확률이 아닙니다. 매출은 1인당 구매 수량을 **밀키스 딸기·밀키스 제로 두 제품**에서 빌려 와 모든 컨셉에 적용했으므로 순서만 참고하고, 특히 쥬시모먼트 매출은 과대일 수 있습니다. 카니발은 빼지 않은 총매출입니다. "
               "대조: 밀키스 딸기 최고 85억 · 밀키스 수박 70억 · 프레시원 22억 · 델몬트 프리미엄 포도 31억 (제품자산DB).")
    s1, s2, s3, s4, s5 = st.tabs(["근거", "조건 바꾸기", "구매층", "데이터 문제", "검토 이력"])
    with s1:
        _why()
    with s2:
        w = _whatif()
        st.dataframe(w.style.format({"지금": "{:.1%}", "바꾸면": "{:.1%}"}), width="stretch", hide_index=True)
        st.caption("기본 가정에서 하나만 바꾼 결과입니다.")
        ev, tp = G.elasticity()
        c = st.columns(2)
        c[0].markdown("**행사 때 판매 배수** (같은 제품·같은 달의 행사 없는 곳 대비)")
        c[0].dataframe(pd.DataFrame({"행사": ev.keys(), "배수": ev.values()}).style.format({"배수": "{:.2f}배"}), hide_index=True)
        nm = G.pos().drop_duplicates("제품코드").set_index("제품코드").제품명
        c[1].markdown("**기온 1℃ 오를 때 판매 변화** (행사 없는 달)")
        c[1].dataframe(pd.DataFrame({"제품": [nm[k] for k in tp], "변화": list(tp.values())}).sort_values("변화", ascending=False).style.format({"변화": "{:+.1%}"}), hide_index=True)
        st.caption("여름에 강한 것은 컵얼음커피와 밀키스류이고, 캔커피는 더울수록 줄어듭니다. 프리즈의 경쟁 상대는 캔커피보다 컵얼음커피에 가깝습니다(추론).")
    with s3:
        _who()
    with s4:
        st.dataframe(G.audit(), width="stretch", hide_index=True, height=560)
    with s5:
        c = st.columns([3, 1])
        memo = c[0].text_input("검토 메모", placeholder="예: 프리즈 냉동 입점 협의 전 기준")
        if c[1].button("지금 결과 저장"):
            save(memo, assume, t)
            st.success("저장했습니다.")
        word = st.text_input("이력·자산 찾기", placeholder="예: 수박, 냉동, 프리미엄")
        h = history(word)
        st.markdown("**우리 검토 이력**")
        st.dataframe(h, width="stretch", hide_index=True) if len(h) else st.caption("아직 없습니다.")
        st.markdown("**제품자산 DB (운영·단종·개발중단)**")
        d = G.sheet(G.A, "4_제품자산DB").astype(str).replace("nan", "")
        st.dataframe(d[d.apply(lambda x: word in " ".join(x), axis=1)] if word else d, width="stretch", hide_index=True)


def _why():
    s, i = G.stages()
    st.markdown(f"**① 구매의향(TOP2)을 보정하면 순위가 바뀝니다.** 응답자 304명 → 정제 후 {i['응답자']}명, 유효표본 {i['유효표본']:.0f}명")
    st.dataframe(s.style.format("{:.1%}"), width="stretch")
    st.caption(f"오차는 유효표본 기준 약 ±{100 * 1.96 * np.sqrt(.24 / i['유효표본']):.0f}%p입니다. 가중 뒤 세 컨셉의 의향 차이는 대부분 이 안에 있습니다. 쥬시모먼트는 의향이 가장 높지만 3,000원을 낼 수 있다는 사람은 {s.loc['쥬시모먼트', '제시 가격을 낼 수 있음']:.0%}뿐입니다.")
    v, er = G.validate()
    st.markdown(f"**② 과거 출시 실적으로 환산합니다.** 하나씩 빼고 맞힌 평균 오차: 의향만 쓰면 **{er['의향만 환산']:.0%}**, 취급점포율까지 쓰면 **{er['의향 × 점포율 × k']:.0%}**")
    st.dataframe(v.style.format({c: "{:.1%}" for c in v.columns[1:]}), width="stretch", hide_index=True)
    st.caption("제품이 5개뿐이고 모두 가상 데이터라 '이 식이 현실에서 맞는다'는 증거는 아닙니다.")
    m = G.market()
    st.markdown("**③ 살 수 있는 비율.** " + " · ".join(f"{c} 이론상 {G.reach(c):.0%}" for c in G.CON[1:]) + " (채널 비중 × 냉동·냉장 인프라) × 입점률 가정. 밀키스 화채는 과거 밀키스 취급점포율을 그대로 씁니다.")
    st.markdown("**④ 1인당 구매 수량** " + " · ".join(f"{k} {x:.1f}개" for k, x in m["1인당 수량"].items()) + f" (19~59세 {m['인구'] / 1e4:,.0f}만 명 기준, 1+1 포함)")


def _who():
    st.markdown("패널 273명의 답으로 배운 모델을 **KREI 실응답자 기반 가상 소비자**(19~59세)에 옮겨, 고른 집단의 의향을 봅니다.")
    f, cs = {}, st.columns(4)
    for k, (n, (c, v)) in enumerate([x for x in U.CATS.items() if x[0] in ("연령대", "성별", "권역", "가구", "새로움 추구", "건강 관심", "가격 비교", "제로음료 빈도")]):
        x = cs[k % 4].multiselect(n, [o for o in v if o not in ("10대", "60대", "70대")], key="gw_" + c, placeholder="전체")
        if x:
            f[c] = x
    w, info = U.select(f)
    p = G.personas(w)
    if not len(p):
        st.warning("고른 조건에 19~59세가 없습니다.")
        return
    base, au = G.personas(), G.persona_model()
    r = pd.DataFrame([{"컨셉": c, "고른 집단": np.average(p[c], weights=p.w), "19~59세 전체": np.average(base[c], weights=base.w), "패널 실측(가중)": G.stages()[0]["+ 모집단 가중"][c], "모델 판별력(AUC)": au[c][1]} for c in G.CON])
    st.dataframe(r.style.format({c: "{:.0%}" for c in r.columns[1:4]} | {"모델 판별력(AUC)": "{:.2f}"}), width="stretch", hide_index=True)
    st.caption(f"자료 근거 {info['신호']} · 고른 집단의 실응답자 {info['실응답자']:,}명. 패널 응답자는 273명뿐이라 세부 집단 숫자는 ±10%p 이상 흔들립니다. 쥬시모먼트는 유도 문항 보정 전 값입니다. AUC 0.5는 찍기, 1은 완벽.")
    g = base.groupby(["age10", "sex"])[G.CON + ["w"]].apply(lambda d: pd.Series({c: np.average(d[c], weights=d.w) for c in G.CON})).reset_index()
    fig = go.Figure([go.Bar(x=g.age10 + " " + g.sex, y=g[c], name=c, marker_color=COL[c]) for c in G.CON])
    fig.update_layout(height=320, barmode="group", yaxis_tickformat=".0%", margin=dict(t=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
