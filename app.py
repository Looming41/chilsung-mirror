"""칠성 미러마켓 대시보드.  실행: .venv/bin/streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mirror import consumer as C
from mirror import records as R, sim, stats as S

st.set_page_config(page_title="칠성 미러마켓", page_icon="🥤", layout="wide")


@st.cache_resource
def pool():
    return C.load_pool()


# ---------------- 사이드바: 실험 설계 ----------------
sb = st.sidebar
sb.title("🥤 칠성 미러마켓")
sb.caption("가상 소비자 시장에 신제품을 먼저 출시해 보는 시뮬레이터")
concept = sb.text_area("제품 콘셉트", "자두맛 탄산음료. 새콤달콤한 자두 향에 톡 쏘는 탄산, 500ml 페트.")

sb.subheader("타깃")
ages = sb.slider("연령", 19, 80, (20, 39))
sexes = sb.multiselect("성별", ["남자", "여자"], ["여자"])

sb.subheader("시나리오 변수 (요인설계)")
prices = sb.multiselect("가격(원)", [1500, 1800, 2000, 2200, 2500], [1500, 1800, 2200])
zero = sb.multiselect("당", ["제로 슈거", "일반"], ["제로 슈거"])
flavor = sb.multiselect("맛 포지션", ["새로운 맛", "익숙한 맛"], ["새로운 맛"])
package = sb.multiselect("패키지", ["비주얼(숏폼형)", "클래식"], ["비주얼(숏폼형)", "클래식"])
copy = sb.multiselect("광고 문구 톤", ["재미", "건강"], ["재미", "건강"])
infl = sb.multiselect("인플루언서 협업", ["협업", "광고만"], ["협업", "광고만"])

sb.subheader("시뮬레이션")
n = sb.slider("페르소나 수 (n)", 50, 3000, 1000, 50)
sb.caption(f"±3%p·95% 신뢰 기준 필요 표본 = {S.sample_size():,}명 · 현재 오차한계 ≈ ±{100 * 1.96 * np.sqrt(0.25 / n):.1f}%p")
reps = sb.slider("반복 횟수 (몬테카를로)", 5, 100, 30, 5)
say_do = sb.slider("말-행동 전환율 (설문 의향 → 실제 구매)", 0.1, 1.0, 0.35, 0.05,
                   help="설문에서 '사겠다'고 한 사람이 실제 노출 시 구매하는 비율. 출시 데이터로 보정하는 값.")
ad_reach = sb.slider("일일 광고 도달률", 0.0, 0.05, 0.01, 0.005)
gkey = sb.text_input("Gemini API 키", type="password", key="gemini_key", placeholder="AIza...",
                     help="Google AI Studio(aistudio.google.com)에서 무료 발급. 이 브라우저 세션 메모리에만 보관되고 저장되지 않습니다.")
gem_models = []
if gkey.strip():
    C.GEMINI_KEY = gkey.strip()
    try:
        gem_models = st.cache_data(C.gemini_models, show_spinner=False)(C.GEMINI_KEY)
        sb.success(f"Gemini 연결됨 · 모델 {len(gem_models)}개")
    except Exception as e:
        sb.error(f"Gemini 키 확인 실패: {type(e).__name__}")
llm_ok = C.llm_available()
mode = sb.radio("응답 엔진", ["rule", "llm"], format_func=lambda m: {"rule": "규칙 기반 (빠름)", "llm": "LLM + SSR"}[m],
                disabled=not llm_ok, help=None if llm_ok else "Gemini 키를 넣거나 Claude에 로그인하면 켜집니다.")
options = gem_models + C.CLAUDE_MODELS + ["gpt-5-mini"]
model = (sb.selectbox("LLM 모델", options,
                      help="Gemini 무료 등급은 분당·일일 호출 한도가 있어 n을 작게(50~100) 잡으세요. "
                           "Claude: Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.1/$0.5 (100만 토큰당)")
         if mode == "llm" else options[0])
if mode == "llm":
    try:
        reply = st.cache_data(C.ping, show_spinner="모델 시험 호출 중…")(model)
        sb.success(f"✅ {model} 응답 확인: {reply[:30]}")
    except Exception as e:
        print(f"[ping] {model}: {type(e).__name__}: {e}", flush=True)
        sb.error(f"❌ {model} 호출 실패 — {type(e).__name__}: {str(e)[:300]}")

levels = dict(price=prices, zero=[1 if z == "제로 슈거" else 0 for z in zero], flavor=flavor,
              package=package, copy=copy, influencer=[i == "협업" for i in infl])
n_sc = int(np.prod([len(v) for v in levels.values()]))
sb.caption(f"시나리오 {n_sc}개 × 반복 {reps}회 = 시뮬레이션 {n_sc * reps:,}번")
if mode == "llm":
    calls = n * n_sc // max(1, len(infl))
    tin, tout = calls * C.EST_TOKENS[0], calls * C.EST_TOKENS[1]
    price = ("무료 등급 (한도 내)" if model.startswith("gemini")
             else f"약 ${C.cost(model, tin, tout):,.2f} (₩{C.cost(model, tin, tout) * 1450:,.0f})")
    sb.warning(f"예상: LLM 호출 {calls:,}회 · 입력 {tin / 1e6:.2f}M / 출력 {tout / 1e6:.2f}M 토큰 · {price}")

if sb.button("▶ 실험 실행", type="primary", width="stretch", disabled=n_sc == 0 or not sexes):
    bar = st.progress(0.0, "시작")
    try:
        before = dict(C.USAGE)
        st.session_state.res = sim.run(pool(), levels, n=n, reps=reps, ages=ages, sexes=tuple(sexes), say_do=say_do,
                                       ad_reach=ad_reach, mode=mode, concept=concept, model=model,
                                       progress=lambda f, t: bar.progress(min(f, 1.0), t))
        st.session_state.concept = concept
        used = {k: C.USAGE[k] - before[k] for k in before}
        st.session_state.usage = (model, used) if used["calls"] else None
    except Exception as e:  # LLM 인증·요금 한도 등은 화면에 그대로 보여준다
        import traceback
        traceback.print_exc()
        st.error(f"실행 실패: {type(e).__name__}: {str(e)[:500]}")
    bar.empty()


def show_validation():
    """실데이터 검증: KREI 2025 식품소비행태조사(실제 성인 5,874명)로 채점."""
    from mirror import records as R
    k = st.cache_data(R.load, show_spinner=False)()
    _, rep = st.cache_data(R.zero_model, show_spinner=False)()
    st.markdown("페르소나마다 **같은 성별·나이·시도의 실제 응답자 기록**(KREI 2025 식품소비행태조사, 성인 5,874명)을 "
                "통계적 매칭(hot-deck)으로 붙였습니다. 성향 점수와 음료 습관은 모두 이 실기록에서 나옵니다.")
    c = st.columns(3)
    c[0].metric("실제 응답자", f"{len(k):,}명", "KREI 2025 원자료", delta_color="off")
    c[1].metric("제로 수용성 모델 AUC", f"{rep['auc']['성향+인구통계']:.3f}",
                f"인구통계만 {rep['auc']['인구통계만']:.3f} (5겹 교차검증)", delta_color="off")
    c[2].metric("척도 신뢰도 Cronbach α", " / ".join(f"{v:.2f}" for v in k.attrs["alpha"].values()),
                "새로움·가격·건강·SNS", delta_color="off")
    st.caption("ML 계수(제로음료 월 2회 이상, 가중 로지스틱 회귀): " +
               ", ".join(f"{n} {v:+.2f}" for n, v in rep["coef"].items()))
    st.subheader("채점: 보지 않은 실제 응답자 절반의 집단별 제로음료 이용률을 맞히기")
    vm = "gemini-3.5-flash-lite" if C.GEMINI_KEY else (model if mode == "llm" else None)
    use_llm = st.checkbox("LLM 단독 추정도 비교 (호출 1회, 결과 캐시)", value=bool(vm), disabled=not vm)
    try:
        tab, met = st.cache_data(R.validate, show_spinner="채점 중…")(vm if use_llm else None)
    except Exception as e:
        st.warning(f"LLM 비교 실패({type(e).__name__}) — 시뮬레이터만 채점합니다.")
        tab, met = st.cache_data(R.validate, show_spinner="채점 중…")(None)
    long = tab.reset_index(names="집단").melt("집단", var_name="방법", value_name="비율")
    st.plotly_chart(px.bar(long, x="집단", y="비율", color="방법", barmode="group", height=360,
                           color_discrete_map={"정답(실제 응답자)": "#222", "시뮬레이터": "#E60012",
                                               "기준선(전체평균)": "#bbb", "LLM 단독": "#5b8def"})
                    .update_yaxes(tickformat=".0%"), width="stretch")
    st.dataframe(met.style.format({"평균 오차(%p)": "{:.1f}", "차이 재현 r": "{:.2f}", "차이 기울기": "{:.2f}"}),
                 width="stretch")
    st.subheader("음료 선택 모델 (KREI 2023~2025, 17,536명)")
    import json, os
    from mirror import choice as M
    if os.path.exists("data/choice_validation.json"):
        v = json.load(open("data/choice_validation.json"))
        m = st.cache_resource(M.model)()
        st.markdown("공통 리커트 33문항 → 병렬분석으로 **심리 요인 7개** → 사람별 속성 선호(탄산·제로·단맛·과일…)를 학습한 "
                    "이질적 조건부 로짓(1~3순위 순위 확장). 데이터에 없던 속성 조합도 새 선택지로 넣을 수 있다.")
        c = st.columns(2)
        with c[0]:
            st.caption("① 범주를 통째로 빼고 학습 → '신제품'처럼 넣어 예측 (누가 고르나 = 12집단 상관)")
            st.dataframe(pd.DataFrame(v["loco"]).style.format({"실제 점유율": "{:.1%}", "예측 점유율": "{:.1%}",
                                                                "집단 상관 r": "{:.2f}", "집단 오차(%p)": "{:.1f}"}),
                         width="stretch", hide_index=True, height=300)
        with c[1]:
            st.caption("② 시간 검증: 2023~24로 학습 → 2025 집단별 점유율")
            st.dataframe(pd.DataFrame(v["temporal"]), width="stretch", hide_index=True)
            st.caption("③ 개인 트윈: 실제 응답자 60명의 제로음료 빈도를 가리고 맞히기")
            st.dataframe(pd.DataFrame(v["twin"]).style.format({"정확히 맞힘": "{:.0%}", "1단계 이내": "{:.0%}"}),
                         width="stretch", hide_index=True)
            st.caption("④ 1순위 예측 5겹 교차검증")
            st.dataframe(pd.DataFrame(v["cv"]), width="stretch", hide_index=True)
        L = m["loadings"].copy()
        L.columns = m["names"]
        L.index = [f"{n + 1}. {M._short(i)}" for n, i in enumerate(L.index)]
        st.plotly_chart(px.imshow(L.T, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
                                  title="심리 요인 적재량 (요인분석, varimax)", height=380), width="stretch")
        st.caption("읽는 법: 일부 범주만 '누가 고르나'가 맞고(가당 탄산·유제품) 탄산수·차·이온·가공우유는 상관이 음수다. 시간 예측은 2024 값 그대로 쓰는 것보다 나쁘다. "
                   "커피·음료전문점처럼 고유 속성 범주는 수준을 못 맞힘. 개인 단위(n=60)는 LLM 35%. "
                   "분포 모양만 그럴듯 — 그래서 개인 판단은 학습모델 확률, LLM은 이유·말투를 맡는다.")
    st.subheader("백테스트: 실제 출시 실적과 비교 (성인 전체, 광고만, 익숙한 맛·클래식 1,700원)")
    from mirror import backtest as B
    sd, rows = st.cache_data(B.report, show_spinner="백테스트 중…")()
    bt = pd.DataFrame(rows)
    bt["판정"] = np.where((bt._lo <= bt["보정 후"]) & (bt["보정 후"] <= bt._hi), "✅ 범위 안", "❌ 범위 밖")
    st.dataframe(bt.drop(columns=["_lo", "_hi"]).style.format({"보정 전(say_do 0.35)": "{:.1%}", "보정 후": "{:.1%}"}),
                 width="stretch", hide_index=True)
    st.caption(f"펩시 제로 라임으로 말→실구매 전환율을 보정(={sd:.3f}, 기본값 0.35와 거의 같음)하고, 밀키스 제로는 보지 않고 예측. "
               "판매 개수→시용률은 근거 자료 없는 1인당 구매 개수 가정으로 변환했고, 중간에 범위를 벗어나 효용 기준점을 바꿔 맞춘 뒤라 독립 검증이 아니다. 두 제품이 요인공간에서 같아 제품 간 차이는 검증 못 함.")
    st.caption("차이 재현 r·기울기: 집단 간 차이의 방향·크기를 얼마나 맞히는지(1이 이상적). "
               "주의: 시뮬레이터는 기증자의 같은 문항 답을 복사하므로 '집단평균 복사'와 비슷한 게 당연하다(표본오차 수준). "
               "이 문항의 정답을 본 적 없는 LLM과의 비교는 공정하지 않으며 예측력의 증거가 아니다.")


res = st.session_state.get("res")
if res is None:
    st.title("칠성 미러마켓")
    st.info("왼쪽에서 콘셉트와 시나리오 변수를 정하고 **▶ 실험 실행**을 누르세요.")
    st.markdown("""
- **페르소나**: NVIDIA Nemotron-Personas-Korea(통계청 분포 기반 합성 한국인) 3만 명 풀에서 연령×성별 **층화추출**
- **응답**: 규칙 기반 효용 모델 또는 LLM 자유 서술 → **SSR**(의미 유사도)로 5점 척도 분포
- **확산**: 멱법칙 팔로우 네트워크 위 4주 일 단위 확산, 게시물 영향력 지수 감쇠, 부정 후기는 전환을 깎음
- **통계**: 부트스트랩 95% CI, 두 비율 z검정 + Holm 보정, 베이지안 1등 확률, 반복별 승률, Bass 모형 적합
""")
    st.header("🔬 실데이터 검증")
    show_validation()
    st.stop()

tb = res["table"]
pct = lambda x: f"{100 * x:.1f}%"

# 마지막 결과를 로컬 파일로 남김(분석·재현용, data/ 는 git 제외). API 키는 넣지 않는다.
try:
    import json
    with open("data/last_run.json", "w") as f:
        json.dump({"concept": st.session_state.get("concept"), "usage": st.session_state.get("usage"),
                   "table": tb.drop(columns=["zero", "influencer"], errors="ignore").to_dict("records"),
                   "texts": {k: v[:200] for k, v in res["texts"].items()},
                   "personas": res["base"][["age", "sex", "province", "occupation"]].head(200).to_dict("records"),
                   "p_top2": {lb: res["p_base"][j][:200].round(4).tolist() for j, lb in enumerate(res["labels"])}},
                  f, ensure_ascii=False, default=str)
except OSError:
    pass
top = tb.iloc[0]

st.title("실험 결과")
st.caption(f"페르소나 {res['n']:,}명 × 반복 {res['reps']}회 · 콘셉트: {st.session_state.get('concept', '')[:60]}")
if u := st.session_state.get("usage"):
    m, d = u
    st.caption(f"🔢 실제 사용량 ({m}): 호출 {d['calls']:,}회 · 입력 {d['in']:,} / 출력 {d['out']:,} 토큰 · "
               f"호출당 평균 {d['in'] // d['calls']:,} / {d['out'] // d['calls']:,} · ${C.cost(m, d['in'], d['out']):,.3f}")
c = st.columns(4)
c[0].metric("🏆 1위 시나리오", top.시나리오.split("·")[0], top.시나리오, delta_color="off")
c[1].metric("승률 (4주 구매 1위 빈도)", pct(top.승률))
c[2].metric("4주 구매율", pct(top["4주구매율"]), f"95% CI {pct(top.구매_하한)}~{pct(top.구매_상한)}", delta_color="off")
c[3].metric("바이럴 확률 (n*>1)", pct(top.바이럴확률), f"분기비 {top.분기비:.2f}", delta_color="off")

t1, t2, t3, t4, t5, t6 = st.tabs(["📊 시나리오 순위", "🧪 통계 검정", "📈 바이럴 확산", "👥 세그먼트", "💬 페르소나",
                                  "🔬 실데이터 검증"])

with t1:
    with st.expander("🥤 자기잠식: 1위 시나리오 신제품이 어느 음료에서 1순위 수요를 뺏어오나 (선택모델)", expanded=True):
        from mirror import choice as M
        j0 = res["labels"].index(top.시나리오)
        cb = M.cannibal(C.zmat(res["base"]), res["attrs"][j0])
        st.metric("신제품 1순위 점유(이 표본)", f'{cb.loc["신제품", "변화(%p)"]:.2f}%p', delta_color="off")
        rel = cb.drop(["신제품"])["잠식률(%)"].sort_values()
        st.plotly_chart(px.bar(rel, orientation="h", height=380,
                               labels={"value": "잠식률: 자기 1순위 점유율 중 잃는 비율(%)", "index": ""},
                               color=rel < rel.median(), color_discrete_map={True: "#E60012", False: "#999"})
                        .update_layout(showlegend=False), width="stretch")
        st.caption("신제품 속성: " + ", ".join(M.ATTR_KO[k] for k, v in res["attrs"][j0].items() if v) +
                   " · 칠성 기존 제품(가당 탄산=칠성사이다, 제로 탄산=칠성사이다 제로·펩시 제로) 잠식 여부를 볼 것. "
                   "λ 재추정값 1.0 = 일반 로짓(묶음 대체 효과 없음). 잠식은 속성 선호 차이에서만 나오며 검증되지 않았다.")
    show = tb.head(15)
    fig = go.Figure()
    fig.add_bar(y=show.시나리오, x=show.승률, orientation="h", marker_color="#E60012",
                text=[pct(v) for v in show.승률], textposition="outside")
    fig.update_layout(title="시나리오별 승률 — 같은 난수 조건에서 4주 구매 1위를 한 비율", height=80 + 28 * len(show),
                      yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=".0%"), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for col, lo, hi, name, color in [("구매의향", "의향_하한", "의향_상한", "설문 구매의향 (4~5점)", "#9AA0A6"),
                                     ("4주구매율", "구매_하한", "구매_상한", "4주 실제 구매율", "#E60012")]:
        fig.add_scatter(y=show.시나리오, x=show[col], mode="markers", name=name, marker=dict(size=10, color=color),
                        error_x=dict(type="data", symmetric=False, array=show[hi] - show[col], arrayminus=show[col] - show[lo]))
    fig.update_layout(title="말 vs 행동 — 95% 부트스트랩 신뢰구간", height=80 + 28 * len(show),
                      yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=".0%"), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(tb[["시나리오", "승률", "4주구매율", "구매_하한", "구매_상한", "구매의향", "의향1등확률",
                     "바이럴확률", "분기비", "부정비율"]].style.format(
        {k: "{:.1%}" for k in ["승률", "4주구매율", "구매_하한", "구매_상한", "구매의향", "의향1등확률", "바이럴확률", "부정비율"]}
        | {"분기비": "{:.2f}"}), width="stretch", hide_index=True)

    # 요인별 주효과
    eff = []
    for f, name in [("price", "가격"), ("zero", "당"), ("flavor", "맛"), ("package", "패키지"), ("copy", "문구"), ("influencer", "인플루언서")]:
        g = tb.groupby(f)["4주구매율"].mean()
        if len(g) > 1:
            eff += [{"요인": name, "수준": str(k), "4주구매율": v} for k, v in g.items()]
    if eff:
        fig = px.bar(pd.DataFrame(eff), x="수준", y="4주구매율", facet_col="요인", color="요인",
                     title="요인별 주효과 (다른 변수 평균)", height=320)
        fig.update_xaxes(matches=None, title=None).update_yaxes(tickformat=".1%")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch")

with t2:
    a, b = st.columns([1, 2])
    with a:
        st.subheader("표본 크기 계산기")
        e = st.number_input("허용 오차(%p)", 1.0, 10.0, 3.0, 0.5)
        conf = st.selectbox("신뢰수준", [0.90, 0.95, 0.99], 1)
        st.metric("필요 표본 수", f"{S.sample_size(e / 100, conf):,}명")
        st.caption("n = z²·p(1−p)/e², p=0.5(최대 분산) 가정")
    with b:
        st.subheader("베이지안 1등 확률 (설문 구매의향)")
        pb = tb.sort_values("의향1등확률", ascending=False).head(8)
        st.plotly_chart(px.bar(pb, x="의향1등확률", y="시나리오", orientation="h", height=300)
                        .update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=".0%")),
                        width="stretch")
        st.caption("각 시나리오 의향률에 Beta(1+k, 1+n−k) 사후분포 → 2만 번 추출해 1위 빈도")
    st.subheader(f"쌍별 검정: 설문 1위 「{res['best_label']}」 vs 나머지")
    st.caption("두 비율 z검정(양측), 다중비교는 Holm 보정. 효과크기 Cohen h: 0.2 작음 · 0.5 중간 · 0.8 큼")
    if len(res["tests"]):
        st.dataframe(res["tests"].style.format({"1위 의향": "{:.1%}", "대상 의향": "{:.1%}", "차이(%p)": "{:.1f}",
                                                "Cohen h": "{:.3f}", "p": "{:.4f}", "Holm 보정 p": "{:.4f}"}),
                     width="stretch", hide_index=True)
    st.info("⚠️ LLM/규칙 페르소나는 독립 표본이 아니므로 이 신뢰구간은 '시뮬레이션 내부의 불확실성'입니다. "
            "실제 출시 데이터와의 백테스트로 '말-행동 전환율' 등을 보정해야 현실 오차를 말할 수 있습니다.")

with t3:
    pick = st.multiselect("비교할 시나리오", res["labels"], list(tb.시나리오.head(3)))
    days = np.arange(1, res["curves"]["buy"].shape[2] + 1)
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    for i, lb in enumerate(pick):
        j = res["labels"].index(lb)
        cum = res["curves"]["buy"][j].cumsum(1) / res["n"]
        lo, mid, hi = np.quantile(cum, [0.1, 0.5, 0.9], axis=0)
        col = colors[i % len(colors)]
        fig.add_scatter(x=np.r_[days, days[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                        fillcolor=col, opacity=0.15, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=days, y=mid, name=lb, line=dict(color=col, width=3))
    fig.update_layout(title="누적 구매율 (중앙값, 음영 = 반복 간 10~90%)", xaxis_title="출시 후 일", yaxis=dict(tickformat=".1%"), height=380)
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for i, lb in enumerate(pick):
        j = res["labels"].index(lb)
        col = colors[i % len(colors)]
        fig.add_scatter(x=days, y=res["curves"]["pos"][j].mean(0), name=f"긍정 · {lb[:18]}", line=dict(color=col))
        fig.add_scatter(x=days, y=res["curves"]["neg"][j].mean(0), name=f"부정 · {lb[:18]}", line=dict(color=col, dash="dot"))
    fig.update_layout(title="일별 SNS 언급량 (실선 긍정 / 점선 부정, 반복 평균)", xaxis_title="출시 후 일", height=360)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        rows = [{"시나리오": lb, "혁신계수 p (외부 유입)": f[0], "모방계수 q (입소문)": f[1], "q/p": f[1] / f[0]}
                for lb in pick if (f := res["bass"].get(lb))]
        st.subheader("Bass 확산 모형 적합")
        st.caption("q/p가 클수록 광고보다 입소문이 확산을 이끈다")
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, width="stretch")
    with b:
        st.subheader("누가 먼저 샀나 (초기 구매자 20명)")
        sp = res["spreaders"][res["spreaders"].시나리오.isin(pick)].copy()
        sp["팔로워 규모"] = pd.cut(sp.팔로워, [0, 5, 20, 60, 10**6], labels=["~5", "6~20", "21~60", "60+ (인플루언서급)"])
        st.plotly_chart(px.histogram(sp, x="팔로워 규모", color="시나리오", barmode="group", height=300)
                        .update_layout(showlegend=False), width="stretch")

with t4:
    lb = st.selectbox("시나리오", res["labels"], res["labels"].index(top.시나리오))
    seg = res["segs"][lb]
    st.plotly_chart(px.imshow(seg, text_auto=".1%", color_continuous_scale="Reds", aspect="auto",
                              labels=dict(color="구매의향"), title="연령대 × 성별 설문 구매의향 (4~5점 확률)"),
                    width="stretch")
    j = res["labels"].index(lb)
    trait = res["base"][["health", "novelty", "social", "drink", "thrift", "zero_aff"]].assign(의향=res["p_base"][j])
    corr = trait.corr()["의향"].drop("의향").rename({"health": "건강 관심", "novelty": "새로움 추구", "social": "SNS 활용",
                                                   "drink": "탄산 선호", "thrift": "가격 민감", "zero_aff": "제로 수용성(ML)"})
    st.plotly_chart(px.bar(corr, orientation="h", title="성향 점수와 구매의향의 상관", height=280)
                    .update_layout(showlegend=False, xaxis_title="피어슨 r", yaxis_title=None), width="stretch")

with t5:
    lb = st.selectbox("시나리오 ", res["labels"], res["labels"].index(top.시나리오))
    j = res["labels"].index(lb)
    base = res["base"]
    view = base[["age", "sex", "province", "occupation", "persona"]].assign(
        구매의향=res["p_base"][j], 반응=res["texts"].get(lb, [""] * len(base)),
        실기록=base.fav + " · 제로 " + base.zero_code.map(R.ZERO_FREQ))
    which = st.radio("보기", ["안 살 사람", "살 사람"], horizontal=True)
    view = view.sort_values("구매의향", ascending=which == "안 살 사람").head(30)
    st.dataframe(view.rename(columns={"age": "나이", "sex": "성별", "province": "지역", "occupation": "직업", "persona": "소개"})
                 .style.format({"구매의향": "{:.1%}"}), width="stretch", hide_index=True)
    st.subheader("페르소나에게 직접 묻기")
    i = st.selectbox("페르소나", view.index, format_func=lambda i: f"{base.age[i]}세 {base.sex[i]} · {base.occupation[i]}")
    q = st.text_input("질문", "이거 왜 안 샀어요? 뭐가 바뀌면 사겠어요?")
    if st.button("묻기"):
        if C.llm_available():
            sc = res["scenarios"][j]
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(C.ask_why(base.loc[i], st.session_state.concept, sc, q, model))
        else:
            st.chat_message("assistant").write(f"(규칙 기반 근거) {res['texts'][lb][i]}")
            st.caption("Claude에 로그인하면 이 페르소나가 직접 대답합니다.")

with t6:
    show_validation()
