"""현장 설문 화면(휴대폰용). 이름·연락처를 묻지 않는다. 실행은 설문 보정 탭의 '설문 페이지 열기' 또는:
streamlit run mirror/field_page.py --server.port 8520 --server.address 0.0.0.0
"""
import os
import random
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mirror import field as F  # noqa: E402

st.set_page_config(page_title="1분 음료 설문", layout="centered")
s = F.study()
if not s:
    st.info("아직 열린 설문이 없습니다.")
    st.stop()
if st.session_state.get("done"):
    st.success("고맙습니다! 답이 저장됐어요.")
    if st.button("다음 사람", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.stop()
if "cards" not in st.session_state:
    st.session_state.cards = random.sample(list(s["scs"]), min(F.MAXCARD, len(s["scs"])))

st.title("1분 음료 설문")
st.caption("이름·연락처는 묻지 않으며, 신제품 연구용 통계로만 씁니다. 만 14세 이상만 참여해 주세요. 언제든 그만둘 수 있습니다.")
with st.form("f"):
    a = {"age": st.number_input("나이(만)", 14, 79, None, placeholder="예: 23"),
         "sex": st.radio("성별", ["여자", "남자"], None, horizontal=True),
         "zero": st.radio("제로 음료를 얼마나 자주 마시나요?", F.ZERO[::-1], None),
         "reject": st.radio("제로 음료의 맛이나 감미료가 싫어서 피하는 편인가요?", ["아니오", "예"], None, horizontal=True),
         "fav": st.selectbox("가장 좋아하는 음료 종류", F.FAV, None, placeholder="골라 주세요")}
    for k, (t, m) in F.IT.items():
        a[k] = st.radio(t + ("" if m == 3 else "  (1 전혀 아니다 ~ 5 매우 그렇다)"), list(range(1, m + 1)), None, horizontal=True)
    st.markdown(f"#### 편의점에서 이 신제품을 봤다면?\n**{s['concept']}**")
    rate = {}
    for k in st.session_state.cards:
        x = s["scs"][k]
        st.markdown(f"**{x['price']:,}원 · {'제로 슈거' if x['zero'] else '일반(설탕)'} · {x['flavor']}** · {x['package'].split('(')[0]} 패키지 · '{x['copy']}' 강조" + (" · 한정 수량" if x.get("stock") else ""))
        rate[k] = st.radio("살 것 같나요? (1 절대 안 산다 ~ 5 꼭 산다)", [1, 2, 3, 4, 5], None, horizontal=True, key="r_" + k)
    if st.form_submit_button("보내기", type="primary", width="stretch"):
        if any(v is None for v in [*a.values(), *rate.values()]):
            st.error("빠진 문항이 있어요.")
        else:
            try:
                F.save(s, a | {"reject": a["reject"] == "예"}, rate)
                st.session_state.done = True
                st.rerun()
            except ValueError as e:
                st.error(str(e))
