"""칠성 신제품 검증.  실행: .venv/bin/streamlit run app.py"""
import streamlit as st

from mirror.given_ui import render

st.set_page_config(page_title="칠성 신제품 검증", layout="wide", initial_sidebar_state="collapsed")
render()
