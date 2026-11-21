# 칠성 미러마켓

한국 소비자 페르소나 수천 명이 사는 가상 시장에 신제품 콘셉트를 먼저 출시해 보고, 여러 시나리오(가격·패키지·문구·인플루언서) 중 **4주 뒤 가장 많이 팔릴 확률이 높은 안**을 통계적으로 고르는 시뮬레이터.

## 실행

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py          # 첫 실행 시 페르소나 데이터(약 220MB) 자동 다운로드
ant auth login                          # 선택: Claude로 LLM + SSR 응답 모드, 페르소나 대화 (또는 ANTHROPIC_API_KEY)
```

## 구조

| 파일 | 역할 |
|---|---|
| `mirror/consumer.py` | 페르소나 풀(NVIDIA Nemotron-Personas-Korea, 통계청 분포 기반) 로드, 연령×성별 층화추출, 반응 모델(규칙 기반 효용 / LLM 자유서술 → SSR) |
| `mirror/viral.py` | 멱법칙 팔로우 네트워크, 4주 일 단위 확산(게시물 영향력 지수 감쇠, 부정 후기 페널티), 분기비 n*, Bass 모형 적합 |
| `mirror/stats.py` | 표본 크기, 부트스트랩 CI, 두 비율 z검정, Holm 보정, Cohen h, 베이지안 1등 확률, TVD |
| `mirror/sim.py` | 요인설계 시나리오 × 몬테카를로 반복(공통 난수) → 결과 요약 |
| `app.py` | Streamlit 대시보드: 시나리오 순위 · 통계 검정 · 바이럴 확산 · 세그먼트 · 페르소나 대화 |

## 핵심 지표

- **승률**: 같은 난수 조건의 각 반복에서 4주 구매율 1위를 한 비율
- **설문 구매의향**: 5점 척도 4~5점 확률 (말) vs **4주 구매율**: 확산 시뮬레이션 결과 (행동)
- **바이럴 확률**: 초기 14일 게시물 1건당 직접 유발 구매 수(분기비 n*)가 1을 넘은 반복의 비율
- **말-행동 전환율**: 설문 의향 → 실제 구매 전환. 실제 출시 데이터로 보정해야 하는 핵심 파라미터

## 자체 검사

```bash
for m in stats viral consumer sim; do .venv/bin/python -m mirror.$m; done
```

## 한계

페르소나는 독립 표본이 아니므로 신뢰구간은 시뮬레이션 내부 불확실성이다. 규칙 기반 모드의 계수는 가정값이며, 과거 출시 제품 백테스트로 보정해야 현실 예측력을 말할 수 있다.

## 참고

- Maier et al. (2025), *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338
- Zhao et al. (2015), *SEISMIC: A Self-Exciting Point Process Model for Predicting Tweet Popularity*
- Bass (1969), *A New Product Growth Model for Consumer Durables*
- NVIDIA, *Nemotron-Personas-Korea* (CC BY 4.0)
