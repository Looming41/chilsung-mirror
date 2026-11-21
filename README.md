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

## 실데이터 앵커 (기록 앵커 시민)
페르소나마다 **같은 성별·나이(±3세)·시도의 실제 응답자 기록**을 hot-deck 통계적 매칭으로 붙인다.
출처: 한국농촌경제연구원 [2025 식품소비행태조사](https://krei.re.kr/foodSurvey/page/460) 성인가구원 원자료(n=5,874, 가중치 포함). 처음 실행 시 자동 다운로드(`mirror/records.py`).

- 성향 척도(실측 상관으로 구성, Cronbach α): 새로움 추구 .68 · 알뜰 계획형 .75 · 건강 관심 .60 · SNS 활용 .85
- ML: '제로음료 월 2회 이상' 가중 로지스틱 회귀, 5겹 CV AUC 0.741 (인구통계만 0.722)
- 검증(보지 않은 실제 응답자 절반, 연령대×성별 12집단 제로음료 이용률):

| 방법 | 평균 오차 | 집단 차이 재현 r |
|---|---|---|
| 시뮬레이터(실기록 매칭) | **3.6%p** | 0.95 |
| 전체 평균 복사 | 11.1%p | — |
| LLM 단독(Gemini 3.5 Flash Lite) | 33.6%p | 0.91 (2~3배 과대) |

근거: LLM 페르소나는 집단 차이를 고정관념으로 채우고 수준을 왜곡한다(Pew 2026, Twin-2K-500, digital-citizens 등). 그래서 수준·성향은 실기록이, 반응 문장은 LLM 이 맡는다.

### 백테스트: 실제 출시 실적 (`python -m mirror.backtest`)
| 출시 | 기간 | 실제 시용률(추정 범위) | 시뮬레이터 | 판정 |
|---|---|---|---|---|
| 펩시 제로 라임 (보정용) | 40일 | 2.4~4.0% | 2.9% | 범위 안 |
| 밀키스 제로 (보지 않고 예측) | 150일 | 7.7~20.6% | 10.3% | 범위 안 |

출처: 아시아경제 2021-02-24(선판매 40일 200만 개), 비즈워치 2023-11-30(5개월 2,600만 캔). 판매 개수→시용률은 성인 4,200만 명, 1인당 구매 1.2~2개 / 3~8개 가정.
보정된 말→실구매 전환율 0.338 ≈ 기본값 0.35. 한계: 두 제품이 요인공간에서 같아 시간 동학·절대 수준만 검증되고 제품 간 차이는 검증 안 됨.
