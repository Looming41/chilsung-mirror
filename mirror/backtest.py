"""실제 출시 실적으로 절대 수준 보정 + 보지 않은 출시로 검증.

보정(학습): 펩시 제로슈거 라임 — 편의점·온라인 선판매 40일 200만 개 (아시아경제 2021-02-24)
검증(예측): 밀키스 제로 — 출시 5개월 누적 2,600만 캔 (비즈워치 2023-11-30)
참고 기준: 평균 신제품 첫해 가구 침투율 ≈1.5% (NIQ 2026, 유럽 5개국)

판매 개수 → '한 번이라도 산 성인 비율(시용률)' 변환에는 1인당 구매 개수 가정이 필요 → 범위로 다룬다.
한계: 두 제품 모두 시뮬레이터 요인공간에선 '익숙한 맛·클래식·광고'로 같아서, 검증되는 건 시간 동학과 절대 수준뿐.
      페르소나 기록은 2025년(제로 대중화 이후)이라 2021년 출시엔 시용을 과대평가하는 쪽으로 기운다.
"""
import numpy as np

from . import consumer as C
from . import sim

ADULTS = 42_000_000          # 만 19~79세 인구(약, 2025 주민등록 기준 근사)
SC = dict(price=[1700], zero=[1], flavor=["익숙한 맛"], package=["클래식"], copy=["재미"], influencer=[False])

BENCH = {   # 이름: (일수, 판매 개수, 1인당 구매 개수 범위, 출처)
    "펩시 제로 라임(보정용)": (40, 2_000_000, (1.2, 2.0), "아시아경제 2021-02-24, 선판매 40일 200만 개"),
    "밀키스 제로(검증용)": (150, 26_000_000, (3.0, 8.0), "비즈워치 2023-11-30, 5개월 2,600만 캔"),
}


def target(name):
    days, units, (lo, hi), _ = BENCH[name]
    return units / ADULTS / hi, units / ADULTS / lo          # 시용률 범위(하한, 상한)


def trial(pool, say_do, days, n=2000, reps=6, seed=0):
    res = sim.run(pool, SC, n=n, reps=reps, ages=(19, 79), say_do=say_do, days=days, seed=seed)
    return float(res["table"]["4주구매율"].iloc[0])          # 열 이름과 달리 days 일 누적 시용률


def calibrate(pool=None, iters=10):
    """펩시 제로 라임 40일 시용률 범위의 기하평균에 맞도록 say_do(말→실구매 전환) 이분 탐색."""
    pool = C.load_pool() if pool is None else pool
    lo_t, hi_t = target("펩시 제로 라임(보정용)")
    goal = (lo_t * hi_t) ** 0.5
    lo, hi = 1e-4, 1.0
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if trial(pool, mid, 40) < goal else (lo, mid)
    return (lo * hi) ** 0.5, goal


def report(pool=None):
    pool = C.load_pool() if pool is None else pool
    sd, goal = calibrate(pool)
    rows = []
    for name, (days, units, _, src) in BENCH.items():
        lo, hi = target(name)
        rows.append({"출시": name, "기간(일)": days, "실제 시용률 범위": f"{lo:.1%}~{hi:.1%}",
                     "보정 전(say_do 0.35)": trial(pool, 0.35, days), "보정 후": trial(pool, sd, days),
                     "출처": src, "_lo": lo, "_hi": hi})
    return sd, rows


if __name__ == "__main__":
    import time
    t = time.time()
    sd, rows = report()
    print(f"보정된 say_do = {sd:.4f}  ({time.time() - t:.0f}s)")
    for r in rows:
        inside = r["_lo"] <= r["보정 후"] <= r["_hi"]
        print(f"{r['출시']:<16} {r['기간(일)']:>4}일  실제 {r['실제 시용률 범위']:<12} "
              f"보정 전 {r['보정 전(say_do 0.35)']:.1%}  보정 후 {r['보정 후']:.1%}  {'범위 안' if inside else '범위 밖'}")
