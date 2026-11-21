"""통계 엔진: 표본 크기, 부트스트랩, 비율 검정 + Holm 보정, 베이지안 1등 확률, 분포 비교."""
import numpy as np
from scipy import stats as st


def sample_size(margin=0.03, conf=0.95, p=0.5):
    """비율 추정에 필요한 표본 수 n = z² p(1-p) / e²."""
    z = st.norm.ppf(1 - (1 - conf) / 2)
    return int(np.ceil(z**2 * p * (1 - p) / margin**2))


def bootstrap_ci(x, B=2000, conf=0.95, rng=None):
    """평균의 퍼센타일 부트스트랩 신뢰구간."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    means = rng.choice(x, (B, len(x)), replace=True).mean(1)
    a = (1 - conf) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def two_prop_ztest(k1, n1, k2, n2):
    """두 비율 차이 z검정(양측). 반환: z, p."""
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return float(z), float(2 * st.norm.sf(abs(z)))


def cohens_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def holm(pvals):
    """Holm-Bonferroni 보정 p값."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def prob_best(k, n, draws=20000, rng=None):
    """Beta(1+k, 1+n-k) 사후분포에서 뽑아 각 시나리오가 1등일 확률."""
    rng = rng or np.random.default_rng(0)
    k, n = np.asarray(k), np.asarray(n)
    samples = rng.beta(1 + k[:, None], 1 + (n - k)[:, None], (len(k), draws))
    wins = np.bincount(samples.argmax(0), minlength=len(k))
    return wins / draws


def tvd(p, q):
    """두 이산분포의 총변동거리."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


if __name__ == "__main__":
    assert sample_size() == 1068
    z, p = two_prop_ztest(340, 1000, 290, 1000)
    assert 0.01 < p < 0.03, p
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    pb = prob_best([100, 200], [1000, 1000])
    assert pb[1] > 0.99
    m, lo, hi = bootstrap_ci(np.r_[np.ones(300), np.zeros(700)])
    assert lo < 0.3 < hi and abs(m - 0.3) < 1e-9
    assert tvd([1, 0], [0, 1]) == 1.0
    print("stats ok")
