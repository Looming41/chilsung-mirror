"""바이럴 엔진: 멱법칙 팔로우 네트워크 위의 일 단위 확산(Hawkes식 감쇠) + Bass 적합.

- 광고 노출 = Bass 혁신 계수 p 역할 (외부 유입)
- 게시물 → 팔로워 노출 = 모방 계수 q 역할 (입소문)
- 게시물의 영향력은 시간이 지나며 지수 감쇠 (자기흥분 과정의 커널)
- 분기비 n* = 게시 1건이 직접 만든 구매자 수의 평균. n* > 1 이면 자가 확산.
"""
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit


def build_network(n, m=3, seed=0):
    """Barabási–Albert 그래프: 소수가 팔로워를 독점하는 SNS 구조."""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    nbrs = [np.fromiter(G.neighbors(i), int) for i in range(n)]
    deg = np.array([len(a) for a in nbrs])
    return nbrs, deg


def simulate(intent, post_rate, satisfaction, nbrs, deg, *, days=28, ad_reach=0.01,
             influencer_seed=False, shortform=1.0, decay=0.6, neg_penalty=0.5, early=14, rng=None):
    """한 번의 4주 확산을 돌린다.

    intent[i]       : 페르소나 i가 '노출됐을 때' 살 확률
    post_rate[i]    : 산 뒤 SNS에 올릴 확률
    satisfaction[i] : 마셔본 뒤 만족 확률 (불만이면 부정 게시물)
    shortform       : 패키지/문구가 숏폼에 맞을수록 > 1 (공유 확률 배수)
    """
    rng = rng or np.random.default_rng()
    n = len(intent)
    bought = np.zeros(n, bool)
    neg_seen = np.zeros(n)                      # 본 부정 게시물 수
    active = []                                 # [(작성자, 남은 영향력, 긍정여부)]
    children = {}                               # 작성자 -> 직접 유발 구매 수
    first_day = np.full(n, -1)
    out = {k: np.zeros(days) for k in ("buy", "pos", "neg", "exposed")}

    if influencer_seed:                          # 최상위 팔로워 계정이 1일차에 게시
        top = int(deg.argmax())
        active.append((top, 1.0, True))
        children[top] = 0

    for d in range(days):
        exposures = {}                           # 노출된 사람 -> 원인 작성자 (마지막 것)
        for i in np.flatnonzero(rng.random(n) < ad_reach):
            exposures[int(i)] = None
        for author, w, positive in active:
            seen = nbrs[author][rng.random(len(nbrs[author])) < w * shortform * 0.5]
            for j in seen:
                if positive:
                    exposures[int(j)] = author
                else:
                    neg_seen[j] += 1
        out["exposed"][d] = len(exposures)

        new_posts = []
        for j, src in exposures.items():
            if bought[j]:
                continue
            p = intent[j] * neg_penalty ** neg_seen[j]
            if rng.random() < p:
                bought[j] = True
                first_day[j] = d
                out["buy"][d] += 1
                if src in children:
                    children[src] += 1
                if rng.random() < min(1.0, post_rate[j] * shortform):
                    pos = rng.random() < satisfaction[j]
                    new_posts.append((j, 1.0, pos))
                    if d < early:
                        children.setdefault(j, 0)
                    out["pos" if pos else "neg"][d] += 1

        active = [(a, w * decay, s) for a, w, s in active if w * decay > 0.05] + new_posts

    # 초기 14일 게시물만(포화 전), 시드 계정 제외, 분모 +1 로 소표본 수축
    organic = [c for a, c in children.items() if not (influencer_seed and a == int(deg.argmax()))]
    n_star = float(sum(organic) / (len(organic) + 1))
    out["n_star"] = n_star
    out["adopters"] = int(bought.sum())
    out["first_day"] = first_day
    return out


def bass_cum(t, p, q, m):
    e = np.exp(-(p + q) * t)
    return m * (1 - e) / (1 + (q / p) * e)


def fit_bass(daily):
    """일별 구매 수에 Bass 모델 적합 → (p, q, m). 실패하면 None."""
    y = np.cumsum(daily)
    t = np.arange(1, len(y) + 1)
    if y[-1] <= 0:
        return None
    try:
        (p, q, m), _ = curve_fit(bass_cum, t, y, p0=[0.01, 0.3, y[-1] * 1.5],
                                 bounds=([1e-5, 0, y[-1]], [1, 3, y[-1] * 50]), maxfev=5000)
        return float(p), float(q), float(m)
    except RuntimeError:
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    nbrs, deg = build_network(800, seed=1)
    n = len(deg)
    lo = simulate(np.full(n, 0.05), np.full(n, 0.1), np.full(n, 0.8), nbrs, deg, rng=rng)
    hi = simulate(np.full(n, 0.5), np.full(n, 0.6), np.full(n, 0.9), nbrs, deg,
                  influencer_seed=True, shortform=1.5, rng=np.random.default_rng(1))
    assert hi["adopters"] > lo["adopters"], (lo["adopters"], hi["adopters"])
    assert hi["n_star"] > lo["n_star"]
    t = np.arange(1, 29)
    synth = np.diff(np.r_[0, bass_cum(t, 0.02, 0.4, 500)])
    p, q, m = fit_bass(synth)
    assert abs(p - 0.02) < 0.005 and abs(q - 0.4) < 0.05, (p, q, m)
    print("viral ok", lo["adopters"], hi["adopters"], round(hi["n_star"], 2))
