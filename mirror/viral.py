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
             influencer_seed=False, shortform=1.0, decay=0.6, neg_penalty=0.5, early=14, rng=None,
             conform=0.0, fatigue=0.0, scarce=0.0, stock=None, trait=1.0):
    """한 번의 확산을 돌린다.

    intent[i]       : 페르소나 i가 '노출됐을 때' 살 확률
    post_rate[i]    : 산 뒤 SNS에 올릴 확률
    satisfaction[i] : 마셔본 뒤 만족 확률 (불만이면 부정 게시물)
    shortform       : 패키지/문구가 숏폼에 맞을수록 > 1 (공유 확률 배수)

    심리 규칙(로그 오즈에 더함, 기본 0 = 끔). trait[i] 는 사람별 민감도(평균 1 근처)
    conform : 동조. 서로 다른 지인의 추천을 2번째·3번째 볼 때마다 더해짐(3번에서 멈춤)
    fatigue : 피로. 4번째 노출부터 한 번마다 깎임(10번까지)
    scarce  : 희소성. stock(전체 인원 대비 물량)을 정한 한정 판매일 때만. 물량이 떨어지면 더 못 산다(missed)

    난수는 사람·관계·날짜마다 미리 정해져 있어, 같은 rng 씨앗이면 조건이 달라도 같은 운을 쓴다.
    """
    rng = rng or np.random.default_rng()
    n = len(intent)
    src, dst = np.repeat(np.arange(n), deg), np.concatenate(nbrs)
    u_post, u_sat, order = rng.random(n), rng.random(n), rng.random(n)
    bought, first_day = np.zeros(n, bool), np.full(n, -1)
    first_seen, via_friend, post_day, post_pos = np.full(n, -1), np.zeros(n, bool), np.full(n, -1), np.zeros(n, bool)   # 한 사람씩의 기록
    neg_seen, expo = np.zeros(n), np.zeros(n)
    wp, wn = np.zeros(n), np.zeros(n)            # 게시물의 남은 영향력(긍정·부정)
    seen = np.zeros(len(src), bool)              # 이 관계에서 추천을 본 적이 있나
    kids, is_parent = np.zeros(n), np.zeros(n, bool)   # 게시자별 직접 유발 구매 수
    left = np.inf if stock is None else int(round(stock * n))
    lg = np.log(np.clip(intent, 1e-9, 1 - 1e-9) / (1 - np.clip(intent, 1e-9, 1 - 1e-9)))
    out = {k: np.zeros(days) for k in ("buy", "pos", "neg", "exposed")}
    out["missed"] = 0
    top = int(deg.argmax())
    if influencer_seed:                          # 최상위 팔로워 계정이 1일차에 게시
        wp[top], is_parent[top] = 1.0, True

    for d in range(days):
        u_ad, u_buy, u_edge = rng.random(n), rng.random(n), rng.random(len(src))
        hit = u_edge < wp[src] * shortform * 0.5
        neg_seen += np.bincount(dst[1 - u_edge < wn[src] * shortform * 0.5], minlength=n)
        ex = u_ad < ad_reach
        fr = np.zeros(n, bool)
        fr[dst[hit]] = True
        ex |= fr
        fresh = ex & (first_seen < 0)
        first_seen[fresh], via_friend[fresh] = d, fr[fresh]
        out["exposed"][d] = ex.sum()
        seen |= hit
        expo += ex
        p = np.asarray(intent)
        if conform or fatigue or (scarce and stock is not None):
            sig = np.bincount(dst[seen], minlength=n)
            z = lg + trait * (conform * np.clip(np.minimum(sig, 3) - 1, 0, None) + (scarce if stock is not None else 0.0)) - fatigue * np.clip(np.minimum(expo, 10) - 3, 0, None)
            p = 1 / (1 + np.exp(-z))
        new = np.flatnonzero(ex & ~bought & (u_buy < p * neg_penalty ** neg_seen))
        if len(new) > left:                      # 물량 부족: 먼저 온 사람부터
            new = new[np.argsort(order[new])]
            out["missed"] += len(new) - int(left)
            new = new[:int(left)]
        left -= len(new)
        bought[new], first_day[new] = True, d
        out["buy"][d] = len(new)
        credit = np.full(n, -1)
        credit[dst[hit]] = src[hit]
        c = credit[new]
        np.add.at(kids, c[(c >= 0) & is_parent[c]], 1)
        wp, wn = wp * decay, wn * decay
        wp[wp <= 0.05], wn[wn <= 0.05] = 0, 0
        post = new[u_post[new] < np.minimum(1.0, np.asarray(post_rate)[new] * shortform)]
        good = u_sat[post] < np.asarray(satisfaction)[post]
        wp[post[good]], wn[post[~good]] = 1.0, 1.0
        post_day[post], post_pos[post] = d, good
        if d < early:
            is_parent[post] = True
        out["pos"][d], out["neg"][d] = good.sum(), (~good).sum()

    # 초기 14일 게시물만(포화 전), 시드 계정 제외, 분모 +1 로 소표본 수축
    if influencer_seed:
        is_parent[top] = False
    out["n_star"] = float(kids[is_parent].sum() / (is_parent.sum() + 1))
    out["adopters"] = int(bought.sum())
    out["first_day"] = first_day
    out["log"] = {"seen": first_seen, "friend": via_friend, "post": post_day, "pos": post_pos, "neg_seen": neg_seen}
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
    # 심리 규칙: 동조는 늘리고, 피로는 줄이고, 한정 물량은 상한을 지킨다. 같은 씨앗이면 같은 운
    run = lambda **k: simulate(np.full(n, .2), np.full(n, .4), np.full(n, .9), nbrs, deg, rng=np.random.default_rng(3), **k)
    tot = lambda **k: sum(simulate(np.full(n, .2), np.full(n, .4), np.full(n, .9), nbrs, deg, rng=np.random.default_rng(q), ad_reach=.05, shortform=1.6, **k)["adopters"] for q in range(8))
    assert run()["adopters"] == run()["adopters"] and tot(conform=1.0) > tot() > tot(fatigue=.6)
    lim = run(stock=.1, scarce=.8, ad_reach=.1)
    assert lim["adopters"] == round(.1 * n) and lim["missed"] > 0
    solo = lambda **k: simulate(np.full(n, .2), np.zeros(n), np.full(n, .9), nbrs, deg, rng=np.random.default_rng(3), ad_reach=.05, **k)["first_day"] >= 0
    assert (solo(scarce=.8, stock=1.0) | ~solo()).all()      # 같은 운이면 조건만 좋아졌을 때 사던 사람이 빠지지 않는다(입소문을 끈 상태)
    t = np.arange(1, 29)
    synth = np.diff(np.r_[0, bass_cum(t, 0.02, 0.4, 500)])
    p, q, m = fit_bass(synth)
    assert abs(p - 0.02) < 0.005 and abs(q - 0.4) < 0.05, (p, q, m)
    print("viral ok", lo["adopters"], hi["adopters"], round(hi["n_star"], 2))
