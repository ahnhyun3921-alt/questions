"""나답 질문 평가 모델: 릿지 규제 베타-이항 회귀(문항 단위).

왜 이 모델인가
- 결과는 문항마다 '답변 a / 결론 n(= 답변 + 교체)' 이고 n이 작다(대부분 10 미만).
- 같은 특성의 문항끼리도 진짜 답변율이 다르다(과산포). 베타-이항은 문항별 진짜 답변율 θ_i 가
  Beta(μ_i·φ, (1-μ_i)·φ) 를 따른다고 보고, μ_i = sigmoid(x_i·β) 를 특성으로 설명한다.
- φ(정밀도)를 데이터로 추정하므로, 문항별 사후 추정 (a_i + φμ_i)/(n_i + φ) 는
  '특성으로 예측한 값' 쪽으로 데이터가 정한 만큼 당겨진다(경험적 베이즈). 고정 K=10 대체.
- 특성이 많고 서로 겹치므로 β 에 릿지(L2) 규제를 두고, 강도 λ 는 문항 단위 교차검증으로 고른다.
- 새 질문은 θ ~ Beta(μφ, (1-μ)φ) 이므로 예측 구간(80%)을 함께 낸다.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln
from scipy.stats import beta as beta_dist


def _nll(params, X, a, n, lam):
    k = X.shape[1]
    b, g = params[:k], params[k]
    mu = np.clip(expit(X @ b), 1e-6, 1 - 1e-6)
    phi = np.exp(g)
    al, be = mu * phi, (1 - mu) * phi
    ll = betaln(a + al, n - a + be) - betaln(al, be)
    return -(ll.sum()) + lam * np.sum(b[1:] ** 2)


def bb_loglik(a, n, mu, phi):
    """문항별 베타-이항 로그우도(조합 항 포함) — 모델 비교용."""
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    al, be = mu * phi, (1 - mu) * phi
    comb = gammaln(n + 1) - gammaln(a + 1) - gammaln(n - a + 1)
    return comb + betaln(a + al, n - a + be) - betaln(al, be)


class BetaBinomialRidge:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, a, n):
        X = np.column_stack([np.ones(len(X)), X])
        p0 = a.sum() / max(1, n.sum())
        x0 = np.zeros(X.shape[1] + 1)
        x0[0] = np.log(p0 / (1 - p0))
        x0[-1] = np.log(10.0)
        r = minimize(_nll, x0, args=(X, a, n, self.lam), method="L-BFGS-B")
        self.coef_ = r.x[:-1]
        self.phi_ = float(np.exp(r.x[-1]))
        return self

    def mu(self, X):
        return expit(np.column_stack([np.ones(len(X)), X]) @ self.coef_)

    def posterior_mean(self, X, a, n):
        m = self.mu(X)
        return (a + self.phi_ * m) / (n + self.phi_)

    def prob_below(self, X, a, n, thr):
        """사후 P(θ_i < thr) — 퇴출/수정 판단용."""
        m = self.mu(X)
        return beta_dist.cdf(thr, a + self.phi_ * m, n - a + self.phi_ * (1 - m))

    def interval(self, X, q=(0.1, 0.9)):
        """새 질문(관측 없음)의 진짜 답변율 예측 구간."""
        m = self.mu(X)
        return [beta_dist.ppf(qq, self.phi_ * m, self.phi_ * (1 - m)) for qq in q]


def kfold_eval(X, a, n, make, k=5, seed=0):
    """문항 단위 k겹 교차검증: 보지 않은 문항의 베타-이항 로그우도 합과 제곱오차."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(a))
    folds = np.array_split(idx, k)
    ll, sq, w, preds = 0.0, 0.0, 0.0, np.zeros(len(a))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        m = make().fit(X[tr], a[tr], n[tr])
        mu = m.mu(X[f])
        preds[f] = mu
        ll += bb_loglik(a[f], n[f], mu, m.phi_).sum()
        sq += np.sum(n[f] * ((a[f] / np.maximum(n[f], 1)) - mu) ** 2)
        w += n[f].sum()
    return dict(ll=float(ll), brier=float(sq / w), preds=preds)
